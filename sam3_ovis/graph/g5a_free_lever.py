#!/usr/bin/env python3
"""G5a 免费杠杆消融（协议 2026-11-22-g5a-protocol.md）。

五臂（B0/L1/L2/L3/COMBO）× 46 val 图：逐类首遍候选生成（prompt 变体 /
floor / 分辨率）→ 落盘 → 类内 NMS → mAP + 双口径召回 + FP/图 + 配对
bootstrap。断点续跑：臂内逐图落盘，重跑跳过已有。

Run:  source sam3_ovis/scripts/load_environment.sh
      CUDA_VISIBLE_DEVICES=0 "$PYTHON" sam3_ovis/graph/g5a_free_lever.py \
          [--arms B0 L1 L2 L3 COMBO] [--eval-only]
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402
from graph.g1_qbridge_probe import box_iou_xyxy

CACHE_ROOT = Path("/data/wangcheng/geoevi-g2a-cache")
OUT_ROOT = Path("/data/wangcheng/geoevi-r1/freeler")
LOG_ROOT = Path(os.environ.get(
    "SAM3_OVIS_LOG_ROOT",
    str(Path(__file__).resolve().parents[1] / "logs")))
RES_DIR = LOG_ROOT / "g5a"
NMS_IOU = 0.5
BOOT_N = 2000
SEED = 20261122

ARMS = {
    "B0": {"prompt": "raw", "floor": 0.05, "res": 1008},
    "L1": {"prompt": "space", "floor": 0.05, "res": 1008},
    "L2": {"prompt": "raw", "floor": 0.01, "res": 1008},
    "L3": {"prompt": "raw", "floor": 0.05, "res": 1344},
    "COMBO": {"prompt": "space", "floor": 0.01, "res": 1344},
}


def prompt_of(cname, style):
    return cname.replace("_", " ") if style == "space" else cname


def nms_image(cands, key="score"):
    kept = []
    for cls in {c["cls"] for c in cands}:
        group = sorted([c for c in cands if c["cls"] == cls],
                       key=lambda c: -c[key])
        for c in group:
            if all(box_iou_xyxy(c["box"], k["box"]) < NMS_IOU
                   for k in kept if k["cls"] == cls):
                kept.append(c)
    return kept


def gen_arm(arm, cfg, gt, spec, root, model_root_device):
    device = "cuda"
    from sam3.model.sam3_image_processor import Sam3Processor
    from graph.g4b_gpu_probes import set_grid
    # 全局块 RoPE 按 1008/14=72 缓存；1344 需重建（窗口块不受影响，
    # 96 可被 24 整除——g4b 审计已验证该方案）
    set_grid(model_root_device, cfg["res"] // 14, device)
    processor = Sam3Processor(model_root_device, device=device,
                              resolution=cfg["res"],
                              confidence_threshold=cfg["floor"])
    man = json.load(open(CACHE_ROOT / "manifest.json"))
    split = json.load(open(CACHE_ROOT / "group_split.json"))
    val_groups = {g for g, s in split.items() if s == "val"}
    imgs = [im for im in gt.dataset["images"]
            if im["file_name"].replace(".png", "").split("_")[0]
            in val_groups]
    cat_ids = gt.getCatIds()
    cat_names = [gt.loadCats([c])[0]["name"] for c in cat_ids]

    arm_dir = OUT_ROOT / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "meta.json").write_text(json.dumps(
        {"arm": arm, "cfg": cfg, "n_imgs": len(imgs)}, indent=1))
    t0 = time.time()
    n_done = 0
    for im in sorted(imgs, key=lambda x: x["id"]):
        out_path = arm_dir / f"cands_{im['id']}.json"
        if out_path.exists():
            continue
        image = Image.open(root / spec.images_dir /
                           im["file_name"]).convert("RGB")
        W, H = image.size
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)
            cands = []
            for cname in cat_names:
                state = processor.set_text_prompt(
                    prompt=prompt_of(cname, cfg["prompt"]), state=state)
                sc = state["scores"]
                for j in range(int(sc.shape[0])
                               if sc is not None else 0):
                    s = float(sc[j])
                    if s < cfg["floor"]:
                        continue
                    m = state["masks"][j, 0].cpu().numpy().astype(bool)
                    if not m.any():
                        continue
                    ys, xs = np.where(m.any(1))[0], np.where(m.any(0))[0]
                    rle = mask_util.encode(np.asfortranarray(
                        m.astype(np.uint8)))
                    cands.append({
                        "cls": cname, "score": s,
                        "box": [int(xs[0]), int(ys[0]),
                                int(xs[-1]) + 1, int(ys[-1]) + 1],
                        "rle": {"size": list(rle["size"]),
                                "counts": rle["counts"]
                                .decode("utf-8")}})
        out_path.write_text(json.dumps({
            "img_id": im["id"], "W": W, "H": H, "cands": cands}))
        n_done += 1
        if n_done % 10 == 0:
            print(f"[{arm}] {n_done} imgs {time.time() - t0:.0f}s",
                  flush=True)
        torch.cuda.empty_cache()
    print(f"[{arm}] gen done ({len(imgs)} imgs)", flush=True)


def eval_arms(gt, name_to_id):
    id_to_name = {v: k for k, v in name_to_id.items()}
    arm_recs = {}
    for arm in sorted(OUT_ROOT.glob("*")):
        if not arm.is_dir():
            continue
        recs = []
        for p in sorted(arm.glob("cands_*.json")):
            recs.append(json.loads(p.read_text()))
        arm_recs[arm.name] = recs

    gt_by_img = {}
    val_ids = set()
    for r in arm_recs[list(arm_recs)[0]]:
        val_ids.add(r["img_id"])
    for a in gt.dataset["annotations"]:
        if a["image_id"] in val_ids:
            gt_by_img.setdefault(a["image_id"], []).append(a)
    val_ids = sorted(val_ids)

    sub = COCO()
    sub.dataset = {
        "images": [im for im in gt.dataset["images"]
                   if im["id"] in set(val_ids)],
        "annotations": [a for a in gt.dataset["annotations"]
                        if a["image_id"] in set(val_ids)],
        "categories": gt.dataset["categories"]}
    sub.createIndex()

    def mAP(dets):
        if not dets:
            return 0.0
        dt = sub.loadRes([dict(d) for d in dets])
        ev = COCOeval(sub, dt, iouType="segm")
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate(); ev.accumulate(); ev.summarize()
        return float(ev.stats[0])

    def per_image_mAP(dets):
        out = {}
        for img_id in val_ids:
            d = [dict(x) for x in dets if x["image_id"] == img_id]
            s = COCO()
            s.dataset = {
                "images": [im for im in sub.dataset["images"]
                           if im["id"] == img_id],
                "annotations": [a for a in sub.dataset["annotations"]
                                if a["image_id"] == img_id],
                "categories": sub.dataset["categories"]}
            s.createIndex()
            if d:
                dt = s.loadRes(d)
                ev = COCOeval(s, dt, iouType="segm")
                with contextlib.redirect_stdout(io.StringIO()):
                    ev.evaluate(); ev.accumulate(); ev.summarize()
                out[img_id] = float(ev.stats[0])
            else:
                out[img_id] = 0.0
        return out

    results = {}
    per_store = {}
    for arm, recs in arm_recs.items():
        dets, n_cand, n_fp = [], 0, 0
        rec_bbox = rec_ctr = rec_same = 0
        n_gt = 0
        for r in recs:
            kept = nms_image(r["cands"])
            n_cand += len(r["cands"])
            for c in kept:
                dets.append({"image_id": r["img_id"],
                             "category_id": name_to_id[c["cls"]],
                             "segmentation": c["rle"],
                             "score": c["score"]})
            gts = gt_by_img.get(r["img_id"], [])
            n_gt += len(gts)
            for a in gts:
                gx, gy, gw, gh = a["bbox"]
                gbox = (gx, gy, gx + gw, gy + gh)
                cx, cy = gx + gw / 2, gy + gh / 2
                best_iou, best_j = 0.0, -1
                hit_ctr = False
                same = False
                for j, c in enumerate(kept):
                    iou = box_iou_xyxy(c["box"], gbox)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= 0.5:
                    rec_bbox += 1
                    same = kept[best_j]["cls"] == id_to_name[
                        a["category_id"]]
                for c in kept:
                    m = mask_util.decode({
                        "size": c["rle"]["size"],
                        "counts": c["rle"]["counts"].encode()})
                    if m[int(cy), int(cx)]:
                        hit_ctr = True
                        if c["cls"] == id_to_name[a["category_id"]]:
                            same = True
                        break
                if hit_ctr:
                    rec_ctr += 1
                if same:
                    rec_same += 1
            n_fp += sum(1 for c in kept if not any(
                box_iou_xyxy(c["box"], (a["bbox"][0], a["bbox"][1],
                                        a["bbox"][0] + a["bbox"][2],
                                        a["bbox"][1] + a["bbox"][3])) >= 0.5
                for a in gts))
        results[arm] = {
            "mAP": round(mAP(dets), 4),
            "recall_bbox_pct": round(100 * rec_bbox / n_gt, 2),
            "recall_center_pct": round(100 * rec_ctr / n_gt, 2),
            "recall_same_class_pct": round(100 * rec_same / n_gt, 2),
            "fp_per_img": round(n_fp / len(recs), 1),
            "raw_cands_per_img": round(n_cand / len(recs), 1),
            "n_gt": n_gt}
        per_store[arm] = per_image_mAP(dets)
        print(f"[{arm}] {results[arm]}", flush=True)

    if "B0" in per_store:
        rng = np.random.default_rng(SEED)
        n = len(val_ids)
        for arm in per_store:
            if arm == "B0":
                continue
            d = np.array([per_store[arm][i] - per_store["B0"][i]
                          for i in val_ids])
            samples = np.array([d[rng.integers(0, n, n)].mean()
                                for _ in range(BOOT_N)])
            lo, hi = np.percentile(samples, [2.5, 97.5])
            results[arm]["vs_B0"] = {
                "delta_map_pt": round(100 * float(d.mean()), 2),
                "ci95_pt": [round(100 * float(lo), 2),
                            round(100 * float(hi), 2)]}
            print(f"[{arm}] mAP Δ={100 * d.mean():+.2f}pt "
                  f"CI[{100 * lo:+.2f},{100 * hi:+.2f}]", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / "ablation.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False))
    print(f"-> {RES_DIR / 'ablation.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DATASETS["iSAID"]
    gt = COCO(str(root / spec.val_json))
    name_to_id = {c["name"]: c["id"]
                  for c in gt.loadCats(gt.getCatIds())}

    if not args.eval_only:
        from sam3.model_builder import build_sam3_image_model
        model = build_sam3_image_model(
            checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False,
            enable_inst_interactivity=False).cuda().eval()
        for arm in args.arms:
            gen_arm(arm, ARMS[arm], gt, spec, root, model)
        del model
        torch.cuda.empty_cache()
    eval_arms(gt, name_to_id)


if __name__ == "__main__":
    main()
