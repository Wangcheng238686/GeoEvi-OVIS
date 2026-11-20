#!/usr/bin/env python3
"""G4a 结论可信度审计（协议 2026-11-22-g4a-protocol.md）。

A：G3b 六臂 + G3c 适配器在类内 NMS 池上重评 mAP（v3 口径）。
B：G3c 适配器逐候选 Δscore 验尸（TP 类/错类/FP 分账）。
C：NMS 池损失分解（召回/FP 比/oracle 换类上限/oracle 分数上限）。

Run:  source sam3_ovis/scripts/load_environment.sh
      CUDA_VISIBLE_DEVICES=0 "$PYTHON" sam3_ovis/graph/g4a_audit.py
"""

import contextlib
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402
from graph.g1_qbridge_probe import box_iou_xyxy

CACHE_ROOT = Path("/data/wangcheng/geoevi-g2a-cache")
CAND_DIR = CACHE_ROOT / "cands"
LOG_ROOT = Path(os.environ.get(
    "SAM3_OVIS_LOG_ROOT",
    str(Path(__file__).resolve().parents[1] / "logs")))
OUT_DIR = LOG_ROOT / "g4a"
NMS_IOU = 0.5
CAT_NAMES = None  # 延迟初始化


def load_cands():
    man_ids = {int(k) for k in json.load(
        open(CACHE_ROOT / "manifest.json"))["images"]}
    out = []
    for p in sorted(CAND_DIR.glob("cands_*.json")):
        rec = json.loads(p.read_text())
        if rec["img_id"] in man_ids:
            out.append(rec)
    return out


def nms_image(cands, key="score"):
    """类内 NMS（v3 口径：逐类、分数降序贪心、mask IoU>=0.5 抑制）。
    用 bbox IoU 近似（登记：与 v3 mask-IoU 的差异，同一近似用于全部臂，
    比较有效性不受影响）。"""
    kept = []
    for cls in {c["cls"] for c in cands}:
        group = sorted([c for c in cands if c["cls"] == cls],
                       key=lambda c: -c[key])
        for c in group:
            if all(box_iou_xyxy(c["box"], k["box"]) < NMS_IOU
                   for k in kept if k["cls"] == cls):
                kept.append(c)
    return kept


def eval_mAP(gt, dets, img_ids):
    sub = COCO()
    sub.dataset = {
        "images": [im for im in gt.dataset["images"]
                   if im["id"] in set(img_ids)],
        "annotations": [a for a in gt.dataset["annotations"]
                        if a["image_id"] in set(img_ids)],
        "categories": gt.dataset["categories"]}
    sub.createIndex()
    if not dets:
        return 0.0
    dt = sub.loadRes([dict(d) for d in dets])
    ev = COCOeval(sub, dt, iouType="segm")
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0])


def per_image_mAP(gt, dets, img_ids):
    out = {}
    for img_id in img_ids:
        out[img_id] = eval_mAP(
            gt, [d for d in dets if d["image_id"] == img_id], [img_id])
    return out


def main():
    global CAT_NAMES
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DATASETS["iSAID"]
    gt = COCO(str(root / spec.val_json))
    name_to_id = {c["name"]: c["id"]
                  for c in gt.loadCats(gt.getCatIds())}
    id_to_name = {v: k for k, v in name_to_id.items()}
    base = torch.load(sorted(CACHE_ROOT.glob("img_*.pt"))[0],
                      weights_only=False, map_location="cpu")
    CAT_NAMES = sorted(base["texts"].keys())

    recs = load_cands()
    val_recs = [r for r in recs if r["split"] == "val"]
    val_ids = [r["img_id"] for r in val_recs]
    gt_by_img = {}
    for a in gt.dataset["annotations"]:
        if a["image_id"] in set(val_ids):
            gt_by_img.setdefault(a["image_id"], []).append(a)
    print(f"[g4a] val imgs={len(val_recs)}", flush=True)

    # ---------------- A：NMS 后重评 ----------------
    mg = json.load(open(LOG_ROOT / "g3b" / "map_gate.json"))
    alphas = mg["alphas"]

    def margin(c, ev_key):
        r = np.array(c["ev"][ev_key], np.float32) \
            if c["ev"][ev_key] is not None else None
        if r is None:
            return 0.0
        k = CAT_NAMES.index(c["cls"])
        return float(r[k] - np.delete(r, k).max())

    def dets_of(recs_sel, key):
        out = []
        for r in recs_sel:
            tmp = [dict(c) for c in r["cands"]]
            for c in tmp:
                c["_s"] = c[key]
            for c in nms_image(tmp, key="_s"):
                out.append({"image_id": r["img_id"],
                            "category_id": name_to_id[c["cls"]],
                            "segmentation": c["rle"],
                            "score": c["_s"]})
        return out

    # 先给每个候选挂上各臂分数（NMS 前算好，臂内共用）
    for r in val_recs:
        for c in r["cands"]:
            c["_sA0"] = c["score"]
            c["_sA1"] = c["score"] + alphas["m_self"] * margin(c, "self")
            c["_sA2"] = c["score"] + alphas["m_ctx"] * margin(c, "ctx")
            c["_sA5"] = c["score"] + alphas["m_dilated"] * margin(
                c, "dilated")
    arms = {"A0": "_sA0", "A1": "_sA1", "A2": "_sA2", "A5": "_sA5"}

    # G3c 适配器分数（GPU，seed 0；并留 Δscore 供 B 验尸）
    device = "cuda"
    from sam3.model_builder import build_sam3_image_model
    from graph.g2a_train import maskpool
    from graph.g3c_adapter_confirm import build_modules, delta_null
    from graph.g2a_train import load_cache as load_g2a

    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False,
        enable_inst_interactivity=True).cuda().eval()
    trk = model.inst_interactive_predictor.model
    reader, writer, dps = (trk.transformer.encoder,
                           trk.maskmem_backbone, model.dot_prod_scoring)
    for p in list(reader.parameters()) + list(writer.parameters()):
        p.requires_grad_(False)
    for p in dps.parameters():
        p.requires_grad_(False)
    reader, writer, dps = reader.cuda(), writer.cuda(), dps.cuda().float()
    del model
    torch.cuda.empty_cache()
    ck = torch.load(LOG_ROOT / "g3c" / "ckpt_s0.pt", weights_only=True)
    from torch import nn
    bridge = nn.Linear(256, 256, bias=False).cuda()
    bridge.load_state_dict(ck["bridge"])
    reader.load_state_dict(ck["reader"])

    imgs = load_g2a()
    by_id = {im["img_id"]: im for im in imgs}
    deltas = {}
    with torch.no_grad():
        for im in imgs:
            if im["split"] == "val":
                deltas[im["img_id"]] = delta_null(reader, im)
    n_shift = {"tp": [], "wrong": [], "fp": []}
    with torch.no_grad():
        for r in val_recs:
            img = by_id[r["img_id"]]
            grid = tuple(img["grid"])
            delta = deltas[r["img_id"]]
            gts = gt_by_img.get(r["img_id"], [])
            for c in r["cands"]:
                tr = img["texts"][c["cls"]]
                bpx = tr["boxes_px"].float()
                bj, bi = -1, 0.0
                for j in range(bpx.shape[0]):
                    iou = box_iou_xyxy(c["box"], bpx[j].tolist())
                    if iou > bi:
                        bj, bi = j, iou
                c["_sAD"] = c["score"]
                if bi >= 0.5:
                    m = mask_util.decode({
                        "size": c["rle"]["size"],
                        "counts": c["rle"]["counts"].encode()}) \
                        .astype(bool)
                    pv = maskpool(delta, m, grid)
                    dq = bridge(pv)
                    q = tr["q32"]
                    p_eff = dps.prompt_mlp(tr["p32"]) \
                        if getattr(dps, "prompt_mlp", None) is not None \
                        else tr["p32"]
                    proj_p = dps.prompt_proj(
                        dps.mean_pool_text(p_eff, tr["pm"]))[0]
                    s_new = float((dps.hs_proj(q[bj] + dq)
                                   .matmul(proj_p) * dps.scale)
                                  .clamp(-12, 12))
                    c["_sAD"] = s_new
                    # B 验尸：GT 匹配分账
                    best_iou, best_cat = 0.0, None
                    for a in gts:
                        gx, gy, gw, gh = a["bbox"]
                        iou = box_iou_xyxy(
                            c["box"], (gx, gy, gx + gw, gy + gh))
                        if iou > best_iou:
                            best_iou, best_cat = iou, a["category_id"]
                    if best_iou >= 0.5:
                        if best_cat == name_to_id[c["cls"]]:
                            n_shift["tp"].append(s_new - c["score"])
                        else:
                            n_shift["wrong"].append(s_new - c["score"])
                    else:
                        n_shift["fp"].append(s_new - c["score"])
    arms["AD(G3c-s0)"] = "_sAD"

    results = {"A_nms_mAP": {}, "B_autopsy": {}, "C_decomp": {}}
    per_imgs = {}
    for arm, key in arms.items():
        dets = dets_of(val_recs, key)
        m = eval_mAP(gt, dets, val_ids)
        results["A_nms_mAP"][arm] = round(m, 4)
        per_imgs[arm] = per_image_mAP(gt, dets, val_ids)
        print(f"[A/NMS] {arm}: mAP={m:.4f}", flush=True)
    rng = np.random.default_rng(20261122)
    n = len(val_ids)
    for a in ("A2", "A1", "A5", "AD(G3c-s0)"):
        d = np.array([per_imgs[a][i] - per_imgs["A0"][i] for i in val_ids])
        samples = np.array([d[rng.integers(0, n, n)].mean()
                            for _ in range(2000)])
        lo, hi = np.percentile(samples, [2.5, 97.5])
        results["A_nms_mAP"][f"{a}-A0"] = {
            "delta_pt": round(100 * float(d.mean()), 2),
            "ci95": [round(float(lo) * 100, 3), round(float(hi) * 100, 3)]}
        print(f"[A/NMS] {a}-A0: {100 * d.mean():+.2f}pt "
              f"CI[{100 * lo:+.2f},{100 * hi:+.2f}]", flush=True)

    # ---------------- B：验尸 ----------------
    for k, v in n_shift.items():
        v = np.array(v)
        results["B_autopsy"][k] = {
            "n": len(v),
            "mean_shift": round(float(v.mean()), 4),
            "median_shift": round(float(np.median(v)), 4),
            "frac_down": round(float((v < 0).mean()), 3)}
    print(f"[B] {results['B_autopsy']}", flush=True)

    # ---------------- C：损失分解（NMS 池，A0 分数）----------------
    n_gt, hit_same, hit_any, n_fp = 0, 0, 0, 0
    for r in val_recs:
        gts = gt_by_img.get(r["img_id"], [])
        kept = nms_image(r["cands"])
        used = set()
        for a in gts:
            n_gt += 1
            gx, gy, gw, gh = a["bbox"]
            best_j, best_iou = -1, 0.0
            for j, c in enumerate(kept):
                iou = box_iou_xyxy(c["box"], (gx, gy, gx + gw, gy + gh))
                if iou > best_iou:
                    best_j, best_iou = j, iou
            if best_iou >= 0.5:
                hit_any += 1
                if kept[best_j]["cls"] == id_to_name[a["category_id"]]:
                    hit_same += 1
        n_fp += sum(1 for c in kept
                    if not any(box_iou_xyxy(
                        c["box"], (a["bbox"][0], a["bbox"][1],
                                   a["bbox"][0] + a["bbox"][2],
                                   a["bbox"][1] + a["bbox"][3])) >= 0.5
                        for a in gts))
    # oracle 换类：每个保留候选换成其最佳 GT 的类，分数不变
    dets_orc = []
    for r in val_recs:
        gts = gt_by_img.get(r["img_id"], [])
        for c in nms_image(r["cands"]):
            best_iou, best_cat = 0.0, c["cls"]
            for a in gts:
                gx, gy, gw, gh = a["bbox"]
                iou = box_iou_xyxy(c["box"], (gx, gy, gx + gw, gy + gh))
                if iou > best_iou:
                    best_iou, best_cat = iou, a["category_id"]
            cat = best_cat if best_iou >= 0.5 \
                else name_to_id[c["cls"]]
            dets_orc.append({"image_id": r["img_id"],
                             "category_id": cat,
                             "segmentation": c["rle"],
                             "score": c["_sA0"]})
    results["C_decomp"] = {
        "n_gt": n_gt,
        "recall_any_class_pct": round(100 * hit_any / n_gt, 2),
        "recall_same_class_pct": round(100 * hit_same / n_gt, 2),
        "wrong_class_gap_pt": round(100 * (hit_any - hit_same) / n_gt, 2),
        "n_fp_per_img": round(n_fp / len(val_recs), 1),
        "oracle_reclass_mAP": round(
            eval_mAP(gt, dets_orc, val_ids), 4)}
    print(f"[C] {results['C_decomp']}", flush=True)

    (OUT_DIR / "audit.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False))
    print(f"-> {OUT_DIR / 'audit.json'}", flush=True)


if __name__ == "__main__":
    main()
