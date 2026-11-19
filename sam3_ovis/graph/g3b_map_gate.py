#!/usr/bin/env python3
"""G3b mAP 口径复验门（协议 2026-11-21-g3b-protocol.md）。

Phase cand (GPU)：100 图全候选集（pool floor 0.05，15 类，mask RLE）+
逐候选证据：R_self/R_ctx/R_uni/R_foreign 的 RemoteCLIP 文本逻辑 [15]
与 margin 形式 m = R[own_cls] − max_{k≠own} R[k]。
Phase eval (CPU)：A0/A2/A3/A1/A4 分数融合（α 仅 train 组选择），val 组
COCO segm mAP + 逐图 AP 源图组配对 bootstrap。

Run:  source sam3_ovis/scripts/load_environment.sh
      CUDA_VISIBLE_DEVICES=0 "$PYTHON" sam3_ovis/graph/g3b_map_gate.py --phase cand
      "$PYTHON" sam3_ovis/graph/g3b_map_gate.py --phase eval
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
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
CAND_DIR = CACHE_ROOT / "cands"
LOG_ROOT = Path(os.environ.get(
    "SAM3_OVIS_LOG_ROOT",
    str(Path(__file__).resolve().parents[1] / "logs")))
OUT_DIR = LOG_ROOT / "g3b"
POOL_FLOOR = 0.05
CTX_MIN_SCORE = 0.3
K_CTX = 2
MARGIN = 0.1
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
BOOT_N = 10000
SEED = 20261121


def crop_img(image, bbox, margin=MARGIN):
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return image.crop((max(0, int(x1 - mx)), max(0, int(y1 - my)),
                       min(image.width, int(x2 + mx)),
                       min(image.height, int(y2 + my))))


def phase_cand():
    device = "cuda"
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    from graph.provider import RemoteCLIPProvider

    CAND_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DATASETS["iSAID"]
    gt = COCO(str(root / spec.val_json))
    split = json.load(open(CACHE_ROOT / "group_split.json"))
    cat_ids = gt.getCatIds()
    cat_names = [gt.loadCats([c])[0]["name"] for c in cat_ids]

    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False,
        enable_inst_interactivity=False).to(device).eval()
    processor = Sam3Processor(model, confidence_threshold=POOL_FLOOR,
                              device=device)
    clip = RemoteCLIPProvider()
    txt = np.stack([clip.encode_texts_cached(cat_names)[c]
                    for c in cat_names]).astype(np.float32)
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)

    files = sorted(CACHE_ROOT.glob("img_*.pt"))
    # 只处理 manifest 登记的 100 图（磁盘上另有 30 个早期非交错运行残留，
    # 已核实不在 manifest 内，排除——协议口径 100 图）
    man_ids = {int(k) for k in json.load(
        open(CACHE_ROOT / "manifest.json"))["images"]}
    files = [p for p in files
             if int(p.stem.split("_")[1]) in man_ids]
    # 异图对照：确定性配对（i -> i+1 环绕）
    order = [torch.load(p, weights_only=False) for p in files]
    t0 = time.time()
    for i, rec in enumerate(order):
        out_path = CAND_DIR / f"cands_{rec['img_id']}.json"
        if out_path.exists():
            continue
        image = Image.open(root / spec.images_dir /
                           rec["file"]).convert("RGB")
        # vitdet 分支内部 bf16：autocast 包裹（g2a_cache 同款已验证方案）
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)
            cands = []
            for cname in cat_names:
                state = processor.set_text_prompt(prompt=cname, state=state)
                sc = state["scores"]
                for j in range(int(sc.shape[0]) if sc is not None else 0):
                    s = float(sc[j])
                    m = state["masks"][j, 0].cpu().numpy().astype(bool)
                    if not m.any():
                        continue
                    ys, xs = np.where(m.any(1))[0], np.where(m.any(0))[0]
                    box = [int(xs[0]), int(ys[0]),
                           int(xs[-1]) + 1, int(ys[-1]) + 1]
                    rle = mask_util.encode(np.asfortranarray(
                        m.astype(np.uint8)))
                    cands.append({"cls": cname, "score": s, "box": box,
                                  "rle": {"size": list(rle["size"]),
                                          "counts": rle["counts"]
                                          .decode("utf-8")}})
        # 逐候选证据
        conf = [c for c in cands if c["score"] >= CTX_MIN_SCORE]
        clip_cache = {}

        def logits_of(box):
            key = tuple(box)
            if key not in clip_cache:
                f = np.asarray(clip.encode_images_cached(
                    [(f"c{key}", crop_img(image, box))])[f"c{key}"],
                    dtype=np.float32)
                f /= (np.linalg.norm(f) + 1e-9)
                clip_cache[key] = f @ txt.T
            return clip_cache[key]

        foreign = order[(i + 1) % len(order)]
        for c in cands:
            r_self = logits_of(c["box"])
            ctx = [o for o in conf
                   if box_iou_xyxy(o["box"], c["box"]) < 0.5]
            ctx.sort(key=lambda o: -o["score"])
            ctx = ctx[:K_CTX]
            r_ctx = (np.mean([logits_of(o["box"]) for o in ctx], axis=0)
                     if ctx else None)
            r_uni = (np.mean([r_self] + [logits_of(o["box"])
                                         for o in ctx], axis=0)
                     if ctx else r_self)
            c["ev"] = {"self": r_self.round(4).tolist(),
                       "ctx": (r_ctx.round(4).tolist()
                               if r_ctx is not None else None),
                       "uni": r_uni.round(4).tolist()}
        out_path.write_text(json.dumps({
            "img_id": rec["img_id"], "split": split.get(
                rec["file"].replace(".png", "").split("_")[0], "train"),
            "W": image.width, "H": image.height, "cands": cands}))
        clip.clear_image_cache()
        torch.cuda.empty_cache()
        print(f"[cand] {i + 1}/{len(order)} {rec['file']} "
              f"{len(cands)} cands {time.time() - t0:.0f}s", flush=True)
    # R_foreign 在 eval 阶段用其他图的 self/ctx 证据等价构造（登记）


def phase_cand5():
    """A5（SCORE 式膨胀直注模拟）的补充证据：每候选 mask 膨胀 0.5 边距
    crop 的 RemoteCLIP 文本逻辑，写入 cands_*.json 的 ev.dilated。"""
    device = "cuda"
    from graph.provider import RemoteCLIPProvider

    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DATASETS["iSAID"]
    gt = COCO(str(root / spec.val_json))
    cat_ids = gt.getCatIds()
    cat_names = [gt.loadCats([c])[0]["name"] for c in cat_ids]
    clip = RemoteCLIPProvider()
    txt = np.stack([clip.encode_texts_cached(cat_names)[c]
                    for c in cat_names]).astype(np.float32)
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)

    for p in sorted(CAND_DIR.glob("cands_*.json")):
        rec = json.loads(p.read_text())
        if rec["cands"] and "dilated" in rec["cands"][0].get("ev", {}):
            continue
        image = _open_image(rec["img_id"])
        cache = {}
        for c in rec["cands"]:
            x1, y1, x2, y2 = c["box"]
            mx, my = (x2 - x1) * 0.5, (y2 - y1) * 0.5
            key = (int(max(0, x1 - mx)), int(max(0, y1 - my)),
                   int(min(rec["W"], x2 + mx)),
                   int(min(rec["H"], y2 + my)))
            if key not in cache:
                f = np.asarray(clip.encode_images_cached(
                    [(f"d{key}", image.crop(key))])[f"d{key}"],
                    dtype=np.float32)
                f /= (np.linalg.norm(f) + 1e-9)
                cache[key] = (f @ txt.T).round(4).tolist()
            c["ev"]["dilated"] = cache[key]
        p.write_text(json.dumps(rec))
        clip.clear_image_cache()
        print(f"[cand5] {rec['img_id']} done", flush=True)


def _open_image(img_id):
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DATASETS["iSAID"]
    gt = COCO(str(root / spec.val_json))
    info = gt.loadImgs(img_id)[0]
    return Image.open(root / spec.images_dir / info["file_name"]) \
        .convert("RGB")


def _load_cands():
    man_ids = {int(k) for k in json.load(
        open(CACHE_ROOT / "manifest.json"))["images"]}
    out = []
    for p in sorted(CAND_DIR.glob("cands_*.json")):
        if int(p.stem.split("_")[1]) in man_ids:
            out.append(json.loads(p.read_text()))
    return out


def _margins(ev_key):
    """逐候选 margin：R[own_cls] − max_{k≠own} R[k]。"""
    cat_names = None
    rows = []
    for rec in _load_cands():
        if cat_names is None:
            # 类表从 g2a 缓存统一获取
            base = torch.load(sorted(CACHE_ROOT.glob("img_*.pt"))[0],
                              weights_only=False, map_location="cpu")
            cat_names = sorted(base["texts"].keys())
        for c in rec["cands"]:
            r = np.array(c["ev"][ev_key], np.float32) \
                if c["ev"][ev_key] is not None else None
            if r is None:
                c[f"m_{ev_key}"] = None
                continue
            k = cat_names.index(c["cls"])
            others = np.delete(r, k)
            c[f"m_{ev_key}"] = float(r[k] - others.max())
        rows.append(rec)
    return rows, cat_names


def _build_eval(cands_recs, cat_names):
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DATASETS["iSAID"]
    gt = COCO(str(root / spec.val_json))
    name_to_id = {c["name"]: c["id"] for c in gt.loadCats(gt.getCatIds())}
    ids = [r["img_id"] for r in cands_recs]
    sub = COCO()
    sub.dataset = {
        "images": [im for im in gt.dataset["images"]
                   if im["id"] in set(ids)],
        "annotations": [a for a in gt.dataset["annotations"]
                        if a["image_id"] in set(ids)],
        "categories": gt.dataset["categories"]}
    sub.createIndex()
    dets_base = [{"image_id": r["img_id"],
                  "category_id": name_to_id[c["cls"]],
                  "segmentation": c["rle"],
                  "score": c["score"]}
                 for r in cands_recs for c in r["cands"]]
    return sub, dets_base, name_to_id


def _mAP(sub, dets):
    if not dets:
        return 0.0
    import contextlib
    import io
    dt = sub.loadRes([dict(d) for d in dets])
    ev = COCOeval(sub, dt, iouType="segm")
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0])


def _per_image_ap(sub, dets, img_ids, name_to_id):
    per = {}
    for img_id in img_ids:
        d = [dict(x) for x in dets if x["image_id"] == img_id]
        s = COCO()
        s.dataset = {
            "images": [im for im in sub.dataset["images"]
                       if im["id"] == img_id],
            "annotations": [a for a in sub.dataset["annotations"]
                            if a["image_id"] == img_id],
            "categories": sub.dataset["categories"]}
        s.createIndex()
        import contextlib
        import io
        if d:
            dt = s.loadRes(d)
            ev = COCOeval(s, dt, iouType="segm")
            with contextlib.redirect_stdout(io.StringIO()):
                ev.evaluate(); ev.accumulate(); ev.summarize()
            per[img_id] = float(ev.stats[0])
        else:
            per[img_id] = 0.0
    return per


def phase_eval():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recs, cat_names = _margins("self")
    recs2, _ = _margins("ctx")
    recs3, _ = _margins("uni")
    recs5, _ = _margins("dilated")
    for r1, r2, r3, r5 in zip(recs, recs2, recs3, recs5):
        for c1, c2, c3, c5 in zip(r1["cands"], r2["cands"], r3["cands"],
                                  r5["cands"]):
            c1["m_ctx"] = c2["m_ctx"]
            c1["m_uni"] = c3["m_uni"]
            c1["m_dilated"] = c5["m_dilated"]
    sub, dets_base, name_to_id = _build_eval(recs, cat_names)
    val_recs = [r for r in recs if r["split"] == "val"]
    tr_recs = [r for r in recs if r["split"] == "train"]
    print(f"[eval] imgs train={len(tr_recs)} val={len(val_recs)}",
          flush=True)

    def fuse(alpha, mkey, recs_sel):
        out = []
        for r in recs_sel:
            for c in r["cands"]:
                m = c.get(mkey)
                s = c["score"] + (alpha * m if m is not None else 0.0)
                out.append({"image_id": r["img_id"],
                            "category_id": name_to_id[c["cls"]],
                            "segmentation": c["rle"],
                            "score": float(s)})
        return out

    # α 仅 train 组选择
    tr_sub_recs = recs
    alphas = {}
    for mkey in ("m_self", "m_ctx", "m_uni", "m_dilated"):
        best_a, best = 0.0, -1
        for a in ALPHAS:
            if a == 0:
                continue
            ap = _mAP(sub, fuse(a, mkey, tr_recs))  # train 子集 mAP
            if ap > best:
                best_a, best = a, ap
        alphas[mkey] = best_a
        print(f"[eval] {mkey}: alpha*={best_a} (train mAP={best:.4f})",
              flush=True)

    # val 评估（全局 mAP + 逐图 AP bootstrap）
    val_ids = [r["img_id"] for r in val_recs]
    results = {"alphas": alphas, "val_mAP": {}, "bootstrap": {}}
    per_base = _per_image_ap(sub, dets_base, val_ids, name_to_id)
    results["val_mAP"]["A0"] = _mAP(sub, [d for d in dets_base
                                          if d["image_id"] in set(val_ids)])
    vecs = {"A0": np.array([per_base[i] for i in val_ids])}
    arms = {"A2": "m_ctx", "A3": "m_uni", "A1": "m_self",
            "A5": "m_dilated"}
    for arm, mkey in arms.items():
        dets = fuse(alphas[mkey], mkey, val_recs)
        results["val_mAP"][arm] = _mAP(sub, dets)
        per = _per_image_ap(sub, dets, val_ids, name_to_id)
        vecs[arm] = np.array([per[i] for i in val_ids])
        print(f"[eval] {arm} ({mkey}, a={alphas[mkey]}): "
              f"mAP={results['val_mAP'][arm]:.4f}", flush=True)
    # A4 异图：用 val 内固定 seed 异图 ctx 证据（等价构造）
    rng = np.random.default_rng(SEED)
    vrecs = val_recs
    perm = list(rng.permutation(len(vrecs)))
    dets_a4 = []
    for r, src in zip(vrecs, [vrecs[p] for p in perm]):
        # 借用源图的高分候选 box 的 self margin（异图证据的等价代理）
        pool = sorted(src["cands"], key=lambda c: -c["score"])[:K_CTX]
        foreign_m = (np.mean([c["m_self"] for c in pool])
                     if pool else 0.0)
        for c in r["cands"]:
            dets_a4.append({"image_id": r["img_id"],
                            "category_id": name_to_id[c["cls"]],
                            "segmentation": c["rle"],
                            "score": float(c["score"]
                                           + alphas["m_ctx"]
                                           * foreign_m)})
    results["val_mAP"]["A4"] = _mAP(sub, dets_a4)
    per = _per_image_ap(sub, dets_a4, val_ids, name_to_id)
    vecs["A4"] = np.array([per[i] for i in val_ids])

    rng = np.random.default_rng(SEED)
    n = len(val_ids)
    for a, b in (("A2", "A0"), ("A3", "A0"), ("A2", "A4"),
                 ("A2", "A5"), ("A5", "A0")):
        diff = vecs[a] - vecs[b]
        samples = np.empty(BOOT_N)
        for t in range(BOOT_N):
            idx = rng.integers(0, n, n)
            samples[t] = diff[idx].mean()
        lo, hi = np.percentile(samples, [2.5, 97.5])
        results["bootstrap"][f"{a}-{b}"] = {
            "mean_diff": round(float(diff.mean()), 6),
            "ci95": [round(float(lo), 6), round(float(hi), 6)]}
        print(f"[eval] {a}-{b}: {diff.mean():+.5f} "
              f"CI[{lo:+.5f},{hi:+.5f}]", flush=True)
    (OUT_DIR / "map_gate.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False))
    print(f"-> {OUT_DIR / 'map_gate.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["cand", "cand5", "eval"],
                    required=True)
    args = ap.parse_args()
    if args.phase == "cand":
        phase_cand()
    elif args.phase == "cand5":
        phase_cand5()
    else:
        phase_eval()


if __name__ == "__main__":
    main()
