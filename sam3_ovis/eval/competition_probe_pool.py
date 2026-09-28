"""Competition-mechanism probe on REAL P0 candidates (pool-based, T1 suppl).

Same mechanism as competition_probe_source.py (design-v3.0 §3) but driven by
the P0 candidate pool of a TEST set where cross-class rival pairs actually
exist (NWPU: 476 wrong-class kept candidates; sports-field confusions).

For each kept candidate (class-wise NMS, raw-score order):
  rivals C_i = kept candidates of OTHER classes with mask IoU >= 0.5
  z_i(y)     = RemoteCLIP cos(crop_i, text y)
  d_i        = max(0, max_{y in C_i} z_i(y) - z_i(y_own) - delta)
  v_i        = u_i * exp(-lambda * d_i)
Measured vs GT: wrong-class candidates suppressed (desired) vs TP candidates
suppressed (cost), per (delta, lambda).

Run:  source sam3_ovis/scripts/load_environment.sh
      "$PYTHON" sam3_ovis/eval/competition_probe_pool.py --dataset NWPU
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402
from graph.config import THRESHOLDS, norm_text  # noqa: E402


def _arr(seg, h=None, w=None):
    if isinstance(seg, dict):
        r = dict(seg)
        if isinstance(r["counts"], str):
            r["counts"] = r["counts"].encode("utf-8")
        return mask_util.decode(r).astype(bool)
    objs = seg if isinstance(seg[0], list) else [seg]
    return mask_util.decode(mask_util.merge(
        mask_util.frPyObjects(objs, h, w))).astype(bool)


def _tight_bbox(m):
    ys, xs = np.any(m, axis=1), np.any(m, axis=0)
    if not ys.any():
        return None
    return (int(np.where(xs)[0][0]), int(np.where(ys)[0][0]),
            int(np.where(xs)[0][-1]) + 1, int(np.where(ys)[0][-1]) + 1)


def _iou_cropped(m1, b1, m2, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = int((m1[iy1:iy2, ix1:ix2] & m2[iy1:iy2, ix1:ix2]).sum())
    if inter == 0:
        return 0.0
    return inter / float(int(m1.sum()) + int(m2.sum()) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="NWPU")
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.0, 0.05])
    ap.add_argument("--lambdas", nargs="+", type=float,
                    default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from graph.provider import RemoteCLIPProvider
    from PIL import Image

    spec = DATASETS[args.dataset]
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    log_root = Path(os.environ["SAM3_OVIS_LOG_ROOT"])
    from pycocotools.coco import COCO

    gt = COCO(str(root / spec.val_json))
    h_w = {im["id"]: (im["height"], im["width"]) for im in gt.dataset["images"]}
    cat_names = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}

    # pool per (image, class), class-wise NMS
    pool = defaultdict(list)
    with open(log_root / "zeroshot" / args.dataset / "predictions.jsonl") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                pool[(d["image_id"], d["category_id"])].append(d)

    kept_by_pair = {}
    for (img_id, cat_id), dets in pool.items():
        h, w = h_w[img_id]
        order = sorted(dets, key=lambda d: -d["score"])
        kept, kept_m, kept_b = [], [], []
        for d in order:
            m = _arr(d["segmentation"], h, w)
            b = _tight_bbox(m)
            if b is None:
                continue
            dup = False
            for km, kb in zip(kept_m, kept_b):
                if _iou_cropped(m, b, km, kb) >= 0.5:
                    dup = True
                    break
            if not dup:
                d = dict(d)
                d["mask"], d["bbox"] = m, b
                kept.append(d)
                kept_m.append(m)
                kept_b.append(b)
        kept_by_pair[(img_id, cat_id)] = kept
    # NOTE crop protocol: candidate crops use bbox + 10% margin on BOTH
    # domains (aligned with competition_probe_source.py); the tight-bbox
    # crop used in the first NWPU probe run was a cross-domain protocol
    # mismatch flagged in the 2026-09-07 audit and is fixed here.

    # GT masks per (image, class)
    gt_masks = defaultdict(list)
    for a in gt.dataset["annotations"]:
        h, w = h_w[a["image_id"]]
        gt_masks[(a["image_id"], a["category_id"])].append(
            _arr(a["segmentation"], h, w))

    # per kept candidate: kind (tp/wrong) + rival classes (cross-class kept,
    # IoU >= 0.5)
    recs = []
    for (img_id, cat_id), kept in kept_by_pair.items():
        gts = gt_masks.get((img_id, cat_id), [])
        rivals_by = defaultdict(list)
        for (img2, cat2), kept2 in kept_by_pair.items():
            if img2 != img_id or cat2 == cat_id:
                continue
            for d2 in kept2:
                rivals_by[img_id].append((cat2, d2))
        for d in kept:
            m, b = d["mask"], d["bbox"]
            best_same = max((_iou_cropped(m, b, gm, _tight_bbox(gm))
                             for gm in gts), default=0.0)
            # cross-class GT overlap: the candidate DOES cover an object of
            # a different class (true wrong-class candidate per the
            # registered definition); without it the candidate is background
            best_cross = 0.0
            if best_same < 0.5:
                for (img2, cat2), gts2 in gt_masks.items():
                    if img2 != img_id:
                        continue
                    for gm in gts2:
                        if _iou_cropped(m, b, gm, _tight_bbox(gm)) > best_cross:
                            best_cross = _iou_cropped(m, b, gm, _tight_bbox(gm))
            if best_same >= 0.5:
                kind = "tp"
            elif best_cross >= 0.5:
                kind = "wrong"
            else:
                kind = "background"
            rivals = []
            for cat2, d2 in rivals_by[img_id]:
                m2, b2 = d2["mask"], d2["bbox"]
                if _iou_cropped(m, b, m2, b2) >= 0.5:
                    rivals.append(cat_names[cat2])
            if not rivals:
                continue
            recs.append({"kind": kind, "u": norm_text(d["score"]),
                         "cat": cat_names[cat_id], "img": img_id,
                         "bbox": list(b),
                         "rival_classes": sorted(set(rivals))})

    print(f"scored recs (with rivals): {len(recs)}")
    by_kind = defaultdict(lambda: [0, 0])
    for r in recs:
        by_kind[r["kind"]][1] += 1
    for k, (s, t) in by_kind.items():
        print(f"  {k}: {t}")

    # CLIP z per candidate per class (own + rivals)
    provider = RemoteCLIPProvider()
    img_cache = {}
    all_names = sorted({r["cat"] for r in recs} |
                       {c for r in recs for c in r["rival_classes"]})
    txt = provider.encode_texts_cached(all_names)

    for i, r in enumerate(recs):
        img_id = r["img"]
        if img_id not in img_cache:
            info = gt.loadImgs(img_id)[0]
            img_cache[img_id] = Image.open(
                root / spec.images_dir / info["file_name"]).convert("RGB")
        image = img_cache[img_id]
        x, y, x2, y2 = r["bbox"]
        mx, my = (x2 - x) * 0.1, (y2 - y) * 0.1
        cx1 = max(0, int(x - mx))
        cy1 = max(0, int(y - my))
        cx2 = min(image.width, int(x2 + mx))
        cy2 = min(image.height, int(y2 + my))
        crop = image.crop((cx1, cy1, cx2, cy2))
        key = f"{img_id}/{i}"
        f = provider.encode_images_cached([(key, crop)])[key]
        r["z_own"] = float(np.dot(f, txt[r["cat"]]))
        r["z_rival_max"] = max((float(np.dot(f, txt[c]))
                                for c in r["rival_classes"]), default=None)
        if i % 200 == 0:
            print(f"  CLIP {i}/{len(recs)}", flush=True)

    table = {}
    for delta in args.deltas:
        for lam in args.lambdas:
            s = defaultdict(lambda: [0, 0])
            for r in recs:
                if r["z_rival_max"] is None:
                    continue
                d_i = max(0.0, r["z_rival_max"] - r["z_own"] - delta)
                v = r["u"] * float(np.exp(-lam * d_i))
                s[r["kind"]][1] += 1
                s[r["kind"]][0] += int(v < r["u"])
            key = f"d{delta}_l{lam}"
            table[key] = {
                "wrong_suppressed": s["wrong"][0],
                "wrong_total": s["wrong"][1],
                "tp_suppressed": s["tp"][0],
                "tp_total": s["tp"][1],
                "background_suppressed": s["background"][0],
                "background_total": s["background"][1],
                "wrong_rate": round(s["wrong"][0] / max(s["wrong"][1], 1), 4),
                "tp_rate": round(s["tp"][0] / max(s["tp"][1], 1), 4),
                "background_rate": round(s["background"][0]
                                         / max(s["background"][1], 1), 4),
            }
            print(f"delta={delta} lambda={lam}: "
                  f"错类压低 {s['wrong'][0]}/{s['wrong'][1]}"
                  f"={table[key]['wrong_rate']:.3f} | "
                  f"TP误伤 {s['tp'][0]}/{s['tp'][1]}"
                  f"={table[key]['tp_rate']:.3f} | "
                  f"背景 {s['background'][0]}/{s['background'][1]}"
                  f"={table[key]['background_rate']:.3f}")

    out = Path(args.out or (log_root / "competition_probe_pool.json"))
    out.write_text(json.dumps({"dataset": args.dataset, "table": table,
                               "n_recs": len(recs)}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
