"""S3.5 ablation evaluation: per-variant mAP/recall + paired bootstrap CI.

Point estimates: full-set COCO segm mAP/AP50 + G1 recall proxy per variant
(registered protocol). Uncertainty: paired bootstrap (1,000 resamples of
images, seed 20260903) over the per-image AP statistic (micro over classes,
IoU 0.5 all-point AP on each image); CI is for A2-A0 / A2-A1 mean-delta of
that statistic, reported alongside the full-set mAP deltas. Images without
GT of any class contribute only false-positive mass and are skipped by the
per-image AP (undefined), as is conventional.

Run:  source sam3_ovis/scripts/load_environment.sh
      "$PYTHON" sam3_ovis/eval/run_ablation_eval.py --datasets NWPU SOTA FAST
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util  # noqa: E402

MAINLINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAINLINE))

from eval.score_eval import (  # noqa: E402
    compute_diagnostics, evaluate_dataset, load_gt_coco,
)
from data.score_datasets import DATASETS  # noqa: E402

VARIANTS = ["A0", "A1", "A2",
            "A3-competition", "A3-context", "A3-scene", "A3-risk"]
BOOT_N = 1000
BOOT_SEED = 20260903
MATCH_IOU = 0.5


def load_jsonl(path: Path) -> list:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _rle_arr(seg):
    from pycocotools import mask as mask_util

    r = dict(seg)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode("utf-8")
    return mask_util.decode(r)


def gt_masks_by_image(gt):
    out = defaultdict(list)
    for a in gt.dataset["annotations"]:
        seg = a["segmentation"]
        if isinstance(seg, dict):
            r = dict(seg)
            if isinstance(r["counts"], str):
                r["counts"] = r["counts"].encode("utf-8")
            m = mask_util.decode(r)
        else:
            im = gt.imgs[a["image_id"]]
            objs = seg if isinstance(seg[0], list) else [seg]
            rle = mask_util.merge(mask_util.frPyObjects(objs, im["height"],
                                                        im["width"]))
            m = mask_util.decode(rle)
        out[a["image_id"]].append((a["category_id"], m.astype(bool)))
    return out


IOU_THRESHOLDS = [round(0.5 + 0.05 * k, 2) for k in range(10)]


def per_image_maps(gt_masks, preds):
    """Per-image micro mAP@[.50:.95] over classes (all-point PR per
    threshold, averaged over the 10 COCO thresholds). Accepts a flat
    prediction list or a pre-grouped {image_id: [dets]}.

    Images WITH GT but empty/failed predictions score 0.0 (counted — a
    complete miss is exactly what completeness-sensitive comparisons must
    see). Images with no GT are skipped (undefined)."""
    if isinstance(preds, list):
        grouped = defaultdict(list)
        for p in preds:
            grouped[p["image_id"]].append(p)
        preds = grouped
    aps = {}
    for img_id, gt_list in gt_masks.items():
        scored = sorted(preds.get(img_id, []), key=lambda d: -d["score"])
        if not gt_list:
            continue
        if not scored:
            aps[img_id] = 0.0
            continue
        dmasks = [_rle_arr(d["segmentation"]).astype(bool) for d in scored]
        n_pos = len(gt_list)
        ap_t = []
        for thr in IOU_THRESHOLDS:
            matched = [False] * n_pos
            tp, fp = [], []
            for di, d in enumerate(scored):
                m = dmasks[di]
                best_iou, best_j = 0.0, -1
                for j, (cid, gm) in enumerate(gt_list):
                    if cid != d["category_id"] or matched[j]:
                        continue
                    inter = int((m & gm).sum())
                    if inter == 0:
                        continue
                    iou = inter / float((m | gm).sum())
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= thr:
                    matched[best_j] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
            ctp, cfp = np.cumsum(tp), np.cumsum(fp)
            recall = ctp / n_pos
            precision = ctp / np.maximum(ctp + cfp, 1)
            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([0.0], precision, [0.0]))
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap_t.append(float(np.sum((mrec[idx + 1] - mrec[idx])
                                     * mpre[idx + 1])))
        aps[img_id] = float(np.mean(ap_t))
    return aps


def paired_bootstrap(ap_by_variant: dict, img_ids, n=BOOT_N, seed=BOOT_SEED):
    """Mean-delta CI between A2 and each base over image resamples."""
    rng = np.random.default_rng(seed)
    out = {}
    for base in ("A0", "A1"):
        a2, b0 = ap_by_variant.get("A2", {}), ap_by_variant.get(base, {})
        common = [i for i in img_ids if i in a2 and i in b0]
        if not common:
            out[f"A2-{base}"] = {"mean_delta": None, "ci95": None,
                                 "n_images": 0}
            continue
        d = np.array([a2[i] - b0[i] for i in common])
        means = np.empty(n)
        for k in range(n):
            idx = rng.integers(0, len(d), size=len(d))
            means[k] = d[idx].mean()
        lo, hi = np.percentile(means, [2.5, 97.5])
        out[f"A2-{base}"] = {"mean_delta": round(float(d.mean()), 6),
                             "ci95": [round(float(lo), 6), round(float(hi), 6)],
                             "n_images": len(common)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["NWPU", "SOTA", "FAST"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from pycocotools import mask as mask_util  # noqa: F401

    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    log_root = Path(os.environ["SAM3_OVIS_LOG_ROOT"])
    table = {}
    for ds in args.datasets:
        spec = DATASETS[ds]
        gt = load_gt_coco(str(root / spec.val_json))
        gt_masks = gt_masks_by_image(gt)
        table[ds] = {}
        preds_by_variant = {}
        for v in VARIANTS:
            p = log_root / "ablation" / ds / f"{v}.jsonl"
            if not p.exists():
                print(f"[{ds}] {v}: MISSING")
                continue
            preds = load_jsonl(p)
            preds_by_variant[v] = preds
            summary, per_class = evaluate_dataset(gt, preds)
            diag = compute_diagnostics(gt, preds)
            table[ds][v] = {
                "mAP": round(summary["segm/mAP"], 4),
                "AP50": round(summary["segm/mAP_50"], 4),
                "recall_class_avg": round(diag["recall_proxy_class_avg"], 4),
                "recall_micro": round(diag["recall_proxy_micro"], 4),
                "n_preds": len(preds),
            }
            r = table[ds][v]
            print(f"[{ds}] {v:14s} mAP={r['mAP']:.4f} AP50={r['AP50']:.4f} "
                  f"G1={r['recall_class_avg']:.4f} "
                  f"micro={r['recall_micro']:.4f} n={r['n_preds']}")

        # paired bootstrap on per-image AP
        ap_by = {v: per_image_maps(gt_masks, preds_by_variant[v])
                 for v in preds_by_variant}
        table[ds]["bootstrap"] = paired_bootstrap(ap_by, list(gt_masks.keys()))
        for pair, st in table[ds]["bootstrap"].items():
            if st["mean_delta"] is not None:
                print(f"[{ds}] {pair}: per-image AP delta "
                      f"{st['mean_delta']:+.4f} (95% CI "
                      f"{st['ci95'][0]:+.4f}..{st['ci95'][1]:+.4f}, "
                      f"n={st['n_images']} images)")

    if args.out:
        Path(args.out).write_text(json.dumps(table, indent=1))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
