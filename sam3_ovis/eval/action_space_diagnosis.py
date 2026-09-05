"""T1 action-space diagnosis: how much of the P0 failure is cross-class
confusion — i.e., the space a same-position relative-evidence mechanism
(design-v3.0 §3) could possibly address? (Step 1 of the verification
sequence in v3-literature-lessons.md.)

Protocol (registered before numbers were produced, 2026-09-05):
- Pool: P0 predictions (raw score >= 0.05), then class-wise standard NMS
  (raw-score order, mask IoU >= 0.5 suppress) — the v3 working set.
- Each kept candidate vs GT (any class):
    TP           best same-class IoU >= 0.5
    wrong-class  otherwise, best any-class IoU >= 0.5  <- addressable by
                                                          class switching
    background   otherwise                               <- not addressable
- Each GT instance without a same-class kept candidate at IoU >= 0.5:
    recoverable  some kept candidate of ANOTHER class has IoU >= 0.5 with it
    partial      best cross-class IoU in [0.3, 0.5)
    true miss    no cross-class candidate >= 0.3          <- not addressable
- Ideal ceiling = (TP + recoverable) / total GT: the recall a PERFECT
  same-position class-switching mechanism would reach. Nothing above this
  is reachable without new instances (red line).

Run:  $PYTHON sam3_ovis/eval/action_space_diagnosis.py [--datasets ...]
Out:  notes/reports/v3-feasibility-data.json + printed table.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402

NMS_IOU = 0.5
MATCH_IOU = 0.5
PARTIAL_IOU = 0.3


def _arr(seg, h=None, w=None):
    if isinstance(seg, dict):
        r = dict(seg)
        if isinstance(r["counts"], str):
            r["counts"] = r["counts"].encode("utf-8")
        return mask_util.decode(r).astype(bool)
    objs = seg if isinstance(seg[0], list) else [seg]
    return mask_util.decode(mask_util.merge(
        mask_util.frPyObjects(objs, h, w))).astype(bool)


def class_nms(dets, iou_thr=NMS_IOU):
    order = sorted(dets, key=lambda d: -d["score"])
    kept, kept_m = [], []
    for d in order:
        m = _arr(d["segmentation"])
        dup = False
        for km in kept_m:
            inter = int((m & km).sum())
            if inter and inter / float((m | km).sum()) >= iou_thr:
                dup = True
                break
        if not dup:
            kept.append(d)
            kept_m.append(m)
    return kept, kept_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["NWPU", "SOTA", "FAST"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(os.environ.get("SAM3_OVIS_DATA_ROOT", ""))
    log_root = Path(os.environ["SAM3_OVIS_LOG_ROOT"])
    report = {}
    for ds in args.datasets:
        spec = DATASETS[ds]
        gt = COCO(str(root / spec.val_json))
        name_by_cat = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
        h_w = {im["id"]: (im["height"], im["width"])
               for im in gt.dataset["images"]}

        pool = defaultdict(list)
        with open(log_root / "zeroshot" / ds / "predictions.jsonl") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    pool[d["image_id"]].append(d)

        cand_stats = defaultdict(int)   # candidate-side
        gt_stats = defaultdict(int)     # GT-side
        confusion_pairs = defaultdict(int)
        n_gt = 0

        for img_id in gt.getImgIds():
            if img_id not in pool:
                continue
            h, w = h_w[img_id]
            kept, kept_m = class_nms(pool[img_id])
            gt_list = gt.loadAnns(gt.getAnnIds(imgIds=[img_id]))
            gt_m = [(a["category_id"], _arr(a["segmentation"], h, w))
                    for a in gt_list]
            n_gt += len(gt_m)

            # candidate side
            claimed = set()
            for ki, (d, m) in enumerate(zip(kept, kept_m)):
                best_cat, best_any, best_same = None, 0.0, 0.0
                for cid, gm in gt_m:
                    inter = int((m & gm).sum())
                    if inter == 0:
                        continue
                    iou = inter / float((m | gm).sum())
                    if iou > best_any:
                        best_any, best_cat = iou, cid
                    if cid == d["category_id"]:
                        best_same = max(best_same, iou)
                if best_same >= MATCH_IOU:
                    cand_stats["tp"] += 1
                    claimed.add(best_cat)
                elif best_any >= MATCH_IOU:
                    cand_stats["wrong_class"] += 1
                    confusion_pairs[(name_by_cat[d["category_id"]],
                                     name_by_cat[best_cat])] += 1
                elif best_any >= PARTIAL_IOU:
                    cand_stats["partial_overlap"] += 1
                else:
                    cand_stats["background"] += 1

            # GT side: misses decomposed
            for cid, gm in gt_m:
                same_hit = False
                best_cross, best_cross_cat = 0.0, None
                for ki, (d, m) in enumerate(zip(kept, kept_m)):
                    inter = int((m & gm).sum())
                    if inter == 0:
                        continue
                    iou = inter / float((m | gm).sum())
                    if d["category_id"] == cid and iou >= MATCH_IOU:
                        same_hit = True
                        break
                    if d["category_id"] != cid and iou > best_cross:
                        best_cross, best_cross_cat = iou, d["category_id"]
                if same_hit:
                    gt_stats["covered_same_class"] += 1
                elif best_cross >= MATCH_IOU:
                    gt_stats["recoverable_by_switch"] += 1
                    confusion_pairs[("GT:" + name_by_cat[cid],
                                     name_by_cat[best_cross_cat])] += 1
                elif best_cross >= PARTIAL_IOU:
                    gt_stats["partial_miss"] += 1
                else:
                    gt_stats["true_miss"] += 1

        tp = cand_stats["tp"]
        rec = gt_stats["recoverable_by_switch"]
        covered = gt_stats["covered_same_class"]
        ceiling = (covered + rec) / max(n_gt, 1)
        recall_now = covered / max(n_gt, 1)
        row = {
            "n_gt": n_gt,
            "candidates": dict(cand_stats),
            "gt": dict(gt_stats),
            "recall_now": round(recall_now, 4),
            "ideal_ceiling_with_switch": round(ceiling, 4),
            "headroom_share_of_misses": round(
                rec / max(n_gt - covered, 1), 4),
            "top_confusion_pairs": sorted(
                ((k, v) for k, v in confusion_pairs.items()),
                key=lambda x: -x[1])[:10],
        }
        report[ds] = row
        print(f"== {ds} ==")
        print(f"  GT 实例 {n_gt} | 当前同类覆盖 {covered} (召回 {recall_now:.3f})")
        print(f"  候选侧: TP={cand_stats['tp']} 错类={cand_stats['wrong_class']} "
              f"部分重叠={cand_stats['partial_overlap']} "
              f"背景={cand_stats['background']}")
        print(f"  漏检分解: 可换类纠正={rec} 部分重叠漏检="
              f"{gt_stats['partial_miss']} 真漏检={gt_stats['true_miss']}")
        print(f"  理想换类上限 recall={ceiling:.3f} "
              f"(漏检中可纠正占比 {row['headroom_share_of_misses']:.2f})")
        print("  Top 混淆对 (真类 -> 被判类):")
        for (a, b), n in row["top_confusion_pairs"][:6]:
            print(f"    {a} -> {b}: {n}")

    out = Path(__file__).resolve().parents[2] / "notes/reports/v3-feasibility-data.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
