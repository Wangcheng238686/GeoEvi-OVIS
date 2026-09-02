"""SCORE cross-dataset evaluation for the sam3_ovis mainline.

Thin wrapper in the spirit of geoevi_ovis/utils/coco_eval_utils.py (that file
is single-class and tensor-facing; SCORE needs multi-class COCO-json I/O).
The original file stays untouched (C5-v2 freeze rule).

Protocol (registered 2026-09-02, cross-check against SCORE paper table in
S2.1 before any gate decision):
- Metric = COCO instance segm AP@[.50:.05:.95] per test dataset (standard
  pycocotools defaults: maxDets=[1,10,100], areaRng=all), averaged over
  categories; "Average" column = unweighted mean over datasets.
- Source: SCORE (ICCV 2025, arXiv 2507.12857) §4.1 "We follow previous work
  [39] to evaluate ... with mean Average Precision (mAP) for open vocabulary
  instance segmentation" — i.e., the FreeSeg/FC-CLIP COCO-style protocol.
- Diagnostics (self-use only, reported separately from the SCORE protocol):
  per-class instance recall proxy @IoU0.5, precision proxy, conditional
  mask-IoU quality. See eval/README.md.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# COCO IO helpers
# ---------------------------------------------------------------------------

def load_gt_coco(gt_json_path: str):
    """Load a ground-truth COCO-instance json as pycocotools COCO."""
    from pycocotools.coco import COCO

    return COCO(gt_json_path)


def load_predictions(pred_json_path: str) -> List[dict]:
    """Load predictions in COCO results format (list of dicts)."""
    import json

    with open(pred_json_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# SCORE protocol: per-dataset segm mAP + per-class AP
# ---------------------------------------------------------------------------

def evaluate_dataset(
    coco_gt,
    predictions: List[dict],
    iou_type: str = "segm",
) -> Tuple[OrderedDict, OrderedDict]:
    """Run COCOeval on one dataset.

    Args:
        coco_gt: pycocotools COCO ground truth.
        predictions: COCO results list ({image_id, category_id, score,
            segmentation}); segmentation is RLE (counts as utf8 str) or polygon.
        iou_type: "segm" (protocol default) or "bbox".

    Returns:
        (summary, per_class): summary maps metric name -> float for the whole
        dataset; per_class maps category name -> AP@[.50:.95] for that class.
    """
    from pycocotools.cocoeval import COCOeval

    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:  # empty prediction set: valid all-zero outcome, not an error
        coco_dt = coco_gt.loadRes([])

    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    prefix = iou_type
    summary = OrderedDict()
    names = [
        "mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l",
        "AR@1", "AR@10", "AR@100", "AR_s", "AR_m", "AR_l",
    ]
    for name, val in zip(names, coco_eval.stats):
        summary[f"{prefix}/{name}"] = float(val)

    # Per-class AP@[.50:.95]: average the precision tensor over IoU
    # thresholds, recall points, area ranges and maxDets, skipping the -1
    # sentinel for classes with no GT / no DT.
    precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
    per_class = OrderedDict()
    cat_ids = coco_eval.params.catIds
    for k, cat_id in enumerate(cat_ids):
        pr = precision[:, :, k, 0, 0]
        pr = pr[pr > -1]
        name = coco_gt.cats[cat_id]["name"]
        per_class[name] = float(pr.mean()) if pr.size else float("nan")
    return summary, per_class


def cross_dataset_average(summaries: Dict[str, OrderedDict]) -> float:
    """SCORE "Average" column: unweighted mean of per-dataset mAP."""
    vals = [s["segm/mAP"] for s in summaries.values()]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Diagnostics (self-use; see module docstring)
# ---------------------------------------------------------------------------

def compute_diagnostics(
    coco_gt,
    predictions: List[dict],
    iou_thr: float = 0.5,
    score_attr: str = "score",
) -> OrderedDict:
    """Class-averaged instance recall / precision proxies + conditional IoU.

    - recall proxy: GT instances covered by >=1 prediction of the same class
      at IoU >= iou_thr, per class, then averaged (design-v2 G1 metric shape).
    - precision proxy: predictions one-to-one greedy-matched (by descending
      score) to an unmatched same-class GT at IoU >= iou_thr, per class,
      then averaged.
    - conditional mask IoU: mean IoU over matched pairs (mask quality of
      already-localized instances).

    All three are computed per class and averaged over classes that have at
    least one GT instance (matches "类平均" wording in implementation-v2
    S1.4); overall micro-averaged variants are reported alongside.
    """
    from pycocotools import mask as mask_util

    dets_by: Dict[Tuple[int, int], List[dict]] = {}
    for dt in predictions:
        dets_by.setdefault((dt["image_id"], dt["category_id"]), []).append(dt)
    for dets in dets_by.values():
        dets.sort(key=lambda d: -float(d[score_attr]))

    def _to_rle(seg):
        if isinstance(seg, dict):
            rle = dict(seg)
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("utf-8")
            return rle
        raise ValueError(
            "predictions must be COCO RLE (SAM3 masks are encoded to RLE "
            "upstream); polygon predictions are not supported by the "
            "diagnostics path"
        )

    gt_by: Dict[Tuple[int, int], List[dict]] = {}
    for ann in coco_gt.dataset["annotations"]:
        gt_by.setdefault((ann["image_id"], ann["category_id"]), []).append(ann)

    cat_ids = sorted({a["category_id"] for a in coco_gt.dataset["annotations"]})

    per_class = {cid: {"n_gt": 0, "recall_hits": 0, "n_dt": 0, "tp": 0,
                       "iou_sum": 0.0} for cid in cat_ids}

    for (image_id, cat_id), gts in gt_by.items():
        key_stats = per_class.setdefault(
            cat_id, {"n_gt": 0, "recall_hits": 0, "n_dt": 0, "tp": 0,
                     "iou_sum": 0.0})
        key_stats["n_gt"] += len(gts)
        dets = dets_by.get((image_id, cat_id), [])
        key_stats["n_dt"] += len(dets)
        if not dets:
            continue
        gt_rles = []
        for g in gts:
            seg = g["segmentation"]
            if isinstance(seg, dict):
                rle = dict(seg)
                if isinstance(rle["counts"], str):
                    rle["counts"] = rle["counts"].encode("utf-8")
                gt_rles.append(rle)
            else:
                h, w = g.get("segmentation_hw") or _image_hw(coco_gt, image_id)
                objs = seg if isinstance(seg[0], list) else [seg]
                gt_rles.append(mask_util.merge(mask_util.frPyObjects(objs, h, w)))
        dt_rles = [_to_rle(d["segmentation"]) for d in dets]
        ious = mask_util.iou(dt_rles, gt_rles, [0] * len(gt_rles))
        ious = np.atleast_2d(np.asarray(ious))

        matched_gt = set()
        for r in range(ious.shape[0]):
            best_g = int(np.argmax(ious[r]))
            best_iou = float(ious[r, best_g])
            if best_iou >= iou_thr and best_g not in matched_gt:
                matched_gt.add(best_g)
                key_stats["tp"] += 1
                key_stats["iou_sum"] += best_iou
        key_stats["recall_hits"] += len(matched_gt)

    per_class_out = OrderedDict()
    recall_c, precision_c, iou_c = [], [], []
    micro_gt = micro_hits = micro_dt = micro_tp = 0
    micro_iou = 0.0
    for cid, st in per_class.items():
        name = coco_gt.cats[cid]["name"]
        recall = st["recall_hits"] / st["n_gt"] if st["n_gt"] else float("nan")
        precision = st["tp"] / st["n_dt"] if st["n_dt"] else float("nan")
        cond_iou = st["iou_sum"] / st["tp"] if st["tp"] else float("nan")
        per_class_out[name] = {
            "recall_proxy@%.2f" % iou_thr: recall,
            "precision_proxy@%.2f" % iou_thr: precision,
            "conditional_iou": cond_iou,
            "n_gt": st["n_gt"], "n_dt": st["n_dt"], "tp": st["tp"],
            "recall_hits": st["recall_hits"],
        }
        if st["n_gt"]:
            recall_c.append(recall)
            micro_gt += st["n_gt"]
            micro_hits += st["recall_hits"]
        if st["n_dt"]:
            precision_c.append(precision)
            iou_c.append(cond_iou)
            micro_dt += st["n_dt"]
            micro_tp += st["tp"]
            micro_iou += st["iou_sum"]

    def _mean(xs):
        xs = [x for x in xs if not np.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")

    return OrderedDict([
        ("recall_proxy_class_avg", _mean(recall_c)),
        ("precision_proxy_class_avg", _mean(precision_c)),
        ("conditional_iou_class_avg", _mean(iou_c)),
        ("recall_proxy_micro", micro_hits / micro_gt if micro_gt else float("nan")),
        ("precision_proxy_micro", micro_tp / micro_dt if micro_dt else float("nan")),
        ("conditional_iou_micro", micro_iou / micro_tp if micro_tp else float("nan")),
        ("per_class", per_class_out),
    ])


def _image_hw(coco_gt, image_id) -> Tuple[int, int]:
    img = coco_gt.imgs[image_id]
    return int(img["height"]), int(img["width"])


def run_score_eval(
    gt_json_path: str,
    pred_json_path: Optional[str] = None,
    predictions: Optional[List[dict]] = None,
) -> OrderedDict:
    """One-call entry: dataset mAP + diagnostics from GT json + predictions."""
    coco_gt = load_gt_coco(gt_json_path)
    if predictions is None:
        assert pred_json_path is not None, "need pred_json_path or predictions"
        predictions = load_predictions(pred_json_path)
    summary, per_class = evaluate_dataset(coco_gt, predictions)
    diag = compute_diagnostics(coco_gt, predictions)
    return OrderedDict([
        ("summary", summary),
        ("per_class_ap", per_class),
        ("diagnostics", diag),
    ])
