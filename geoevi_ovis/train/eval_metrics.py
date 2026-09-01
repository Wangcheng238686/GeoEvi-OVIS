"""Extracted from train/train.py — behaviour-preserving split (C-level)."""
import logging
import math
import os
import subprocess
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

import argparse
import gc
import cv2
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmengine.structures import InstanceData  # noqa: F401

logger = logging.getLogger("portable_sam_fusion")



def _mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Convert binary mask (H, W) to COCO RLE format."""
    from pycocotools import mask as mask_util

    mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_util.encode(mask_fortran)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def _masks_to_rles(binary_masks: np.ndarray) -> List[dict]:
    """Batch-encode N binary masks as JSON-safe COCO RLE records."""
    from pycocotools import mask as mask_util

    masks = np.asarray(binary_masks, dtype=np.uint8)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape [N,H,W], got {masks.shape}")
    if len(masks) == 0:
        return []

    # pycocotools batch encode expects [H,W,N] in Fortran order. Encoding here
    # lets validation release dense 1024x1024 masks immediately per image.
    encoded = mask_util.encode(
        np.asfortranarray(np.moveaxis(masks, 0, -1))
    )
    if isinstance(encoded, dict):
        encoded = [encoded]
    rles = []
    for rle in encoded:
        rle = dict(rle)
        counts = rle.get("counts")
        if isinstance(counts, bytes):
            rle["counts"] = counts.decode("utf-8")
        rles.append(rle)
    return rles

def _xyxy_to_xywh(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    out = bbox.copy()
    out[..., 2] = bbox[..., 2] - bbox[..., 0]
    out[..., 3] = bbox[..., 3] - bbox[..., 1]
    return out

def build_coco_gt_and_dt(
    all_gt: List[dict],
    all_dt: List[dict],
    img_metas_list: List[dict],
    score_key: str = "scores",
) -> Tuple:
    """Build pycocotools COCO objects for GT and detections."""
    from pycocotools.coco import COCO

    images = []
    annotations = []
    predictions = []
    ann_id = 1

    for img_id, (gt, dt, meta) in enumerate(
        zip(all_gt, all_dt, img_metas_list), start=1
    ):
        h, w = meta["img_shape"][:2]
        images.append({"id": img_id, "height": int(h), "width": int(w)})

        # GT annotations
        gt_bboxes = gt["bboxes"]
        gt_labels = gt["labels"]
        gt_rles = gt.get("rles")
        gt_masks = gt.get("masks")
        if gt_rles is None:
            if hasattr(gt_masks, "masks"):
                gt_masks = gt_masks.masks
            gt_rles = _masks_to_rles(gt_masks)

        num_gt = len(gt_labels)
        for i in range(num_gt):
            bbox_xywh = _xyxy_to_xywh(gt_bboxes[i]).tolist()
            area = float(bbox_xywh[2] * bbox_xywh[3])
            seg_rle = gt_rles[i]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": bbox_xywh,
                    "area": area,
                    "segmentation": seg_rle,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        # Predictions
        dt_bboxes = dt["bboxes"]
        if score_key not in dt:
            raise KeyError(
                f"Configured COCO score key {score_key!r} is missing from "
                "predictions; refusing a silent score fallback"
            )
        dt_scores = dt[score_key]
        dt_rles = dt.get("rles")
        dt_masks = dt.get("masks")
        if dt_rles is None:
            if hasattr(dt_masks, "masks"):
                dt_masks = dt_masks.masks
            dt_rles = _masks_to_rles(dt_masks)

        num_dt = len(dt_scores)
        for i in range(num_dt):
            bbox_xywh = _xyxy_to_xywh(dt_bboxes[i]).tolist()
            seg_rle = dt_rles[i]
            predictions.append(
                {
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": bbox_xywh,
                    "score": float(dt_scores[i]),
                    "segmentation": seg_rle,
                }
            )

    # Build COCO GT
    gt_dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "building"}],
    }
    coco_gt = COCO()
    coco_gt.dataset = gt_dataset
    coco_gt.createIndex()

    # Build COCO DT
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {"images": images, "annotations": [], "categories": gt_dataset["categories"]}
        coco_dt.createIndex()

    return coco_gt, coco_dt

def run_coco_eval(coco_gt, coco_dt, iou_type: str = "bbox", max_dets=None) -> OrderedDict:
    """Run COCOeval and return metrics dict."""
    from pycocotools.cocoeval import COCOeval

    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    if max_dets is not None:
        coco_eval.params.maxDets = max_dets
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    if max_dets is not None:
        # summarize() hardcodes maxDets=100 for the primary AP stat and
        # yields -1 when 100 is absent from params.maxDets. Recompute the
        # IoU-averaged AP from the accumulated precision array at the
        # custom cap (no reliance on private summarize helpers).
        try:
            _mind = list(coco_eval.params.maxDets).index(list(max_dets)[-1])
            _prec = coco_eval.eval["precision"]
            _aps = []
            for _k in range(_prec.shape[2]):
                _p = _prec[:, :, _k, 0, _mind]
                _v = _p[_p > -1]
                if _v.size:
                    _aps.append(float(_v.mean()))
            if _aps:
                coco_eval.stats[0] = sum(_aps) / len(_aps)
        except Exception:
            # A silent -1 here would poison best-model selection for the whole
            # run (constant metric -> best never updates); surface it loudly.
            logger.warning(
                "run_coco_eval custom-maxDets recompute failed for "
                "iou_type=%s max_dets=%s; mAP may read -1",
                iou_type,
                list(max_dets),
                exc_info=True,
            )

    metric_names = [
        "mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l",
        "AR@1", "AR@10", "AR@100", "AR_s", "AR_m", "AR_l",
    ]
    results = OrderedDict()
    prefix = "bbox" if iou_type == "bbox" else "segm"
    for name, val in zip(metric_names, coco_eval.stats):
        results[f"{prefix}/{name}"] = float(val)
    return results

def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}

def _metric_value(metrics: Dict[str, float], metric_name: str) -> Optional[float]:
    if not metric_name:
        return None
    value = metrics.get(metric_name)
    if value is None:
        return None
    return float(value)

def _weighted_metric_value(metrics: Dict[str, float], spec: str) -> Optional[float]:
    """Parse '0.5*bbox/mAP_75+0.5*segm/mAP_75' style metric expressions."""
    total = 0.0
    found = False
    for raw_part in spec.split("+"):
        part = raw_part.strip()
        if not part:
            continue
        if "*" in part:
            weight_text, metric_name = part.split("*", 1)
            weight = float(weight_text.strip())
            metric_name = metric_name.strip()
        else:
            weight = 1.0
            metric_name = part
        value = _metric_value(metrics, metric_name)
        if value is None:
            return None
        total += weight * value
        found = True
    return total if found else None

def _metric_expression_value(metrics: Dict[str, float], spec: str) -> Optional[float]:
    if "+" in spec or "*" in spec:
        return _weighted_metric_value(metrics, spec)
    return _metric_value(metrics, spec)

def _extract_instances_numpy(
    instances, img_shape: Tuple[int, int], encode_masks: bool = False
) -> dict:
    """Extract InstanceData, optionally replacing dense masks with COCO RLE."""
    bboxes = instances.bboxes
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()
    labels = instances.labels
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    scores = None
    if hasattr(instances, "scores"):
        scores = instances.scores
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

    masks = None
    if hasattr(instances, "masks"):
        masks = instances.masks
        if hasattr(masks, "masks"):
            masks = masks.masks
        elif hasattr(masks, "to_ndarray"):
            masks = masks.to_ndarray()
        elif isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()

    # Ensure masks shape matches img_shape
    h, w = img_shape[:2]
    if masks is not None and len(masks) > 0 and masks.shape[1:] != (h, w):
        resized = []
        for m in masks:
            resized.append(
                cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            )
        masks = np.stack(resized, axis=0)

    if masks is None or len(masks) == 0:
        masks = np.zeros((0, h, w), dtype=np.uint8)

    out = {"bboxes": bboxes, "labels": labels}
    if encode_masks:
        mask_count = len(masks)
        out["rles"] = _masks_to_rles(masks)
        out["mask_count"] = mask_count
        out["mask_fill_sum"] = (
            float(np.count_nonzero(masks)) / float(max(1, h * w))
            if mask_count > 0
            else 0.0
        )
    else:
        out["masks"] = masks
    if scores is not None:
        out["scores"] = scores
    if hasattr(instances, "mask_scores"):
        mask_scores = instances.mask_scores
        if isinstance(mask_scores, torch.Tensor):
            mask_scores = mask_scores.detach().cpu().numpy()
        out["mask_scores"] = mask_scores
    return out

def _summarize_mask_density(all_dt: List[dict]) -> Tuple[float, float]:
    """Return average binary mask fill ratio and masks per validation image."""
    weighted_fill = 0.0
    total_masks = 0
    for dt in all_dt:
        if "mask_count" in dt and "mask_fill_sum" in dt:
            total_masks += int(dt["mask_count"])
            weighted_fill += float(dt["mask_fill_sum"])
            continue
        masks = dt.get("masks")
        if masks is None or len(masks) == 0:
            continue
        pixels_per_mask = int(np.prod(masks.shape[1:]))
        weighted_fill += float(np.count_nonzero(masks)) / max(1, pixels_per_mask)
        total_masks += len(masks)
    return (
        weighted_fill / max(total_masks, 1),
        total_masks / max(len(all_dt), 1),
    )

