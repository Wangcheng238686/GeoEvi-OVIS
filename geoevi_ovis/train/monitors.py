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



def _log_trainable_state(model: torch.nn.Module):
    model_core = model.module if hasattr(model, "module") else model
    roi_head = getattr(model_core, "roi_head", None)
    mask_head = getattr(roi_head, "mask_head", None)
    checks = {
        "backbone": getattr(model_core, "backbone", None),
        "shared_image_embedding": getattr(model_core, "shared_image_embedding", None),
        "mask_decoder": getattr(mask_head, "mask_decoder", None) if mask_head is not None else None,
        "no_mask_embed": getattr(mask_head, "no_mask_embed", None) if mask_head is not None else None,
        "prompt_encoder": getattr(mask_head, "prompt_encoder", None) if mask_head is not None else None,
        "shape_injector": getattr(mask_head, "shape_injector", None) if mask_head is not None else None,
        "p2_boundary_refiner": getattr(
            mask_head, "p2_boundary_refiner", None
        ) if mask_head is not None else None,
    }
    for name, module in checks.items():
        if module is None:
            logger.info("Trainable state %-22s absent", name)
            continue
        if isinstance(module, torch.nn.Parameter):
            total = module.numel()
            trainable = module.numel() if module.requires_grad else 0
        else:
            params = list(module.parameters())
            total = sum(p.numel() for p in params)
            trainable = sum(p.numel() for p in params if p.requires_grad)
        logger.info("Trainable state %-22s %d / %d", name, trainable, total)

def _log_prompt_debug_stats(
    model: torch.nn.Module,
    epoch: int,
    step: int,
    enabled_prefixes: Optional[set] = None,
) -> None:
    model_core = model.module if hasattr(model, "module") else model
    getter = getattr(model_core, "get_prompt_debug_stats", None)
    if getter is None:
        return
    stats = getter()
    if not stats:
        return
    groups: Dict[str, Dict[str, float]] = {}
    for key, value in sorted(stats.items()):
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = float(value.detach().float().item())
        if not isinstance(value, (int, float)):
            continue
        prefix, _, name = key.partition("/")
        if not name:
            continue
        if enabled_prefixes is not None and prefix not in enabled_prefixes:
            continue
        groups.setdefault(prefix, {})[name] = float(value)
    for prefix, values in groups.items():
        payload = " ".join(f"{k}={v:.4g}" for k, v in values.items())
        logger.info("[DEBUG-%s] epoch=%d step=%d %s", prefix, epoch + 1, step, payload)

_DENSE_RESIDUAL_MONITOR_KEYS = (
    "DENSE/residual_alpha",
    "DENSE/source_delta_norm",
    "DENSE/applied_delta_norm",
    "DENSE/source_delta_ratio",
    "DENSE/applied_delta_ratio",
    "DENSE/canvas_mean",
    "DENSE/canvas_std",
    "DENSE/canvas_min",
    "DENSE/canvas_max",
    "GAUSSIAN/valid_ratio",
    "GAUSSIAN/foreground_area_ratio",
    "GAUSSIAN/edt_max",
    "GAUSSIAN/center_x_normalized",
    "GAUSSIAN/center_y_normalized",
    "GAUSSIAN/canvas_area",
)

def _accumulate_dense_residual_monitor(
    model: torch.nn.Module,
    sums: Dict[str, float],
    counts: Dict[str, int],
) -> None:
    """Accumulate detached dense-residual diagnostics from the latest forward."""
    model_core = model.module if hasattr(model, "module") else model
    getter = getattr(model_core, "get_prompt_debug_stats", None)
    if getter is None:
        return
    stats = getter() or {}
    for key in _DENSE_RESIDUAL_MONITOR_KEYS:
        value = stats.get(key)
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = float(value.detach().float().item())
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        sums[key] += float(value)
        counts[key] += 1

def _reduce_dense_residual_monitor(
    sums: Dict[str, float],
    counts: Dict[str, int],
    device: torch.device,
    distributed: bool,
) -> Dict[str, float]:
    """Return rank-aggregated epoch means for the fixed dense monitor fields."""
    packed = torch.tensor(
        [
            item
            for key in _DENSE_RESIDUAL_MONITOR_KEYS
            for item in (sums[key], float(counts[key]))
        ],
        dtype=torch.float64,
        device=device,
    )
    if distributed and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    result = {}
    for index, key in enumerate(_DENSE_RESIDUAL_MONITOR_KEYS):
        total = float(packed[2 * index].item())
        count = float(packed[2 * index + 1].item())
        if count > 0:
            result[key] = total / count
    return result

_P2BR_EPOCH_FIELDS = (
    "roi_count",
    "valid_support_count",
    "rejected_support_count",
    "search_coverage_sum",
    "delta_abs_sum",
    "delta_nonzero_sum",
    "delta_saturated_sum",
    "projected_feature_norm_sum",
    "highpass_feature_norm_sum",
    "raw_dice_sum",
    "raw_iou_sum",
    "refined_dice_sum",
    "refined_iou_sum",
    "raw_boundary_tp",
    "raw_boundary_fp",
    "raw_boundary_fn",
    "refined_boundary_tp",
    "refined_boundary_fp",
    "refined_boundary_fn",
    "boundary_loss_local_sum",
    "boundary_loss_valid_roi_count",
    "boundary_loss_support_pixels",
    "boundary_loss_foreground_pixels",
)

def _stat_scalar(stats: Dict[str, Any], key: str) -> Optional[float]:
    value = stats.get(key)
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = float(value.detach().float().item())
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)

def _accumulate_p2br_epoch_monitor(
    model: torch.nn.Module, sums: Dict[str, float]
) -> None:
    model_core = model.module if hasattr(model, "module") else model
    getter = getattr(model_core, "get_prompt_debug_stats", None)
    if getter is None:
        return
    stats = getter() or {}
    roi_count = _stat_scalar(stats, "P2BR/roi_count")
    if roi_count is None:
        return
    direct = {
        "roi_count": "P2BR/roi_count",
        "valid_support_count": "P2BR/valid_support_count",
        "rejected_support_count": "P2BR/rejected_support_count",
        "search_coverage_sum": "P2BR/search_coverage_sum",
        "delta_abs_sum": "P2BR/delta_abs_sum",
        "delta_nonzero_sum": "P2BR/delta_nonzero_sum",
        "delta_saturated_sum": "P2BR/delta_saturated_sum",
        "projected_feature_norm_sum": "P2BR/projected_feature_norm_sum",
        "highpass_feature_norm_sum": "P2BR/highpass_feature_norm_sum",
        "raw_boundary_tp": "COARSE/raw_boundary_tp",
        "raw_boundary_fp": "COARSE/raw_boundary_fp",
        "raw_boundary_fn": "COARSE/raw_boundary_fn",
        "refined_boundary_tp": "COARSE/refined_boundary_tp",
        "refined_boundary_fp": "COARSE/refined_boundary_fp",
        "refined_boundary_fn": "COARSE/refined_boundary_fn",
        "boundary_loss_local_sum": "COARSE/boundary_loss_local_sum",
        "boundary_loss_valid_roi_count": "COARSE/loss_valid_roi_count",
        "boundary_loss_support_pixels": "COARSE/loss_support_pixel_sum",
        "boundary_loss_foreground_pixels": "COARSE/loss_support_foreground_sum",
    }
    for out_key, stat_key in direct.items():
        value = _stat_scalar(stats, stat_key)
        if value is not None:
            sums[out_key] += value
    for out_key, stat_key in (
        ("raw_dice_sum", "COARSE/raw_dice_score"),
        ("raw_iou_sum", "COARSE/raw_iou"),
        ("refined_dice_sum", "COARSE/refined_dice_score"),
        ("refined_iou_sum", "COARSE/refined_iou"),
    ):
        value = _stat_scalar(stats, stat_key)
        if value is not None:
            sums[out_key] += value * roi_count

def _reduce_p2br_epoch_monitor(
    sums: Dict[str, float], device: torch.device, distributed: bool
) -> Dict[str, float]:
    packed = torch.tensor(
        [sums[key] for key in _P2BR_EPOCH_FIELDS],
        dtype=torch.float64,
        device=device,
    )
    if distributed and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    values = {
        key: float(packed[index].item())
        for index, key in enumerate(_P2BR_EPOCH_FIELDS)
    }
    roi_count = values["roi_count"]
    if roi_count <= 0:
        return {}
    valid_count = max(values["valid_support_count"], 1.0)
    loss_count = max(values["boundary_loss_valid_roi_count"], 1.0)

    def _boundary_f1(prefix: str) -> float:
        tp = values[f"{prefix}_boundary_tp"]
        fp = values[f"{prefix}_boundary_fp"]
        fn = values[f"{prefix}_boundary_fn"]
        return (2.0 * tp) / max(2.0 * tp + fp + fn, 1e-12)

    return {
        "roi_count": roi_count,
        "valid_support_ratio": values["valid_support_count"] / roi_count,
        "rejected_support_ratio": values["rejected_support_count"] / roi_count,
        "search_coverage": values["search_coverage_sum"] / roi_count,
        "delta_abs": values["delta_abs_sum"] / roi_count,
        "delta_nonzero_ratio": values["delta_nonzero_sum"] / roi_count,
        "delta_saturated_ratio": values["delta_saturated_sum"] / roi_count,
        "projected_feature_norm": values["projected_feature_norm_sum"] / roi_count,
        "highpass_feature_norm": values["highpass_feature_norm_sum"] / roi_count,
        "raw_dice": values["raw_dice_sum"] / roi_count,
        "raw_iou": values["raw_iou_sum"] / roi_count,
        "refined_dice": values["refined_dice_sum"] / roi_count,
        "refined_iou": values["refined_iou_sum"] / roi_count,
        "raw_boundary_f1": _boundary_f1("raw"),
        "refined_boundary_f1": _boundary_f1("refined"),
        "boundary_loss_raw": values["boundary_loss_local_sum"] / loss_count,
        "boundary_support_fg_ratio": values["boundary_loss_foreground_pixels"]
        / max(values["boundary_loss_support_pixels"], 1.0),
        "valid_count_for_norm": valid_count,
    }

