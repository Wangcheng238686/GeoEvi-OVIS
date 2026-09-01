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



_CANONICAL_DETECTION_LOSS_KEYS = {
    "loss_rpn_cls",
    "loss_rpn_bbox",
    "loss_cls",
    "loss_bbox",
    "loss_bbox_cls",
    "loss_bbox_reg",
}

def _set_model_current_epoch(model, epoch_number: int) -> None:
    """Set 1-based epoch on modules with epoch-dependent schedules."""
    core = model.module if hasattr(model, "module") else model
    seen = set()
    for module in core.modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        setter = getattr(module, "set_current_epoch", None)
        if callable(setter):
            setter(int(epoch_number))

def _detection_loss_weight(args, epoch_number: int) -> float:
    if epoch_number <= int(args.det_loss_stage1_end):
        return float(args.det_loss_weight_stage1)
    if epoch_number <= int(args.det_loss_stage2_end):
        return float(args.det_loss_weight_stage2)
    return float(args.det_loss_weight_stage3)

def _scale_loss_value(value, scale: float):
    if isinstance(value, torch.Tensor):
        return value * float(scale)
    if isinstance(value, list):
        return [_scale_loss_value(item, scale) for item in value]
    if isinstance(value, tuple):
        return tuple(_scale_loss_value(item, scale) for item in value)
    return value

def _apply_detection_loss_weight(loss_dict: dict, scale: float, device) -> dict:
    """Scale detector losses as one group without changing their ratios."""
    for key in list(loss_dict.keys()):
        if key in _CANONICAL_DETECTION_LOSS_KEYS or key.startswith("loss_rpn_"):
            loss_dict[key] = _scale_loss_value(loss_dict[key], scale)
    loss_dict["schedule/det_weight"] = torch.tensor(
        float(scale), device=device
    )
    return loss_dict

def _materialize_p2_boundary_refiner_loss(
    loss_dict: dict,
    model: torch.nn.Module,
    distributed: bool,
) -> dict:
    """Convert local ROI sums into a four-rank global per-ROI mean.

    DDP averages gradients across ranks. Multiplying the local sum by
    ``world_size / global_count`` therefore yields the exact mean over all
    valid ROIs in the current micro-batch. Gradient accumulation keeps the
    historical equal-micro-batch contract.
    """
    local_sum = loss_dict.pop("_p2br_local_sum", None)
    local_count = loss_dict.pop("_p2br_valid_count", None)
    if local_sum is None and local_count is None:
        return loss_dict
    if not torch.is_tensor(local_sum) or not torch.is_tensor(local_count):
        raise TypeError("P2 boundary refiner loss metadata must be tensors")
    global_count = local_count.detach().float().clone()
    world_size = 1
    if distributed and dist.is_initialized():
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()

    model_core = model.module if hasattr(model, "module") else model
    mask_head = getattr(getattr(model_core, "roi_head", None), "mask_head", None)
    refiner = getattr(mask_head, "p2_boundary_refiner", None)
    if refiner is None:
        raise RuntimeError("received P2 refiner loss metadata without the module")
    if float(global_count.item()) > 0:
        normalized = local_sum * (
            float(world_size) / float(global_count.item())
        )
    else:
        normalized = local_sum * 0.0
    loss_dict["loss_p2_boundary_refiner"] = (
        float(refiner.boundary_loss_weight) * normalized
    )
    return loss_dict

