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



def _build_optimizer(
    model: torch.nn.Module,
    lr: float,
    sat_backbone_lr_mult: float,
    sat_other_lr_mult: float,
    bbox_head_lr_mult: float,
    mask_decoder_lr_mult: float = 0.05,
    no_mask_lr_mult: float = 0.1,
    prompt_encoder_lr_mult: float = 0.1,
    shape_prior_lr_mult: float = 1.0,
    p2_boundary_refiner_lr_mult: float = 1.0,
    weight_decay: float = 0.05,
) -> optim.Optimizer:
    params_sat_backbone = []
    names_sat_backbone = []
    params_mask_decoder = []
    names_mask_decoder = []
    params_no_mask = []
    names_no_mask = []
    params_prompt_encoder = []
    names_prompt_encoder = []
    params_shape_prior = []
    names_shape_prior = []
    params_p2_boundary_refiner = []
    names_p2_boundary_refiner = []
    params_bbox_head = []
    names_bbox_head = []
    params_sat_other = []
    names_sat_other = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        match_name = name.replace("_fsdp_wrapped_module.", "").replace("module.", "")
        if match_name.startswith("backbone.") or match_name.startswith("shared_image_embedding."):
            params_sat_backbone.append(param)
            names_sat_backbone.append(name)
        elif "roi_head.mask_head.mask_decoder" in match_name:
            params_mask_decoder.append(param)
            names_mask_decoder.append(name)
        elif "roi_head.mask_head.no_mask_embed" in match_name:
            params_no_mask.append(param)
            names_no_mask.append(name)
        elif "roi_head.mask_head.prompt_encoder" in match_name:
            params_prompt_encoder.append(param)
            names_prompt_encoder.append(name)
        elif "roi_head.mask_head.p2_boundary_refiner" in match_name:
            params_p2_boundary_refiner.append(param)
            names_p2_boundary_refiner.append(name)
        elif (
            "roi_head.mask_head.shape_injector" in match_name
            or "roi_head.mask_head.shape_prompt_scale" in match_name
            or "roi_head.mask_head.roi_dense" in match_name
        ):
            params_shape_prior.append(param)
            names_shape_prior.append(name)
        elif (
            bbox_head_lr_mult > 0
            and match_name.startswith("roi_head.bbox_head.")
        ):
            params_bbox_head.append(param)
            names_bbox_head.append(name)
        else:
            params_sat_other.append(param)
            names_sat_other.append(name)

    param_groups = []
    audit_groups = []
    if params_sat_backbone:
        param_groups.append(
            {
                "params": params_sat_backbone,
                "lr": lr * sat_backbone_lr_mult,
                "name": "sat_backbone",
            }
        )
        audit_groups.append(("sat_backbone", params_sat_backbone, names_sat_backbone, lr * sat_backbone_lr_mult, weight_decay))
    if params_mask_decoder:
        param_groups.append(
            {
                "params": params_mask_decoder,
                "lr": lr * mask_decoder_lr_mult,
                "weight_decay": weight_decay,
                "name": "mask_decoder",
            }
        )
        audit_groups.append(("mask_decoder", params_mask_decoder, names_mask_decoder, lr * mask_decoder_lr_mult, weight_decay))
    if params_no_mask:
        param_groups.append(
            {
                "params": params_no_mask,
                "lr": lr * no_mask_lr_mult,
                "weight_decay": weight_decay,
                "name": "no_mask_embed",
            }
        )
        audit_groups.append(("no_mask_embed", params_no_mask, names_no_mask, lr * no_mask_lr_mult, weight_decay))
    if params_prompt_encoder:
        param_groups.append(
            {
                "params": params_prompt_encoder,
                "lr": lr * prompt_encoder_lr_mult,
                "weight_decay": 0.0,
                "name": "prompt_encoder",
            }
        )
        audit_groups.append(("prompt_encoder", params_prompt_encoder, names_prompt_encoder, lr * prompt_encoder_lr_mult, 0.0))
    if params_shape_prior:
        param_groups.append(
            {
                "params": params_shape_prior,
                "lr": lr * shape_prior_lr_mult,
                "name": "shape_prior",
            }
        )
        audit_groups.append(("shape_prior", params_shape_prior, names_shape_prior, lr * shape_prior_lr_mult, weight_decay))
    if params_p2_boundary_refiner:
        param_groups.append(
            {
                "params": params_p2_boundary_refiner,
                "lr": lr * p2_boundary_refiner_lr_mult,
                "weight_decay": weight_decay,
                "name": "p2_boundary_refiner",
            }
        )
        audit_groups.append(
            (
                "p2_boundary_refiner",
                params_p2_boundary_refiner,
                names_p2_boundary_refiner,
                lr * p2_boundary_refiner_lr_mult,
                weight_decay,
            )
        )
    if params_bbox_head:
        param_groups.append(
            {
                "params": params_bbox_head,
                "lr": lr * bbox_head_lr_mult,
                "name": "bbox_head",
            }
        )
        audit_groups.append(("bbox_head", params_bbox_head, names_bbox_head, lr * bbox_head_lr_mult, weight_decay))
    if params_sat_other:
        param_groups.append(
            {
                "params": params_sat_other,
                "lr": lr * sat_other_lr_mult,
                "name": "sat_other",
            }
        )
        audit_groups.append(("sat_other", params_sat_other, names_sat_other, lr * sat_other_lr_mult, weight_decay))

    if not param_groups:
        raise RuntimeError(
            "No trainable parameters found. Check freezing / requires_grad."
        )

    if int(os.environ.get("RANK", "0")) == 0:
        for group_name, params, names, group_lr, group_wd in audit_groups:
            count = sum(p.numel() for p in params)
            first = names[0] if names else "<none>"
            last = names[-1] if names else "<none>"
            logger.info(
                "Optimizer group %-15s params=%d tensors=%d lr=%.3e wd=%.3e first=%s last=%s",
                group_name,
                count,
                len(params),
                group_lr,
                group_wd,
                first,
                last,
            )

    return optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, eps=1e-6)

def _set_norm_eval(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(
            m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm, torch.nn.GroupNorm)
        ):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

def _get_model_core(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP/FSDP to reach the underlying model."""
    return model.module if hasattr(model, "module") else model

class ExponentialMovingAverage:
    """EMA shadow weights used only for eval/export, never for backprop."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0,1), got {decay}")
        self.decay = float(decay)
        model_core = _get_model_core(model)
        state = model_core.state_dict()
        self.ema_state = {k: v.detach().clone() for k, v in state.items()}
        self.backup_state = None
        # Frozen tensors (params with requires_grad=False, plus buffers) must
        # be copied verbatim, never averaged: averaging a constant across tens
        # of thousands of steps accumulates fp32 rounding drift (~1e-6 per
        # the fast-150 run), which trips strict pretrained-value contract
        # checks (no_mask_embed) when the EMA state is loaded for inference.
        _frozen = {n for n, p in model_core.named_parameters() if not p.requires_grad}
        _frozen.update(n for n, _ in model_core.named_buffers())
        self._frozen_keys = frozenset(k for k in state if k in _frozen)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        model_state = _get_model_core(model).state_dict()
        for key, value in model_state.items():
            value_detached = value.detach()
            if key not in self.ema_state:
                self.ema_state[key] = value_detached.clone()
                continue
            ema_value = self.ema_state[key]
            if (
                torch.is_floating_point(value_detached)
                and key not in self._frozen_keys
            ):
                ema_value.mul_(self.decay).add_(value_detached, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(value_detached)

        stale_keys = [k for k in self.ema_state.keys() if k not in model_state]
        for stale_key in stale_keys:
            self.ema_state.pop(stale_key, None)

    def apply_to(self, model: torch.nn.Module):
        model_core = _get_model_core(model)
        current_state = model_core.state_dict()
        self.backup_state = {k: v.detach().clone() for k, v in current_state.items()}
        model_core.load_state_dict(self.ema_state, strict=False)
        _assert_no_mask_embedding_contract(model_core, "EMA state load")

    def restore(self, model: torch.nn.Module):
        if self.backup_state is None:
            return
        model_core = _get_model_core(model)
        model_core.load_state_dict(self.backup_state, strict=False)
        _assert_no_mask_embedding_contract(model_core, "EMA state restore")
        self.backup_state = None

    def state_dict(self) -> Dict:
        return {
            "decay": self.decay,
            "ema_state": {k: v.detach().clone() for k, v in self.ema_state.items()},
        }

    def load_state_dict(self, state: Dict):
        self.decay = float(state.get("decay", self.decay))
        loaded = state.get("ema_state", {})
        default_device = None
        for tensor in self.ema_state.values():
            default_device = tensor.device
            break

        for key in list(self.ema_state.keys()):
            if key in loaded:
                target = self.ema_state[key]
                source = loaded[key].detach().to(device=target.device, dtype=target.dtype)
                target.copy_(source)
        for key, value in loaded.items():
            if key not in self.ema_state:
                source = value.detach()
                if default_device is not None:
                    source = source.to(device=default_device)
                self.ema_state[key] = source.clone()

