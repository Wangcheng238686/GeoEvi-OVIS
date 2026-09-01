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



def _capture_git_state() -> Dict[str, str]:
    """Fingerprint the executed code: HEAD, dirty flag, tracked-diff sha.

    Complements the runner's resolved_hyperparameters git_* lines so both the
    training log and every checkpoint pin the exact working-tree state, not
    just the commit (a dirty tree silently diverging from HEAD is what made
    the resumed paper run hard to trace).
    """
    import subprocess

    def _git(*argv: str) -> str:
        try:
            return subprocess.run(
                ["git", *argv],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except Exception:
            return ""

    head = _git("rev-parse", "HEAD") or "unknown"
    status = _git("status", "--porcelain")
    dirty = "1" if status else "0"
    code_files = ",".join(
        line.split(maxsplit=1)[1].strip()
        for line in status.splitlines()
        if line.split(maxsplit=1)[-1].strip().endswith((".py", ".sh", ".md"))
    )[:400]
    try:
        diff_sha = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        import hashlib

        diff_sha16 = (
            hashlib.sha256(diff_sha.stdout.encode("utf-8")).hexdigest()[:16]
            if diff_sha.stdout
            else "none"
        )
    except Exception:
        diff_sha16 = "unknown"
    return {
        "git_commit": head,
        "git_dirty": dirty,
        "git_dirty_code_files": code_files or "none",
        "git_diff_sha16": diff_sha16,
    }

def _resolve_ddp_timeout_seconds() -> int:
    raw_value = os.environ.get(
        "TORCH_DDP_TIMEOUT_SECONDS", os.environ.get("NCCL_TIMEOUT", "1800")
    )
    try:
        timeout_seconds = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "TORCH_DDP_TIMEOUT_SECONDS must be a positive integer, "
            f"got {raw_value!r}"
        ) from exc
    if timeout_seconds < 1:
        raise ValueError(
            "TORCH_DDP_TIMEOUT_SECONDS must be >= 1, "
            f"got {timeout_seconds}"
        )
    return timeout_seconds

def _resolve_ddp_control_timeout_seconds() -> int:
    raw_value = os.environ.get("TORCH_DDP_CONTROL_TIMEOUT_SECONDS", "86400")
    try:
        timeout_seconds = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "TORCH_DDP_CONTROL_TIMEOUT_SECONDS must be a positive integer, "
            f"got {raw_value!r}"
        ) from exc
    if timeout_seconds < 1:
        raise ValueError(
            "TORCH_DDP_CONTROL_TIMEOUT_SECONDS must be >= 1, "
            f"got {timeout_seconds}"
        )
    return timeout_seconds

def _init_distributed() -> Dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    launched_by_torchrun = os.environ.get("LOCAL_RANK", None) is not None
    should_init = (world_size > 1) or launched_by_torchrun
    if should_init and not dist.is_initialized():
        # Bind each worker before NCCL creates communicators. Initializing NCCL
        # while every process still points at cuda:0 can leave rank 0 with an
        # avoidable memory imbalance.
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=_resolve_ddp_timeout_seconds()),
        )
    distributed = dist.is_initialized()
    control_group = None
    if distributed and world_size > 1:
        # Validation is intentionally rank-0-only and can take longer than the
        # training NCCL timeout. Keep epoch control on CPU/Gloo so idle ranks do
        # not hold an NCCL collective open throughout validation/COCO eval.
        control_group = dist.new_group(
            ranks=list(range(world_size)),
            backend="gloo",
            timeout=timedelta(seconds=_resolve_ddp_control_timeout_seconds()),
        )
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "distributed": int(distributed),
        "control_group": control_group,
    }

def _broadcast_stop_training(
    stop_training: bool, control_group: Optional[Any]
) -> bool:
    """Synchronize rank-0's epoch decision without occupying NCCL."""
    if not dist.is_initialized():
        return bool(stop_training)
    if control_group is None:
        raise RuntimeError(
            "distributed epoch control requires the CPU/Gloo control group"
        )
    stop_tensor = torch.tensor(
        [1 if stop_training else 0], device="cpu", dtype=torch.int32
    )
    dist.broadcast(stop_tensor, src=0, group=control_group)
    return bool(stop_tensor.item())

def _is_main_process(rank: int) -> bool:
    return rank == 0

