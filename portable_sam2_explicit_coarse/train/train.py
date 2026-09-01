#!/usr/bin/env python

import argparse
import gc
import logging
import math
import os
import sys
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler

from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmengine.structures import InstanceData


import sys as _sys
_ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

# Behaviour-preserving split: helpers moved to sibling modules.
from train.checkpointing import (  # noqa: F401
    _assert_no_mask_embedding_contract,
    _load_checkpoint,
    _load_init_checkpoint,
    _save_checkpoint,
    _save_verified_best_checkpoint,
)
from train.distributed import (  # noqa: F401
    _broadcast_stop_training,
    _capture_git_state,
    _init_distributed,
    _is_main_process,
    _resolve_ddp_control_timeout_seconds,
    _resolve_ddp_timeout_seconds,
)
from train.eval_metrics import (  # noqa: F401
    _env_flag,
    _extract_instances_numpy,
    _mask_to_rle,
    _metric_expression_value,
    _metric_value,
    _masks_to_rles,
    _summarize_mask_density,
    _weighted_metric_value,
    _xyxy_to_xywh,
    build_coco_gt_and_dt,
    run_coco_eval,
)
from train.losses import (  # noqa: F401
    _apply_detection_loss_weight,
    _detection_loss_weight,
    _materialize_p2_boundary_refiner_loss,
    _scale_loss_value,
    _set_model_current_epoch,
)
from train.monitors import (  # noqa: F401
    _accumulate_dense_residual_monitor,
    _accumulate_p2br_epoch_monitor,
    _log_prompt_debug_stats,
    _log_trainable_state,
    _reduce_dense_residual_monitor,
    _reduce_p2br_epoch_monitor,
    _stat_scalar,
)
from train.optim import (  # noqa: F401
    ExponentialMovingAverage,
    _build_optimizer,
    _get_model_core,
    _set_norm_eval,
)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("portable_sam_fusion")


# ---------------------------------------------------------------------------
# COCO-style evaluation helpers (from inference_rsprompter_fusion.py)
# ---------------------------------------------------------------------------





















































































def main():
    parser = argparse.ArgumentParser(
        description="portable_sam2_explicit_coarse: single-stream WHU-1024 SAM2 training"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--data-root", type=str, default="/data/wangcheng/dataset/university-test/S"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--checkpoint-dir", type=str, default="/tmp/portable_sam_fusion_ckpts"
    )
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=False,
        help="Load weights via --resume-from and run COCO evaluation once on the "
        "WHU test split (2.2 test/test + test.json), then exit. No training. "
        "Requires --resume-from pointing at a checkpoint with a matching "
        "architecture contract.",
    )
    parser.add_argument(
        "--allow-cross-arch-init",
        action="store_true",
        help="Explicitly allow INIT_FROM from a different or legacy architecture.",
    )
    parser.add_argument(
        "--init-exclude-prefixes",
        type=str,
        default="",
        help="Comma-separated model state prefixes excluded from --init-from",
    )
    parser.add_argument("--gpu-id", type=int, default=0)

    parser.add_argument(
        "--use-whu-coco",
        action="store_true",
        default=False,
        help="使用 WHU 公开数据集（COCO 格式，自带 train/validation 划分）",
    )
    parser.add_argument(
        "--subset-ratio",
        type=float,
        default=None,
        help="Deprecated compatibility alias: set both train and val subset ratios.",
    )
    parser.add_argument(
        "--train-subset-ratio",
        type=float,
        default=1.0,
        help="Deterministic fraction of WHU training images to load; default 1.0.",
    )
    parser.add_argument(
        "--val-subset-ratio",
        type=float,
        default=1.0,
        help="Deterministic fraction of WHU validation images to load; default 1.0.",
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512])

    parser.add_argument("--use-fsdp", action="store_true")
    parser.add_argument("--fsdp-sharding", type=str, default="FULL_SHARD")
    parser.add_argument("--fsdp-min-num-params", type=int, default=1000000)

    parser.add_argument("--freeze-bn", action="store_true")

    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=1,
        help="Validation batch size (legacy WHU1024 baseline default: 1)",
    )
    parser.add_argument("--val-every-n-epochs", type=int, default=1)
    parser.add_argument(
        "--compute-val-loss",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Whether validation also runs a second loss forward on positive-GT "
            "batches. COCO prediction/evaluation is unaffected."
        ),
    )
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-start-epoch", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=5e-4)
    parser.add_argument(
        "--early-stopping-smooth-window",
        type=int,
        default=5,
        help="Moving-average window for the resolved early-stopping metric.",
    )
    parser.add_argument("--det-loss-stage1-end", type=int, default=5)
    parser.add_argument("--det-loss-stage2-end", type=int, default=10)
    parser.add_argument("--det-loss-weight-stage1", type=float, default=1.0)
    parser.add_argument("--det-loss-weight-stage2", type=float, default=1.0)
    parser.add_argument("--det-loss-weight-stage3", type=float, default=1.0)
    parser.add_argument(
        "--max-nonfinite-batches",
        type=int,
        default=0,
        help="Abort after this many synchronized non-finite batches; 0 disables abort.",
    )

    parser.add_argument("--sat-backbone-lr-mult", type=float, default=1.0)
    parser.add_argument("--sat-other-lr-mult", type=float, default=1.0)
    parser.add_argument(
        "--bbox-head-lr-mult",
        type=float,
        default=0.0,
        help="Use >0 to place bbox_head in a separate optimizer group",
    )
    parser.add_argument("--mask-decoder-lr-mult", type=float, default=1.0)
    parser.add_argument("--no-mask-lr-mult", type=float, default=1.0)
    parser.add_argument("--prompt-encoder-lr-mult", type=float, default=0.1)
    parser.add_argument("--shape-prior-lr-mult", type=float, default=1.0)
    parser.add_argument("--p2-boundary-refiner-lr-mult", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Number of warmup epochs for learning rate scheduling")
    parser.add_argument("--warmup-iters", type=int, default=100, help="Number of warmup optimizer steps (overrides warmup-epochs if > 0)")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay for optimizer")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for regularization")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # --- AMP / EMA (训练基础设施) ---
    parser.add_argument("--amp", type=int, default=0, help="Enable AMP mixed precision when set to 1")
    parser.add_argument("--ema-enabled", type=int, default=1, help="Enable EMA shadow weights")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay in (0,1)")
    parser.add_argument("--ema-update-every", type=int, default=1, help="Update EMA every N optimizer steps")
    parser.add_argument("--ema-eval", type=int, default=1, help="Use EMA shadow weights for validation")
    parser.add_argument("--ema-save-best", type=int, default=1, help="Save best_model using EMA shadow weights")
    parser.add_argument("--ema-eval-start-epoch", type=int, default=5, help="Start using EMA for eval after this epoch")
    parser.add_argument("--prompt-debug-stats", type=int, choices=[0, 1], default=0,
                        help="Print shape-point/coarse-mask prompt diagnostics")
    parser.add_argument("--prompt-debug-stats-interval", type=int, default=50,
                        help="Training iteration interval for prompt diagnostics")

    args = parser.parse_args()

    if int(args.det_loss_stage1_end) < 1:
        raise ValueError("det_loss_stage1_end must be >= 1")
    if int(args.det_loss_stage2_end) <= int(args.det_loss_stage1_end):
        raise ValueError(
            "det_loss_stage2_end must be greater than det_loss_stage1_end"
        )
    for _name in (
        "det_loss_weight_stage1",
        "det_loss_weight_stage2",
        "det_loss_weight_stage3",
    ):
        if float(getattr(args, _name)) < 0:
            raise ValueError(f"{_name} must be non-negative")
    if int(args.early_stopping_start_epoch) < 1:
        raise ValueError("early_stopping_start_epoch must be >= 1")
    if int(args.early_stopping_patience) < 0:
        raise ValueError("early_stopping_patience must be >= 0")
    if float(args.early_stopping_min_delta) < 0:
        raise ValueError("early_stopping_min_delta must be >= 0")
    if int(args.early_stopping_smooth_window) < 1:
        raise ValueError("early_stopping_smooth_window must be >= 1")
    if int(args.val_batch_size) < 1:
        raise ValueError("val_batch_size must be >= 1")
    if int(args.val_every_n_epochs) < 1:
        raise ValueError("val_every_n_epochs must be >= 1")

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    import mmdet.models
    from mmengine.config import Config
    from mmengine.registry import MODELS as MMENGINE_MODELS
    from mmdet.registry import MODELS
    from mmdet.models.data_preprocessors import DetDataPreprocessor

    if "DetDataPreprocessor" not in MMENGINE_MODELS:
        MMENGINE_MODELS.register_module(
            name="DetDataPreprocessor", module=DetDataPreprocessor
        )

    import rsprompter

    # === 固定随机种子 (可复现) ===
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # CUDNN_BENCHMARK=1 trades bit-level reproducibility for conv autotuning
    # on fixed-size inputs (public runners keep the deterministic default).
    cudnn_benchmark = os.environ.get("CUDNN_BENCHMARK", "0") == "1"
    torch.backends.cudnn.deterministic = not cudnn_benchmark
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"[seed] Random seed fixed to {args.seed} "
            f"| cudnn.deterministic={not cudnn_benchmark} "
            f"| cudnn.benchmark={cudnn_benchmark}"
        )

    dist_info = _init_distributed()
    distributed = bool(dist_info.get("distributed", 0))
    rank = int(dist_info["rank"])
    control_group = dist_info.get("control_group")

    if torch.cuda.is_available():
        device = (
            f"cuda:{dist_info['local_rank']}" if distributed else f"cuda:{args.gpu_id}"
        )
    else:
        device = "cpu"

    is_main = _is_main_process(rank)
    checkpoint_dir = Path(args.checkpoint_dir)
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "DDP process-group timeout: %d seconds",
            _resolve_ddp_timeout_seconds(),
        )
        if control_group is not None:
            logger.info(
                "DDP validation control: backend=gloo timeout=%d seconds",
                _resolve_ddp_control_timeout_seconds(),
            )

    cfg = Config.fromfile(args.config)
    from rsprompter.architecture_contract import architecture_contract

    resolved_architecture = architecture_contract(cfg.model.to_dict())
    sam2_checkpoint_path = os.environ.get("SAM2_CKPT", "")
    sam2_identity = [sam2_checkpoint_path, ""]
    if is_main and sam2_checkpoint_path:
        from rsprompter.architecture_contract import file_sha256

        sam2_identity[1] = file_sha256(sam2_checkpoint_path)
    if distributed and dist.is_initialized():
        dist.broadcast_object_list(
            sam2_identity, src=0, group=control_group
        )
    if is_main:
        logger.info(
            "Architecture contract: id=%s schema=%d fingerprint=%s",
            resolved_architecture["architecture_id"],
            resolved_architecture["schema_version"],
            resolved_architecture["model_fingerprint"],
        )

    model = MODELS.build(cfg.model)
    model.to(device)

    model_for_preproc = model.module if hasattr(model, "module") else model
    data_preprocessor = model_for_preproc.data_preprocessor
    evaluation_mask_head = model_for_preproc.roi_head.mask_head
    segm_score_mode = getattr(evaluation_mask_head, "segm_score_mode", "detector")
    segm_score_key = getattr(evaluation_mask_head, "segm_score_key", "scores")
    if segm_score_mode != "detector" or segm_score_key != "scores":
        raise RuntimeError("segm score contract must be detector/scores")
    if is_main:
        logger.info(
            "Evaluation score contract: bbox=scores segm=%s mode=%s",
            segm_score_key,
            segm_score_mode,
        )

    if args.freeze_bn:
        _set_norm_eval(model)

    if distributed and not args.use_fsdp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        # find_unused_parameters=True：PE freeze_all=True 后有参数不收梯度，
        # 且 mask_head 的 ROI 数量随 batch 动态变化，必须开启避免 DDP hang。
        model = DDP(
            model,
            device_ids=[int(device.split(":")[1])]
            if device.startswith("cuda")
            else None,
            find_unused_parameters=True,
        )

    if distributed and args.use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy

        sharding = getattr(
            ShardingStrategy, args.fsdp_sharding, ShardingStrategy.FULL_SHARD
        )

        model = FSDP(
            model,
            sharding_strategy=sharding,
            device_id=int(device.split(":")[1]) if device.startswith("cuda") else None,
            use_orig_params=True,
            min_num_params=int(args.fsdp_min_num_params),
        )

    from data import create_train_loader

    train_subset_ratio = float(args.train_subset_ratio)
    val_subset_ratio = float(args.val_subset_ratio)
    if args.subset_ratio is not None:
        if args.train_subset_ratio != 1.0 or args.val_subset_ratio != 1.0:
            raise ValueError(
                "--subset-ratio cannot be combined with --train-subset-ratio "
                "or --val-subset-ratio"
            )
        train_subset_ratio = val_subset_ratio = float(args.subset_ratio)
    for name, value in (
        ("train_subset_ratio", train_subset_ratio),
        ("val_subset_ratio", val_subset_ratio),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1], got {value}")

    # --- train-time augmentation knobs (env-driven; defaults preserve the
    # historical hflip-only protocol, individual wrappers opt in) ---
    train_flip_prob = float(os.environ.get("TRAIN_FLIP_PROB", "0.5"))
    train_vflip_prob = float(os.environ.get("TRAIN_VFLIP_PROB", "0.0"))
    train_ms_resize_prob = float(
        os.environ.get("TRAIN_MULTI_SCALE_RESIZE_PROB", "0.0")
    )
    train_ms_mode = os.environ.get("TRAIN_MULTI_SCALE_MODE", "value")
    train_ms_img_scale_spec = os.environ.get(
        "TRAIN_MULTI_SCALE_IMG_SCALE",
        "896:896,960:960,1024:1024,1088:1088,1152:1152",
    )
    train_ms_img_scale = None
    if train_ms_img_scale_spec.strip():
        train_ms_img_scale = []
        for part in train_ms_img_scale_spec.split(","):
            w_text, h_text = part.strip().split(":", 1)
            train_ms_img_scale.append((int(w_text), int(h_text)))
    for name, value in (
        ("TRAIN_FLIP_PROB", train_flip_prob),
        ("TRAIN_VFLIP_PROB", train_vflip_prob),
        ("TRAIN_MULTI_SCALE_RESIZE_PROB", train_ms_resize_prob),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0,1], got {value}")
    if train_ms_mode not in ("value", "range"):
        raise ValueError(
            f"TRAIN_MULTI_SCALE_MODE must be value or range, got {train_ms_mode}"
        )

    train_loader, val_loader, train_dataset = (
        create_train_loader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            flip_prob=train_flip_prob,
            vflip_prob=train_vflip_prob,
            gaussian_noise_prob=0.0,
            gaussian_noise_std=0.02,
            random_erasing_prob=0.0,
            random_erasing_scale=(0.02, 0.15),
            random_erasing_ratio=(0.3, 3.3),
            # Multi-scale jitter resizes BEFORE the fixed resize back to
            # image_size: the network (and frozen SAM2) always receives its
            # native 1024x1024 input; only object scale varies. Kept off by
            # default — random_erasing stays disabled because it erases image
            # pixels without updating GT masks (label corruption for
            # instance segmentation).
            multi_scale_resize_prob=train_ms_resize_prob,
            multi_scale_mode=train_ms_mode,
            multi_scale_img_scale=train_ms_img_scale,
            image_size=tuple(args.image_size),
            distributed=distributed,
            rank=rank,
            world_size=int(dist_info.get("world_size", 1)),
            val_ratio=args.val_ratio,
            val_batch_size=args.val_batch_size,
            dataset_format="whu_coco" if args.use_whu_coco else "labelme",
            whu_train_ann_file=os.environ.get(
                "WHU_TRAIN_ANN_FILE", "2.4 annotation/annotation/train.json"
            ),
            whu_val_ann_file=os.environ.get(
                "WHU_VAL_ANN_FILE", "2.4 annotation/annotation/validation.json"
            ),
            whu_train_img_subdir=os.environ.get(
                "WHU_TRAIN_IMG_SUBDIR", "2.1 train/train"
            ),
            whu_val_img_subdir=os.environ.get(
                "WHU_VAL_IMG_SUBDIR", "2.3 valid/validation"
            ),
            train_subset_ratio=train_subset_ratio,
            val_subset_ratio=val_subset_ratio,
            seed=int(args.seed),
        )
    )
    # Preserve the proven WHU1024 baseline validation contract: rank 0 runs
    # the complete validation loader while the other DDP ranks wait at the
    # existing end-of-epoch synchronization point.  Iterating the unsharded
    # loader on every rank duplicates full-resolution mask accumulation and
    # can exhaust host memory.
    if distributed and not is_main:
        val_loader = None
    if is_main:
        logger.info(
            "Validation policy: rank0_only=%d batch_size=%d every_n_epochs=%d compute_loss=%d",
            int(distributed),
            int(args.val_batch_size),
            int(args.val_every_n_epochs),
            int(args.compute_val_loss),
        )

    num_batches_per_epoch = len(train_loader)
    if args.max_train_batches > 0:
        num_batches_per_epoch = min(num_batches_per_epoch, args.max_train_batches)

    # 问题 10：scheduler 必须按 optimizer step 计数，而不是 mini-batch 数。
    # 之前 cosine_t_max / warmup_iters 都按 mini-batch 算，但 scheduler.step() 只在
    # optimizer.step() 后调用一次（grad_accum 组才一次），导致 grad_accum>1 时
    # cosine 永远走不到 eta_min、warmup 需要 accum 倍 epoch 才结束。
    grad_accum_for_sched = max(1, int(args.grad_accum_steps))
    optimizer_steps_per_epoch = math.ceil(num_batches_per_epoch / grad_accum_for_sched)
    total_optimizer_steps = max(1, args.epochs * optimizer_steps_per_epoch)

    if is_main:
        logger.info(
            "Dataset size: %d | mini-batch/epoch=%d | grad_accum=%d | "
            "optimizer_steps/epoch=%d | total_optimizer_steps=%d",
            len(train_dataset), num_batches_per_epoch, grad_accum_for_sched,
            optimizer_steps_per_epoch, total_optimizer_steps,
        )

    optimizer = _build_optimizer(
        model.module if hasattr(model, "module") else model,
        lr=args.lr,
        sat_backbone_lr_mult=args.sat_backbone_lr_mult,
        sat_other_lr_mult=args.sat_other_lr_mult,
        bbox_head_lr_mult=args.bbox_head_lr_mult,
        mask_decoder_lr_mult=args.mask_decoder_lr_mult,
        no_mask_lr_mult=args.no_mask_lr_mult,
        prompt_encoder_lr_mult=args.prompt_encoder_lr_mult,
        shape_prior_lr_mult=args.shape_prior_lr_mult,
        p2_boundary_refiner_lr_mult=args.p2_boundary_refiner_lr_mult,
        weight_decay=args.weight_decay,
    )
    if is_main:
        _log_trainable_state(model)

    # 问题 10：warmup / cosine 全部按 optimizer step 计数（与 scheduler.step() 调用频率一致）。
    # --warmup-iters 现在表示 optimizer step 数（而非 mini-batch），语义更准确；
    # grad_accum==1 时 optimizer step == mini-batch，行为与旧版完全一致。
    if args.warmup_iters > 0:
        warmup_optimizer_steps = args.warmup_iters
        if is_main:
            logger.info(f"Using user-specified warmup: {warmup_optimizer_steps} optimizer steps")
    elif args.warmup_epochs > 0:
        warmup_optimizer_steps = args.warmup_epochs * optimizer_steps_per_epoch
        if is_main:
            logger.info(f"Using epoch-based warmup: {args.warmup_epochs} epochs "
                        f"({warmup_optimizer_steps} optimizer steps)")
    else:
        warmup_optimizer_steps = min(50, optimizer_steps_per_epoch)
        if is_main:
            logger.info(f"Using default warmup: {warmup_optimizer_steps} optimizer steps")

    cosine_t_max = max(1, total_optimizer_steps - warmup_optimizer_steps)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_t_max, eta_min=args.lr * 0.001
    )
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.001, total_iters=warmup_optimizer_steps
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_optimizer_steps],
    )
    # 保留旧变量名供 config_snapshot 引用（向后兼容）
    warmup_iters = warmup_optimizer_steps

    # AMP scaler (--amp 0 关闭混合精度，用 fp32 训练)
    amp_enabled = bool(args.amp) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # === 构建训练 config 快照 (保存到 checkpoint, 推理时可直接加载) ===
    # Historical WHU1024 baseline reports/selects bbox/mAP as its primary bbox
    # metric.  Keep mAP_75 in the detailed metrics, but do not silently use it
    # as the headline or best-bbox checkpoint criterion.
    save_bbox_best_metric = os.environ.get("SAVE_BBOX_BEST_METRIC", "bbox/mAP")
    early_stopping_metric_spec = os.environ.get(
        "EARLY_STOPPING_METRIC", "segm/mAP"
    )
    resolved_mask_head_cfg = cfg.model.roi_head.mask_head
    resolved_point_warmup = dict(
        resolved_mask_head_cfg.get("point_warmup_cfg", {})
    )
    resolved_shape_point_miner = dict(
        resolved_mask_head_cfg.get("shape_point_miner_cfg", {})
    )
    resolved_coarse_loss = dict(
        resolved_mask_head_cfg.get("coarse_mask_loss_cfg", {})
    )
    resolved_shape_prior = dict(
        resolved_mask_head_cfg.get("shape_prior_cfg", {})
    )
    config_snapshot = {
        "checkpoint_schema_version": 2,
        "architecture_contract": dict(resolved_architecture),
        "config_path": str(args.config),
        "training_args": dict(vars(args)),
        "train_augmentation": {
            "flip_prob": train_flip_prob,
            "vflip_prob": train_vflip_prob,
            "multi_scale_resize_prob": train_ms_resize_prob,
            "multi_scale_mode": train_ms_mode,
            "multi_scale_img_scale": train_ms_img_scale,
        },
        # HEAD alone cannot identify the executed code when the working tree
        # is dirty; carry the dirty flag + diff fingerprint so a checkpoint
        # uniquely pins its code state (same fields as the runner's
        # resolved_hyperparameters git_* lines).
        "git_state": _capture_git_state(),
        "data_config": {
            "dataset_format": "whu_coco" if args.use_whu_coco else "labelme",
            "data_root": str(args.data_root),
            "image_size": list(args.image_size),
            "single_class": True,
            "validation": {
                "ann_file": "2.4 annotation/annotation/validation.json",
                "image_subdir": "2.3 valid/validation",
            },
            "test": {
                "ann_file": "2.4 annotation/annotation/test.json",
                "image_subdir": "2.2 test/test",
            },
        },
        "runtime_config": {
            "sam2_repo": os.environ.get("SAM2_REPO", ""),
            "sam2_checkpoint": os.environ.get("SAM2_CKPT", ""),
            "sam2_checkpoint_sha256": sam2_identity[1],
            "sam2_model_size": os.environ.get("SAM2_MODEL_SIZE", ""),
            "ddp_timeout_seconds": _resolve_ddp_timeout_seconds(),
            "ddp_control_backend": "gloo" if control_group is not None else "none",
            "ddp_control_timeout_seconds": _resolve_ddp_control_timeout_seconds(),
        },
        "train_subset_ratio": float(train_subset_ratio),
        "val_subset_ratio": float(val_subset_ratio),
        "explicit_prompt_mode": os.environ.get("EXPLICIT_PROMPT_MODE", ""),
        "explicit_use_box_prompt": os.environ.get("EXPLICIT_USE_BOX_PROMPT", ""),
        "use_shape_dense": os.environ.get("USE_SHAPE_DENSE", ""),
        "prompt_mode": os.environ.get("PROMPT_MODE", ""),
        "neck_type": os.environ.get("NECK_TYPE", ""),
        "prompt_dense_mode": os.environ.get("PROMPT_DENSE_MODE", ""),
        "save_bbox_best_metric": save_bbox_best_metric,
        "early_stopping_metric_spec": early_stopping_metric_spec,
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_start_epoch": int(args.early_stopping_start_epoch),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "early_stopping_smooth_window": int(
            args.early_stopping_smooth_window
        ),
        "det_loss_schedule": {
            "stage1_end": int(args.det_loss_stage1_end),
            "stage2_end": int(args.det_loss_stage2_end),
            "stage1_weight": float(args.det_loss_weight_stage1),
            "stage2_weight": float(args.det_loss_weight_stage2),
            "stage3_weight": float(args.det_loss_weight_stage3),
        },
        "point_warmup": {
            "enabled": bool(resolved_point_warmup.get("enabled", False)),
            "no_point_epochs": int(
                resolved_point_warmup.get("no_point_epochs", 0)
            ),
            "one_pair_epochs": int(
                resolved_point_warmup.get("one_pair_epochs", 0)
            ),
            "full_2p2n_start_epoch": int(
                resolved_point_warmup.get("full_2p2n_start_epoch", 1)
            ),
        },
        "shape_point_mining": {
            "adaptive_validity": bool(
                resolved_shape_point_miner.get("adaptive_validity", False)
            ),
        },
        "shape_loss_schedule": {
            "mode": resolved_coarse_loss.get("schedule_mode", "fixed"),
            "fixed_weight": float(resolved_coarse_loss.get("weight", 0.0)),
            "ranges": [
                dict(item)
                for item in resolved_coarse_loss.get("weight_schedule", [])
            ],
        },
        "shape_dense_gate": {
            "mode": resolved_shape_prior.get("shape_scale_mode", ""),
            "init": resolved_shape_prior.get("prompt_scale_init", ""),
        },
        "freeze_decoder": os.environ.get("FREEZE_DECODER", ""),
        "freeze_no_mask": os.environ.get("FREEZE_NO_MASK", ""),
        "shape_prior_loss_weight": float(
            resolved_mask_head_cfg.get("shape_prior_loss_weight", 0.0)
        ),
        "p2_boundary_refiner": dict(
            resolved_mask_head_cfg.get("p2_boundary_refiner_cfg", {})
        ),
        "bbox_arch": os.environ.get("AP75_BBOX_ARCH", ""),
        "bbox_iou_loss_weight": os.environ.get("AP75_BBOX_IOU_LOSS_WEIGHT", ""),
        "init_from": args.init_from or "",
        "init_exclude_prefixes": args.init_exclude_prefixes,
        "prompt_debug_stats": args.prompt_debug_stats,
        "prompt_debug_stats_interval": args.prompt_debug_stats_interval,
        "seed": args.seed,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "epochs": args.epochs,
        "warmup_iters": warmup_iters,
        # 问题 10：scheduler 真实进度依据（optimizer step 维度）
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps * max(1, int(os.environ.get("WORLD_SIZE", "1"))),
        "weight_decay": args.weight_decay,
        "lr_mults": {
            "sat_backbone": args.sat_backbone_lr_mult,
            "sat_other": args.sat_other_lr_mult,
            "bbox_head": args.bbox_head_lr_mult,
            "mask_decoder": args.mask_decoder_lr_mult,
            "no_mask": args.no_mask_lr_mult,
            "prompt_encoder": args.prompt_encoder_lr_mult,
            "shape_prior": args.shape_prior_lr_mult,
            "p2_boundary_refiner": args.p2_boundary_refiner_lr_mult,
        },
        "image_size": list(args.image_size),
        "mask_head_config": dict(cfg.model.roi_head.mask_head)
            if hasattr(cfg.model.roi_head, "mask_head") else {},
        "neck_config": dict(cfg.model.get("neck", {})),
        # 完整解析后的 cfg.model (含 type/backbone/roi_head/...): 推理时可据此直接重建模型,
        # 无需再传 .py config 或重设环境变量。新字段, 旧 ckpt 没有时推理脚本自动 fallback。
        "model_config": cfg.model.to_dict(),
    }

    start_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_segm_map = 0.0  # Track best segm mAP
    best_bbox_score = 0.0
    best_early_stopping_score = float("-inf")
    early_counter = 0
    early_stopping_metric_history = []
    history = {"epochs": [], "train_losses": [], "val_losses": [], "val_metrics": [], "learning_rates": []}

    # EMA shadow weights (eval/save-best only, never for backprop)
    ema = None
    ema_update_steps = 0
    ema_enabled = bool(args.ema_enabled) and (0.0 < float(args.ema_decay) < 1.0)
    if ema_enabled:
        ema = ExponentialMovingAverage(model, decay=float(args.ema_decay))
        if is_main:
            logger.info(
                "EMA enabled: decay=%.4f update_every=%d eval=%d save_best=%d eval_start_epoch=%d",
                float(args.ema_decay), int(args.ema_update_every),
                int(args.ema_eval), int(args.ema_save_best), int(args.ema_eval_start_epoch),
            )

    if args.resume_from and args.init_from:
        raise ValueError("--resume-from and --init-from are mutually exclusive")

    if args.init_from:
        exclude_prefixes = tuple(
            prefix.strip()
            for prefix in args.init_exclude_prefixes.split(",")
            if prefix.strip()
        )
        init_stats = _load_init_checkpoint(
            Path(args.init_from),
            model,
            exclude_prefixes=exclude_prefixes,
            expected_architecture=resolved_architecture,
            allow_cross_arch=args.allow_cross_arch_init,
        )
        if is_main:
            logger.info(
                "Initialized model from %s loaded=%d excluded=%d shape_mismatch=%d unexpected=%d missing=%d migrated=%d prefixes=%s",
                args.init_from,
                init_stats["loaded"],
                init_stats["excluded"],
                init_stats["shape_mismatch"],
                init_stats["unexpected"],
                init_stats["missing"],
                init_stats["migrated"],
                exclude_prefixes or "<none>",
            )
            if init_stats.get("migration_pairs"):
                logger.info(
                    "Checkpoint key migration (INIT_FROM): %s",
                    init_stats["migration_pairs"],
                )

    if args.resume_from:
        ckpt = _load_checkpoint(
            Path(args.resume_from),
            model,
            optimizer,
            scheduler,
            scaler,
            expected_architecture=resolved_architecture,
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = ckpt.get("best_metrics", {}) or {}
        best_val_loss = float(best.get("best_val_loss", best_val_loss))
        best_train_loss = float(best.get("best_train_loss", best_train_loss))
        best_segm_map = float(best.get("best_segm_map", best_segm_map))
        best_bbox_score = float(best.get("best_bbox_score", best_bbox_score))
        best_early_stopping_score = float(
            best.get("best_early_stopping_score", best_early_stopping_score)
        )
        early_counter = int(best.get("early_counter", early_counter))
        early_stopping_metric_history = list(
            best.get(
                "early_stopping_metric_history",
                early_stopping_metric_history,
            )
        )
        history = ckpt.get("history", history) or history
        if ema is not None and "ema_state" in ckpt:
            ema.load_state_dict(ckpt["ema_state"])
            if is_main:
                logger.info("Restored EMA shadow weights from checkpoint")
        # 恢复 config 快照 (从 checkpoint 加载, 保证续训一致)
        if "config_snapshot" in ckpt:
            saved_snapshot = ckpt["config_snapshot"]
            if isinstance(saved_snapshot, dict):
                # Preserve the resumed run's recorded values while retaining
                # new schema fields for checkpoints produced by newer code.
                config_snapshot = {**config_snapshot, **saved_snapshot}
                config_snapshot["checkpoint_schema_version"] = 2
                config_snapshot["architecture_contract"] = dict(
                    resolved_architecture
                )
        if is_main:
            logger.info(
                "Resumed from %s (start_epoch=%d, best_segm_map=%.4f, best_bbox_score=%.4f)",
                args.resume_from,
                start_epoch,
                best_segm_map,
                best_bbox_score,
            )

    # ===== --test-only: evaluate on the held-out WHU test split, then exit =====
    if args.test_only:
        if not args.resume_from:
            raise SystemExit("--test-only requires --resume-from to load weights.")
        from data import create_test_loader

        test_loader = create_test_loader(
            data_root=args.data_root,
            ann_file="2.4 annotation/annotation/test.json",
            image_subdir="2.2 test/test",
            image_size=tuple(args.image_size),
            batch_size=int(args.val_batch_size),
            num_workers=4,
            seed=int(args.seed),
        )
        # Non-main ranks skip iteration to match the rank0_only validation
        # contract (no NCCL collective held open during COCO eval).
        if distributed and not is_main:
            test_loader = None

        model.eval()
        if args.freeze_bn:
            _set_norm_eval(model)

        all_gt: List[dict] = []
        all_dt: List[dict] = []
        all_img_metas: List[dict] = []

        if test_loader is not None:
            if is_main:
                logger.info(
                    "Test-only evaluation: checkpoint=%s, test split=%d images",
                    args.resume_from,
                    len(test_loader.dataset),
                )
            model_for_eval = model.module if hasattr(model, "module") else model
            # Runtime-only max_per_img override for dense scenes (>100
            # instances per tile). Mutates the live test_cfg objects, so the
            # config file and the architecture fingerprint stay untouched.
            _test_max_per_img = int(os.environ.get("TEST_MAX_PER_IMG", "0") or 0)
            if _test_max_per_img > 0:
                _model_test_cfg = getattr(model_for_eval, "test_cfg", None)
                if _model_test_cfg is not None and "rcnn" in _model_test_cfg:
                    _model_test_cfg["rcnn"]["max_per_img"] = _test_max_per_img
                _roi_test_cfg = getattr(
                    getattr(model_for_eval, "roi_head", None), "test_cfg", None
                )
                if _roi_test_cfg is not None and "max_per_img" in _roi_test_cfg:
                    _roi_test_cfg["max_per_img"] = _test_max_per_img
                if is_main:
                    logger.info(
                        "TEST_MAX_PER_IMG=%d override applied (model-side cap)",
                        _test_max_per_img,
                    )
            _test_eval_max_dets = (
                [1, 10, _test_max_per_img] if _test_max_per_img > 0 else None
            )
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                        break
                    imgs = batch["imgs"].to(device)
                    img_metas = batch["img_metas"]
                    gt_bboxes = [b.to(device) for b in batch["gt_bboxes"]]
                    gt_labels = [l.to(device) for l in batch["gt_labels"]]
                    gt_masks = batch["gt_masks"]

                    data_samples = []
                    for i in range(len(imgs)):
                        ds = DetDataSample()
                        ds.set_metainfo(img_metas[i])
                        gt_instances = InstanceData()
                        num_inst = len(gt_labels[i])
                        if num_inst > 0:
                            valid_mask = gt_labels[i] >= 0
                            num_valid = int(valid_mask.sum().item())
                            if num_valid > 0:
                                valid_indices = (
                                    valid_mask.nonzero().squeeze(-1)[:num_valid].cpu()
                                )
                                gt_instances.bboxes = gt_bboxes[i][valid_indices]
                                gt_instances.labels = gt_labels[i][valid_indices]
                                masks = gt_masks[i][valid_indices.cpu().numpy()]
                                gt_instances.masks = BitmapMasks(
                                    masks, *img_metas[i]["img_shape"]
                                )
                            else:
                                gt_instances.bboxes = torch.zeros(
                                    (0, 4), dtype=torch.float32, device=device
                                )
                                gt_instances.labels = torch.zeros(
                                    (0,), dtype=torch.int64, device=device
                                )
                                gt_instances.masks = BitmapMasks(
                                    np.zeros(
                                        (0, *img_metas[i]["img_shape"]), dtype=np.uint8
                                    ),
                                    *img_metas[i]["img_shape"],
                                )
                        else:
                            gt_instances.bboxes = torch.zeros(
                                (0, 4), dtype=torch.float32, device=device
                            )
                            gt_instances.labels = torch.zeros(
                                (0,), dtype=torch.int64, device=device
                            )
                            gt_instances.masks = BitmapMasks(
                                np.zeros(
                                    (0, *img_metas[i]["img_shape"]), dtype=np.uint8
                                ),
                                *img_metas[i]["img_shape"],
                            )
                        ds.gt_instances = gt_instances
                        data_samples.append(ds.to(device))

                    # Mirror the validation loop: run the model's
                    # data_preprocessor (padding/normalization) before predict.
                    processed = model_for_eval.data_preprocessor(
                        {"inputs": imgs, "data_samples": data_samples},
                        training=False,
                    )
                    model_inputs = processed["inputs"]
                    data_samples = processed["data_samples"]

                    outputs = model_for_eval.predict(
                        model_inputs, data_samples, rescale=False
                    )
                    for j, output in enumerate(outputs):
                        meta = img_metas[j]
                        img_shape = meta["img_shape"]
                        gt_inst = data_samples[j].gt_instances
                        gt_np = _extract_instances_numpy(
                            gt_inst, img_shape, encode_masks=True
                        )
                        all_gt.append(gt_np)
                        pred_inst = (
                            output.pred_instances
                            if hasattr(output, "pred_instances")
                            else output
                        )
                        pred_np = _extract_instances_numpy(
                            pred_inst, img_shape, encode_masks=True
                        )
                        all_dt.append(pred_np)
                        all_img_metas.append(meta)

        if is_main:
            total_gt = sum(len(g["labels"]) for g in all_gt)
            total_dt = sum(len(d["scores"]) for d in all_dt if "scores" in d)
            avg_mask_fill, avg_masks_per_image = _summarize_mask_density(all_dt)
            logger.info(
                "Test samples: GT=%d, DT=%d, avg_masks_per_image=%.2f, avg_mask_fill=%.4f",
                total_gt,
                total_dt,
                avg_masks_per_image,
                avg_mask_fill,
            )
            if total_gt > 0:
                coco_gt, coco_bbox_dt = build_coco_gt_and_dt(
                    all_gt, all_dt, all_img_metas, score_key="scores"
                )
                if True:
                    bbox_metrics = run_coco_eval(
                        coco_gt, coco_bbox_dt, iou_type="bbox",
                        max_dets=_test_eval_max_dets,
                    )
                    logger.info(
                        "Test bbox/mAP: %.4f bbox/mAP_75: %.4f",
                        bbox_metrics.get("bbox/mAP", 0.0),
                        bbox_metrics.get("bbox/mAP_75", 0.0),
                    )
                if segm_score_key == "scores":
                    coco_segm_dt = coco_bbox_dt
                else:
                    _, coco_segm_dt = build_coco_gt_and_dt(
                        all_gt,
                        all_dt,
                        all_img_metas,
                        score_key=segm_score_key,
                    )
                segm_metrics = run_coco_eval(
                    coco_gt, coco_segm_dt, iou_type="segm",
                    max_dets=_test_eval_max_dets,
                )
                logger.info(
                    "Test segm/mAP: %.4f segm/mAP_50: %.4f segm/mAP_75: %.4f segm/mAP_s: %.4f segm/mAP_m: %.4f",
                    segm_metrics.get("segm/mAP", 0.0),
                    segm_metrics.get("segm/mAP_50", 0.0),
                    segm_metrics.get("segm/mAP_75", 0.0),
                    segm_metrics.get("segm/mAP_s", 0.0),
                    segm_metrics.get("segm/mAP_m", 0.0),
                )

                # Full COCO stat dump: every metric key of both evals, plus a
                # test_metrics.json next to the checkpoint dir for scripting.
                _full_test_metrics = {}
                _full_test_metrics.update(bbox_metrics)
                _full_test_metrics.update(segm_metrics)
                for _mk, _mv in _full_test_metrics.items():
                    logger.info("Test metric %s: %.6f", _mk, float(_mv))
                try:
                    import json as _json

                    _metrics_path = checkpoint_dir / "test_metrics.json"
                    with _metrics_path.open("w", encoding="utf-8") as _mh:
                        _json.dump(
                            {
                                "checkpoint": str(args.resume_from),
                                "test_eval_max_dets": _test_eval_max_dets,
                                "metrics": _full_test_metrics,
                            },
                            _mh,
                            indent=2,
                        )
                    logger.info("Full test metrics written: %s", _metrics_path)
                except Exception:
                    logger.warning("Failed to write test_metrics.json", exc_info=True)

                # Optional prediction visualization (TEST_VIS_OUT=<dir>).
                # Records reuse the infer_from_checkpoint predictions.json
                # contract (model-frame RLE + absolute file_name), which
                # scripts/visualize_instances.py pred mode rescales onto the
                # stored images automatically.
                _vis_out = os.environ.get("TEST_VIS_OUT", "").strip()
                if _vis_out:
                    _vis_thr = float(os.environ.get("TEST_VIS_SCORE_THR", "0.3"))
                    _vis_limit = int(os.environ.get("TEST_VIS_LIMIT", "50"))
                    _vis_records = []
                    for _dt, _meta in zip(all_dt, all_img_metas):
                        _fn = str(
                            _meta.get("img_path") or _meta.get("filename") or ""
                        )
                        _rles = _dt.get("rles")
                        if _rles is None and _dt.get("masks") is not None:
                            _masks = _dt["masks"]
                            if hasattr(_masks, "masks"):
                                _masks = _masks.masks
                            _rles = _masks_to_rles(_masks)
                        for _bb, _sc, _rl in zip(
                            _dt["bboxes"], _dt["scores"], _rles or []
                        ):
                            if float(_sc) < _vis_thr:
                                continue
                            _counts = _rl["counts"]
                            if isinstance(_counts, bytes):
                                _counts = _counts.decode("utf-8")
                            _vis_records.append(
                                {
                                    "file_name": _fn,
                                    "category_id": 1,
                                    "bbox": [
                                        float(_bb[0]),
                                        float(_bb[1]),
                                        float(_bb[2] - _bb[0]),
                                        float(_bb[3] - _bb[1]),
                                    ],
                                    "score": float(_sc),
                                    "segmentation": {
                                        "size": _rl["size"],
                                        "counts": _counts,
                                    },
                                }
                            )
                    _vis_dir = Path(_vis_out).expanduser().resolve()
                    _vis_dir.mkdir(parents=True, exist_ok=True)
                    _pred_json = _vis_dir / "predictions.json"
                    import json as _vis_json

                    with _pred_json.open("w", encoding="utf-8") as _vh:
                        _vis_json.dump(_vis_records, _vh)
                    logger.info(
                        "Test visualization records: %d (score>=%.2f) -> %s",
                        len(_vis_records),
                        _vis_thr,
                        _pred_json,
                    )
                    import subprocess
                    import sys as _sys

                    _project_root = Path(__file__).resolve().parents[1]
                    _vis_cmd = [
                        _sys.executable,
                        "scripts/visualize_instances.py",
                        "pred",
                        "--pred-json",
                        str(_pred_json),
                        "--out-dir",
                        str(_vis_dir / "images"),
                        "--limit",
                        str(_vis_limit),
                    ]
                    logger.info("Visualization: %s", " ".join(_vis_cmd))
                    subprocess.run(
                        _vis_cmd, check=False, cwd=str(_project_root)
                    )
            else:
                logger.warning("No valid GT in test split; skipping COCO eval.")
            logger.info("Test-only evaluation complete. Exiting (no training).")
        # Ensure all ranks reach the exit together.
        if dist_info.get("world_size", 1) > 1:
            dist.barrier()
        return

    nonfinite_batches = 0
    # Validation order is deterministic within a run. Cache compact GT RLE
    # records after the first validation instead of rebuilding them every epoch.
    cached_val_gt_records: Optional[List[dict]] = None
    for epoch in range(start_epoch, args.epochs):
        epoch_number = epoch + 1
        _set_model_current_epoch(model, epoch_number)
        det_loss_weight = _detection_loss_weight(args, epoch_number)
        if distributed:
            sampler = getattr(train_loader, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

        model.train()
        if args.freeze_bn:
            _set_norm_eval(model)

        total_loss = 0.0
        loss_meter = {}
        dense_monitor_sums = {key: 0.0 for key in _DENSE_RESIDUAL_MONITOR_KEYS}
        dense_monitor_counts = {key: 0 for key in _DENSE_RESIDUAL_MONITOR_KEYS}
        p2br_monitor_sums = {key: 0.0 for key in _P2BR_EPOCH_FIELDS}
        num_batches = 0
        grad_accum = args.grad_accum_steps
        train_batches_limit = len(train_loader)
        if args.max_train_batches > 0:
            train_batches_limit = min(train_batches_limit, args.max_train_batches)
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break

            imgs = batch["imgs"].to(device)
            img_metas = batch["img_metas"]
            gt_bboxes = [b.to(device) for b in batch["gt_bboxes"]]
            gt_labels = [l.to(device) for l in batch["gt_labels"]]
            gt_masks = batch["gt_masks"]

            # Prepare data_samples and move to device
            data_samples: List[DetDataSample] = []
            for i in range(len(imgs)):
                ds = DetDataSample()
                ds.set_metainfo(img_metas[i])
                gt_instances = InstanceData()
                num_inst = len(gt_labels[i])
                if num_inst > 0:
                    valid_mask = gt_labels[i] >= 0
                    num_valid = int(valid_mask.sum().item())
                    if num_valid > 0:
                        valid_indices = (
                            valid_mask.nonzero().squeeze(-1)[:num_valid].cpu()
                        )
                        gt_instances.bboxes = gt_bboxes[i][valid_indices]
                        gt_instances.labels = gt_labels[i][valid_indices]
                        masks = gt_masks[i][valid_indices.cpu().numpy()]
                        gt_instances.masks = BitmapMasks(
                            masks, *img_metas[i]["img_shape"]
                        )
                    else:
                        gt_instances.bboxes = torch.zeros(
                            (0, 4), dtype=torch.float32, device=device
                        )
                        gt_instances.labels = torch.zeros(
                            (0,), dtype=torch.int64, device=device
                        )
                        gt_instances.masks = BitmapMasks(
                            np.zeros((0, *img_metas[i]["img_shape"]), dtype=np.uint8),
                            *img_metas[i]["img_shape"],
                        )
                else:
                    gt_instances.bboxes = torch.zeros(
                        (0, 4), dtype=torch.float32, device=device
                    )
                    gt_instances.labels = torch.zeros(
                        (0,), dtype=torch.int64, device=device
                    )
                    gt_instances.masks = BitmapMasks(
                        np.zeros((0, *img_metas[i]["img_shape"]), dtype=np.uint8),
                        *img_metas[i]["img_shape"],
                    )
                ds.gt_instances = gt_instances
                data_samples.append(ds.to(device))

            # Match the historical WHU1024 baseline: empty-GT training batches
            # still enter the detector loss. A rank-local early continue here
            # would make DDP ranks execute different collective sequences when
            # their DistributedSampler shards contain different empty images.

            # Synchronize data preprocessing for both images and ground truth
            processed = data_preprocessor(
                {"inputs": imgs, "data_samples": data_samples}, training=True
            )
            imgs = processed["inputs"]
            data_samples = processed["data_samples"]

            model_inputs = imgs

            model_for_loss = model.module if hasattr(model, "module") else model

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss_dict = model_for_loss.loss(model_inputs, data_samples)
                loss_dict = _materialize_p2_boundary_refiner_loss(
                    loss_dict, model, distributed
                )
                loss_dict = _apply_detection_loss_weight(
                    loss_dict, det_loss_weight, device
                )
                loss_parts = []
                for k, v in loss_dict.items():
                    if "loss" not in k:
                        continue
                    if isinstance(v, torch.Tensor):
                        if v.requires_grad:
                            loss_parts.append(v)
                    elif isinstance(v, (list, tuple)):
                        loss_parts.extend(
                            x
                            for x in v
                            if isinstance(x, torch.Tensor)
                            and (x.requires_grad)
                        )
                loss = sum(loss_parts) if loss_parts else torch.tensor(0.0, device=device)
                # 审查 3.6 / 问题 10：loss 归一化按当前 batch 所属 accumulation group 的
                # 真实长度，而非固定 grad_accum。之前只在最后一个 batch 修正分母，导致
                # 末组前面的 batch（如 batch_idx=4,5，末组真实只有 3 个）仍除以 grad_accum，
                # 该组总梯度被低估。
                group_start = (batch_idx // grad_accum) * grad_accum
                group_end = min(group_start + grad_accum, train_batches_limit)
                current_group_size = max(1, group_end - group_start)
                loss_for_backward = loss / float(current_group_size)

            # Record detailed losses
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    val = v.detach().item()
                    loss_meter[k] = loss_meter.get(k, 0.0) + val
                elif isinstance(v, list):
                    # Handle case where loss might be a list of tensors (though rare in final output dict)
                    val = sum(
                        x.detach().item() for x in v if isinstance(x, torch.Tensor)
                    )
                    loss_meter[k] = loss_meter.get(k, 0.0) + val
            _accumulate_dense_residual_monitor(
                model, dense_monitor_sums, dense_monitor_counts
            )
            _accumulate_p2br_epoch_monitor(model, p2br_monitor_sums)

            finite_flag = torch.isfinite(loss).to(dtype=torch.int32)
            if distributed and dist.is_initialized():
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            if finite_flag.item() == 0:
                nonfinite_batches += 1
                if is_main:
                    nonfinite_parts = [
                        name
                        for name, value in loss_dict.items()
                        if isinstance(value, torch.Tensor)
                        and (value.requires_grad)
                        and not torch.isfinite(value).all()
                    ]
                    logger.error(
                        "Non-finite loss detected (epoch=%d count=%d parts=%s).",
                        epoch + 1,
                        nonfinite_batches,
                        nonfinite_parts or "remote-rank",
                    )
                optimizer.zero_grad(set_to_none=True)
                if (
                    args.max_nonfinite_batches > 0
                    and nonfinite_batches >= args.max_nonfinite_batches
                ):
                    raise FloatingPointError(
                        "Non-finite batch limit reached: "
                        f"{nonfinite_batches}/{args.max_nonfinite_batches}"
                    )
                continue

            global_step = epoch * max(1, train_batches_limit) + batch_idx + 1
            if (
                is_main
                and args.prompt_debug_stats
                and args.prompt_debug_stats_interval > 0
                and global_step % args.prompt_debug_stats_interval == 0
            ):
                enabled_prefixes = {"SP", "COARSE", "JITTER", "GATE", "DENSE", "CONTEXT", "BOX"}
                _log_prompt_debug_stats(model, epoch, global_step, enabled_prefixes)
            scaler.scale(loss_for_backward).backward()

            # Step optimizer every grad_accum batches or at the last batch
            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == train_batches_limit:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                ema_update_steps += 1
                if ema is not None and (ema_update_steps % max(1, int(args.ema_update_every)) == 0):
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                # Step scheduler after optimizer step (iteration-level)
                scheduler.step()
                # 问题 10：每 50 个 optimizer step 打印进度，便于确认 cosine/warmup 走对
                if is_main and ema_update_steps % 50 == 0:
                    _cur_lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        "epoch %d | mini-batch %d/%d | optimizer_step %d/%d | lr=%.3e",
                        epoch + 1, batch_idx + 1, train_batches_limit,
                        ema_update_steps, total_optimizer_steps, _cur_lr,
                    )

            total_loss += float(loss.detach().item())
            num_batches += 1
        avg_loss = total_loss / max(1, num_batches)
        # Collect LRs for all groups
        lr_groups = {}
        for group in optimizer.param_groups:
            name = group.get("name", "unknown")
            lr_groups[name] = group["lr"]

        dense_monitor_epoch = _reduce_dense_residual_monitor(
            dense_monitor_sums,
            dense_monitor_counts,
            device=device,
            distributed=distributed,
        )
        p2br_monitor_epoch = _reduce_p2br_epoch_monitor(
            p2br_monitor_sums, device=device, distributed=distributed
        )

        if is_main:
            loss_str = []
            for k, v in loss_meter.items():
                avg_k = v / max(1, num_batches)
                loss_str.append(f"{k}={avg_k:.4f}")
            logger.info(f"Epoch {epoch + 1} detailed losses: {', '.join(loss_str)}")
            if dense_monitor_epoch:
                logger.info(
                    "Epoch %d dense residual monitor: alpha=%.6f "
                    "source_delta_norm=%.4f applied_delta_norm=%.4f "
                    "source_delta_ratio=%.6f applied_delta_ratio=%.6f",
                    epoch_number,
                    dense_monitor_epoch.get("DENSE/residual_alpha", float("nan")),
                    dense_monitor_epoch.get("DENSE/source_delta_norm", float("nan")),
                    dense_monitor_epoch.get("DENSE/applied_delta_norm", float("nan")),
                    dense_monitor_epoch.get("DENSE/source_delta_ratio", float("nan")),
                    dense_monitor_epoch.get("DENSE/applied_delta_ratio", float("nan")),
                )
                if "GAUSSIAN/valid_ratio" in dense_monitor_epoch:
                    logger.info(
                        "Epoch %d Gaussian dense prompt: valid=%.2f%% "
                        "fg_area=%.4f edt_max=%.4f center=(%.4f,%.4f) "
                        "canvas_area=%.4f canvas[min/mean/max]=%.4f/%.4f/%.4f",
                        epoch_number,
                        100.0 * dense_monitor_epoch["GAUSSIAN/valid_ratio"],
                        dense_monitor_epoch.get(
                            "GAUSSIAN/foreground_area_ratio", float("nan")
                        ),
                        dense_monitor_epoch.get("GAUSSIAN/edt_max", float("nan")),
                        dense_monitor_epoch.get(
                            "GAUSSIAN/center_x_normalized", float("nan")
                        ),
                        dense_monitor_epoch.get(
                            "GAUSSIAN/center_y_normalized", float("nan")
                        ),
                        dense_monitor_epoch.get(
                            "GAUSSIAN/canvas_area", float("nan")
                        ),
                        dense_monitor_epoch.get("DENSE/canvas_min", float("nan")),
                        dense_monitor_epoch.get("DENSE/canvas_mean", float("nan")),
                        dense_monitor_epoch.get("DENSE/canvas_max", float("nan")),
                    )
            if p2br_monitor_epoch:
                logger.info(
                    "Epoch %d P2 boundary refiner: raw_dice=%.4f refined_dice=%.4f "
                    "raw_iou=%.4f refined_iou=%.4f raw_boundary_f1=%.4f "
                    "refined_boundary_f1=%.4f",
                    epoch_number,
                    p2br_monitor_epoch["raw_dice"],
                    p2br_monitor_epoch["refined_dice"],
                    p2br_monitor_epoch["raw_iou"],
                    p2br_monitor_epoch["refined_iou"],
                    p2br_monitor_epoch["raw_boundary_f1"],
                    p2br_monitor_epoch["refined_boundary_f1"],
                )
                logger.info(
                    "Epoch %d P2 boundary support: valid=%.2f%% rejected=%.2f%% "
                    "coverage=%.4f delta_abs=%.6f nonzero=%.2f%% saturated=%.2f%% "
                    "p2_norm=%.4f highpass_norm=%.4f support_fg=%.2f%%",
                    epoch_number,
                    100.0 * p2br_monitor_epoch["valid_support_ratio"],
                    100.0 * p2br_monitor_epoch["rejected_support_ratio"],
                    p2br_monitor_epoch["search_coverage"],
                    p2br_monitor_epoch["delta_abs"],
                    100.0 * p2br_monitor_epoch["delta_nonzero_ratio"],
                    100.0 * p2br_monitor_epoch["delta_saturated_ratio"],
                    p2br_monitor_epoch["projected_feature_norm"],
                    p2br_monitor_epoch["highpass_feature_norm"],
                    100.0 * p2br_monitor_epoch["boundary_support_fg_ratio"],
                )
            averaged_components = {
                key: value / max(1, num_batches) for key, value in loss_meter.items()
            }
            det_component = sum(
                value for key, value in averaged_components.items()
                if key in _CANONICAL_DETECTION_LOSS_KEYS or key.startswith("loss_rpn_")
            )
            final_mask_component = float(averaged_components.get("loss_mask", 0.0))
            shape_component = float(averaged_components.get("loss_shape_prior", 0.0))
            refiner_component = float(
                averaged_components.get("loss_p2_boundary_refiner", 0.0)
            )
            component_total = (
                det_component + final_mask_component + shape_component
                + refiner_component
            )
            if component_total > 0:
                logger.info(
                    "Epoch %d primary loss shares: det=%.2f%% final_mask=%.2f%% "
                    "shape=%.2f%% p2_boundary=%.2f%%",
                    epoch_number,
                    100.0 * det_component / component_total,
                    100.0 * final_mask_component / component_total,
                    100.0 * shape_component / component_total,
                    100.0 * refiner_component / component_total,
                )
            logger.info(
                "Epoch %d schedule: det_loss_weight=%.3f early_stop_start=%d min_delta=%.4g",
                epoch_number, det_loss_weight,
                int(args.early_stopping_start_epoch),
                float(args.early_stopping_min_delta),
            )
            # Debug: log current learning rates
            lr_str = ", ".join([f"{k}={v:.2e}" for k, v in lr_groups.items()])
            logger.info(f"Epoch {epoch + 1} learning rates: {lr_str}")

        val_loss = None
        val_metrics = None
        ema_applied_for_eval = False
        use_ema_this_epoch = (
            ema is not None
            and bool(args.ema_eval)
            and (epoch + 1) >= max(1, int(args.ema_eval_start_epoch))
        )
        if (
            val_loader is not None
            and (epoch + 1) % int(args.val_every_n_epochs) == 0
        ):
            if use_ema_this_epoch:
                ema.apply_to(model)
                ema_applied_for_eval = True
            model.eval()
            if args.freeze_bn:
                _set_norm_eval(model)

            collect_gt_records = cached_val_gt_records is None
            all_gt: List[dict] = (
                [] if collect_gt_records else cached_val_gt_records
            )
            all_dt: List[dict] = []
            all_img_metas: List[dict] = []
            total_val_loss = 0.0
            val_batches = 0
            skipped_no_gt_val_loss_batches = 0
            seen_val_batches = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                        break
                    seen_val_batches += 1

                    imgs = batch["imgs"].to(device)
                    img_metas = batch["img_metas"]
                    gt_bboxes = [b.to(device) for b in batch["gt_bboxes"]]
                    gt_labels = [l.to(device) for l in batch["gt_labels"]]
                    gt_masks = batch["gt_masks"]

                    # Prepare data_samples and move to device
                    data_samples = []
                    for i in range(len(imgs)):
                        ds = DetDataSample()
                        ds.set_metainfo(img_metas[i])
                        gt_instances = InstanceData()
                        num_inst = len(gt_labels[i])
                        if num_inst > 0:
                            valid_mask = gt_labels[i] >= 0
                            num_valid = int(valid_mask.sum().item())
                            if num_valid > 0:
                                valid_indices = (
                                    valid_mask.nonzero().squeeze(-1)[:num_valid].cpu()
                                )
                                gt_instances.bboxes = gt_bboxes[i][valid_indices]
                                gt_instances.labels = gt_labels[i][valid_indices]
                                masks = gt_masks[i][valid_indices.cpu().numpy()]
                                gt_instances.masks = BitmapMasks(
                                    masks, *img_metas[i]["img_shape"]
                                )
                            else:
                                gt_instances.bboxes = torch.zeros(
                                    (0, 4), dtype=torch.float32, device=device
                                )
                                gt_instances.labels = torch.zeros(
                                    (0,), dtype=torch.int64, device=device
                                )
                                gt_instances.masks = BitmapMasks(
                                    np.zeros(
                                        (0, *img_metas[i]["img_shape"]), dtype=np.uint8
                                    ),
                                    *img_metas[i]["img_shape"],
                                )
                        else:
                            gt_instances.bboxes = torch.zeros(
                                (0, 4), dtype=torch.float32, device=device
                            )
                            gt_instances.labels = torch.zeros(
                                (0,), dtype=torch.int64, device=device
                            )
                            gt_instances.masks = BitmapMasks(
                                np.zeros(
                                    (0, *img_metas[i]["img_shape"]), dtype=np.uint8
                                ),
                                *img_metas[i]["img_shape"],
                            )
                        ds.gt_instances = gt_instances
                        data_samples.append(ds.to(device))

                    # Optional validation loss is a second model forward. Public
                    # ablation runners disable it for fast point screening; all
                    # images still participate in prediction and COCO evaluation.
                    val_batch_has_valid_gt = False
                    if args.compute_val_loss:
                        for ds in data_samples:
                            labels = getattr(
                                ds.gt_instances, "labels", torch.tensor([])
                            )
                            if labels.numel() > 0 and (labels >= 0).any():
                                val_batch_has_valid_gt = True
                                break
                        if not val_batch_has_valid_gt:
                            skipped_no_gt_val_loss_batches += 1

                    # Synchronize data preprocessing for both images and ground truth
                    processed = data_preprocessor(
                        {"inputs": imgs, "data_samples": data_samples}, training=False
                    )
                    imgs_processed = processed["inputs"]
                    data_samples = processed["data_samples"]

                    model_inputs = imgs_processed

                    model_for_eval = model.module if hasattr(model, "module") else model

                    # ---- Compute loss ----
                    if args.compute_val_loss and val_batch_has_valid_gt:
                        loss_dict = model_for_eval.loss(model_inputs, data_samples)
                        loss_dict = _apply_detection_loss_weight(
                            loss_dict, det_loss_weight, device
                        )
                        val_loss_parts = []
                        for k, v in loss_dict.items():
                            if "loss" not in k.lower():
                                continue
                            if isinstance(v, torch.Tensor):
                                val_loss_parts.append(v)
                            elif isinstance(v, (list, tuple)):
                                val_loss_parts.extend(
                                    x for x in v if isinstance(x, torch.Tensor)
                                )
                        loss = (
                            sum(val_loss_parts)
                            if val_loss_parts
                            else torch.tensor(0.0, device=device)
                        )
                        total_val_loss += float(loss.detach().item())
                        val_batches += 1

                    # ---- Run prediction for COCO evaluation ----
                    outputs = model_for_eval.predict(model_inputs, data_samples, rescale=False)

                    # Collect GT and predictions
                    for j, output in enumerate(outputs):
                        meta = img_metas[j]
                        img_shape = meta["img_shape"]

                        # GT - from data_samples
                        if collect_gt_records:
                            gt_inst = data_samples[j].gt_instances
                            gt_np = _extract_instances_numpy(
                                gt_inst, img_shape, encode_masks=True
                            )
                            all_gt.append(gt_np)

                        # Predictions - from output.pred_instances (InstanceData)
                        pred_inst = output.pred_instances if hasattr(output, "pred_instances") else output
                        pred_np = _extract_instances_numpy(
                            pred_inst, img_shape, encode_masks=True
                        )
                        all_dt.append(pred_np)

                        all_img_metas.append(meta)

            if args.compute_val_loss and skipped_no_gt_val_loss_batches > 0:
                logger.warning(
                    (
                        "Epoch %d skipped val loss on batches with no valid GT: "
                        "%d/%d (%.2f%%); these batches were still included in "
                        "COCO evaluation"
                    ),
                    epoch + 1,
                    skipped_no_gt_val_loss_batches,
                    seen_val_batches,
                    100.0
                    * skipped_no_gt_val_loss_batches
                    / max(1, seen_val_batches),
                )

            val_loss = (
                total_val_loss / max(1, val_batches)
                if args.compute_val_loss
                else None
            )

            if collect_gt_records:
                cached_val_gt_records = all_gt
                logger.info(
                    "Cached validation GT as RLE records: images=%d",
                    len(cached_val_gt_records),
                )
            elif len(all_gt) != len(all_dt):
                raise RuntimeError(
                    "Cached validation GT count no longer matches predictions: "
                    f"GT={len(all_gt)} DT={len(all_dt)}"
                )

            # ---- COCO-style evaluation ----
            val_metrics = {}
            total_gt = sum(len(g["labels"]) for g in all_gt)
            total_dt = sum(len(d["scores"]) for d in all_dt if "scores" in d)
            avg_mask_fill, avg_masks_per_image = _summarize_mask_density(all_dt)

            if is_main:
                logger.info(
                    (
                        "Validation samples: GT=%d, DT=%d, "
                        "avg_masks_per_image=%.2f, avg_mask_fill=%.4f"
                    ),
                    total_gt,
                    total_dt,
                    avg_masks_per_image,
                    avg_mask_fill,
                )

            if total_gt > 0:
                # Honor TEST_MAX_PER_IMG during per-epoch validation so that
                # best-model selection runs at the same maxDets contract as
                # the final --test-only report (previously this env only
                # affected the test-only path, silently leaving best selection
                # at the COCO default maxDets=100). The model-side cap mirrors
                # the test-only override: dense tiles would otherwise batch
                # hundreds of instances through the SAM2 decoder (~10 GB
                # spike) only for the eval to keep the top 150 anyway. Ranking
                # is score-based, so metrics are identical either way.
                _val_max_dets = None
                _val_md_env = int(os.environ.get("TEST_MAX_PER_IMG", "0") or 0)
                if _val_md_env > 0:
                    _val_max_dets = [1, 10, _val_md_env]
                    _val_model_for_eval = (
                        model.module if hasattr(model, "module") else model
                    )
                    _val_test_cfg = getattr(_val_model_for_eval, "test_cfg", None)
                    if _val_test_cfg is not None and "rcnn" in _val_test_cfg:
                        _val_test_cfg["rcnn"]["max_per_img"] = _val_md_env
                    _val_roi_cfg = getattr(
                        getattr(_val_model_for_eval, "roi_head", None),
                        "test_cfg",
                        None,
                    )
                    if _val_roi_cfg is not None and "max_per_img" in _val_roi_cfg:
                        _val_roi_cfg["max_per_img"] = _val_md_env
                # The segmentation score contract is stored in cfg.model and
                # consumed identically by validation and checkpoint inference.
                # Bbox candidate selection/ranking always remains detector-score based.
                coco_gt, coco_bbox_dt = build_coco_gt_and_dt(
                    all_gt, all_dt, all_img_metas, score_key="scores"
                )
                if True:
                    bbox_metrics = run_coco_eval(
                        coco_gt, coco_bbox_dt, iou_type="bbox",
                        max_dets=_val_max_dets,
                    )
                    val_metrics.update(bbox_metrics)
                    logger.info(
                        "Validation bbox/mAP: %.4f bbox/mAP_75: %.4f",
                        bbox_metrics.get("bbox/mAP", 0.0),
                        bbox_metrics.get("bbox/mAP_75", 0.0),
                    )
                if segm_score_key == "scores":
                    coco_segm_dt = coco_bbox_dt
                else:
                    _, coco_segm_dt = build_coco_gt_and_dt(
                        all_gt,
                        all_dt,
                        all_img_metas,
                        score_key=segm_score_key,
                    )
                segm_metrics = run_coco_eval(
                    coco_gt, coco_segm_dt, iou_type="segm",
                    max_dets=_val_max_dets,
                )
                val_metrics.update(segm_metrics)
                logger.info(
                    "Validation segm/mAP: %.4f",
                    segm_metrics.get("segm/mAP", 0.0),
                )

        # Match the baseline's post-validation cleanup and reduce allocator
        # fragmentation across long runs.  Non-main ranks reach this point
        # early and then wait at the existing epoch-end DDP synchronization.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        stop_training = False
        if is_main:
            history["epochs"].append(epoch)
            history["train_losses"].append(avg_loss)
            history["val_losses"].append(val_loss)
            history["val_metrics"].append(val_metrics)
            history["learning_rates"].append(lr_groups)

            # Update best metrics tracking
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss

            if avg_loss < best_train_loss:
                best_train_loss = avg_loss

            # Check if this is the best model based on segm/mAP
            is_best = False
            is_bbox_best = False
            current_segm_map = 0.0
            current_bbox_score = None
            if val_metrics is not None and "segm/mAP" in val_metrics:
                current_segm_map = val_metrics["segm/mAP"]
                if current_segm_map > best_segm_map:
                    best_segm_map = current_segm_map
                    is_best = True
                    if is_main:
                        logger.info(f"New best segm/mAP: {best_segm_map:.4f}")

            if val_metrics is not None and save_bbox_best_metric:
                current_bbox_score = _metric_value(val_metrics, save_bbox_best_metric)
                if current_bbox_score is not None and current_bbox_score > best_bbox_score:
                    best_bbox_score = current_bbox_score
                    is_bbox_best = True
                    logger.info("New best %s: %.4f", save_bbox_best_metric, best_bbox_score)

            current_early_stopping_score = None
            if val_metrics is not None:
                current_early_stopping_score = _metric_expression_value(
                    val_metrics, early_stopping_metric_spec
                )
                if current_early_stopping_score is None:
                    raise RuntimeError(
                        "Early-stopping metric could not be resolved: "
                        f"{early_stopping_metric_spec!r}. Available metrics: "
                        f"{sorted(val_metrics.keys())}"
                    )
                if current_early_stopping_score is not None:
                    early_stopping_metric_history.append(
                        float(current_early_stopping_score)
                    )
                    smooth_window = max(
                        1, int(args.early_stopping_smooth_window)
                    )
                    smoothed_early_stopping_score = float(
                        np.mean(
                            early_stopping_metric_history[-smooth_window:]
                        )
                    )
                    improved = (
                        smoothed_early_stopping_score
                        > best_early_stopping_score
                        + float(args.early_stopping_min_delta)
                    )
                    if improved:
                        best_early_stopping_score = (
                            smoothed_early_stopping_score
                        )
                        early_counter = 0
                        logger.info(
                            "New best early-stop metric (%s): raw=%.4f "
                            "smooth(%d)=%.4f",
                            early_stopping_metric_spec,
                            current_early_stopping_score,
                            smooth_window,
                            best_early_stopping_score,
                        )
                    elif epoch_number >= int(args.early_stopping_start_epoch):
                        early_counter += 1
                        logger.info(
                            "Early-stop metric (%s): raw=%.4f smooth(%d)=%.4f "
                            "best=%.4f counter=%d",
                            early_stopping_metric_spec,
                            current_early_stopping_score,
                            smooth_window,
                            smoothed_early_stopping_score,
                            best_early_stopping_score,
                            early_counter,
                        )
                    else:
                        logger.info(
                            "Early-stop metric (%s): raw=%.4f smooth(%d)=%.4f; "
                            "counter disabled before epoch %d",
                            early_stopping_metric_spec,
                            current_early_stopping_score,
                            smooth_window,
                            smoothed_early_stopping_score,
                            int(args.early_stopping_start_epoch),
                        )

            best_metrics = {
                "best_train_loss": best_train_loss,
                "best_val_loss": best_val_loss,
                "best_segm_map": best_segm_map,
                "best_bbox_score": best_bbox_score,
                "best_bbox_metric": save_bbox_best_metric,
                "best_early_stopping_score": best_early_stopping_score,
                "early_stopping_metric": early_stopping_metric_spec,
                "early_counter": int(early_counter),
                "early_stopping_start_epoch": int(args.early_stopping_start_epoch),
                "early_stopping_min_delta": float(args.early_stopping_min_delta),
                "early_stopping_smooth_window": int(
                    args.early_stopping_smooth_window
                ),
                "early_stopping_metric_history": list(
                    early_stopping_metric_history
                ),
            }

            # Save best model: write to new file first, verify, then delete old.
            # Only rank 0 does checkpoint I/O; all ranks synchronize after this
            # block below. Placing a barrier inside `if is_main` deadlocks DDP.
            ema_state_to_save = ema.state_dict() if ema is not None else None
            if is_best:
                _save_verified_best_checkpoint(
                    checkpoint_dir,
                    "best_model",
                    "segm/mAP",
                    best_segm_map,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_metrics,
                    history,
                    config_snapshot,
                    scaler,
                    ema_state=ema_state_to_save,
                )


            if is_bbox_best and current_bbox_score is not None:
                _save_verified_best_checkpoint(
                    checkpoint_dir,
                    "best_bbox_model",
                    save_bbox_best_metric,
                    current_bbox_score,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_metrics,
                    history,
                    config_snapshot,
                    scaler,
                    ema_state=ema_state_to_save,
                )

            if epoch == args.epochs - 1 and _env_flag("SAVE_LAST_MODEL", "0"):
                last_path = checkpoint_dir / f"last_model_epoch{epoch + 1}.pth"
                _save_checkpoint(
                    last_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_metrics,
                    history,
                    config_snapshot,
                    scaler,
                )
                logger.info("Saved final checkpoint: %s", last_path)

            # Periodic last_checkpoint.pth: overwrite every epoch so a crash can
            # be resumed from the most recent epoch without losing progress.
            # Controlled by SAVE_LAST_CHECKPOINT (default on). Unlike best_model,
            # this is not gated on metric improvement.
            if _env_flag("SAVE_LAST_CHECKPOINT", "1") and epoch < args.epochs - 1:
                _last_ckpt = checkpoint_dir / "last_checkpoint.pth"
                _save_checkpoint(
                    _last_ckpt,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_metrics,
                    history,
                    config_snapshot,
                    scaler,
                    ema_state=ema_state_to_save,
                )

            # Restore training weights after EMA-based eval/save
            if ema_applied_for_eval:
                ema.restore(model)
                ema_applied_for_eval = False

            # Logging: metric evaluation and early stopping do not depend on
            # whether the optional validation-loss forward is enabled.
            lr_str = " ".join([f"{k}={v:.2e}" for k, v in lr_groups.items()])
            if val_metrics is not None:
                segm_info = f" segm/mAP={current_segm_map:.4f}" if current_segm_map > 0 else ""
                bbox_info = (
                    f" {save_bbox_best_metric}={current_bbox_score:.4f}"
                    if current_bbox_score is not None
                    else ""
                )
                val_info = (
                    f"{val_loss:.6f}" if val_loss is not None else "disabled"
                )
                logger.info(
                    "Epoch %d/%d | train=%.6f val=%s%s%s | best_segm=%.4f best_bbox=%.4f | %s",
                    epoch + 1,
                    args.epochs,
                    avg_loss,
                    val_info,
                    segm_info,
                    bbox_info,
                    best_segm_map,
                    best_bbox_score,
                    lr_str,
                )
            else:
                logger.info(
                    "Epoch %d/%d | train=%.6f | %s",
                    epoch + 1,
                    args.epochs,
                    avg_loss,
                    lr_str,
                )

            if (
                current_early_stopping_score is not None
                and epoch_number >= int(args.early_stopping_start_epoch)
                and args.early_stopping_patience > 0
                and early_counter >= int(args.early_stopping_patience)
            ):
                logger.info(
                    "Early stopping triggered (patience=%d)",
                    int(args.early_stopping_patience),
                )
                stop_training = True

        # Rank 0 decides whether to stop after its rank-0-only validation. The
        # CPU/Gloo broadcast doubles as the epoch boundary: non-main ranks wait
        # here without keeping an NCCL collective alive during long COCO eval.
        if distributed and dist.is_initialized() and control_group is not None:
            stop_training = _broadcast_stop_training(
                stop_training, control_group
            )
        if stop_training:
            break

    if distributed and dist.is_initialized():
        if control_group is not None:
            dist.destroy_process_group(control_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
