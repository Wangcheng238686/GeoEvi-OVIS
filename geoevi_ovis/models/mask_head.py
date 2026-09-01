"""Split from rsprompter/models_sam2.py — behaviour-preserving (registry
names, attribute names and state_dict are unchanged)."""
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import ConfigDict
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from torch import Tensor

from mmcv.ops import RoIAlign
from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.models.task_modules import SamplingResult
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.structures.mask.mask_target import mask_target
from mmdet.utils import ConfigType, InstanceList, OptConfigType

from .ckpt_utils import load_module_state_dict_strict
from . import models_sam2 as _ms2  # lazy sam2 names (Hiera/ImageEncoder/FpnNeck/MaskDecoder)
from .models_sam2 import (
    SAM2_AVAILABLE,
    SAM2_REPO,
    PositionEmbeddingSine,
    _load_sam2_checkpoint,
    _resolve_sam2_checkpoint_path,
    _load_pretrained_no_mask_embedding,
)
from .mask_decoder import RSSAM2MaskDecoderWrapper  # noqa: F401


from .mask_head_targets import MaskHeadTargetsMixin

# NOTE: methods/classes below were moved verbatim from models_sam2.py.

class _DiscardedIoUPredictionHead(nn.Module):
    """Keep SAM2's decoder contract without computing an unused IoU MLP."""

    def __init__(self, num_outputs: int):
        super().__init__()
        self.num_outputs = int(num_outputs)

    def forward(self, x: Tensor) -> Tensor:
        return x.new_zeros((x.shape[0], self.num_outputs))

class _UnusedPromptEmbedding(nn.Module):
    """Parameter-free device anchor for PromptEncoder's unused point labels."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    @property
    def weight(self) -> Tensor:
        return self._device_anchor

    def forward(self, _: Tensor) -> Tensor:
        raise RuntimeError("Pruned point-label embedding cannot encode point prompts")

class MaskHeadPromptMixin:
    """Prompt/dense-canvas helpers, diagnostics and contracts."""

    def assert_no_mask_embedding_contract(self, context: str = "runtime") -> None:
        """Fail if a required frozen SAM2 no-mask embedding drifted or fell back."""
        if not self.load_no_mask_pretrained:
            return
        if not self.no_mask_pretrained_loaded:
            raise RuntimeError(
                f"{context}: pretrained no-mask embedding was requested but not loaded"
            )
        if self.no_mask_embed is None:
            raise RuntimeError(f"{context}: required MaskHead no_mask_embed is absent")
        expected = self._pretrained_no_mask_reference
        actual = self.no_mask_embed.detach()
        if tuple(actual.shape) != (1, 256, 1, 1):
            raise RuntimeError(
                f"{context}: invalid MaskHead no_mask_embed shape {tuple(actual.shape)}"
            )
        if not torch.isfinite(actual).all():
            raise RuntimeError(f"{context}: MaskHead no_mask_embed contains non-finite values")
        if expected.numel() != actual.numel():
            raise RuntimeError(f"{context}: pretrained no-mask reference is unavailable")
        max_diff = float(
            (actual.float().cpu() - expected.float().cpu()).abs().max().item()
        )
        if max_diff != 0.0:
            raise RuntimeError(
                f"{context}: MaskHead no_mask_embed no longer matches the SAM2 "
                f"pretrained value (max_abs_diff={max_diff:.6g}); reject silent "
                "zero/random/checkpoint overwrite"
            )
        if self.no_mask_embed.requires_grad != (not self.freeze_no_mask_embed):
            raise RuntimeError(
                f"{context}: no_mask_embed requires_grad="
                f"{self.no_mask_embed.requires_grad} "
                f"does not match freeze_no_mask_embed={self.freeze_no_mask_embed}"
            )

    def init_weights(self) -> None:
        BaseModule.init_weights(self)

    def _shape_prior_to_prompt_mask(
        self,
        roi_mask_logits: Tensor,
        boxes: Tensor,
    ):
        """批次2 子任务3：ROI-local coarse logits → full-image PE mask grid。

        处理顺序：ROI-local transform → resize 到 bbox 对应 prompt-grid → paste 到
        full-image canvas → bbox 外 outside_fill_logit → 最终 clamp（若非 None）。
        """
        from .dense_prompt_utils import (
            gaussian_prompt_from_roi_coarse,
            paste_roi_to_full_canvas,
            transform_coarse_prompt,
        )
        mask_h, mask_w = self._pe_mask_input_size()
        transform = str(self.dense_prompt_cfg.get("transform", "raw_logits"))
        if transform == "gaussian_edt":
            return gaussian_prompt_from_roi_coarse(
                roi_mask_logits,
                boxes,
                mask_h=mask_h,
                mask_w=mask_w,
                image_size=(
                    self.prompt_encoder_image_size,
                    self.prompt_encoder_image_size,
                ),
                foreground_threshold=float(
                    self.dense_prompt_cfg.get("foreground_threshold", 0.5)
                ),
                omega=float(self.dense_prompt_cfg.get("gaussian_omega", 15.0)),
                gamma=float(self.dense_prompt_cfg.get("gaussian_gamma", 4.0)),
            )
        # 1. ROI-local transform（在 resize/paste 之前完成）
        transformed = transform_coarse_prompt(
            roi_mask_logits,
            transform=transform,
            gamma=self.dense_prompt_cfg.get("gamma", 1.0),
            strength=self.dense_prompt_cfg.get("strength", 1.0),
            clamp_range=self.dense_prompt_cfg.get("clamp_range", None),
            temperature=self.dense_prompt_cfg.get("temperature", 1.0),
            detach_input=bool(self.dense_prompt_cfg.get("detach_input", False)),
        )
        # 2. paste 到 full-image canvas（PE mask 尺寸从 PE 属性读，不硬编码）
        prompt = paste_roi_to_full_canvas(
            transformed, boxes, mask_h, mask_w,
            image_size=(self.prompt_encoder_image_size, self.prompt_encoder_image_size),
            outside_fill_logit=float(self.dense_prompt_cfg.get("outside_fill_logit", 0.0)),
        )
        valid = torch.ones(
            roi_mask_logits.shape[0],
            device=roi_mask_logits.device,
            dtype=torch.bool,
        )
        return prompt, valid, {}

    def _select_explicit_prompt_inputs(
        self,
        coords: Tensor,
        labels: Tensor,
        boxes: Tensor,
        masks: Optional[Tensor],
    ):
        """Select SAM2 PromptEncoder inputs for the explicit-mask ablation.

        The detector boxes remain the internal geometry for ROI extraction and
        coordinate mapping in every mode.  This method controls only which
        prompt modalities are exposed to the frozen PromptEncoder.
        """
        if self.prompt_sparse_mode != "shape_point":
            raise RuntimeError(
                "_select_explicit_prompt_inputs is only valid for shape_point"
            )
        # A warm-up/all-invalid batch must be a true no-point prompt. Passing
        # four label=-1 entries would create four learned not-a-point tokens.
        points = (
            None
            if not self.explicit_use_point_prompt or bool((labels < 0).all())
            else (coords, labels)
        )
        boxes_for_pe = boxes if self.explicit_use_box_prompt else None
        masks_for_pe = masks if self.explicit_use_dense_prompt else None
        return points, boxes_for_pe, masks_for_pe

    @staticmethod
    def _neutralize_invalid_point_tokens(
        sparse_embeddings: Tensor, labels: Tensor
    ) -> Tensor:
        """Zero invalid point slots while preserving any following box tokens."""
        point_count = int(labels.shape[1])
        point_tokens = sparse_embeddings[:, :point_count]
        valid = (labels >= 0).unsqueeze(-1).to(dtype=point_tokens.dtype)
        point_tokens = point_tokens * valid
        return torch.cat(
            [point_tokens, sparse_embeddings[:, point_count:]], dim=1
        )

    def _effective_shape_dense_alpha(self):
        """批次2 子任务4：返回 shape dense gate 的有效值（三模式统一）。

        - legacy_unbounded：self.shape_prompt_scale（可训练无界，旧实现）
        - fixed：self.shape_dense_alpha（buffer，不训练）
        - global_sigmoid：sigmoid(self.shape_dense_alpha_raw) ∈ (0,1)
        - use_shape_dense=False：返回 None（forward 里据此跳过 dense 混合）
        """
        if not self.use_shape_dense:
            return None
        if self.shape_scale_mode == "legacy_unbounded":
            return self.shape_prompt_scale
        elif self.shape_scale_mode == "fixed":
            return self.shape_dense_alpha
        elif self.shape_scale_mode == "global_sigmoid":
            return torch.sigmoid(self.shape_dense_alpha_raw)
        return None

    def _dense_embedding_box_support(
        self, boxes: Tensor, spatial_size: Tuple[int, int], dtype: torch.dtype
    ) -> Tensor:
        """Restrict dense-prompt embedding changes to the proposal box."""
        height, width = int(spatial_size[0]), int(spatial_size[1])
        yy = (
            torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 0.5
        ) * (float(self.prompt_encoder_image_size) / float(height))
        xx = (
            torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 0.5
        ) * (float(self.prompt_encoder_image_size) / float(width))
        x1, y1, x2, y2 = boxes.detach().unbind(dim=1)
        support = (
            (xx[None, None, None, :] >= x1[:, None, None, None])
            & (xx[None, None, None, :] < x2[:, None, None, None])
            & (yy[None, None, :, None] >= y1[:, None, None, None])
            & (yy[None, None, :, None] < y2[:, None, None, None])
        )
        return support.to(dtype=dtype)

    def _pe_mask_input_size(self) -> Tuple[int, int]:
        """批次2 子任务3：读取 PromptEncoder 的 mask 输入尺寸（不硬编码）。

        优先 self.prompt_encoder.mask_input_size（SAM2 PE 属性）；
        若 PE 不存在则 fallback 到 prompt_encoder_embed_size*4（旧逻辑）。
        不假设正方形（返回 (H,W)）。
        """
        pe = getattr(self, "prompt_encoder", None)
        if pe is not None and hasattr(pe, "mask_input_size"):
            mis = pe.mask_input_size
            return (int(mis[0]), int(mis[1]))
        es = self.prompt_encoder_embed_size
        return (es * 4, es * 4)

    @staticmethod
    def _debug_delta_ratio(after: Optional[Tensor], before: Optional[Tensor]) -> Optional[float]:
        if after is None or before is None:
            return None
        with torch.no_grad():
            denom = before.detach().float().norm().clamp_min(1e-6)
            ratio = (after.detach().float() - before.detach().float()).norm() / denom
            return float(ratio.item())

    def set_current_epoch(self, epoch_number: int) -> None:
        """Synchronize epoch-dependent point/loss schedules (1-based)."""
        self.current_epoch = max(1, int(epoch_number))
        if self.shape_point_miner is not None:
            self.shape_point_miner.set_current_epoch(self.current_epoch)
        if self.coarse_mask_loss_module is not None and hasattr(
            self.coarse_mask_loss_module, "set_current_epoch"
        ):
            self.coarse_mask_loss_module.set_current_epoch(self.current_epoch)

    def _shape_point_gt_stats(self, coarse_targets: Tensor) -> Dict[str, Tensor]:
        local_yx = self._last_shape_point_local_yx
        labels = self._last_shape_point_labels
        if local_yx is None or labels is None or coarse_targets.shape[0] == 0:
            zero = coarse_targets.new_zeros(())
            return {
                "SP/p1_in_gt_fg_count": zero, "SP/p1_gt_valid_count": zero,
                "SP/p2_in_gt_fg_count": zero, "SP/p2_gt_valid_count": zero,
                "SP/n1_in_gt_bg_count": zero, "SP/n1_gt_valid_count": zero,
                "SP/n2_in_gt_bg_count": zero, "SP/n2_gt_valid_count": zero,
            }
        if local_yx.shape[0] != coarse_targets.shape[0]:
            raise RuntimeError(
                "Shape-point/target ROI count mismatch: "
                f"{local_yx.shape[0]} vs {coarse_targets.shape[0]}"
            )
        h, w = coarse_targets.shape[-2:]
        y = local_yx[..., 0].clamp(0, h - 1)
        x = local_yx[..., 1].clamp(0, w - 1)
        batch = torch.arange(coarse_targets.shape[0], device=coarse_targets.device)[:, None]
        sampled = coarse_targets[batch, y, x] >= 0.5
        out: Dict[str, Tensor] = {}
        for slot, name, positive in (
            (0, "p1", True), (1, "p2", True),
            (2, "n1", False), (3, "n2", False),
        ):
            valid = labels[:, slot] >= 0
            correct = sampled[:, slot] if positive else ~sampled[:, slot]
            suffix = "in_gt_fg" if positive else "in_gt_bg"
            out[f"SP/{name}_{suffix}_count"] = (correct & valid).sum().detach()
            out[f"SP/{name}_gt_valid_count"] = valid.sum().detach()
        return out

    def get_prompt_debug_stats(self) -> Dict[str, float]:
        """Forward + GT-point diagnostics collected during the latest pass."""
        return dict(getattr(self, "_last_prompt_debug_stats", {}) or {})



@MODELS.register_module()
class RSPrompterAnchorMaskHeadSAM2(
    MaskHeadTargetsMixin, MaskHeadPromptMixin, FCNMaskHead, BaseModule
):
    def __init__(
        self,
        sam2_mask_decoder,
        in_channels,
        roi_feat_size=14,
        per_pointset_point=5,
        with_sincos=True,
        multimask_output=False,
        attention_similarity=None,
        target_embedding=None,
        output_attentions=None,
        class_agnostic=False,
        # ===== Prompt 编码迁移 (阶段1) =====
        # sparse prompt 两模式: "point"(MLP point_emb 桥接) / "shape_point"(显式 coarse→PE)
        prompt_sparse_mode: str = "point",
        explicit_prompt_mode: Optional[str] = None,
        prompt_encoder_enabled: bool = False,      # 是否接回 SAM2 原生 PromptEncoder
        prompt_encoder_image_size: int = 512,      # input_image_size, 与图一致
        prompt_encoder_embed_size: int = 16,       # image_embedding_size, 实测 512→16×16
        prompt_encoder_cfg: Optional[Dict] = None, # 问题 3：PE 严格加载/冻结统一配置（见下）
        dense_prompt_mode: str = "none",           # "none"(no_mask 常量) / "residual"(+ROI 残差, 零初始化)
        prune_unused_prompt_components: bool = False,
        shape_base_dense_mode: str = "no_mask",
        # 问题 1（审查重构）：拆分为 coarse / final 两个独立坐标语义，禁止一个开关同时控制。
        # - coarse_mask_coordinate_mode：coarse mask（ShapePriorInjector 输出）的坐标系，
        #   固定 "roi_local"（coarse mask 在 ROI 局部生成与监督）。
        # - final_mask_coordinate_mode：最终实例 mask 的监督/后处理坐标系。
        #   B0/B1 用 "roi_local" 复现旧口径；显式 coarse 路线使用 SAM2 原生
        #   "full_image" 网格，训练对齐整图 target，推理只 resize 一次。
        final_mask_coordinate_mode: str = "roi_local",
        final_mask_loss_cfg: Optional[Dict] = None,
        roi_mask_size: int = 28,
        # 问题 13：mask_head 级别的 checkpoint 严格加载配置（透传给 PE；decoder/backbone 各自接收）
        checkpoint_load_cfg: Optional[Dict] = None,
        # 批次2 子任务1：coarse mask 独立强监督配置（替换旧 _shape_prior_aux_loss）
        coarse_mask_loss_cfg: Optional[Dict] = None,
        # 批次2 子任务3：dense prompt transform 配置
        dense_prompt_cfg: Optional[Dict] = None,
        restrict_dense_prompt_to_box: bool = True,
        # P2: explicit point miner / training-only box prompt jitter configs
        shape_point_miner_cfg: Optional[Dict] = None,
        point_warmup_cfg: Optional[Dict] = None,
        mask_prompt_box_cfg: Optional[Dict] = None,
        box_prompt_cfg: Optional[Dict] = None,
        # ===== 冻结策略 =====
        freeze_mask_decoder: bool = False,         # SAM2 MaskDecoder 可训练 (canonical 契约 §10/§14.1)
        freeze_no_mask_embed: bool = True,         # 冻结 no_mask_embed (True=加载ckpt并冻结 / False=零初始化可训练)
        load_no_mask_pretrained: bool = True,       # False 复现旧 MLP 基线的零初始化 no_mask_embed
        # ===== 阶段2: shape prior =====
        shape_prior_cfg: Optional[Dict] = None,    # dict(enabled=True, context_source="visual", ...) 或 None
        shape_prior_loss_weight: float = 0.5,       # 辅助 dice_bce loss 权重
        p2_boundary_refiner_cfg: Optional[Dict] = None,
        loss_mask: ConfigType = dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
        init_cfg=None,
        *args,
        **kwargs,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        self.in_channels = in_channels
        self.roi_feat_size = roi_feat_size
        self.per_pointset_point = per_pointset_point
        self.with_sincos = with_sincos
        self.multimask_output = multimask_output
        self.attention_similarity = attention_similarity
        self.target_embedding = target_embedding
        self.output_attentions = output_attentions

        # Prompt 编码开关
        self.prompt_sparse_mode = str(prompt_sparse_mode)
        if self.prompt_sparse_mode not in ("point", "shape_point"):
            raise ValueError(
                "prompt_sparse_mode must be 'point' or 'shape_point', got "
                f"{self.prompt_sparse_mode!r}"
            )
        _explicit_mode_was_provided = explicit_prompt_mode is not None
        if self.prompt_sparse_mode == "shape_point":
            if explicit_prompt_mode is None:
                # Legacy shape-point configs always included box tokens; infer
                # only whether the dense mask was enabled.
                explicit_prompt_mode = (
                    "points_box_dense"
                    if bool((shape_prior_cfg or {}).get("use_shape_dense", True))
                    else "points_box"
                )
            self.explicit_prompt_mode = str(explicit_prompt_mode).strip().lower()
            _valid_explicit_modes = {"points_box", "points_box_dense"}
            if self.explicit_prompt_mode not in _valid_explicit_modes:
                raise ValueError(
                    "explicit_prompt_mode must be points_box or "
                    f"points_box_dense, got {self.explicit_prompt_mode!r}"
                )
            self.explicit_use_point_prompt = self.explicit_prompt_mode in {
                "points", "points_box", "points_box_dense"
            }
            self.explicit_use_box_prompt = self.explicit_prompt_mode in {
                "box", "points_box", "points_box_dense"
            }
            self.explicit_use_dense_prompt = (
                self.explicit_prompt_mode in {"mask", "points_box_dense"}
            )
            if _explicit_mode_was_provided:
                _cfg_use_dense = bool(
                    (shape_prior_cfg or {}).get("use_shape_dense", False)
                )
                if _cfg_use_dense != self.explicit_use_dense_prompt:
                    raise ValueError(
                        "explicit_prompt_mode and shape_prior_cfg.use_shape_dense "
                        "must agree: mode="
                        f"{self.explicit_prompt_mode!r}, use_shape_dense={_cfg_use_dense}"
                    )
        else:
            self.explicit_prompt_mode = None
            self.explicit_use_point_prompt = False
            self.explicit_use_box_prompt = False
            self.explicit_use_dense_prompt = False
        self.prompt_encoder_enabled = bool(prompt_encoder_enabled)
        self.dense_prompt_mode = str(dense_prompt_mode)
        self.point_warmup_cfg = dict(point_warmup_cfg or {})
        self.mask_prompt_box_cfg = dict(mask_prompt_box_cfg or {})
        self.prompt_encoder_image_size = int(prompt_encoder_image_size)
        self.prompt_encoder_embed_size = int(prompt_encoder_embed_size)
        self.prompt_encoder_mask_size = self.prompt_encoder_embed_size * 4
        self.prune_unused_prompt_components = bool(prune_unused_prompt_components)
        # 问题 1（审查重构）：coarse / final 双坐标语义，禁止混用。
        self.coarse_mask_coordinate_mode = "roi_local"
        self.final_mask_coordinate_mode = str(final_mask_coordinate_mode).lower()
        if self.final_mask_coordinate_mode not in ("full_image", "roi_local"):
            raise ValueError(
                "final_mask_coordinate_mode must be 'full_image' or 'roi_local', "
                f"got {self.final_mask_coordinate_mode!r}"
            )
        # Coarse logits are always ROI-local.  Final SAM2 logits may retain the
        # historical ROI-local interpretation for the frozen B0/B1 reproduction,
        # or use their native full-image grid in the explicit-coarse route.
        assert self.coarse_mask_coordinate_mode == "roi_local", (
            "coarse_mask_coordinate_mode must be 'roi_local' in current architecture "
            "(coarse mask is generated and supervised per-ROI)"
        )
        _final_loss_cfg = dict(final_mask_loss_cfg or {})
        _allowed_final_loss_keys = {"mode", "eps"}
        _unknown_final_loss_keys = set(_final_loss_cfg) - _allowed_final_loss_keys
        if _unknown_final_loss_keys:
            raise ValueError(
                "Unknown final_mask_loss_cfg keys: "
                f"{sorted(_unknown_final_loss_keys)}"
            )
        _final_loss_cfg.setdefault("mode", "standard")
        _final_loss_cfg.setdefault("eps", 1e-6)
        _final_loss_cfg["mode"] = str(_final_loss_cfg["mode"]).strip().lower()
        if _final_loss_cfg["mode"] != "standard":
            raise ValueError(
                "final_mask_loss_cfg.mode must be standard, "
                f"got {_final_loss_cfg['mode']!r}"
            )
        self.final_mask_loss_cfg = ConfigDict(_final_loss_cfg)
        self.roi_mask_size = int(roi_mask_size)
        # 问题 3：PromptEncoder 严格加载/冻结统一配置。
        # 但若提供了 prompt_encoder_cfg，则以 cfg 为准（更细粒度）。
        self.checkpoint_load_cfg = dict(checkpoint_load_cfg or {})
        _pe_cfg = dict(prompt_encoder_cfg or {})
        _allowed_pe_cfg_keys = {
            "require_pretrained",
            "ckpt_path",
            "freeze_all",
        }
        _unknown_pe_cfg_keys = set(_pe_cfg) - _allowed_pe_cfg_keys
        if _unknown_pe_cfg_keys:
            raise ValueError(
                "Unknown prompt_encoder_cfg keys: "
                f"{sorted(_unknown_pe_cfg_keys)}"
            )
        # 旧参数 → cfg 默认值（仅当 cfg 未显式指定时）
        _pe_cfg.setdefault("require_pretrained", True)
        _pe_cfg.setdefault("freeze_all", True)
        self.prompt_encoder_cfg = _pe_cfg
        # 暴露给后续加载逻辑使用的扁平字段（保持旧代码可读）
        self._pe_require_pretrained = bool(_pe_cfg.get("require_pretrained", True))
        self._pe_cfg_path = _pe_cfg.get("ckpt_path")  # None 表示复用 decoder ckpt
        # Segmentation COCO ranking is fixed to the detector score.
        self.segm_score_mode = "detector"
        self.segm_score_key = "scores"
        self.shape_base_dense_mode = str(shape_base_dense_mode)
        self.load_no_mask_pretrained = bool(load_no_mask_pretrained)
        self.freeze_no_mask_embed = bool(freeze_no_mask_embed)
        if self.shape_base_dense_mode not in {"no_mask", "zero"}:
            raise ValueError(
                "shape_base_dense_mode must be 'no_mask' or 'zero', got "
                f"{self.shape_base_dense_mode!r}"
            )

        self.mask_decoder = RSSAM2MaskDecoderWrapper(**sam2_mask_decoder)
        if self.prune_unused_prompt_components:
            decoder = self.mask_decoder.decoder
            if decoder is None or not hasattr(decoder, "iou_prediction_head"):
                raise ValueError("Cannot prune missing SAM2 IoU prediction head")
            decoder.iou_prediction_head = _DiscardedIoUPredictionHead(
                decoder.num_mask_tokens
            )

        if self.shape_base_dense_mode == "no_mask":
            # C1-C5 require the exact SAM2 embedding.  Missing paths/keys used
            # to fall through to zeros here, even when the config explicitly
            # requested pretrained loading.  That is now a hard error.
            _no_mask_ckpt = (
                self._pe_cfg_path or sam2_mask_decoder.get("checkpoint_path")
            )
            if self.load_no_mask_pretrained:
                _no_mask = _load_pretrained_no_mask_embedding(_no_mask_ckpt)
                self.no_mask_pretrained_loaded = True
            else:
                _no_mask = torch.zeros(1, 256, 1, 1)
                self.no_mask_pretrained_loaded = False
            self.no_mask_embed = nn.Parameter(_no_mask)
            self.no_mask_embed.requires_grad_(not freeze_no_mask_embed)
            self.register_buffer(
                "_pretrained_no_mask_reference",
                _no_mask.detach().clone()
                if self.no_mask_pretrained_loaded
                else torch.empty(0),
                persistent=False,
            )
            if is_main_process():
                src = "SAM2 ckpt" if self.no_mask_pretrained_loaded else "intentional zeros"
                state = "frozen" if freeze_no_mask_embed else "trainable"
                abs_mean = float(self.no_mask_embed.detach().abs().mean().item())
                print(
                    f"[MaskHead] Initialized no_mask_embed from {src}, "
                    f"{state}, abs_mean={abs_mean:.8f}"
                )
        else:
            self.no_mask_pretrained_loaded = False
            self.register_parameter("no_mask_embed", None)
            self.register_buffer(
                "_pretrained_no_mask_reference", torch.empty(0), persistent=False
            )
            if is_main_process():
                print("[MaskHead] base dense disabled")

        # Freeze mask_decoder (默认为 True, 对齐原始行为)
        # freeze_mask_decoder=False 时: 解码器可训练 (对齐参考项目 B 的 decoder_freeze=False 行为)
        if freeze_mask_decoder:
            for _p in self.mask_decoder.parameters():
                _p.requires_grad_(False)
            self.mask_decoder.eval()
        # else: decoder 保持可训练, 由父类 decoder_freeze 开关二次控制

        # === SAM2 原生 PromptEncoder (可选) ===
        # box 通道: proposal box 坐标 → 2 个角点 token [N,2,256]
        if self.prompt_encoder_enabled:
            if SAM2_REPO and SAM2_REPO not in sys.path:
                sys.path.insert(0, SAM2_REPO)
            from sam2.modeling.sam.prompt_encoder import PromptEncoder

            self.prompt_encoder = PromptEncoder(
                embed_dim=256,
                image_embedding_size=(int(prompt_encoder_embed_size), int(prompt_encoder_embed_size)),
                input_image_size=(int(prompt_encoder_image_size), int(prompt_encoder_image_size)),
                mask_in_chans=16,
            )
            # 问题 3：冻结策略由 prompt_encoder_cfg 统一控制。
            # - 默认 freeze_all=True：PE 全部冻结（canonical 契约 §9.3/§14.2）。
            # - freeze_all=False：sparse_only 解冻 box 角点 point_embeddings[2]/[3]；
            #   shape_point 模式额外解冻 point_embeddings[0]/[1]（前景点/背景点）。
            # 其余冻结 → 避免 DDP unused params, 避免 dense 路径悄悄变成 PE no_mask
            _pe_freeze_all = bool(self.prompt_encoder_cfg.get("freeze_all", True))
            if _pe_freeze_all:
                _pe_trainable_indices = ()
            else:
                _pe_trainable_indices = (2, 3)
                if self.prompt_sparse_mode == "shape_point":
                    _pe_trainable_indices = (
                        (0, 1, 2, 3)
                        if self.explicit_use_box_prompt
                        else (0, 1)
                    )
            for _idx, _emb in enumerate(self.prompt_encoder.point_embeddings):
                _emb.weight.requires_grad_(_idx in _pe_trainable_indices)
            self.prompt_encoder.not_a_point_embed.weight.requires_grad_(False)
            self.prompt_encoder.no_mask_embed.weight.requires_grad_(False)
            # The dense-mask convolution stack stays frozen unconditionally:
            # the densefix-unfreeze ablation that could train it was removed.
            for _p in self.prompt_encoder.mask_downscaling.parameters():
                _p.requires_grad_(False)
            # 加载预训练权重 (对比项目遗漏, 本计划增量)
            # 问题 3：require_pretrained 默认 True——ckpt 不存在或 missing key 时直接 raise，
            # 避免静默随机初始化导致 PE 几何编码错误却继续训练。
            if self._pe_require_pretrained:
                _ck_path = _resolve_sam2_checkpoint_path(
                    self._pe_cfg_path or sam2_mask_decoder.get("checkpoint_path")
                )
                if _ck_path and os.path.exists(_ck_path):
                    _ck = _load_sam2_checkpoint(_ck_path)
                    _pe_state = {
                        k[len("sam_prompt_encoder."):]: v
                        for k, v in _ck.items()
                        if k.startswith("sam_prompt_encoder.")
                    }
                    if not _pe_state:
                        if self._pe_require_pretrained:
                            raise RuntimeError(
                                f"[MaskHead] sam_prompt_encoder.* keys not found in {_ck_path}, "
                                "but require_pretrained=True. Check checkpoint integrity / model_size match."
                            )
                        elif is_main_process():
                            print(f"[MaskHead] WARNING: no sam_prompt_encoder.* keys in {_ck_path}")
                    else:
                        load_module_state_dict_strict(
                            self.prompt_encoder,
                            _pe_state,
                            module_name="PromptEncoder",
                            strict_reproduction=self.checkpoint_load_cfg.get("strict_reproduction", False),
                            allowed_missing_patterns=tuple(
                                self.checkpoint_load_cfg.get("allowed_missing_patterns", [])
                            ),
                            allowed_unexpected_patterns=tuple(
                                self.checkpoint_load_cfg.get("allowed_unexpected_patterns", [])
                            ),
                        )
                        if self.load_no_mask_pretrained:
                            _pe_no_mask = (
                                self.prompt_encoder.no_mask_embed.weight.detach()
                                .reshape_as(self.no_mask_embed)
                            )
                            if not torch.equal(
                                _pe_no_mask.cpu(), self.no_mask_embed.detach().cpu()
                            ):
                                raise RuntimeError(
                                    "PromptEncoder and MaskHead loaded different "
                                    "pretrained no-mask embeddings"
                                )
                else:
                    _msg = (f"PromptEncoder pretrained weights not found at {_ck_path}. "
                            f"require_pretrained={self._pe_require_pretrained}.")
                    if self._pe_require_pretrained:
                        raise RuntimeError(
                            f"[MaskHead] {_msg} Set prompt_encoder_cfg.require_pretrained=False "
                            "to allow random init, or provide a valid SAM2 checkpoint."
                        )
                    elif is_main_process():
                        print(f"[MaskHead] WARNING: {_msg} PE will be randomly initialized.")
            if self.prune_unused_prompt_components:
                if not (shape_prior_cfg and shape_prior_cfg.get("enabled", False)):
                    raise ValueError(
                        "prune_unused_prompt_components requires shape_prior_cfg.enabled=True"
                    )
                # shape_point 模式需要 point_embeddings[0]/[1]（前景点/背景点），不能 prune
                if self.prompt_sparse_mode == "point":
                    # points=None and masks=shape_prompt are invariant in this path.
                    # Indices 2/3 remain because SAM2 uses them for box corners.
                    self.prompt_encoder.point_embeddings[0] = _UnusedPromptEmbedding()
                    self.prompt_encoder.point_embeddings[1] = _UnusedPromptEmbedding()
                    # torch.where 求值两个分支，not_a_point_embed 即使无 label=-1 也会被访问
                    self.prompt_encoder.not_a_point_embed = None
                # Sparse-only shape-point experiments still need SAM2's pretrained
                # no-mask dense embedding.  Prune it only when every forward is
                # guaranteed to provide a shape dense mask.
                if self.explicit_use_dense_prompt:
                    self.prompt_encoder.no_mask_embed = None
        else:
            self.prompt_encoder = None

        # === point_emb (RSPrompter 自造 5 点) ===
        # Only the MLP bridge ("point") instantiates it; shape_point uses the
        # frozen SAM2 PromptEncoder instead.  SAM2 decoder expects
        # sparse_prompt_embeddings with dim=256 (matching image_embeddings).
        # Output: (batch, per_pointset_point, 256)
        num_sincos = 2 if with_sincos else 1
        embed_dim = 256  # Match SAM2 decoder embedding dimension
        if self.prompt_sparse_mode == "point":
            self.point_emb = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(in_channels * roi_feat_size**2 // 4, in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels, in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels, embed_dim * num_sincos * per_pointset_point),
            )
        else:
            self.point_emb = None

        # === ShapePointMiner (shape_point mode) ===
        # Mine hard distance-based positive/negative coordinates from detached
        # ROI-local coarse logits, then encode them with the frozen SAM2 PE.
        if self.prompt_sparse_mode == "shape_point":
            from .shape_prior import ShapePointMiner
            _miner_cfg = dict(shape_point_miner_cfg or {})
            _miner_cfg.setdefault("point_warmup_cfg", self.point_warmup_cfg)
            self.shape_point_miner = ShapePointMiner(**_miner_cfg)
        else:
            self.shape_point_miner = None

        # === dense ROI residual (可选) ===
        # 在 no_mask 常量基底上叠加 per-ROI dense 残差, roi_dense_scale 零初始化 → 启动严格等价 B0
        if self.dense_prompt_mode == "residual":
            self.roi_dense_proj = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, 1),
                nn.GroupNorm(32, embed_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.GroupNorm(32, embed_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, 1),
            )
            self.roi_dense_scale = nn.Parameter(torch.zeros(1))
        else:
            self.roi_dense_proj = None
            self.register_parameter("roi_dense_scale", None)

        # === ShapePriorInjector (阶段2, 可选) ===
        # visual 路径: image_embeddings → context tokens → cross-attn 调制 ROI feat → small mask decoder
        # 产出的 mask_logits 喂给 PE masks= 通道, 产生 shape-aware dense_pe
        # shape_prompt_scale 初始值可配置（默认 0.0 等价无 shape prior，安全热启动）
        self.shape_prior_loss_weight = float(shape_prior_loss_weight)
        # 批次2 子任务1：CoarseMaskLoss（替换旧 _shape_prior_aux_loss）。
        # Canonical weight is epoch-scheduled (0.20 then 0.10); the static
        # shape_prior_loss_weight remains only as enable/fallback behavior.
        _shape_prior_enabled = bool(
            shape_prior_cfg is not None and shape_prior_cfg.get("enabled", False)
        )
        if _shape_prior_enabled:
            from .coarse_mask_loss import CoarseMaskLoss
            _cml_cfg = dict(coarse_mask_loss_cfg or {})
            _cml_cfg.setdefault("weight", self.shape_prior_loss_weight)
            _cml_cfg.setdefault("bce_weight", 1.0)
            _cml_cfg.setdefault("dice_weight", 1.0)
            _cml_cfg.setdefault("boundary_weight", 0.0)
            _cml_cfg.setdefault("distance_weight", 0.0)
            self.coarse_mask_loss_module = CoarseMaskLoss(**_cml_cfg)
        else:
            # Pure RSPrompter MLP mode has no coarse mask and therefore no
            # shape-loss object or schedule in the model graph.
            self.coarse_mask_loss_module = None
        # 批次2 子任务3：dense prompt transform 配置。
        # 默认 raw_logits + outside_fill=0 + clamp_range=None（严格等价旧 _shape_prior_to_prompt_mask）。
        self.dense_prompt_cfg = dict(dense_prompt_cfg or {})
        self.restrict_dense_prompt_to_box = bool(restrict_dense_prompt_to_box)
        self.dense_prompt_cfg.setdefault("transform", "raw_logits")
        self.dense_prompt_cfg.setdefault("outside_fill_logit", 0.0)
        self.dense_prompt_cfg.setdefault("gamma", 1.0)
        self.dense_prompt_cfg.setdefault("strength", 1.0)
        self.dense_prompt_cfg.setdefault("temperature", 1.0)
        self.dense_prompt_cfg.setdefault("clamp_range", None)  # None=不 clamp（raw_logits 兼容）
        self.dense_prompt_cfg.setdefault("detach_input", False)
        self.dense_prompt_cfg.setdefault("foreground_threshold", 0.5)
        self.dense_prompt_cfg.setdefault("gaussian_omega", 15.0)
        self.dense_prompt_cfg.setdefault("gaussian_gamma", 4.0)
        _dense_transform = str(self.dense_prompt_cfg["transform"])
        if _dense_transform not in {
            "raw_logits", "confidence_signed", "gaussian_edt"
        }:
            raise ValueError(f"Unsupported dense prompt transform={_dense_transform!r}")
        if _dense_transform == "gaussian_edt":
            if not bool(self.dense_prompt_cfg["detach_input"]):
                raise ValueError("gaussian_edt requires detach_input=True")
            if self.dense_prompt_cfg["clamp_range"] is not None:
                raise ValueError("gaussian_edt must not be clamped")
        self.shape_point_miner_cfg = dict(shape_point_miner_cfg or {})
        self.box_prompt_cfg = dict(box_prompt_cfg or {})
        self.box_prompt_cfg.setdefault("jitter_prob", 0.0)
        self.box_prompt_cfg.setdefault("jitter_scale", 0.05)
        self.box_prompt_cfg.setdefault("min_box_size", 2.0)
        self.box_prompt_cfg.setdefault("min_iou_with_original", 0.5)
        self.mask_prompt_box_cfg.setdefault("train_box_source", "proposal")
        self.mask_prompt_box_cfg.setdefault("test_box_source", "sabl_refined")
        self.mask_prompt_box_cfg.setdefault("fallback_to_proposal", True)
        self.mask_prompt_box_cfg.setdefault("min_refined_proposal_iou", 0.30)
        self.mask_prompt_box_cfg.setdefault("min_box_size", 2.0)
        self.mask_prompt_box_cfg.setdefault("clamp_to_image", True)
        self._last_shape_point_local_yx = None
        self._last_shape_point_labels = None
        if _shape_prior_enabled:
            import math as _math
            from .shape_prior import ShapePriorInjector
            _mh_keys = {
                "enabled",
                "use_shape_dense",
                "shape_scale_mode",
                "prompt_scale_init",
                "shape_scale_init",
            }  # mask_head 自己处理的字段, 不传给 injector
            _sp_cfg = {k: v for k, v in shape_prior_cfg.items() if k not in _mh_keys}
            _sp_cfg.setdefault("roi_feat_channels", in_channels)
            self.shape_injector = ShapePriorInjector(**_sp_cfg)
            # 批次2 子任务4：shape dense gate 三模式（审查 2.1）。
            # legacy_unbounded=旧行为复现；fixed=固定 gate 消融；
            # global_sigmoid=canonical 有界 gate（默认 init=0.25）。
            self.shape_scale_mode = str(shape_prior_cfg.get("shape_scale_mode", "legacy_unbounded"))
            self.use_shape_dense = bool(shape_prior_cfg.get("use_shape_dense", True))
            if (
                self.prompt_sparse_mode == "shape_point"
                and self.use_shape_dense != self.explicit_use_dense_prompt
            ):
                raise ValueError(
                    "shape_prior_cfg.use_shape_dense must match explicit_prompt_mode: "
                    f"mode={self.explicit_prompt_mode!r}, "
                    f"use_shape_dense={self.use_shape_dense}"
                )
            _scale_init = float(shape_prior_cfg.get("prompt_scale_init",
                                                     shape_prior_cfg.get("shape_scale_init", 0.0)))
            if self.use_shape_dense:
                if self.shape_scale_mode == "legacy_unbounded":
                    # 旧实现原样：可训练无界 Parameter
                    self.shape_prompt_scale = nn.Parameter(torch.tensor(_scale_init))
                    self.register_buffer("shape_dense_alpha", torch.tensor(_scale_init))
                    self.register_parameter("shape_dense_alpha_raw", None)
                elif self.shape_scale_mode == "fixed":
                    # 固定 gate（buffer，不训练）
                    self.register_buffer("shape_dense_alpha", torch.tensor(_scale_init))
                    self.register_parameter("shape_prompt_scale", None)
                    self.register_parameter("shape_dense_alpha_raw", None)
                elif self.shape_scale_mode == "global_sigmoid":
                    # 有界 gate：sigmoid(raw) ∈ (0,1)
                    _eps = 1e-4
                    _init = min(max(_scale_init, _eps), 1.0 - _eps)
                    _raw_init = _math.log(_init / (1.0 - _init))
                    self.shape_dense_alpha_raw = nn.Parameter(torch.tensor(_raw_init))
                    self.register_buffer("shape_dense_alpha", torch.tensor(_scale_init))
                    self.register_parameter("shape_prompt_scale", None)
                else:
                    raise ValueError(
                        f"Unknown shape_scale_mode: {self.shape_scale_mode!r}. "
                        "Expected legacy_unbounded/fixed/global_sigmoid."
                    )
            else:
                # sparse-only：不创建 shape dense gate，gate 不进 optimizer。
                # 注意：shape_injector 保留（shape_point 模式仍需 coarse mask 挖点），
                # 只是 forward 里不把 coarse mask 转 dense_pe（masks_for_pe=None）。
                self.register_parameter("shape_prompt_scale", None)
                self.register_parameter("shape_dense_alpha_raw", None)
                self.register_buffer("shape_dense_alpha", torch.tensor(0.0))
        else:
            self.shape_injector = None
            self.register_parameter("shape_prompt_scale", None)
            self.register_parameter("shape_dense_alpha_raw", None)
            self.register_buffer("shape_dense_alpha", torch.tensor(0.0))
            self.shape_scale_mode = "legacy_unbounded"
            self.use_shape_dense = False

        # C5-v2 is a P2-driven local boundary modifier.  Construct it after all
        # shared random modules and restore the CPU RNG afterwards so enabling
        # the ablation cannot perturb R1-C4 shared initialization.
        self.p2_boundary_refiner_cfg = dict(p2_boundary_refiner_cfg or {})
        if self.p2_boundary_refiner_cfg.get("enabled", False):
            if self.shape_injector is None or self.prompt_encoder is None:
                raise ValueError(
                    "P2BoundaryRefiner requires ShapePrior and PromptEncoder"
                )
            from .p2_boundary_refiner import P2BoundaryRefiner

            _refiner_cfg = dict(self.p2_boundary_refiner_cfg)
            _refiner_cfg.pop("enabled", None)
            _refiner_cfg.setdefault("coarse_size", int(
                (shape_prior_cfg or {}).get("coarse_mask_output_size", 64)
            ))
            with torch.random.fork_rng(devices=[]):
                self.p2_boundary_refiner = P2BoundaryRefiner(**_refiner_cfg)
        else:
            self.p2_boundary_refiner = None

        # DecoderBR is an independent post-decoder residual branch. It is
        # initialized after the baseline modules so old checkpoint loading and
        # random initialization order remain unchanged when it is disabled.
        self.loss_mask = MODELS.build(loss_mask)
        self.class_agnostic = class_agnostic

        self.assert_no_mask_embedding_contract("model construction")

    def forward(
        self,
        x,
        image_embeddings,
        image_positional_embeddings,
        roi_img_ids=None,
        high_res_features=None,
        boxes=None,
        p2_feature=None,
        prompt_rois=None,
    ):
        img_bs = image_embeddings.shape[0]
        roi_bs = x.shape[0]
        image_embedding_size = image_embeddings.shape[-2:]
        debug_stats: Dict[str, float] = {}
        prompt_geometry_boxes = boxes

        # === sparse_embeddings ===
        # point: RSPrompter 自造 point_emb [N,5,256]
        point_embs = None
        if self.point_emb is not None:
            pe = self.point_emb(x)
            pe = einops.rearrange(pe, "b (n c) -> b n c", n=self.per_pointset_point)
            if self.with_sincos:
                pe = torch.sin(pe[..., ::2]) + pe[..., 1::2]
            point_embs = pe  # [N, 5, 256]

        # === shape prior mask (阶段2, 必须在 PE 调用之前计算) ===
        shape_prior_mask_logits = None
        coarse_outputs = None
        shape_prior_prompt_logits = None
        shape_dense_valid = None
        if self.shape_injector is not None and boxes is not None:
            ctx_tokens = self.shape_injector.forward_context_visual(image_embeddings)
            raw_shape_prior_logits, _, _, _ = self.shape_injector.forward_roi(
                ctx_tokens, x, boxes, roi_img_ids,
            )
            debug_stats.update(self.shape_injector.get_debug_stats())
            refined_shape_prior_logits = raw_shape_prior_logits
            refiner_outputs = None
            if self.p2_boundary_refiner is not None:
                if p2_feature is None or prompt_rois is None:
                    raise ValueError(
                        "P2BoundaryRefiner requires full P2 and prompt_rois [N,5]"
                    )
                refiner_outputs = self.p2_boundary_refiner(
                    p2_feature=p2_feature,
                    prompt_rois=prompt_rois,
                    raw_logits=raw_shape_prior_logits,
                )
                refined_shape_prior_logits = refiner_outputs["refined_logits"]
                with torch.no_grad():
                    delta = refiner_outputs["delta_logits"].detach().float()
                    support_valid = refiner_outputs["support_valid"].detach()
                    debug_stats.update({
                        "P2BR/roi_count": delta.new_tensor(float(delta.shape[0])),
                        "P2BR/valid_support_count": support_valid.float().sum(),
                        "P2BR/rejected_support_count": (~support_valid).float().sum(),
                        "P2BR/search_coverage_sum": refiner_outputs["search_coverage_sum"],
                        "P2BR/delta_abs_sum": delta.abs().flatten(1).mean(1).sum(),
                        "P2BR/delta_nonzero_sum": delta.ne(0).float().flatten(1).mean(1).sum(),
                        "P2BR/delta_saturated_sum": delta.abs().ge(0.396).float().flatten(1).mean(1).sum(),
                        "P2BR/projected_feature_norm_sum": refiner_outputs["projected_feature_norm_sum"],
                        "P2BR/highpass_feature_norm_sum": refiner_outputs["highpass_feature_norm_sum"],
                    })
            shape_prior_mask_logits = refined_shape_prior_logits
            coarse_outputs = {
                "raw_logits": raw_shape_prior_logits,
                "refined_logits": refined_shape_prior_logits,
                "delta_logits": None if refiner_outputs is None else refiner_outputs["delta_logits"],
                "raw_boundary_band": None if refiner_outputs is None else refiner_outputs["raw_boundary_band"],
                "search_support": None if refiner_outputs is None else refiner_outputs["search_support"],
                "support_valid": None if refiner_outputs is None else refiner_outputs["support_valid"],
            }
            # Refined coarse logits are the only downstream semantic source;
            # raw logits remain independently supervised by the C4 objective.
            if self.use_shape_dense:
                (
                    shape_prior_prompt_logits,
                    shape_dense_valid,
                    dense_excavation_stats,
                ) = self._shape_prior_to_prompt_mask(shape_prior_mask_logits, boxes)
                for stat_name, stat_value in dense_excavation_stats.items():
                    debug_stats[f"GAUSSIAN/{stat_name}"] = stat_value
            else:
                # True sparse-only: keep the coarse mask for point mining and
                # auxiliary supervision, but skip transform/paste/PE mask work.
                shape_prior_prompt_logits = None

        # === PE 单次调用: box → sparse_pe, mask → dense_pe ===
        dense_pe = None
        if self.prompt_encoder is not None and boxes is not None:
            # 批次2 子任务4：sparse-only(use_shape_dense=False)时不传 mask 给 PE（审查 2.3）
            masks_for_pe = shape_prior_prompt_logits if self.use_shape_dense else None
            # 批次2 子任务4：shape dense gate 三模式有效值（审查 2.1）
            prompt_scale = self._effective_shape_dense_alpha()
            if prompt_scale is not None:
                _gate = prompt_scale.detach().float()
                debug_stats["GATE/shape_dense_alpha_mean"] = _gate.mean()
                debug_stats["GATE/shape_dense_alpha_min"] = _gate.min()
                debug_stats["GATE/shape_dense_alpha_max"] = _gate.max()
            if masks_for_pe is not None and masks_for_pe.numel() > 0:
                _dense_dbg = masks_for_pe.detach().float()
                debug_stats["DENSE/canvas_mean"] = _dense_dbg.mean()
                debug_stats["DENSE/canvas_std"] = _dense_dbg.std()
                debug_stats["DENSE/canvas_min"] = _dense_dbg.min()
                debug_stats["DENSE/canvas_max"] = _dense_dbg.max()
                debug_stats["DENSE/canvas_nonzero_ratio"] = (_dense_dbg != 0).float().mean()
            # === shape_point 模式: 从 shape_prior_mask 挖点，一次 PE 调用编码点+box+mask ===
            # (prompt_sparse_mode 只剩 point/shape_point，point 不带 PromptEncoder)
            if self.prompt_sparse_mode == "shape_point":
                if shape_prior_mask_logits is None:
                    raise ValueError(
                        "prompt_sparse_mode='shape_point' requires shape_prior enabled "
                        "with non-None mask_logits. Got shape_prior_mask_logits=None."
                    )
                coords, labels, point_stats = self.shape_point_miner(
                    shape_prior_mask_logits,
                    prompt_geometry_boxes,
                    self.prompt_encoder_image_size,
                )
                (self._last_shape_point_local_yx,
                 self._last_shape_point_labels) = self.shape_point_miner.get_last_local_points()
                # point_stats 是 prediction-based（不含 GT）；GT-based stats 由上层训练循环算
                for _k, _v in point_stats.items():
                    debug_stats[f"SP/{_k}"] = (
                        _v.detach() if torch.is_tensor(_v) else _v
                    )
                points_for_pe, boxes_for_pe, masks_for_pe = (
                    self._select_explicit_prompt_inputs(
                        coords, labels, prompt_geometry_boxes, masks_for_pe
                    )
                )
                if (
                    points_for_pe is None
                    and boxes_for_pe is None
                    and masks_for_pe is None
                ):
                    # PromptEncoder cannot infer an ROI batch size when every
                    # modality is absent. Construct the true empty sparse set
                    # directly; the canonical base dense embedding is built
                    # below for all ``roi_bs`` instances.
                    sparse_pe = image_embeddings.new_empty((roi_bs, 0, 256))
                    dense_pe_from_pe = None
                else:
                    sparse_pe, dense_pe_from_pe = self.prompt_encoder(
                        points=points_for_pe,
                        boxes=boxes_for_pe,
                        masks=masks_for_pe,
                    )
                    if points_for_pe is not None:
                        sparse_pe = self._neutralize_invalid_point_tokens(
                            sparse_pe, labels
                        )
                debug_stats["PROMPT/use_box_token"] = float(
                    self.explicit_use_box_prompt
                )
                debug_stats["PROMPT/use_point_token"] = float(
                    self.explicit_use_point_prompt
                )
                debug_stats["PROMPT/use_dense_mask"] = float(
                    self.explicit_use_dense_prompt
                )
                # points: [N,4,256]; points_box(_dense): [N,6,256].
            if masks_for_pe is not None:
                dense_pe = dense_pe_from_pe  # [N,256,Hemb,Wemb], 1024输入通常为32x32
        else:
            prompt_scale = None

        if self.prompt_sparse_mode == "point":
            sparse_embeddings = point_embs                                    # [N,5,256]
        elif self.prompt_sparse_mode == "shape_point":
            # PE output is [N,4,256] for points, or [N,6,256] when box tokens are enabled.
            if sparse_pe is None:
                raise ValueError(
                    "prompt_sparse_mode='shape_point' requires prompt_encoder_enabled=True, "
                    "shape_prior enabled, and boxes to be provided"
                )
            sparse_embeddings = sparse_pe                                     # [N,4 or 6,256]
        else:
            raise ValueError(f"Unknown prompt_sparse_mode='{self.prompt_sparse_mode}', "
                             f"expected 'point'/'shape_point'")

        num_roi_per_image = torch.bincount(
            roi_img_ids.long(), minlength=img_bs,
        )[:img_bs]

        # === dense_embeddings ===
        if self.no_mask_embed is None:
            base_dense = image_embeddings.new_zeros(
                roi_bs, 256, image_embedding_size[0], image_embedding_size[1]
            )
        else:
            base_dense = self.no_mask_embed.reshape(1, -1, 1, 1).expand(
                roi_bs, -1, image_embedding_size[0], image_embedding_size[1]
            )
        if self.roi_dense_proj is not None:
            roi_dense = self.roi_dense_proj(x)  # [N,256,14,14]
            roi_dense = F.interpolate(
                roi_dense, size=image_embedding_size,
                mode='bilinear', align_corners=False,
            )
            base_dense = base_dense + self.roi_dense_scale * roi_dense

        if dense_pe is not None:
            if shape_dense_valid is not None:
                from .dense_prompt_utils import replace_invalid_dense_with_base
                dense_pe = replace_invalid_dense_with_base(
                    dense_pe, base_dense, shape_dense_valid
                )
            if self.restrict_dense_prompt_to_box:
                support = self._dense_embedding_box_support(
                    boxes, image_embedding_size, dense_pe.dtype
                )
                dense_pe = base_dense + support * (dense_pe - base_dense)
            # 所有模式统一走门控残差: base_dense + prompt_scale * (dense_pe - base_dense)
            # points_box_dense: bounded global-sigmoid interpolation between
            # pretrained no-mask and shape dense embedding.
            # Legacy/fixed modes remain only for controlled ablations.
            dense_delta = dense_pe - base_dense
            dense_embeddings = base_dense + prompt_scale * dense_delta
            if dense_embeddings.numel() > 0:
                _base_norm = base_dense.detach().float().flatten(1).norm(dim=1)
                _source_delta_norm = dense_delta.detach().float().flatten(1).norm(dim=1)
                _applied_delta_norm = (
                    dense_embeddings.detach().float() - base_dense.detach().float()
                ).flatten(1).norm(dim=1)
                _norm_denom = _base_norm.clamp_min(1e-6)
                debug_stats["DENSE/residual_alpha"] = prompt_scale.detach().float().mean()
                debug_stats["DENSE/base_norm"] = _base_norm.mean()
                debug_stats["DENSE/shape_norm"] = dense_pe.detach().float().flatten(1).norm(dim=1).mean()
                debug_stats["DENSE/final_norm"] = dense_embeddings.detach().float().flatten(1).norm(dim=1).mean()
                debug_stats["DENSE/source_delta_norm"] = _source_delta_norm.mean()
                debug_stats["DENSE/applied_delta_norm"] = _applied_delta_norm.mean()
                debug_stats["DENSE/source_delta_ratio"] = (_source_delta_norm / _norm_denom).mean()
                debug_stats["DENSE/applied_delta_ratio"] = (_applied_delta_norm / _norm_denom).mean()
        else:
            dense_embeddings = base_dense

        # P2BoundaryRefiner has already calibrated ROI-local coarse logits before the
        # frozen PromptEncoder.  No embedding-space residual is applied here.
        image_embeddings = image_embeddings.repeat_interleave(
            num_roi_per_image, dim=0
        )

        if self.prompt_encoder is not None and hasattr(self.prompt_encoder, "get_dense_pe"):
            image_positional_embeddings = self.prompt_encoder.get_dense_pe().to(
                device=image_embeddings.device,
                dtype=image_embeddings.dtype,
            )
            if image_positional_embeddings.shape[0] == 1 and img_bs > 1:
                image_positional_embeddings = image_positional_embeddings.expand(
                    img_bs, -1, -1, -1
                )
        if image_positional_embeddings.dim() == 3:
            image_positional_embeddings = image_positional_embeddings.unsqueeze(0)
        if image_positional_embeddings.dim() == 4:
            image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        else:
            image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        
        if high_res_features is not None:
            high_res_features_expanded = [
                feat.repeat_interleave(num_roi_per_image, dim=0)
                for feat in high_res_features[:2]
            ]
        else:
            high_res_features_expanded = None
        
        low_res_masks, iou_predictions, mask_tokens_out = self.mask_decoder(
            image_embeddings,
            image_positional_embeddings,
            sparse_embeddings,
            dense_embeddings,
            self.multimask_output,
            False,
            high_res_features=high_res_features_expanded,
        )
        h, w = low_res_masks.shape[-2:]
        low_res_masks = low_res_masks.reshape(roi_bs, -1, h, w)
        iou_predictions = iou_predictions.reshape(roi_bs, -1)
        mask_tokens_out = mask_tokens_out.reshape(
            roi_bs, -1, mask_tokens_out.shape[-1]
        )
        quality_predictions = iou_predictions
        # Persist the latest forward diagnostics for the trainer.  Previously
        # these values lived only in this local dictionary, so the logger could
        # not observe the effective dense residual even when debug logging was
        # enabled.  Detach tensors to avoid retaining the autograd graph.
        self._last_prompt_debug_stats = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in debug_stats.items()
        }
        return (
            low_res_masks,
            iou_predictions,
            coarse_outputs,
            quality_predictions,
            mask_tokens_out,
        )

