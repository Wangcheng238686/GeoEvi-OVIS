"""
SAM2 Adapter for Portable SAM Fusion - Simplified Version
直接使用SAM2的build_sam2函数加载模型
"""

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
from mmdet.models import MaskRCNN, StandardRoIHead
from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.structures.mask.mask_target import mask_target
from mmdet.utils import ConfigType, InstanceList, OptConfigType

from .ckpt_utils import load_module_state_dict_strict


SAM2_AVAILABLE = False
PositionEmbeddingSine = None  # safe default; set when sam2 package imports
SAM2_REPO = os.environ.get("SAM2_REPO")
if SAM2_REPO and SAM2_REPO not in sys.path:
    sys.path.insert(0, SAM2_REPO)
try:
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    from sam2.modeling.sam.mask_decoder import MaskDecoder
    SAM2_AVAILABLE = True
except ImportError:
    pass


def _load_sam2_checkpoint(checkpoint_path: str, map_location: str = "cpu") -> Dict:
    """Load SAM2 checkpoint from file."""
    from mmengine.runner.checkpoint import _load_checkpoint
    checkpoint = _load_checkpoint(checkpoint_path, map_location=map_location)
    # SAM2 checkpoint format is {"model": {...}}
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _resolve_sam2_checkpoint_path(checkpoint_path: Optional[str]) -> Optional[str]:
    """Resolve a SAM2 checkpoint path with the same rule for every submodule."""
    if not checkpoint_path or os.path.isabs(checkpoint_path):
        return checkpoint_path
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(project_root, checkpoint_path)


def _load_pretrained_no_mask_embedding(checkpoint_path: Optional[str]) -> Tensor:
    """Load SAM2's no-mask embedding or fail instead of returning zeros."""
    resolved_path = _resolve_sam2_checkpoint_path(checkpoint_path)
    if not resolved_path or not os.path.isfile(resolved_path):
        raise RuntimeError(
            "Pretrained no-mask embedding was requested, but the SAM2 "
            f"checkpoint does not exist: {resolved_path!r}"
        )
    checkpoint = _load_sam2_checkpoint(resolved_path)
    key = "sam_prompt_encoder.no_mask_embed.weight"
    if key not in checkpoint:
        raise RuntimeError(
            f"Pretrained no-mask embedding key {key!r} is missing from "
            f"{resolved_path}. Check checkpoint integrity and SAM2 model size."
        )
    value = checkpoint[key]
    if tuple(value.shape) != (1, 256):
        raise RuntimeError(
            f"Invalid pretrained no-mask embedding shape for {key!r}: "
            f"expected (1, 256), got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Pretrained no-mask embedding {key!r} contains non-finite values")
    return value.detach().reshape(1, 256, 1, 1).clone()



# --- behaviour-preserving split (2026-09-01): component classes moved to
# dedicated modules; re-exported here so `rsprompter.models_sam2` keeps its
# historical import surface (registry names resolve identically, and the
# shared loaders above stay in this namespace for monkeypatch-based tests).
from .vision_encoder import RSSAM2PositionalEmbedding, RSSAM2VisionEncoder  # noqa: E402,F401
from .necks import RSFeatureAggregatorSAM2, RSSAM2PAFPN  # noqa: E402,F401
from .mask_decoder import RSSAM2MaskDecoderWrapper  # noqa: E402,F401
from .mask_head import RSPrompterAnchorMaskHeadSAM2  # noqa: E402,F401
