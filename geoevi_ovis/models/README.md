# models/ — 模型层

| 文件 | 用途 |
|---|---|
| `models.py` | 检测器 `RSPrompterAnchor`（MaskRCNN）与 RoI head `RSPrompterAnchorRoIPromptHead`；`extract_feat` 同时产出检测特征与 SAM2 image embedding。 |
| `models_sam2.py` | SAM2 权重加载器（`_load_sam2_checkpoint` 等）与共享 shim；re-export 全部组件类，保持历史导入面。 |
| `vision_encoder.py` | `RSSAM2VisionEncoder`（Hiera + LoRA）与 `RSSAM2PositionalEmbedding`。 |
| `necks.py` | `RSFeatureAggregatorSAM2`（旧 neck 对照）与 `RSSAM2PAFPN`（主线）。 |
| `mask_decoder.py` | `RSSAM2MaskDecoderWrapper`：SAM2 MaskDecoder 严格加载包装。 |
| `mask_head.py` | `RSPrompterAnchorMaskHeadSAM2`：MLP/coarse 双路线 MaskHead，`__init__` + `forward` 主干，组装自下方 mixin。 |
| `mask_head_targets.py` | `MaskHeadTargetsMixin`：ROI/full-image targets、coarse/final loss、predict 后处理。 |
| `shape_prior.py` | `ShapePriorInjector`（coarse 解码器）、`ShapePointMiner`（2P2N 挖掘）。 |
| `coarse_mask_loss.py` | coarse mask 的 BCE+Dice 组合损失与调度。 |
| `dense_prompt_utils.py` | dense prompt 变换、ROI→整图画布粘贴、Gaussian（EDT）挖掘。 |
| `p2_boundary_refiner.py` | P2-BRR：高频边界约束残差修正器。 |
| `architecture_contract.py` | 架构 ID / SHA256 指纹 / checkpoint 契约校验。 |
| `ckpt_utils.py` | SAM2 子模块 checkpoint 严格加载。 |
| `ckpt_migration.py` | 旧 checkpoint 的 shape-gate 迁移（INIT_FROM 兼容层）。 |
