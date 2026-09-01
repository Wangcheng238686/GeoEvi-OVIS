"""Clean WHU-1024 / SAM2 Base+ detector baseline.

MLP-bridge baseline config (prompt_sparse_mode="point").  The image
embedding is fixed to the official stride-16 / 64x64 contract.  The neck is
selectable between the legacy aggregator and the PAFPN through ``NECK_TYPE``.
"""

import os

_base_ = ["./rsprompter_anchor_satS_v11_sam2_large_full.py"]

sam2_ckpt_path = os.environ.get(
    "SAM2_CKPT",
    "/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt",
)
# Single fixed resolution contract: official stride-16 / 64x64 image
# embedding, ROI-local final-mask postprocessing, detector-score segm ranking.
sam_image_embedding_stride = 16
sam_image_embedding_size = 64
final_mask_coordinate_mode = "roi_local"

model = dict(
    type="RSPrompterAnchor",
    sam_image_embedding_stride=sam_image_embedding_stride,
    data_preprocessor=dict(
        batch_augments=[
            dict(
                type="BatchFixedSizePad",
                size=(1024, 1024),
                img_pad_value=0,
                pad_mask=True,
                mask_pad_value=0,
                pad_seg=False,
            )
        ]
    ),
    shared_image_embedding=dict(
        image_size=1024,
        embedding_stride=sam_image_embedding_stride,
    ),
    backbone=dict(
        checkpoint_path=sam2_ckpt_path,
        init_cfg=dict(type="Pretrained", checkpoint=sam2_ckpt_path),
        model_size="base_plus",
    ),
    neck=dict(
        in_channels="sam2_hiera_base_plus",
    ),
    train_cfg=dict(
        rcnn=dict(mask_size=(1024, 1024)),
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(
            sam2_mask_decoder=dict(checkpoint_path=sam2_ckpt_path),
            prompt_encoder_image_size=1024,
            prompt_encoder_embed_size=sam_image_embedding_size,
            prompt_sparse_mode="point",
            prompt_encoder_enabled=False,
            freeze_mask_decoder=False,
            freeze_no_mask_embed=False,
            load_no_mask_pretrained=False,
            final_mask_coordinate_mode=final_mask_coordinate_mode,
            shape_prior_cfg=dict(enabled=False),
            p2_boundary_refiner_cfg=dict(enabled=False),
        ),
    ),
)
