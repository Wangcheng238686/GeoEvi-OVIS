#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ablation-id", required=True)
    parser.add_argument("--expected-architecture-id", default=None)
    parser.add_argument("--expected-neck", choices=("aggregator", "pafpn"), required=True)
    parser.add_argument("--prompt-route", choices=("mlp", "coarse"), required=True)
    parser.add_argument("--explicit-prompt-mode", required=True)
    parser.add_argument(
        "--p2-boundary-refiner-enabled", type=int, choices=(0, 1), required=True
    )
    parser.add_argument(
        "--expected-final-mask-mode",
        choices=("roi_local", "full_image"),
        required=True,
    )
    parser.add_argument(
        "--expected-image-embed-stride",
        type=int,
        choices=(16,),
        required=True,
    )
    parser.add_argument("--train-subset-ratio", type=float, default=1.0)
    parser.add_argument("--val-subset-ratio", type=float, default=1.0)
    parser.add_argument("--subset-seed", type=int, default=44)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--build-model", action="store_true")
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expected_subset_size(full_size, ratio):
    return full_size if ratio >= 1.0 else max(1, int(full_size * ratio))


def main():
    args = parse_args()
    for name, value in (
        ("train_subset_ratio", args.train_subset_ratio),
        ("val_subset_ratio", args.val_subset_ratio),
    ):
        require(0.0 < value <= 1.0, f"{name} must be in (0,1], got {value}")

    from mmengine import Config

    cfg = Config.fromfile(str(PROJECT_ROOT / args.config))
    from models.architecture_contract import architecture_contract

    resolved_architecture = architecture_contract(cfg.model.to_dict())
    require(
        resolved_architecture["architecture_id"]
        == (args.expected_architecture_id or args.ablation_id),
        "wrapper ABLATION_ID does not match resolved cfg.model: "
        f"{args.expected_architecture_id or args.ablation_id!r} != "
        f"{resolved_architecture['architecture_id']!r}",
    )
    neck = cfg.model.neck
    head = cfg.model.roi_head.mask_head
    expected_neck_type = (
        "RSSAM2PAFPN" if args.expected_neck == "pafpn"
        else "RSFeatureAggregatorSAM2"
    )
    require(neck.type == expected_neck_type, (neck.type, expected_neck_type))
    require(neck.in_channels == "sam2_hiera_base_plus", neck.in_channels)
    expected_embed_size = 1024 // args.expected_image_embed_stride
    require(
        int(cfg.model.sam_image_embedding_stride)
        == args.expected_image_embed_stride,
        "top-level SAM image embedding stride mismatch",
    )
    require(
        int(cfg.model.shared_image_embedding.embedding_stride)
        == args.expected_image_embed_stride,
        "positional embedding stride mismatch",
    )
    require(
        int(head.prompt_encoder_embed_size) == expected_embed_size,
        "PromptEncoder/image embedding spatial size mismatch",
    )
    require(
        head.final_mask_coordinate_mode == args.expected_final_mask_mode,
        "final-mask coordinate mode mismatch",
    )
    final_mask_loss_cfg = head.get("final_mask_loss_cfg", {})
    actual_final_mask_loss_mode = str(
        final_mask_loss_cfg.get("mode", "standard")
    )
    require(
        actual_final_mask_loss_mode == "standard",
        "final-mask loss mode must be standard",
    )
    require(
        getattr(head, "segm_score_mode", "detector") == "detector",
        "segmentation score mode must be detector",
    )

    if args.prompt_route == "mlp":
        require(head.prompt_sparse_mode == "point", head.prompt_sparse_mode)
        require(not bool(head.prompt_encoder_enabled), "MLP route must disable PromptEncoder")
        require(not bool(head.shape_prior_cfg.enabled), "MLP route must disable coarse head")
        require(
            not bool(head.p2_boundary_refiner_cfg.enabled),
            "MLP route must disable P2BoundaryRefiner",
        )
        require(
            not bool(head.load_no_mask_pretrained),
            "MLP baseline must use the legacy zero-initialized no_mask_embed",
        )
        require(
            not bool(head.freeze_no_mask_embed),
            "MLP baseline no_mask_embed must remain trainable",
        )
    else:
        require(head.prompt_sparse_mode == "shape_point", head.prompt_sparse_mode)
        require(bool(head.prompt_encoder_enabled), "coarse route must enable PromptEncoder")
        require(bool(head.shape_prior_cfg.enabled), "coarse route must enable shape prior")
        require(
            head.explicit_prompt_mode == args.explicit_prompt_mode,
            (head.explicit_prompt_mode, args.explicit_prompt_mode),
        )
        expect_dense = args.explicit_prompt_mode in {"mask", "points_box_dense"}
        require(
            bool(head.shape_prior_cfg.use_shape_dense) == expect_dense,
            "dense prompt config mismatch",
        )
        dense_prompt_cfg = head.get("dense_prompt_cfg", {})
        expected_dense_transform = os.environ.get(
            "SHAPE_DENSE_TRANSFORM", "raw_logits"
        )
        expected_dense_detach = os.environ.get("SHAPE_DENSE_DETACH", "0") == "1"
        require(
            dense_prompt_cfg.get("transform", "raw_logits")
            == expected_dense_transform,
            "dense prompt transform mismatch",
        )
        require(
            bool(dense_prompt_cfg.get("detach_input", False))
            == expected_dense_detach,
            "dense prompt detach contract mismatch",
        )
        if expected_dense_transform == "gaussian_edt":
            require(expect_dense, "gaussian_edt requires a dense prompt experiment")
            require(expected_dense_detach, "gaussian_edt must be detached")
            require(
                dense_prompt_cfg.get("clamp_range", None) is None,
                "gaussian_edt must not be clamped",
            )
            gaussian_expected = {
                "foreground_threshold": float(
                    os.environ.get("SHAPE_GAUSSIAN_FOREGROUND_THRESHOLD", "0.5")
                ),
                "gaussian_omega": float(
                    os.environ.get("SHAPE_GAUSSIAN_OMEGA", "15.0")
                ),
                "gaussian_gamma": float(
                    os.environ.get("SHAPE_GAUSSIAN_GAMMA", "4.0")
                ),
            }
            for key, expected in gaussian_expected.items():
                require(
                    math.isclose(float(dense_prompt_cfg[key]), expected),
                    f"Gaussian dense prompt {key} mismatch",
                )
        require(
            bool(head.restrict_dense_prompt_to_box),
            "dense prompt embedding delta must be restricted to the proposal box",
        )
        require(
            bool(head.p2_boundary_refiner_cfg.enabled)
            == bool(args.p2_boundary_refiner_enabled),
            "P2BoundaryRefiner config mismatch",
        )
        refiner_expected = {
            "beta": 0.20,
            "delta_logit_max": float(
                os.environ.get("P2_BOUNDARY_REFINER_DELTA_LOGIT_MAX", "2.0")
            ),
            "projected_channels": int(
                os.environ.get("P2_BOUNDARY_REFINER_PROJECTED_CHANNELS", "64")
            ),
            "mid_channels": int(
                os.environ.get("P2_BOUNDARY_REFINER_MID_CHANNELS", "64")
            ),
            "boundary_loss_weight": float(
                os.environ.get("P2_BOUNDARY_REFINER_LOSS_WEIGHT", "0.05")
            ),
            "spatial_scale": 0.25,
            "roi_output_size": 32,
            "coarse_size": 64,
            "boundary_band_radius": 2,
            "search_band_radius": 4,
            "max_search_coverage": 0.50,
        }
        for key, expected in refiner_expected.items():
            actual = head.p2_boundary_refiner_cfg[key]
            if isinstance(expected, float):
                require(math.isclose(float(actual), expected), (key, actual, expected))
            else:
                require(actual == expected, (key, actual, expected))
        require(
            bool(head.load_no_mask_pretrained),
            "coarse route must retain the pretrained SAM2 no-mask embedding",
        )
        require(
            bool(head.freeze_no_mask_embed),
            "coarse route must freeze the pretrained no-mask embedding",
        )
        expected_adaptive = os.environ.get(
            "SHAPE_POINT_ADAPTIVE_VALIDITY", "1"
        ) == "1"
        require(
            bool(head.shape_point_miner_cfg.adaptive_validity)
            == expected_adaptive,
            "shape-point adaptive-validity mismatch",
        )
        expected_shape_weight = float(
            os.environ.get("SHAPE_PRIOR_LOSS_WEIGHT", "0.10")
        )
        expected_schedule_mode = os.environ.get(
            "SHAPE_LOSS_SCHEDULE_MODE", "fixed"
        )
        require(
            head.coarse_mask_loss_cfg.schedule_mode == expected_schedule_mode,
            "coarse loss schedule mode mismatch",
        )
        require(
            math.isclose(
                float(head.coarse_mask_loss_cfg.weight),
                expected_shape_weight,
            ),
            "coarse loss fixed/fallback weight mismatch",
        )
        schedule = list(head.coarse_mask_loss_cfg.weight_schedule)
        if expected_schedule_mode == "fixed":
            require(not schedule, "fixed coarse loss must have an empty schedule")
        else:
            require(len(schedule) == 2, "two-stage coarse loss requires two ranges")
            require(
                int(schedule[0]["end_epoch"])
                == int(os.environ.get("SHAPE_LOSS_STAGE1_END", "5")),
                "coarse loss stage-1 end mismatch",
            )
            require(
                math.isclose(
                    float(schedule[0]["weight"]),
                    float(os.environ.get("SHAPE_LOSS_WEIGHT_STAGE1", "0.20")),
                ),
                "coarse loss stage-1 weight mismatch",
            )
            require(
                math.isclose(
                    float(schedule[1]["weight"]),
                    float(os.environ.get("SHAPE_LOSS_WEIGHT_STAGE2", "0.10")),
                ),
                "coarse loss stage-2 weight mismatch",
            )

    report = {
        "ablation_id": args.ablation_id,
        "config": args.config,
        "neck": neck.type,
        "prompt_route": args.prompt_route,
        "prompt_sparse_mode": head.prompt_sparse_mode,
        "explicit_prompt_mode": (
            None if args.prompt_route == "mlp" else head.explicit_prompt_mode
        ),
        "prompt_encoder_enabled": bool(head.prompt_encoder_enabled),
        "shape_prior_enabled": bool(head.shape_prior_cfg.enabled),
        "shape_context_fusion": (
            None if args.prompt_route == "mlp" else "roi_only"
        ),
        "dense_prompt_enabled": bool(head.shape_prior_cfg.get("use_shape_dense", False)),
        "dense_prompt_config": (
            None if args.prompt_route == "mlp" else dict(head.dense_prompt_cfg)
        ),
        "p2_boundary_refiner_enabled": bool(
            head.p2_boundary_refiner_cfg.enabled
        ),
        "final_mask_coordinate_mode": head.final_mask_coordinate_mode,
        "final_mask_loss_mode": actual_final_mask_loss_mode,
        "final_mask_loss_config": dict(final_mask_loss_cfg),
        "segm_score_mode": getattr(head, "segm_score_mode", "detector"),
        "segm_score_key": "scores",
        "p2_boundary_refiner_config": dict(head.p2_boundary_refiner_cfg),
        "architecture_contract": resolved_architecture,
        "sam_image_embedding_stride": int(cfg.model.sam_image_embedding_stride),
        "sam_image_embedding_size": expected_embed_size,
        "shape_point_adaptive_validity": (
            None
            if args.prompt_route == "mlp"
            else bool(head.shape_point_miner_cfg.adaptive_validity)
        ),
        "shape_loss_schedule_mode": (
            None
            if args.prompt_route == "mlp"
            else head.coarse_mask_loss_cfg.schedule_mode
        ),
        "shape_prior_loss_weight": (
            None
            if args.prompt_route == "mlp"
            else float(head.coarse_mask_loss_cfg.weight)
        ),
        "train_subset_ratio": args.train_subset_ratio,
        "val_subset_ratio": args.val_subset_ratio,
        "subset_seed": args.subset_seed,
    }

    if args.check_data:
        from data import create_train_loader

        train_loader, val_loader, train_dataset = create_train_loader(
            data_root=args.data_root,
            batch_size=1,
            image_size=(1024, 1024),
            dataset_format="whu_coco",
            whu_train_ann_file="2.4 annotation/annotation/train.json",
            whu_val_ann_file="2.4 annotation/annotation/validation.json",
            whu_train_img_subdir="2.1 train/train",
            whu_val_img_subdir="2.3 valid/validation",
            train_subset_ratio=args.train_subset_ratio,
            val_subset_ratio=args.val_subset_ratio,
            num_workers=0,
            seed=args.subset_seed,
        )
        val_dataset = val_loader.dataset
        require(
            len(train_dataset)
            == expected_subset_size(train_dataset.full_size, args.train_subset_ratio),
            "training subset size mismatch",
        )
        require(
            len(val_dataset)
            == expected_subset_size(val_dataset.full_size, args.val_subset_ratio),
            "validation subset size mismatch",
        )
        report["data"] = {
            "train": [len(train_dataset), train_dataset.full_size],
            "val": [len(val_dataset), val_dataset.full_size],
        }

    if args.build_model:
        from mmengine.registry import init_default_scope

        init_default_scope("mmdet")
        import data  # noqa: F401
        import models  # noqa: F401
        from mmdet.registry import MODELS

        model = MODELS.build(cfg.model)
        mask_head = model.roi_head.mask_head
        require(
            mask_head.segm_score_mode == "detector"
            and mask_head.segm_score_key == "scores",
            "built model segmentation score contract mismatch",
        )
        require(
            int(model.sam_image_embedding_stride)
            == args.expected_image_embed_stride,
            "built model SAM image embedding stride mismatch",
        )
        require(type(model.neck).__name__ == expected_neck_type, type(model.neck).__name__)
        if args.prompt_route == "mlp":
            require(mask_head.point_emb is not None, "MLP point_emb was not constructed")
            require(mask_head.prompt_encoder is None, "MLP unexpectedly has PromptEncoder")
            require(mask_head.shape_injector is None, "MLP unexpectedly has coarse head")
            require(
                bool(mask_head.no_mask_embed.requires_grad),
                "MLP no_mask_embed is not trainable",
            )
            require(
                not bool(mask_head.load_no_mask_pretrained),
                "MLP no_mask_embed unexpectedly loaded SAM2 weights",
            )
            raw_pe = model.shared_image_embedding.get_dense_pe()
            require(
                tuple(raw_pe.shape[-2:]) == (64, 64),
                "MLP legacy positional parameter must retain its 64x64 shape",
            )
            require(
                bool(model.shared_image_embedding.pos_embed.requires_grad),
                "MLP legacy positional embedding is not trainable",
            )
            require(
                int(torch.count_nonzero(raw_pe).item()) == 0,
                "MLP legacy positional embedding is not zero-initialized",
            )
            effective_pe = model.get_image_wide_positional_embeddings(
                size=expected_embed_size
            )
            require(
                tuple(effective_pe.shape[-2:])
                == (expected_embed_size, expected_embed_size),
                "MLP effective positional embedding spatial size mismatch",
            )
        else:
            require(mask_head.point_emb is None, "coarse route constructed legacy MLP")
            require(mask_head.prompt_encoder is not None, "coarse route missing PromptEncoder")
            require(mask_head.shape_injector is not None, "coarse route missing shape head")
            injector = mask_head.shape_injector
            require(
                getattr(injector, "visual_context_proj", None) is None
                and getattr(injector, "spatial_cross_attn", None) is None
                and getattr(injector, "gamma_head", None) is None
                and getattr(injector, "beta_head", None) is None,
                "roi_only injector must not construct FiLM/context parameters",
            )
            require(
                bool(mask_head.no_mask_pretrained_loaded),
                "coarse route silently missed pretrained no_mask_embed",
            )
            require(
                str(mask_head.dense_prompt_cfg.get("transform", "raw_logits"))
                == os.environ.get("SHAPE_DENSE_TRANSFORM", "raw_logits"),
                "built model dense transform mismatch",
            )
            require(
                bool(mask_head.dense_prompt_cfg.get("detach_input", False))
                == (os.environ.get("SHAPE_DENSE_DETACH", "0") == "1"),
                "built model dense detach mismatch",
            )
            require(
                mask_head.no_mask_embed is not None
                and not bool(mask_head.no_mask_embed.requires_grad),
                "coarse route pretrained no_mask_embed must exist and be frozen",
            )
            mask_head.assert_no_mask_embedding_contract(
                f"{args.ablation_id} preflight"
            )
            require(
                tuple(mask_head.prompt_encoder.image_embedding_size)
                == (expected_embed_size, expected_embed_size),
                "built PromptEncoder image embedding size mismatch",
            )
            require(
                tuple(mask_head.prompt_encoder.mask_input_size)
                == (expected_embed_size * 4, expected_embed_size * 4),
                "built PromptEncoder dense-mask input size mismatch",
            )
            dense_transform = str(
                mask_head.dense_prompt_cfg.get("transform", "raw_logits")
            )
            if dense_transform == "gaussian_edt":
                probe_logits = torch.full(
                    (2, 1, 64, 64), -20.0, requires_grad=True
                )
                with torch.no_grad():
                    probe_logits[0, :, 20:44, 20:44] = 20.0
                probe_boxes = torch.tensor(
                    [
                        [128.0, 192.0, 640.0, 704.0],
                        [512.0, 512.0, 896.0, 896.0],
                    ]
                )
                prompt_canvas, prompt_valid, _ = (
                    mask_head._shape_prior_to_prompt_mask(
                        probe_logits, probe_boxes
                    )
                )
                require(
                    tuple(prompt_canvas.shape)
                    == (2, 1, expected_embed_size * 4, expected_embed_size * 4),
                    "Gaussian prompt canvas shape mismatch",
                )
                require(
                    not prompt_canvas.requires_grad,
                    "hard Gaussian prompt unexpectedly retained coarse gradients",
                )
                require(
                    prompt_valid.tolist() == [True, False],
                    "Gaussian empty-coarse validity regression",
                )
                with torch.no_grad():
                    _, encoded_dense = mask_head.prompt_encoder(
                        points=None,
                        boxes=probe_boxes,
                        masks=prompt_canvas,
                    )
                base_dense = mask_head.no_mask_embed.reshape(1, -1, 1, 1).expand_as(
                    encoded_dense
                )
                from models.dense_prompt_utils import (
                    replace_invalid_dense_with_base,
                )
                restored = replace_invalid_dense_with_base(
                    encoded_dense, base_dense, prompt_valid
                )
                require(
                    torch.equal(restored[1], base_dense[1]),
                    "empty Gaussian ROI did not restore exact no-mask embedding",
                )
                report["gaussian_forward_contract"] = "ok"
            require(
                (mask_head.p2_boundary_refiner is not None)
                == bool(args.p2_boundary_refiner_enabled),
                "P2BoundaryRefiner model/config mismatch",
            )
        # Regression guard shared by validation and checkpoint inference.
        image_size = 64
        bbox = torch.tensor([[16.0, 16.0, 48.0, 48.0]])
        probe_logits = torch.full((1, 1, 8, 8), 20.0)
        if args.expected_final_mask_mode == "full_image":
            probe_logits.fill_(-20.0)
            probe_logits[:, :, :4, :4] = 20.0
        pasted = mask_head._predict_by_feat_single(
            mask_preds=probe_logits,
            bboxes=bbox,
            labels=torch.zeros(1, dtype=torch.long),
            img_meta={
                "scale_factor": (1.0, 1.0),
                "ori_shape": (image_size, image_size),
                "img_shape": (image_size, image_size),
            },
            rcnn_test_cfg=cfg.model.test_cfg.rcnn,
            rescale=False,
            activate_map=False,
        )
        if args.expected_final_mask_mode == "roi_local":
            require(bool(pasted[0, 24:40, 24:40].any()), "pasted mask is empty")
            outside = pasted.clone()
            outside[:, 15:49, 15:49] = False
            require(
                not bool(outside.any()),
                "ROI-local mask leaked outside bbox",
            )
            postprocess = "roi_local_bbox_paste"
        else:
            require(bool(pasted[0, 8, 8]), "full-image top-left signal was lost")
            require(
                not bool(pasted[0, 56, 56]),
                "full-image mask was incorrectly pasted through the bbox",
            )
            postprocess = "full_image_resize"
        report["model_build"] = "ok"
        report["mask_postprocess"] = postprocess
        report["no_mask_pretrained_loaded"] = bool(
            mask_head.no_mask_pretrained_loaded
        )
        report["no_mask_abs_mean"] = float(
            mask_head.no_mask_embed.detach().abs().mean().item()
        )
        report["prompt_dense_mask_input_size"] = (
            None
            if mask_head.prompt_encoder is None
            else list(mask_head.prompt_encoder.mask_input_size)
        )
        report["prompt_encoder_trainable_parameters"] = (
            0
            if mask_head.prompt_encoder is None
            else sum(
                parameter.numel()
                for parameter in mask_head.prompt_encoder.parameters()
                if parameter.requires_grad
            )
        )
        needs_model_config_roundtrip = (
            args.prompt_route == "coarse"
            and (
                os.environ.get("SHAPE_DENSE_TRANSFORM", "raw_logits")
                == "gaussian_edt"
                or os.environ.get("SHAPE_DENSE_DETACH", "0") == "1"
            )
        )
        if needs_model_config_roundtrip:
            # Check the same reconstruction path used by schema-v2 checkpoint
            # inference: build from the fully resolved model_config, without
            # relying on the launcher environment, then load strictly.
            from mmengine.config import ConfigDict

            resolved_model_config = cfg.model.to_dict()
            rebuilt_model = MODELS.build(ConfigDict(resolved_model_config))
            rebuilt_model.load_state_dict(model.state_dict(), strict=True)
            rebuilt_mask_head = rebuilt_model.roi_head.mask_head
            require(
                rebuilt_mask_head.dense_prompt_cfg
                == mask_head.dense_prompt_cfg,
                "checkpoint-style reconstruction changed dense_prompt_cfg",
            )
            report["model_config_strict_roundtrip"] = "ok"

    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    print("ablation contract: OK")


if __name__ == "__main__":
    main()
