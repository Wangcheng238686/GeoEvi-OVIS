#!/usr/bin/env python3
"""R1 v2 训练包装器（v1 基础上增加 LoRA 注入与光度增广）。

职责（不改 vendored 文件）：
1. vitdet 融合核补丁（v1 验证过）：无梯度 addmm_act → 数学等价标准算子；
2. build_sam3_image_model 补丁：构建后经 r1_lora.apply_lora 注入
   LoRA r=8（协议 v2 §2）——hydra 实例化时取本模块属性即生效；
3. PhotometricAug（p=0.5 独立光度增广，GOOSE 单项最大增益）：
   配置中 _target_: r1_train_wrapper.PhotometricAug。
4. 委托官方 train.py 的 CLI。

用法：cd sam3 && python r1_train_wrapper.py -c configs/r1_isaid.yaml
"""

import random

import torch
import sam3.model.vitdet as _vd


def _std_mlp_forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.norm(x)
    x = self.fc2(x)
    x = self.drop2(x)
    return x


_vd.Mlp.forward = _std_mlp_forward

import sam3.model_builder as _mb  # noqa: E402

_orig_build = _mb.build_sam3_image_model


def _build_with_lora(**kw):
    m = _orig_build(**kw)
    from r1_lora import apply_lora
    return apply_lora(m, r=8, alpha=16)


_mb.build_sam3_image_model = _build_with_lora


class PhotometricAug:
    """独立 p=0.5 光度增广（亮度/对比度/饱和度/轻微色相）。

    遵循 transforms API 约定 __call__(datapoint)；作用于 PIL 图，
    概率独立于其他增广（不 consistent_transform）。
    """

    def __init__(self, p: float = 0.5, brightness: float = 0.4,
                 contrast: float = 0.4, saturation: float = 0.4,
                 hue: float = 0.1):
        self.p = p
        from torchvision import transforms as _T
        self.jitter = _T.ColorJitter(brightness, contrast,
                                     saturation, hue)

    def __call__(self, datapoint, **kwargs):
        if random.random() < self.p:
            for i, img in enumerate(datapoint.images):
                if img is not None and hasattr(img, "convert"):
                    try:
                        datapoint.images[i] = self.jitter(img.convert("RGB"))
                    except Exception:
                        pass  # 非 PIL/异常：保持原图（登记：保守跳过）
        return datapoint


from sam3.train import train as _train  # noqa: E402
from sam3.train.utils.train_utils import (  # noqa: E402
    register_omegaconf_resolvers,
)
from hydra import initialize_config_module  # noqa: E402
from argparse import ArgumentParser  # noqa: E402


def main():
    initialize_config_module("sam3.train", version_base="1.2")
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument("--use-cluster", type=int, default=None)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--account", type=str, default=None)
    parser.add_argument("--qos", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--num-nodes", type=int, default=None)
    args = parser.parse_args()
    args.use_cluster = (bool(args.use_cluster)
                        if args.use_cluster is not None else None)
    register_omegaconf_resolvers()
    _train.main(args)


if __name__ == "__main__":
    main()
