#!/usr/bin/env python3
"""R1 训练包装器：先打 vitdet 融合核补丁再进官方 train.py（不改 vendored）。

发布版 vitdet.Mlp.forward 使用无梯度融合核（addmm_act），官方训练器
未带补丁（R1 烟雾实测崩溃于 ValueError: Expected grad to be disabled）。
本包装在进程内以数学等价标准算子替换该 forward（G1 M4 已验证正反向
等价可用），随后以原 CLI 委托 sam3.train.train。

用法：python r1_train_wrapper.py -c configs/r1_isaid.yaml [--use-cluster 0]
"""

import sys

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

from sam3.train import train as _train  # noqa: E402
from sam3.train.utils.train_utils import (  # noqa: E402
    makedir, register_omegaconf_resolvers,
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
