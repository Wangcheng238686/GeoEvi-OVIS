#!/usr/bin/env python3
"""R1 v2 的 LoRA 注入（r=8, alpha=16）——训练包装器与评估脚本共用。

冻结全部参数；对目标注意力投影 Linear 原位替换为 LoRALinear
（y = W0x + scale·B(Ax)，B 零初始化 → 起步等价于冻结基座）。
目标（协议 v2 §2）：vision backbone 注意力投影 + 概念 transformer
注意力投影。名字模式：qkv/q_proj/k_proj/v_proj/out_proj/in_proj
且位于 trunk 或 transformer 下。
"""

import torch
import torch.nn as nn

R = 8
ALPHA = 16
NAME_TOKENS = ("qkv", "q_proj", "k_proj", "v_proj",
               "out_proj", "in_proj")
SCOPE = ("backbone.vision_backbone.trunk", "transformer")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = R, alpha: int = ALPHA):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.zeros(
            r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(
            base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.scale = alpha / r

    # 委托基座属性：部分代码直接读 .weight/.bias（如 MHA 快速路径）
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) \
            * self.scale


def is_target(name: str) -> bool:
    return (any(tok in name for tok in NAME_TOKENS)
            and any(name.startswith(s) or f".{s}" in f".{name}"
                    or s in name for s in SCOPE))


def apply_lora(model: nn.Module, r: int = R, alpha: int = ALPHA):
    """冻结基座并注入 LoRA；返回原 model（原位修改）。

    nn.MultiheadAttention 的子 Linear 被排除：其快速路径直接以
    F.linear(weight) 计算，替换后要么崩溃要么 LoRA 被静默绕过
    （烟雾实测 AttributeError，2026-11-21 登记）。
    """
    for p in model.parameters():
        p.requires_grad_(False)
    n_replaced, n_skip_mha = 0, 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.MultiheadAttention):
            n_skip_mha += 1
            continue
        for attr, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                full = f"{name}.{attr}" if name else attr
                if is_target(full):
                    setattr(mod, attr, LoRALinear(child, r, alpha))
                    n_replaced += 1
    print(f"[lora] replaced {n_replaced} Linears "
          f"(r={r}, alpha={alpha}; skipped {n_skip_mha} MHA modules)",
          flush=True)
    return model


def lora_param_names(model: nn.Module):
    return [n for n, p in model.named_parameters()
            if p.requires_grad]
