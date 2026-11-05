"""SAM3 原生 memory 组件契约回归（源自 2026-10-28 接口报告的 CPU 探针）。

来源：notes/reports/2026-10-28-sam3-interface-readiness.md 与临时脚本
/tmp/geoevi_sam3_memory_interface_probe_20261028.py（2026-11-05 移植为
pytest，数字契约不变）。CPU 小张量、无权重、无 GPU。

锁定契约（改动 vendored SAM3 或升级依赖后本文件必须仍全绿，否则视为
接口破坏，须重新核验 interface-readiness 报告）：
- reader：`transformer.encoder` 直接调用，src=[HW,B,256]、prompt=[KHW,B,64]，
  输出 [HW,B,256]；query/memory/reader-q_proj 三处梯度非零；
- reader 拒绝非方形 RoPE（K 与 HW 不匹配）与非空 padding mask；
- writer：SimpleMaskEncoder 输入 [B,256,h,w] 特征 + [B,1,H,W] mask
  （logits 分布契约见报告 §2），输出 [B,64,h/16,w/16]；三处梯度非零。
"""

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "sam3"))
sys.path.insert(0, str(_REPO / "sam3" / "sam3"))
sys.path.insert(0, str(_REPO / "sam3_ovis"))

import torch  # noqa: E402

from sam3.model_builder import _create_tracker_transformer  # noqa: E402
from sam3.model.memory import (  # noqa: E402
    CXBlock, SimpleFuser, SimpleMaskDownSampler, SimpleMaskEncoder,
)
from sam3.model.position_encoding import PositionEmbeddingSine  # noqa: E402

SEED = 20261028


@pytest.fixture(scope="module")
def reader():
    return _create_tracker_transformer().encoder.eval()


def _run_reader(reader, p, pp, x, pos):
    return reader(
        prompt=p, prompt_pos=pp,
        src=[x], src_pos=[pos], src_key_padding_mask=[None],
        feat_sizes=[(2, 2)], num_obj_ptr_tokens=0,
    )["memory"]


def test_reader_shapes_and_grads(reader):
    torch.manual_seed(SEED)
    x = torch.randn(4, 1, 256, requires_grad=True)
    pos = torch.randn_like(x)
    p = torch.randn(8, 1, 64, requires_grad=True)
    pp = torch.zeros_like(p)
    y = _run_reader(reader, p, pp, x, pos)
    assert list(y.shape) == [4, 1, 256]
    # Linear 随机方向避免 LayerNorm 近常数范数损失（报告原注）
    (y * torch.randn_like(y)).sum().backward()
    assert x.grad.norm().item() > 1.0
    assert p.grad.norm().item() > 1.0
    q_proj = reader.layers[0].cross_attn_image.q_proj.weight
    assert q_proj.grad.norm().item() > 1.0


def test_reader_rejects_ragged_key_length(reader):
    torch.manual_seed(SEED)
    x = torch.randn(4, 1, 256)
    pos = torch.randn_like(x)
    p = torch.randn(8, 1, 64)
    pp = torch.zeros_like(p)
    with pytest.raises(RuntimeError):
        _run_reader(reader, p[:5], pp[:5], x, pos)


def test_reader_rejects_padding_mask(reader):
    torch.manual_seed(SEED)
    x = torch.randn(4, 1, 256)
    pos = torch.randn_like(x)
    p = torch.randn(8, 1, 64)
    pp = torch.zeros_like(p)
    with pytest.raises(AssertionError):
        reader(
            prompt=p, prompt_pos=pp,
            src=[x], src_pos=[pos], src_key_padding_mask=[None],
            feat_sizes=[(2, 2)], num_obj_ptr_tokens=0,
            prompt_key_padding_mask=torch.zeros(1, 8, dtype=torch.bool),
        )


def test_writer_shapes_and_grads():
    torch.manual_seed(SEED)
    writer = SimpleMaskEncoder(
        out_dim=64,
        position_encoding=PositionEmbeddingSine(64, normalize=True),
        mask_downsampler=SimpleMaskDownSampler(
            kernel_size=3, stride=2, padding=1, interpol_size=[32, 32]),
        fuser=SimpleFuser(
            CXBlock(dim=256, kernel_size=7, padding=3,
                    layer_scale_init_value=1e-6, use_dwconv=True),
            num_layers=2),
    )
    z = torch.randn(1, 256, 2, 2, requires_grad=True)
    m = torch.randn(1, 1, 32, 32, requires_grad=True)
    w = writer(z, m, skip_mask_sigmoid=False)
    assert list(w["vision_features"].shape) == [1, 64, 2, 2]
    # 布局契约：writer 返回 BCHW 网格 pos [B,64,h,w]；展平为 reader 的
    # [HW,B,64] 是 tracker 包装层（_prepare_memory_conditioned_features）
    # 的职责——两种布局不得混用（interface 报告 §3.1 的核心坑）
    assert list(w["vision_pos_enc"][0].shape) == [1, 64, 2, 2]
    w["vision_features"].square().mean().backward()
    assert z.grad.norm().item() > 0.0
    assert m.grad.norm().item() > 0.0
    assert writer.out_proj.weight.grad.norm().item() > 0.0
