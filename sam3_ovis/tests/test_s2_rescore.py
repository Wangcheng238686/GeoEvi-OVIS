"""S2 分数重标逻辑的合成数据自测（不读任何真实候选，CPU，无 GPU）。

锁定契约（协议 notes/reports/2026-11-24-s2-protocol.md §2）：
- 恒等配置 (T=1, q=0) 不改分数、不改成员、不改输入；
- 温度是单调变换：**类内**相对顺序不变；不保证跨类有序（这正是温度
  的设计目的——改变类间/图间混排，协议 §4 风险项）；
- 逐类阈值：严格小于丢弃、等于保留；缺阈值的类完全不受影响；
- 先过滤后重标；网格封闭为 15 个配置；
- select 阶段源码不引用 val 目录（val 不选参的静态保证）。

Run: "$PYTHON" -m pytest sam3_ovis/tests/test_s2_rescore.py -q
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph import s2_rescore as s2  # noqa: E402

CANDS = [
    {"cls": "A", "score": 0.9, "box": [0, 0, 10, 10], "rle": "r1"},
    {"cls": "A", "score": 0.2, "box": [100, 100, 110, 110], "rle": "r2"},
    {"cls": "B", "score": 0.5, "box": [200, 200, 210, 210], "rle": "r3"},
    {"cls": "B", "score": 0.05, "box": [300, 300, 310, 310], "rle": "r4"},
]


def _sig(out):
    return [(c["cls"], round(c["score"], 6)) for c in out]


def test_identity_config_changes_nothing():
    out = s2.rescore(CANDS, 1.0, 0, {})
    assert _sig(out) == [("A", 0.9), ("A", 0.2), ("B", 0.5), ("B", 0.05)]


def test_input_not_mutated():
    s2.rescore(CANDS, 0.5, 50, {"A": 0.5})
    assert [c["score"] for c in CANDS] == [0.9, 0.2, 0.5, 0.05]


def test_temperature_values():
    assert _sig(s2.rescore(CANDS, 0.5, 0, {})) == [
        ("A", 0.81), ("A", 0.04), ("B", 0.25), ("B", 0.0025)]
    assert _sig(s2.rescore(CANDS, 2.0, 0, {})) == [
        ("A", 0.948683), ("A", 0.447214),
        ("B", 0.707107), ("B", 0.223607)]


def test_temperature_keeps_membership():
    """温度（若被调用）不改变成员——只缩放分数。撤销温度的理由是
    它不改排序（见下），不是它会丢候选。"""
    for T in (0.5, 1.0, 2.0):
        assert len(s2.rescore(CANDS, T, 0, {})) == len(CANDS)


def test_grid_constant_is_identity():
    """网格恒用 TEMP=1.0：rescore 必须走原分数分支，不施任何变换。"""
    assert s2.rescore(CANDS, s2.TEMP, 0, {}) == CANDS


def test_make_tau_per_class_quantile():
    tau = s2.make_tau([{"img_id": 1, "cands": CANDS}], 50)
    assert set(tau) == {"A", "B"}
    assert abs(tau["A"] - 0.55) < 1e-12
    assert abs(tau["B"] - 0.275) < 1e-12


def test_make_tau_zero_means_no_threshold():
    assert s2.make_tau([{"img_id": 1, "cands": CANDS}], 0) == {}


def test_threshold_strict_less_drops_equal_keeps():
    assert _sig(s2.rescore(CANDS, 1.0, 50, {"A": 0.2, "B": 0.5})) == [
        ("A", 0.9), ("A", 0.2), ("B", 0.5)]


def test_class_without_tau_unaffected():
    assert _sig(s2.rescore(CANDS, 1.0, 50, {"A": 0.5})) == [
        ("A", 0.9), ("B", 0.5), ("B", 0.05)]


def test_filter_then_rescore():
    assert _sig(s2.rescore(CANDS, 0.5, 50, {"A": 0.5})) == [
        ("A", 0.81), ("B", 0.25), ("B", 0.0025)]


def test_grid_is_closed_and_matches_protocol():
    """协议 §2.1：温度维度撤销，网格 = 5 个逐类阈值配置。"""
    assert s2.QS == [0, 50, 80, 90, 95]
    assert s2.TEMP == 1.0
    assert not hasattr(s2, "TEMPS"), "温度网格应已撤销"


def test_temperature_never_reorders_any_pair():
    """温度维度被撤销的证据（协议 §2.1）：对全体候选施加同一严格
    单调函数不改变任何一对候选的相对次序，故对 COCO mAP 零影响。
    此测试钉住该事实，防止温度维度被无意重新引入。"""
    import itertools
    base = s2.rescore(CANDS, 1.0, 0, {})
    for T in (0.5, 2.0, 7.0):
        out = s2.rescore(CANDS, T, 0, {})
        for (a0, a1), (b0, b1) in zip(itertools.combinations(base, 2),
                                      itertools.combinations(out, 2)):
            assert (a0["score"] > a1["score"]) == (b0["score"] > b1["score"])


def test_gates_match_protocol():
    assert s2.GATE_FP == 350.0
    assert s2.GATE_MAP_LO == -1.0
    assert s2.BOOT_N == 2000


def _body_src(func):
    """函数体源码（去掉 docstring 与签名），供静态检查用。"""
    src = inspect.getsource(func)
    doc = inspect.getdoc(func) or ""
    if doc:
        src = src.replace(doc, "")
    return src


def test_select_phase_never_touches_val():
    """val 不选参的静态保证：select 的函数体不得引用任何 val 路径、
    不得调用 eval 阶段专用的加载器。"""
    body = _body_src(s2.phase_select)
    for forbidden in ("VAL_ROOT", "OUT_ROOT", "phase_eval",
                      "_load_recs(VAL", "freeler/"):
        assert forbidden not in body, f"select 函数体引用了 {forbidden!r}"
    # 且只从 train 池读
    assert "_load_recs(TRAIN_ROOT" in body
