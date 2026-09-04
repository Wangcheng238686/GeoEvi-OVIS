"""S3.4 unit tests: message passing, duplicate merging, risk modulation.

Run:  $PYTHON sam3_ovis/tests/test_message_passing.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

from graph.builder import build_graph  # noqa: E402
from graph.config import MP  # noqa: E402
from graph.message_passing import (  # noqa: E402
    merge_duplicate_clusters, propagate_reliability, rank_candidates,
    run_graph_inference,
)
from graph.schema import Edge, EdgeKind, Node, NodeKind  # noqa: E402


def rle(x, y, w, h, H=64, W=64):
    m = np.zeros((H, W), dtype=np.uint8)
    m[y:y + h, x:x + w] = 1
    r = mask_util.encode(np.asfortranarray(m))
    r["counts"] = r["counts"].decode("utf-8")
    return r


def test_support_boost_and_competition_penalty():
    # a: supported duplicate of b (IoU 1, scale same) -> reliability boosted
    # c: competes with higher-rel rival -> penalised
    g = build_graph(1, "plane", [
        {"id": "a", "score": 0.5, "rle": rle(0, 0, 20, 20)},
        {"id": "b", "score": 0.9, "rle": rle(0, 0, 20, 20)},
        {"id": "c", "score": 0.5, "rle": rle(40, 40, 10, 10)},
    ], image_size=(64, 64))
    add_rival(g, "cand:c", "helicopter", 0.9, rle(40, 40, 10, 10))
    g.add_edge(Edge(src="cand:c", dst="cand:c@helicopter",
                    kind=EdgeKind.COMPETING_CLASS, weight=1.0))

    rel = propagate_reliability(g)
    base_a = (0.5 - 0.05) / 0.95
    base_c = base_a
    # a gains support from b (rel_b ~ 0.8947): round1+2 boosts it above base
    assert rel["cand:a"] > base_a, (rel["cand:a"], base_a)
    # c is penalised below base
    assert rel["cand:c"] < base_c, (rel["cand:c"], base_c)
    for v in rel.values():
        assert 0.0 <= v <= 1.0
    print("support boost + competition penalty OK")


def test_duplicate_merge_reinforces():
    # two overlapping candidates (IoU 1.0): merged into ONE detection whose
    # score = probabilistic OR of both (standard NMS would discard one)
    g = build_graph(1, "plane", [
        {"id": "hi", "score": 0.9, "rle": rle(0, 0, 20, 20)},
        {"id": "lo", "score": 0.5, "rle": rle(0, 0, 20, 20)},
    ], image_size=(64, 64))
    rel = propagate_reliability(g)
    out = merge_duplicate_clusters(g, rel)
    assert len(out) == 1, out
    r_hi, r_lo = rel["cand:hi"], rel["cand:lo"]
    expect = 1 - (1 - r_hi) * (1 - r_lo)
    assert abs(out[0]["merged_score"] - expect) < 1e-4
    assert out[0]["cand_id"] == "cand:hi"          # representative = highest rel
    assert sorted(out[0]["cluster_members"]) == ["cand:hi", "cand:lo"]
    # reinforcement: merged score exceeds either member's reliability
    assert out[0]["merged_score"] > max(r_hi, r_lo)
    print("duplicate merge reinforces OK")


def test_far_apart_same_scale_not_merged():
    # support edge exists (same scale arm) but no overlap -> separate clusters
    g = build_graph(1, "plane", [
        {"id": "a", "score": 0.9, "rle": rle(0, 0, 20, 20)},
        {"id": "b", "score": 0.9, "rle": rle(40, 40, 20, 20)},
    ], image_size=(64, 64))
    rel = propagate_reliability(g)
    out = merge_duplicate_clusters(g, rel)
    assert len(out) == 2, out
    print("far-apart not merged OK")


def test_risk_modulation_and_alarm():
    g = build_graph(1, "plane", [
        {"id": "h", "score": 0.9, "rle": rle(0, 0, 20, 20)},
        {"id": "l1", "score": 0.2, "rle": rle(40, 0, 10, 10)},
        {"id": "l2", "score": 0.2, "rle": rle(54, 0, 10, 10)},
        {"id": "l3", "score": 0.2, "rle": rle(40, 30, 10, 10)},
    ], image_size=(64, 64))
    out, meta = run_graph_inference(g)
    risk = meta["risk_score"]
    assert risk > 0
    factor = 1 - MP.lambda_risk * risk
    for d in out:
        assert abs(d["final_score"] - d["merged_score"] * factor) < 1e-4
        assert d["risk_modulation"] == round(factor, 6)
    # alarm threshold behaviour
    assert meta["incompleteness_alarm"] == (risk >= MP.theta_alarm)
    print("risk modulation OK")


def test_rival_never_receives_reliability():
    g = build_graph(1, "plane", [
        {"id": "a", "score": 0.9, "rle": rle(0, 0, 20, 20)},
    ], image_size=(64, 64))
    add_rival(g, "cand:a", "helicopter", 0.5, rle(0, 0, 20, 20))
    g.add_edge(Edge(src="cand:a", dst="cand:a@helicopter",
                    kind=EdgeKind.COMPETING_CLASS, weight=0.7))
    rel = propagate_reliability(g)
    assert "cand:a@helicopter" not in rel
    print("rival exclusion OK")


def add_rival(g, cid, cls, score, mask):
    g.add_node(Node(id=f"{cid}@{cls}", kind=NodeKind.CANDIDATE,
                    payload={"score": score, "rle": mask,
                             "rival_class": cls}))


def main():
    test_support_boost_and_competition_penalty()
    test_duplicate_merge_reinforces()
    test_far_apart_same_scale_not_merged()
    test_risk_modulation_and_alarm()
    test_rival_never_receives_reliability()
    print("message passing tests PASSED")


if __name__ == "__main__":
    main()
