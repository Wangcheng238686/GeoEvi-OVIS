"""S3.3 unit tests: completeness metrics (ECR / URD / CCR / risk).

Run:  $PYTHON sam3_ovis/tests/test_completeness.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

from graph.builder import build_graph  # noqa: E402
from graph.completeness import compute_completeness  # noqa: E402
from graph.schema import Edge, EdgeKind, Node, NodeKind  # noqa: E402


def rle(x, y, w, h, H=64, W=64):
    m = np.zeros((H, W), dtype=np.uint8)
    m[y:y + h, x:x + w] = 1
    r = mask_util.encode(np.asfortranarray(m))
    r["counts"] = r["counts"].decode("utf-8")
    return r


def add_rival(g, cid, cls, score, mask):
    g.add_node(Node(id=f"{cid}@{cls}", kind=NodeKind.CANDIDATE,
                    payload={"score": score, "rle": mask,
                             "rival_class": cls}))


def test_all_high_confident():
    g = build_graph(1, "plane", [
        {"id": "a", "score": 0.9, "rle": rle(0, 0, 20, 20)},
        {"id": "b", "score": 0.8, "rle": rle(30, 30, 15, 15)},
    ], image_size=(64, 64))
    r = compute_completeness(g)
    assert r["explained_coverage_rate"] == 1.0
    assert r["unexplained_response_density"] == 0.0
    assert r["conflict_candidate_rate"] == 0.0
    assert r["risk_score"] == 0.0
    assert "completeness" in g.attrs
    print("all-high OK")


def test_low_unmatched_echoes():
    # one high square (400px) + three low squares far away (100px each)
    g = build_graph(1, "plane", [
        {"id": "h", "score": 0.9, "rle": rle(0, 0, 20, 20)},
        {"id": "l1", "score": 0.2, "rle": rle(40, 0, 10, 10)},
        {"id": "l2", "score": 0.2, "rle": rle(54, 0, 10, 10)},
        {"id": "l3", "score": 0.2, "rle": rle(40, 30, 10, 10)},
    ], image_size=(64, 64))
    r = compute_completeness(g)
    # ECR: high union 400 / all union 700
    assert abs(r["explained_coverage_rate"] - 400 / 700) < 1e-6
    # URD: all three low are unmatched -> 3/4
    assert abs(r["unexplained_response_density"] - 0.75) < 1e-6
    assert r["conflict_candidate_rate"] == 0.0
    expect = (1 - 400 / 700) / 3 + 0.75 / 3
    assert abs(r["risk_score"] - expect) < 1e-6
    print("low echoes OK")


def test_duplicate_and_conflict():
    # duplicate: low copy overlapping high at IoU 1.0; rival competing with h
    g = build_graph(1, "plane", [
        {"id": "h", "score": 0.9, "rle": rle(0, 0, 20, 20)},
        {"id": "dup", "score": 0.3, "rle": rle(0, 0, 20, 20)},
    ], image_size=(64, 64))
    add_rival(g, "cand:h", "helicopter", 0.5, rle(0, 0, 20, 20))
    g.add_edge(Edge(src="cand:h", dst="cand:h@helicopter",
                    kind=EdgeKind.COMPETING_CLASS, weight=1.0))
    r = compute_completeness(g)
    # dup is matched by h -> URD numerator excludes it
    assert r["unexplained_response_density"] == 0.0
    # CCR: h (competition) + dup (duplicate) = 2/2
    assert abs(r["conflict_candidate_rate"] - 1.0) < 1e-6
    expect = (1 - 1.0) / 3 + 0.0 / 3 + 1.0 / 3
    assert abs(r["risk_score"] - expect) < 1e-6
    print("duplicate+conflict OK")


def test_rival_excluded_and_empty():
    # rival-only graph pieces must not enter the pool
    g = build_graph(1, "plane", [
        {"id": "a", "score": 0.9, "rle": rle(0, 0, 20, 20)},
    ], image_size=(64, 64))
    add_rival(g, "a", "helicopter", 0.5, rle(0, 0, 20, 20))
    r = compute_completeness(g)
    assert r["n_candidates"] == 1

    # empty pool: registered blind spot -> ECR=1, risk 0
    g2 = build_graph(2, "plane", [], image_size=(64, 64))
    r2 = compute_completeness(g2)
    assert r2["explained_coverage_rate"] == 1.0
    assert r2["unexplained_response_density"] == 0.0
    assert r2["conflict_candidate_rate"] == 0.0
    assert r2["risk_score"] == 0.0
    print("rival-exclusion + empty OK")


def test_domains():
    g = build_graph(1, "plane", [
        {"id": "a", "score": 0.9, "rle": rle(0, 0, 20, 20)},
        {"id": "b", "score": 0.2, "rle": rle(30, 30, 10, 10)},
    ], image_size=(64, 64))
    add_rival(g, "cand:a", "x", 0.4, rle(0, 0, 20, 20))
    g.add_edge(Edge(src="cand:a", dst="cand:a@x",
                    kind=EdgeKind.COMPETING_CLASS, weight=0.6))
    r = compute_completeness(g)
    for k in ("explained_coverage_rate", "unexplained_response_density",
              "conflict_candidate_rate", "risk_score"):
        assert 0.0 <= r[k] <= 1.0, k
    print("domains OK")


def main():
    test_all_high_confident()
    test_low_unmatched_echoes()
    test_duplicate_and_conflict()
    test_rival_excluded_and_empty()
    test_domains()
    print("completeness tests PASSED")


if __name__ == "__main__":
    main()
