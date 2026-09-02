"""S3.1 unit tests: evidence-graph schema + builder structure/serialization.

Run:  $PYTHON sam3_ovis/tests/test_graph_schema.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

from graph.schema import (Edge, EdgeKind, EvidenceGraph, Node,  # noqa: E402
                          NodeKind)
from graph.builder import build_graph, iou_rles  # noqa: E402


def square_rle(x, y, w, h, H=64, W=64):
    m = np.zeros((H, W), dtype=np.uint8)
    m[y:y + h, x:x + w] = 1
    rle = mask_util.encode(np.asfortranarray(m))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def test_schema_roundtrip():
    g = EvidenceGraph(image_id=7, class_name="plane")
    g.add_node(Node("category", NodeKind.CATEGORY, {"phrase": "plane"}))
    g.add_node(Node("cand:1", NodeKind.CANDIDATE, {"score": 0.9}))
    g.add_edge(Edge("category", "cand:1", EdgeKind.TEXT_INSTANCE, 0.9))
    d = g.to_dict()
    g2 = EvidenceGraph.from_dict(d)
    assert g2.image_id == 7 and g2.class_name == "plane"
    assert set(g2.nodes) == {"category", "cand:1"}
    assert len(g2.edges) == 1
    e = g2.edges[0]
    assert e.kind is EdgeKind.TEXT_INSTANCE and abs(e.weight - 0.9) < 1e-9
    nb = g2.neighbors("cand:1", EdgeKind.TEXT_INSTANCE)
    assert len(nb) == 1 and nb[0][0].id == "category"
    print("schema roundtrip OK")


def test_schema_guards():
    g = EvidenceGraph(image_id=1, class_name="x")
    try:
        g.add_edge(Edge("nope", "category", EdgeKind.TEXT_INSTANCE, 0.5))
        raise AssertionError("missing-node edge should fail")
    except ValueError:
        pass
    g.add_node(Node("a", NodeKind.CANDIDATE))
    try:
        g.add_edge(Edge("a", "a", EdgeKind.SAME_CLASS_SUPPORT, 1.5))
        raise AssertionError("weight>1 should fail")
    except ValueError:
        pass
    try:
        g.add_node(Node("a", NodeKind.CANDIDATE))
        raise AssertionError("duplicate id should fail")
    except ValueError:
        pass
    print("schema guards OK")


def test_builder_structure():
    cands = [
        {"id": "a", "score": 0.9, "rle": square_rle(0, 0, 20, 20)},
        {"id": "b", "score": 0.8, "rle": square_rle(0, 0, 22, 22)},
        {"id": "c", "score": 0.7, "rle": square_rle(40, 40, 4, 4)},
    ]
    g = build_graph(image_id=42, class_name="plane", candidates=cands,
                    image_size=(64, 64))
    # nodes: 1 category + 1 scene + 1 unknown + 3 candidates + 3 contexts
    assert len(g.by_kind(NodeKind.CATEGORY)) == 1
    assert len(g.by_kind(NodeKind.SCENE)) == 1
    assert len(g.by_kind(NodeKind.UNKNOWN)) == 1
    assert len(g.by_kind(NodeKind.CANDIDATE)) == 3
    assert len(g.by_kind(NodeKind.LOCAL_CONTEXT)) == 3
    # no exemplar nodes in P1
    assert g.by_kind(NodeKind.EXEMPLAR) == []
    # context edges: one per candidate
    assert len([e for e in g.edges
                if e.kind is EdgeKind.INSTANCE_CONTEXT]) == 3
    # scene coverage edges: one per candidate, area share sane
    scene_edges = [e for e in g.edges if e.kind is EdgeKind.INSTANCE_SCENE]
    assert len(scene_edges) == 3
    # support edges: one undirected edge per unordered pair = C(3,2) = 3
    sup = [e for e in g.edges if e.kind is EdgeKind.SAME_CLASS_SUPPORT]
    assert len(sup) == 3
    # similar-sized pair (20x20 vs 22x22) should have higher support weight
    # than the tiny (4x4) vs large pair
    def w_between(a, b):
        return max(e.weight for e in sup if {e.src, e.dst} == {f"cand:{a}",
                                                               f"cand:{b}"})
    assert w_between("a", "b") > w_between("a", "c")
    # coverage attributes
    assert g.attrs["coverage_ratio"] > 0
    assert "counts" in g.attrs["coverage_rle"]
    # candidate payload geometry
    assert g.nodes["cand:a"].payload["area_px"] == 400.0
    print("builder structure OK")


def test_iou_helper():
    r1 = square_rle(0, 0, 20, 20)
    r2 = square_rle(10, 0, 20, 20)
    assert abs(iou_rles(r1, r2) - (200 / 600)) < 1e-6
    print("iou helper OK")


def main():
    test_schema_roundtrip()
    test_schema_guards()
    test_builder_structure()
    test_iou_helper()
    print("graph schema/builder tests PASSED")


if __name__ == "__main__":
    main()
