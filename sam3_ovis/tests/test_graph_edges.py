"""S3.2 unit tests: each edge family — direction + weight domain (完成判据).

Mock embeddings plant exact cosines so every registered trigger is exercised
deterministically. Run:  $PYTHON sam3_ovis/tests/test_graph_edges.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

from graph.provider import MockProvider  # noqa: E402
from graph import edges as E  # noqa: E402
from graph.schema import EdgeKind, NodeKind  # noqa: E402
from graph.builder import build_graph  # noqa: E402


def rle(x, y, w, h, H=64, W=64):
    m = np.zeros((H, W), dtype=np.uint8)
    m[y:y + h, x:x + w] = 1
    r = mask_util.encode(np.asfortranarray(m))
    r["counts"] = r["counts"].decode("utf-8")
    return r


def mk_graph(cands):
    return build_graph(1, "plane", cands, image_size=(64, 64))


def test_text_instance():
    g = mk_graph([
        {"id": "hi", "score": 0.95, "rle": rle(0, 0, 20, 20)},
        {"id": "lo", "score": 0.10, "rle": rle(30, 30, 10, 10)},
    ])
    p = MockProvider()
    p.set_affinity("plane", "hi", 0.9)   # planted cos
    p.set_affinity("plane", "lo", -0.5)  # planted cos
    cos = {"cand:hi": 0.9, "cand:lo": -0.5}
    E.build_text_instance_edges(g, cos)
    by = {e.dst: e for e in g.edges if e.kind is EdgeKind.TEXT_INSTANCE}
    assert set(by) == {"cand:hi", "cand:lo"}
    # direction: category -> candidate
    assert all(e.src == "category" for e in by.values())
    # weight = 0.5*norm + 0.5*cos01
    hi = by["cand:hi"]
    expect_hi = 0.5 * (0.95 - 0.05) / 0.95 + 0.5 * (0.9 + 1) / 2
    assert abs(hi.weight - expect_hi) < 1e-3, (hi.weight, expect_hi)
    lo = by["cand:lo"]
    expect_lo = 0.5 * (0.10 - 0.05) / 0.95 + 0.5 * ((-0.5 + 1) / 2)
    assert abs(lo.weight - expect_lo) < 1e-3
    assert hi.weight > lo.weight and 0 <= lo.weight <= 1
    print("text_instance OK")


def test_support():
    g = mk_graph([
        {"id": "a", "score": 0.9, "rle": rle(0, 0, 20, 20)},
        {"id": "b", "score": 0.9, "rle": rle(0, 0, 22, 22)},
    ])
    # scale ratio (22*22)/(20*20) = 1.21 -> scale arm weight = 1 - 1.21/3 = 0.597
    p = MockProvider()
    p.set_affinity("__x__", "__x__", 0.0)
    cos_fn = lambda a, b: 0.85  # appearance above theta_s=0.70
    E.build_support_edges(g, cos_fn)
    sup = [e for e in g.edges if e.kind is EdgeKind.SAME_CLASS_SUPPORT]
    assert len(sup) == 1
    expect = max(0.597, (0.85 + 1) / 2)  # appearance arm wins: 0.925
    assert abs(sup[0].weight - expect) < 1e-2, sup[0].weight
    assert sup[0].attrs["appearance_cos"] == 0.85

    # below theta_s: appearance must NOT fire, scale arm stays
    g2 = mk_graph([
        {"id": "a", "score": 0.9, "rle": rle(0, 0, 20, 20)},
        {"id": "b", "score": 0.9, "rle": rle(0, 0, 22, 22)},
    ])
    E.build_support_edges(g2, lambda a, b: 0.5)
    sup2 = [e for e in g2.edges if e.kind is EdgeKind.SAME_CLASS_SUPPORT]
    assert len(sup2) == 1 and "appearance_cos" not in sup2[0].attrs
    assert abs(sup2[0].weight - 0.597) < 1e-2
    print("support OK")


def test_competition_and_unknown():
    # plane candidates (query) vs rival-class "helicopter" candidates
    g = mk_graph([
        {"id": "weak", "score": 0.10, "rle": rle(0, 0, 20, 20)},    # norm 0.053
        {"id": "strong", "score": 0.90, "rle": rle(30, 30, 20, 20)},
    ])
    rivals = ["helicopter"]
    others = {"helicopter": [
        # overlaps weak with IoU 1.0 (rival wins on score)
        {"id": "h1", "score": 0.5, "rle": rle(0, 0, 20, 20)},
        # no overlap with anything
        {"id": "h2", "score": 0.5, "rle": rle(50, 50, 6, 6)},
    ]}
    E.build_competition_edges(g, rivals, others)
    comp = [e for e in g.edges if e.kind is EdgeKind.COMPETING_CLASS]
    # only the overlapping pair passes theta_o=0.30
    assert len(comp) == 1 and comp[0].weight == 1.0
    assert g.nodes["cand:h1@helicopter"].payload["rival_class"] == "helicopter"

    E.build_unknown_edges(g)
    unk = [e for e in g.edges if e.kind is EdgeKind.UNKNOWN_CONFLICT]
    # weak lost the competition AND norm_text=0.0526 < 0.20 -> conflict
    assert len(unk) == 1 and unk[0].dst == "unknown" and unk[0].src == "cand:weak"
    assert abs(unk[0].weight - (1 - 0.0526)) < 1e-2
    # strong: no overlap with rivals, high text -> no conflict edge
    assert all(e.src != "cand:strong" for e in unk)
    print("competition+unknown OK")


def test_context():
    g = mk_graph([
        {"id": "t", "score": 0.95, "rle": rle(20, 20, 10, 10)},
    ])
    prior = {"plane": {"cond": {"ship": 0.8}}}
    # ship candidate overlaps the band of t (band radius ~25px around 10x10)
    evidence = {"cand:t": ["ship"]}
    E.build_context_edges(g, prior, evidence)
    ctx = [e for e in g.edges if e.kind is EdgeKind.INSTANCE_CONTEXT
           and e.dst == "cand:t:ctx" and e.attrs.get("evidence")]
    # structural context edge (weight 0) built by builder; ours carries evidence
    assert len(ctx) == 1
    assert abs(ctx[0].weight - 0.8) < 1e-6 and ctx[0].attrs["prior"] == "matched"

    # unmatched class name -> prior_floor + flagged
    prior2 = {"plane": {"cond": {}}}
    E.build_context_edges(g, prior2, {"cand:t": ["truck"]})
    ctx2 = [e for e in g.edges if e.kind is EdgeKind.INSTANCE_CONTEXT
            and e.attrs.get("evidence") == "truck"]
    assert len(ctx2) == 1
    assert abs(ctx2[0].weight - 0.01) < 1e-6 and ctx2[0].attrs["prior"] == "unmatched"
    print("context OK")


def test_band_evidence_geometry():
    g = mk_graph([
        {"id": "t", "score": 0.95, "rle": rle(20, 20, 10, 10)},
    ])
    others = {
        # inside the band (adjacent to the 10x10 square at 20,20)
        "ship": [{"id": "s1", "score": 0.9, "rle": rle(20, 31, 6, 6)}],
        # far away, outside any band
        "car": [{"id": "c1", "score": 0.9, "rle": rle(60, 60, 3, 3)}],
        # inside band but low confidence (norm 0.05 < 0.5 floor)
        "dam": [{"id": "d1", "score": 0.10, "rle": rle(20, 38, 4, 4)}],
    }
    ev = E.band_evidence(g, others)
    assert ev["cand:t"] == ["ship"], ev
    print("band geometry OK")


def test_scene():
    g = mk_graph([
        {"id": "a", "score": 0.9, "rle": rle(0, 0, 20, 20)},
    ])
    scene_scores = {"airport": 0.3, "harbor": -0.2, "city": 0.1}
    crop_image_cos = {"cand:a": 0.5}
    E.build_scene_edges(g, scene_scores, crop_image_cos)
    scene = g.nodes["scene"]
    top = [s["phrase"] for s in scene.payload["top_scenes"]]
    assert top == ["airport", "city", "harbor"]
    e = [x for x in g.edges if x.kind is EdgeKind.INSTANCE_SCENE
         and x.src == "cand:a" and x.dst == "scene"]
    assert len(e) == 1 and abs(e[0].weight - 0.75) < 1e-6  # cos01(0.5)
    print("scene OK")


def main():
    test_text_instance()
    test_support()
    test_competition_and_unknown()
    test_context()
    test_band_evidence_geometry()
    test_scene()
    print("graph edges tests PASSED")


if __name__ == "__main__":
    main()
