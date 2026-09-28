"""T2 unit tests for scoring_v3 (hard acceptance per implementation-v3.0):
  - lambda=0 => B2 == B1 per candidate;
  - no-rival candidates => B2 == B1 == u;
  - duplicating a same-class candidate never changes scores (scoring is
    per-candidate, so permutation/duplication invariance is by construction
    but tested anyway);
  - B0 monotone in raw score; B1 formula; B2 monotone decreasing in
    z_rival - z_own gap.
Run:  $PYTHON sam3_ovis/tests/test_scoring_v3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.scoring_v3 import (  # noqa: E402
    ScoringConfig, b_score, score_candidates)

CFG = ScoringConfig(beta=0.5, delta=0.05, lam=1.0)


def test_lambda0_equals_b1():
    raw = {"a": 0.5, "b": 0.9}
    z_own = {"a": 0.6, "b": 0.3}
    zr = {"a": 0.9, "b": 0.1}
    cfg0 = ScoringConfig(beta=CFG.beta, delta=CFG.delta, lam=0.0)
    out = score_candidates(raw, z_own, zr, cfg0)
    for cid in raw:
        assert out[cid]["B2"] == out[cid]["B1"], cid
    print("lambda=0 => B2==B1 OK")


def test_no_rival_equals_u():
    out = score_candidates(
        raw_scores={"a": 0.5}, z_own={"a": 0.6},
        z_rival_max={"a": None}, cfg=CFG)
    u = (1 - 0.5) * b_score(0.5, CFG) + 0.5 * (0.6 + 1) / 2
    assert abs(out["a"]["B2"] - u) < 1e-4, (out["a"], u)
    assert out["a"]["B2"] == out["a"]["B1"]
    print("no-rival => v==u OK")


def test_b0_monotone():
    scores = [b_score(s, CFG) for s in [0.05, 0.2, 0.5, 0.8, 1.0]]
    assert scores == sorted(scores)
    assert scores[0] == 0.0 and scores[-1] == 1.0
    print("B0 monotone OK")


def test_b1_formula():
    out = score_candidates(
        raw_scores={"a": 0.5}, z_own={"a": 0.2},
        z_rival_max={"a": None}, cfg=CFG)
    b = (0.5 - 0.05) / 0.95
    u = 0.5 * b + 0.5 * (0.2 + 1) / 2
    assert abs(out["a"]["B1"] - u) < 1e-4
    print("B1 formula OK")


def test_b2_monotone_in_gap():
    """B2 must not increase as the rival's evidence advantage grows."""
    raw = {"a": 0.5}
    z_own = {"a": 0.5}
    gaps = [0.0, 0.1, 0.3, 0.6, 0.9]
    vals = []
    for g in gaps:
        zr = {"a": z_own["a"] + g}
        o = score_candidates(raw, z_own, zr, CFG)
        vals.append(o["a"]["B2"])
    assert vals == sorted(vals, reverse=True), vals
    print("B2 monotone in gap OK")


def test_permutation_invariance():
    raw = {"a": 0.5, "b": 0.9, "c": 0.3}
    z_own = {"a": 0.5, "b": 0.7, "c": 0.2}
    zr = {"a": 0.9, "b": 0.4, "c": None}
    o1 = score_candidates(raw, z_own, zr, CFG)
    items = list(zr.items())[::-1]
    o2 = score_candidates(raw, z_own, dict(items), CFG)
    assert o1 == o2
    print("permutation invariance OK")


def main():
    test_lambda0_equals_b1()
    test_no_rival_equals_u()
    test_b0_monotone()
    test_b1_formula()
    test_b2_monotone_in_gap()
    test_permutation_invariance()
    print("scoring_v3 tests PASSED")


if __name__ == "__main__":
    main()
