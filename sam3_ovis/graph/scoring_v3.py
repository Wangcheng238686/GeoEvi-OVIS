"""v3.0 T2: minimal same-position relative-evidence scoring (design-v3.0 §3).

Implements the four score variants B0/B1/B2/B3 over the P0 candidate pool:

  B0  b_i  = clip((raw_score_i - 0.05)/0.95, 0, 1)          (SAM3 monotone)
  B1  u_i  = (1-beta) b_i + beta (z_i(y_i)+1)/2              (independent
        visual-semantic scoring; no spatial relations)
  B2  d_i  = max(0, max_{y in C_i} z_i(y) - z_i(y_i) - delta)
      v_i  = u_i * exp(-lambda * d_i)                        (relative penalty)
  B3  same as B2 but rivals IGNORE spatial relation entirely (d_i uses ALL
      other-class candidates regardless of overlap) — tests whether the
      spatial restriction matters.

Constraints (hard acceptance, per implementation-v3.0.md T2):
  - lambda = 0 must reduce to B1 exactly (per candidate);
  - no-rival candidates: v_i == u_i exactly;
  - duplicating a same-class candidate never changes any score;
  - permuting candidate order never changes any score;
  - masks/categories are never modified; GT is never read at scoring time.

Run:  $PYTHON sam3_ovis/graph/scoring_v3.py --help
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ScoringConfig:
    beta: float = 0.5
    delta: float = 0.0
    lam: float = 0.0
    pool_floor: float = 0.05
    rival_iou: float = 0.5          # spatial restriction for B2 (C_i)
    crop_margin: float = 0.1        # candidate crop margin (both domains)

    def variant(self, name: str) -> "ScoringConfig":
        """A3-style component removal, by name."""
        if name == "B0":
            return ScoringConfig(beta=0.0, delta=0.0, lam=0.0)
        if name == "B1":
            return ScoringConfig(beta=self.beta, delta=0.0, lam=0.0)
        if name == "B2":
            return self
        if name == "B3":  # no spatial relation: rivals from all classes
            return ScoringConfig(beta=self.beta, delta=self.delta,
                                 lam=self.lam, rival_iou=0.0)
        raise ValueError(name)


VARIANTS = ["B0", "B1", "B2", "B3"]


def b_score(raw_score: float, cfg: ScoringConfig) -> float:
    return float(np.clip((raw_score - cfg.pool_floor)
                         / (1.0 - cfg.pool_floor), 0.0, 1.0))


def z01(cos: float) -> float:
    return (cos + 1.0) / 2.0


def score_candidates(
    raw_scores: Dict[str, float],
    z_own: Dict[str, float],
    z_rival_max: Dict[str, Optional[float]],
    cfg: ScoringConfig,
    z_rival_max_nospatial: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Return {cand_id: {"B0": b, "B1": u, "B2": v, "B3": v_nospatial}}.

    z_own: RemoteCLIP cos of this crop vs its OWN class text.
    z_rival_max: max over spatially-overlapping rival classes (IoU >= 0.5)
                 of cos(crop, rival text); None = no spatial rivals.
    z_rival_max_nospatial: same but WITHOUT the spatial restriction
                 (B3's C_i includes all classes). Defaults to z_rival_max.

    Hard acceptance (implementation-v3.0.md T2): lambda=0 => B2 == B1
    per candidate; no-rival candidates => B2 == B1 == u.
    """
    ns = (z_rival_max_nospatial if z_rival_max_nospatial is not None
          else z_rival_max)
    out: Dict[str, Dict[str, float]] = {}
    for cid, raw in raw_scores.items():
        b = b_score(raw, cfg)
        zc = z01(z_own[cid])
        u = (1.0 - cfg.beta) * b + cfg.beta * zc

        def v_of(zmax: Optional[float], lam: float) -> float:
            if zmax is None:
                return u   # no rivals: v == u exactly
            d = max(0.0, zmax - z_own[cid] - cfg.delta)
            return min(1.0, max(0.0, u * float(np.exp(-lam * d))))

        out[cid] = {
            "B0": round(b, 6),
            "B1": round(u, 6),
            "B2": round(v_of(z_rival_max.get(cid), cfg.lam), 6),
            "B3": round(v_of(ns.get(cid), cfg.lam), 6),
        }
    return out
