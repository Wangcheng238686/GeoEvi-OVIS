"""Completeness metrics (S3.3, docs/graph-thresholds.md §10).

Per (image, query class) graph-level statistics over the registered
candidate pool: raw SAM3 score >= pool_floor (0.05, the P0-registered
pool — enforced here explicitly, not trusted to upstream), rival nodes
excluded:

  ECR  explained coverage rate      - area of high-confidence union over
                                      all-pool union (area base);
  URD  unexplained response density - pool candidates not matched by any
                                      high-confidence candidate, over the
                                      pool (count base);
  CCR  conflict candidate rate      - candidates touching a COMPETING_CLASS
                                      edge or duplicated (same-class
                                      higher-score overlap at IoU >= 0.5),
                                      over the pool;
  R    incompleteness risk score    - equal thirds of (1-ECR), URD, CCR.

The high-confidence floor tau_high lives on the norm_text scale (0.5,
reusing the registered S3.2 context_conf_floor); the pool floor lives on
the raw score scale (0.05, P0 registration) — the two scales are distinct
and both are stated in graph-thresholds.md §10.

R modulates final confidence and alarms only (S3.4) — it never creates or
removes instances. Known blind spot (registered): a class with zero
responses scores ECR=1 / R=0; completeness does not detect total misses.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from graph.config import COMPLETENESS, THRESHOLDS, norm_text
from graph.schema import EdgeKind, NodeKind, EvidenceGraph


def _mask_arr(rle: dict) -> np.ndarray:
    from pycocotools import mask as mask_util

    r = dict(rle)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode("utf-8")
    return mask_util.decode(r)


def compute_completeness(g: EvidenceGraph) -> Dict:
    """Compute ECR / URD / CCR / risk over the registered pool and store the
    report in g.attrs['completeness']. Returns the report."""
    th = COMPLETENESS
    own = [c for c in g.by_kind(NodeKind.CANDIDATE)
           if not c.payload.get("rival_class")]
    pool = [c for c in own
            if float(c.payload.get("score", 0.0)) >= THRESHOLDS.pool_floor]
    n_excluded = len(own) - len(pool)
    n_all = len(pool)

    masks, nt = {}, {}
    for c in pool:
        masks[c.id] = _mask_arr(c.payload["rle"]).astype(bool)
        nt[c.id] = norm_text(c.payload["score"])

    high = [c.id for c in pool if nt[c.id] >= th.tau_high]
    low = [c.id for c in pool if nt[c.id] < th.tau_high]

    # --- ECR (area base) -------------------------------------------------
    all_union = np.zeros_like(next(iter(masks.values()))) if masks else None
    high_union = np.zeros_like(all_union) if masks else None
    for cid, m in masks.items():
        all_union |= m
        if cid in set(high):
            high_union |= m
    if all_union is None or all_union.sum() == 0:
        ecr = 1.0  # registered: zero responses -> vacuously explained
    else:
        ecr = float(high_union.sum()) / float(all_union.sum())

    # --- URD (count base) ------------------------------------------------
    unexplained = 0
    for lid in low:
        matched = False
        for hid in high:
            inter = int((masks[lid] & masks[hid]).sum())
            if inter == 0:
                continue
            iou = inter / float((masks[lid] | masks[hid]).sum())
            if iou >= th.dup_iou:
                matched = True
                break
        if not matched:
            unexplained += 1
    urd = unexplained / n_all if n_all else 0.0

    # --- CCR -------------------------------------------------------------
    in_competition = set()
    for e in g.edges:
        if e.kind is not EdgeKind.COMPETING_CLASS:
            continue
        for ref in (e.src, e.dst):
            node = g.nodes.get(ref)
            if node is not None and node.kind is NodeKind.CANDIDATE \
                    and not node.payload.get("rival_class"):
                in_competition.add(ref)

    order = sorted(pool, key=lambda c: -c.payload["score"])
    duplicated = set()
    for i, ci in enumerate(order):
        for cj in order[:i]:
            if cj.id in duplicated:
                pass  # cj already flagged; still test ci against it
            inter = int((masks[ci.id] & masks[cj.id]).sum())
            if inter == 0:
                continue
            iou = inter / float((masks[ci.id] | masks[cj.id]).sum())
            if iou >= th.dup_iou:
                duplicated.add(ci.id)  # lower-score side is the duplicate
                break
    ccr = (len(in_competition) + len(duplicated - in_competition)) / n_all \
        if n_all else 0.0

    risk = (th.w_unexplained_area * (1.0 - ecr)
            + th.w_unexplained_response * urd
            + th.w_conflict * ccr)

    report = {
        "explained_coverage_rate": round(ecr, 6),
        "unexplained_response_density": round(urd, 6),
        "conflict_candidate_rate": round(ccr, 6),
        "risk_score": round(min(1.0, max(0.0, risk)), 6),
        "n_candidates": n_all,
        "n_pool_excluded": n_excluded,
        "n_high_confidence": len(high),
        "n_low_unexplained": unexplained,
        "n_conflict": len(in_competition) + len(duplicated - in_competition),
    }
    g.attrs["completeness"] = report
    return report
