"""Message passing, duplicate merging, and final ranking (S3.4, §11).

Two rounds of hand-weighted evidence propagation over the evidence graph
produce a per-candidate reliability; support-edge connected components
(restricted to overlapping pairs) form duplicate clusters whose members'
reliabilities combine as a probabilistic OR — the "NMS upgrade": duplicates
reinforce instead of being discarded. The completeness risk R modulates the
final ranking score and raises the incompleteness alarm. No new instances
are ever created here (red line).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from graph.config import MP, THRESHOLDS, norm_text
from graph.schema import EdgeKind, NodeKind, EvidenceGraph


def _mask_arr(rle: dict) -> np.ndarray:
    from pycocotools import mask as mask_util

    r = dict(rle)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode("utf-8")
    return mask_util.decode(r)


def _own_candidates(g: EvidenceGraph):
    return [c for c in g.by_kind(NodeKind.CANDIDATE)
            if not c.payload.get("rival_class")
            and float(c.payload.get("score", 0.0)) >= THRESHOLDS.pool_floor]


def _pair_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    if inter == 0:
        return 0.0
    return inter / float((a | b).sum())


def propagate_reliability(g: EvidenceGraph) -> Dict[str, float]:
    """Two rounds of hand-weighted message passing -> per-candidate
    reliability in [0,1]. Rival candidates participate as message sources
    but receive no reliability of their own."""
    rel: Dict[str, float] = {
        c.id: norm_text(c.payload["score"]) for c in _own_candidates(g)}

    # message channels, precomputed per candidate
    support_msgs: Dict[str, List[Tuple[str, float]]] = {}
    compete_msgs: Dict[str, List[Tuple[str, float]]] = {}
    context_msgs: Dict[str, float] = {}
    scene_msgs: Dict[str, float] = {}
    own_ids = set(rel)
    # rivals are message sources with a STATIC value (their norm_text score);
    # they never receive reliability themselves
    rival_base = {n.id: norm_text(n.payload["score"])
                  for n in g.by_kind(NodeKind.CANDIDATE)
                  if n.payload.get("rival_class")}

    def src(n: str) -> float:
        return rel.get(n, rival_base.get(n, 0.0))

    for e in g.edges:
        if e.kind is EdgeKind.SAME_CLASS_SUPPORT:
            if e.src in own_ids and e.dst in own_ids:
                support_msgs.setdefault(e.src, []).append((e.dst, e.weight))
                support_msgs.setdefault(e.dst, []).append((e.src, e.weight))
        elif e.kind is EdgeKind.COMPETING_CLASS:
            if e.src in own_ids:
                compete_msgs.setdefault(e.src, []).append((e.dst, e.weight))
            if e.dst in own_ids:
                compete_msgs.setdefault(e.dst, []).append((e.src, e.weight))
        elif e.kind is EdgeKind.INSTANCE_CONTEXT:
            if e.src in own_ids and e.weight > 0:
                context_msgs[e.src] = max(context_msgs.get(e.src, 0.0), e.weight)
        elif e.kind is EdgeKind.INSTANCE_SCENE:
            if e.src in own_ids:
                scene_msgs[e.src] = max(scene_msgs.get(e.src, 0.0), e.weight)

    for _ in range(MP.rounds):
        new_rel = dict(rel)
        for cid in own_ids:
            r = rel[cid]
            sup = max((src(n) * w for n, w in support_msgs.get(cid, [])),
                      default=0.0)
            comp = max((src(n) * w for n, w in compete_msgs.get(cid, [])),
                       default=0.0)
            ctx = context_msgs.get(cid, 0.0)
            scn = scene_msgs.get(cid, 0.0)
            v = (r
                 + MP.alpha_support * sup * (1.0 - r)
                 - MP.alpha_compete * comp * r
                 + MP.alpha_context * ctx * (1.0 - r)
                 + MP.alpha_scene * scn * (1.0 - r))
            new_rel[cid] = min(1.0, max(0.0, v))
        rel = new_rel
    return rel


def merge_duplicate_clusters(
    g: EvidenceGraph,
    rel: Dict[str, float],
) -> List[dict]:
    """Support-edge connected components restricted to overlapping pairs
    (IoU >= merge_iou) form duplicate clusters. Per cluster: representative
    mask = highest-rel member; merged score = 1 - prod(1 - rel_i)
    (probabilistic OR — duplicates reinforce). Returns merged detections."""

    cands = _own_candidates(g)
    if not cands:
        return []
    masks = {c.id: _mask_arr(c.payload["rle"]).astype(bool) for c in cands}

    # union-find over overlapping support pairs
    parent = {c.id: c.id for c in cands}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    support_pairs = [(e.src, e.dst) for e in g.edges
                     if e.kind is EdgeKind.SAME_CLASS_SUPPORT
                     and e.src in rel and e.dst in rel]
    for a, b in support_pairs:
        if _pair_iou(masks[a], masks[b]) >= MP.merge_iou:
            union(a, b)

    clusters: Dict[str, List[str]] = {}
    for cid in rel:
        clusters.setdefault(find(cid), []).append(cid)

    merged = []
    for members in clusters.values():
        members.sort(key=lambda m: -rel[m])
        rep = members[0]
        score = 1.0 - float(np.prod([1.0 - rel[m] for m in members]))
        merged.append({
            "cand_id": rep,
            "cluster_members": members,
            "reliability": round(rel[rep], 6),
            "merged_score": round(min(1.0, score), 6),
            "mask_rle": g.nodes[rep].payload["rle"],
            "bbox": g.nodes[rep].payload.get("bbox"),
        })
    merged.sort(key=lambda d: -d["merged_score"])
    return merged


def rank_candidates(
    g: EvidenceGraph,
    merged: List[dict],
    rel: Dict[str, float],
) -> Tuple[List[dict], Dict]:
    """Completeness modulation + alarm. final = rel · (1 - lambda·R)."""
    report = g.attrs.get("completeness") or compute_fallback(g)
    risk = float(report.get("risk_score", 0.0))
    factor = 1.0 - MP.lambda_risk * risk
    alarm = risk >= MP.theta_alarm
    out = []
    for d in merged:
        d = dict(d)
        d["final_score"] = round(d["merged_score"] * factor, 6)
        d["risk_modulation"] = round(factor, 6)
        out.append(d)
    meta = {"risk_score": round(risk, 6),
            "incompleteness_alarm": bool(alarm),
            "n_clusters": len(merged),
            "n_pool_candidates": len(rel)}
    return out, meta


def compute_fallback(g: EvidenceGraph) -> Dict:
    """Import-time-avoiding fallback so rank_candidates can run standalone."""
    from graph.completeness import compute_completeness

    return compute_completeness(g)


def run_graph_inference(g: EvidenceGraph) -> Tuple[List[dict], Dict]:
    """End-to-end: reliability -> merge -> risk-modulated ranking."""
    from graph.completeness import compute_completeness

    compute_completeness(g)
    rel = propagate_reliability(g)
    merged = merge_duplicate_clusters(g, rel)
    return rank_candidates(g, merged, rel)
