"""Edge construction for the evidence graph (S3.2).

Six edge families per docs/graph-thresholds.md (registered 2026-09-03).
Every function takes an EvidenceGraph that already has the structural layer
from builder.build_graph (category/scene/unknown + candidates + contexts)
and ADDS its own edges; weights are plain floats in [0,1].

CLIP-dependent edges are driven by a provider (RemoteCLIPProvider in
production, MockProvider in tests). Geometry-only inputs come from the
graph payloads. Nothing here generates instances; context/scene are soft
scores only (red line 5).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from graph.config import SCENE_PHRASES, THRESHOLDS, cos01, norm_text
from graph.schema import Edge, EdgeKind, EvidenceGraph, Node, NodeKind

try:  # opencv-headless is a runtime dep for the ring band
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

Callable_2 = Callable[[str, str], float]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rle_arr(rle: dict) -> np.ndarray:
    from pycocotools import mask as mask_util

    r = dict(rle)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode("utf-8")
    return mask_util.decode(r).astype(np.uint8)


# Per-image decode cache: pool items reuse the SAME rle dict objects across
# class-graphs of one image, so id()-keyed caching removes the repeated
# decodes that dominate ablation runtime. Cleared between images by
# clear_rle_cache() (caller responsibility).
_ARR_CACHE: Dict[int, tuple] = {}


def clear_rle_cache() -> None:
    _ARR_CACHE.clear()


def _cached_arr_bbox(rle: dict):
    key = id(rle)
    hit = _ARR_CACHE.get(key)
    if hit is None:
        from pycocotools import mask as mask_util

        r = dict(rle)
        if isinstance(r["counts"], str):
            r["counts"] = r["counts"].encode("utf-8")
        arr = mask_util.decode(r)
        bx = mask_util.toBbox(r)
        hit = (arr, [float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])])
        _ARR_CACHE[key] = hit
    return hit


def _edge(g: EvidenceGraph, src: str, dst: str, kind: EdgeKind,
          weight: float, **attrs) -> Optional[Edge]:
    if src not in g.nodes or dst not in g.nodes:
        return None
    e = Edge(src=src, dst=dst, kind=kind, weight=round(min(1.0, max(0.0, weight)), 6),
             attrs=attrs)
    g.edges.append(e)
    return e


def ring_band(mask: np.ndarray, area_px: float) -> np.ndarray:
    """Ring band = dilate(mask, r) minus mask; r = BAND_RATIO * sqrt(area)."""
    r = max(3, int(round(THRESHOLDS.band_ratio * np.sqrt(max(area_px, 1.0)))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dil = cv2.dilate(mask.astype(np.uint8), kernel)
    return (dil.astype(bool) & ~mask.astype(bool))


# ---------------------------------------------------------------------------
# §2 text-instance
# ---------------------------------------------------------------------------

def build_text_instance_edges(
    g: EvidenceGraph,
    crop_text_cos: Dict[str, float],
) -> List[Edge]:
    """weight = W_TEXT_SAM3 * norm_text + W_TEXT_CLIP * cos01(clip_cos)."""
    th = THRESHOLDS
    out = []
    for cand in g.by_kind(NodeKind.CANDIDATE):
        cid = cand.id
        nt = norm_text(cand.payload["score"])
        cc = crop_text_cos.get(cid)
        if cc is None:
            continue
        w = th.w_text_sam3 * nt + th.w_text_clip * cos01(float(cc))
        e = _edge(g, "category", cid, EdgeKind.TEXT_INSTANCE, w,
                  norm_text=round(nt, 6), clip_cos=round(float(cc), 6))
        if e is not None:
            out.append(e)
    return out


# ---------------------------------------------------------------------------
# §3 same-class support — appearance arm; scale arm comes from the builder
# ---------------------------------------------------------------------------

def build_support_edges(
    g: EvidenceGraph,
    crop_crop_cos: Callable_2 = None,
) -> List[Edge]:
    """Upgrade SAME_CLASS_SUPPORT edges with the appearance arm:
    final weight = max(scale_arm, cos01(cos) if cos >= theta_s else 0)."""
    th = THRESHOLDS
    out = []
    for e in g.edges:
        if e.kind is not EdgeKind.SAME_CLASS_SUPPORT:
            continue
        cos = crop_crop_cos(e.src, e.dst) if crop_crop_cos else None
        if cos is not None and cos >= th.theta_s:
            e.weight = round(max(e.weight, cos01(cos)), 6)
            e.attrs["appearance_cos"] = round(float(cos), 6)
            out.append(e)
    return out


# ---------------------------------------------------------------------------
# §4 competing class
# ---------------------------------------------------------------------------

def competing_classes(class_name: str, text_sim: Dict[str, float]) -> List[str]:
    """Class-name CLIP similarity top-K with floor theta_c (excl. self)."""
    th = THRESHOLDS
    ranked = sorted(
        ((c, s) for c, s in text_sim.items()
         if c != class_name and s >= th.theta_c),
        key=lambda x: -x[1])
    return [c for c, _ in ranked[: th.compete_topk]]


def build_competition_edges(
    g: EvidenceGraph,
    rivals: Sequence[str],
    other_candidates: Dict[str, List[dict]],
) -> List[Edge]:
    """Cross-class candidate pairs with IoU >= theta_o get weight=IoU.
    Rival candidates are added as CANDIDATE nodes (id `cand:<id>@<class>`)
    so the graph stays closed under its edges. Pairwise IoU uses the
    per-image decode cache + bbox prefilter."""
    th = THRESHOLDS
    out = []
    added: Dict[str, str] = {}
    for cand in g.by_kind(NodeKind.CANDIDATE):
        m, mb = _cached_arr_bbox(cand.payload["rle"])
        m = m.astype(bool)
        for rival in rivals:
            for oc in other_candidates.get(rival, []):
                rid = f"cand:{oc['id']}@{rival}"
                if rid not in added:
                    if rid not in g.nodes:
                        om, ob = _cached_arr_bbox(oc["rle"])
                        g.add_node(Node(
                            id=rid, kind=NodeKind.CANDIDATE,
                            payload={"score": float(oc["score"]),
                                     "rle": oc["rle"],
                                     "area_px": float(om.sum()),
                                     "bbox": ob,
                                     "rival_class": rival}))
                    added[rid] = rival
                om, ob = _cached_arr_bbox(oc["rle"])
                # bbox prefilter
                if (mb[0] + mb[2] <= ob[0] or ob[0] + ob[2] <= mb[0]
                        or mb[1] + mb[3] <= ob[1]
                        or ob[1] + ob[3] <= mb[1]):
                    continue
                inter = int((m & om).sum())
                if inter == 0:
                    continue
                iou = inter / float((m | om).sum())
                if iou >= th.theta_o:
                    e = _edge(g, cand.id, rid, EdgeKind.COMPETING_CLASS,
                              iou, rival_class=rival)
                    if e is not None:
                        out.append(e)
    return out


# ---------------------------------------------------------------------------
# §5 instance-local context
# ---------------------------------------------------------------------------

def build_context_edges(
    g: EvidenceGraph,
    prior_row: Optional[Dict],
    prior_vocab: set,
    evidence_by_class: Dict[str, List[str]],
) -> List[Edge]:
    """Ring-band soft co-occurrence.

    prior_row: the query class's row ({n_images, cond}) from a priors table;
    cond already covers the full vocabulary (j != i, Laplace-smoothed), so
    an in-vocabulary never-co-occurring class gets its smoothed value —
    `prior_floor` is only a lower clamp. "unmatched" (floor + flag) is
    reserved for evidence classes outside the prior vocabulary. Evidence of
    the query class itself is skipped: same-class proximity is
    SAME_CLASS_SUPPORT's job, and P(i|i) is undefined under j != i smoothing.
    """
    th = THRESHOLDS
    query_class = g.class_name
    out = []
    for cand in g.by_kind(NodeKind.CANDIDATE):
        best_class, best_w, matched = None, 0.0, False
        for ev_class in evidence_by_class.get(cand.id, []):
            if ev_class == query_class:
                continue
            in_vocab = ev_class in prior_vocab
            w = prior_row["cond"].get(ev_class, 0.0) \
                if (prior_row is not None and in_vocab) else 0.0
            if in_vocab and w > best_w:
                best_class, best_w, matched = ev_class, w, True
            elif not in_vocab and best_class is None:
                best_class, best_w = ev_class, th.prior_floor
        if best_class is None:
            continue
        w = max(th.prior_floor, best_w)
        e = _edge(g, cand.id, f"{cand.id}:ctx", EdgeKind.INSTANCE_CONTEXT,
                  w, evidence=best_class, prior_value=round(best_w, 6),
                  prior="matched" if matched else "unmatched")
        if e is not None:
            out.append(e)
    return out


def band_evidence(
    g: EvidenceGraph,
    all_candidates_by_class: Dict[str, List[dict]],
) -> Dict[str, List[str]]:
    """Map each candidate id -> classes with a high-conf candidate whose mask
    overlaps the candidate's ring band."""
    th = THRESHOLDS
    own = {c.id: c for c in g.by_kind(NodeKind.CANDIDATE)}
    out: Dict[str, List[str]] = {}
    for cand in own.values():
        m, _ = _cached_arr_bbox(cand.payload["rle"])
        m = m.astype(bool)
        band = ring_band(m, cand.payload["area_px"])
        ev = []
        for cls, cands in all_candidates_by_class.items():
            for oc in cands:
                nt = norm_text(oc["score"])
                if nt < th.context_conf_floor:
                    continue
                om, _ = _cached_arr_bbox(oc["rle"])
                if (band & om.astype(bool)).sum() > 0:
                    ev.append(cls)
                    break
        if ev:
            out[cand.id] = ev
    return out


# ---------------------------------------------------------------------------
# §6 instance-scene + scene node payload
# ---------------------------------------------------------------------------

def build_scene_edges(
    g: EvidenceGraph,
    scene_scores: Dict[str, float],
    crop_image_cos: Dict[str, float],
) -> List[Edge]:
    scene = g.nodes.get("scene")
    if scene is None:
        return []
    top = sorted(scene_scores.items(), key=lambda x: -x[1])[:3]
    scene.payload["top_scenes"] = [
        {"phrase": p, "score01": round(cos01(s), 6)} for p, s in top]
    out = []
    existing = {e.src: e for e in g.edges
                if e.kind is EdgeKind.INSTANCE_SCENE and e.dst == "scene"}
    for cand in g.by_kind(NodeKind.CANDIDATE):
        c = crop_image_cos.get(cand.id)
        if c is None:
            continue
        w = cos01(float(c))
        e = existing.get(cand.id)
        if e is not None:
            # builder v0 coverage arm becomes an attr (registered doc §8);
            # the registered weight is the scene-representativeness cosine
            e.attrs.setdefault("coverage_area_share", e.weight)
            e.weight = round(w, 6)
            e.attrs["crop_image_cos"] = round(float(c), 6)
            out.append(e)
        else:
            e = _edge(g, cand.id, "scene", EdgeKind.INSTANCE_SCENE, w,
                      crop_image_cos=round(float(c), 6))
            if e is not None:
                out.append(e)
    return out


# ---------------------------------------------------------------------------
# §7 unknown conflict
# ---------------------------------------------------------------------------

def build_unknown_edges(g: EvidenceGraph) -> List[Edge]:
    """lost competition + norm_text < theta_u_text -> cand->unknown edge,
    weight = 1 - norm_text."""
    th = THRESHOLDS
    own = list(g.by_kind(NodeKind.CANDIDATE))
    if not own:
        return []
    out = []
    for cand in own:
        nt = norm_text(cand.payload["score"])
        if nt >= th.theta_u_text:
            continue
        lost = False
        for e in g.edges:
            if e.kind is not EdgeKind.COMPETING_CLASS:
                continue
            if e.src == cand.id or e.dst == cand.id:
                other_id = e.dst if e.src == cand.id else e.src
                other = g.nodes.get(other_id)
                if other is None:
                    continue
                if other.kind is NodeKind.CANDIDATE and \
                        norm_text(other.payload["score"]) > nt:
                    lost = True
                    break
        if lost:
            e = _edge(g, cand.id, "unknown", EdgeKind.UNKNOWN_CONFLICT,
                      1.0 - nt, reason="low_text+lost_competition")
            if e is not None:
                out.append(e)
    return out
