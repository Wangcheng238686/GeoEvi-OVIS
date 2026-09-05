"""Evidence-graph builder (S3.1): instantiate design-v2 §5.1 for one
(image, query class) pair from SAM3 candidates.

Structural v0 (this module):
- 1 CATEGORY node (query phrase), 1 SCENE node, 1 UNKNOWN node;
- 1 CANDIDATE node per candidate (score / bbox / RLE in payload);
- 1 LOCAL_CONTEXT node per candidate (ring-band placeholder payload: the
  band geometry comes from the candidate mask; band feature extraction is
  S3.2's EmbeddingProvider, not built here);
- geometry-only edges computable without any learned feature:
  * INSTANCE_SCENE with weight = candidate area / image area (区域覆盖 proxy,
    stored per candidate; union coverage RLE in graph.attrs["coverage_rle"]);
  * SAME_CLASS_SUPPORT with weight = scale-ratio proximity between two
    candidates (design §5.2 "尺度比 < k" arm only; the appearance arm needs
    S3.2's embedding provider and is added there);
  * UNKNOWN_CONFLICT placeholder edges are NOT created here — conflict
    detection needs S3.2 thresholds and is registered then.

Everything CLIP-dependent (text-instance, appearance support, competing
class, context/scene semantics) is added in S3.2 behind EmbeddingProvider;
thresholds get registered at S3.2 start and are not set in this file.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from graph.config import THRESHOLDS
from graph.schema import (Edge, EdgeKind, EvidenceGraph, Node, NodeKind)

def _from_rle(rle: dict):
    from pycocotools import mask as mask_util

    r = dict(rle)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode("utf-8")
    return r


def candidate_geometry(rle: dict) -> Dict[str, float]:
    m = mask_util_decode(rle)
    return {"area_px": float(m.sum())}


def mask_util_decode(rle: dict):
    from pycocotools import mask as mask_util

    return mask_util.decode(_from_rle(rle))


def iou_rles(rle_a: dict, rle_b: dict) -> float:
    from pycocotools import mask as mask_util

    a, b = _from_rle(rle_a), _from_rle(rle_b)
    return float(mask_util.iou([a], [b], [0])[0][0])


def build_graph(
    image_id: int,
    class_name: str,
    candidates: List[Dict],
    image_size: Optional[tuple] = None,
) -> EvidenceGraph:
    """candidates: [{id, score, rle, bbox?}] with COCO RLE (utf8 counts)."""
    g = EvidenceGraph(image_id=image_id, class_name=class_name)

    cat = Node(id="category", kind=NodeKind.CATEGORY,
               payload={"phrase": class_name})
    scene = Node(id="scene", kind=NodeKind.SCENE, payload={})
    unknown = Node(id="unknown", kind=NodeKind.UNKNOWN, payload={})
    for n in (cat, scene, unknown):
        g.add_node(n)

    coverage = None
    img_area = float(image_size[0] * image_size[1]) if image_size else None

    for cand in candidates:
        cid = f"cand:{cand['id']}"
        geo = candidate_geometry(cand["rle"])
        g.add_node(Node(id=cid, kind=NodeKind.CANDIDATE, payload={
            "score": float(cand["score"]),
            "rle": cand["rle"],
            "bbox": cand.get("bbox"),
            **geo,
        }))
        g.add_node(Node(id=f"{cid}:ctx", kind=NodeKind.LOCAL_CONTEXT,
                        payload={"candidate": cid, "band": "ring"}))
        g.add_edge(Edge(src=cid, dst=f"{cid}:ctx",
                        kind=EdgeKind.INSTANCE_CONTEXT, weight=0.0,
                        attrs={"band_only": True}))
        w_cov = (geo["area_px"] / img_area) if img_area else 0.0
        g.add_edge(Edge(src=cid, dst="scene", kind=EdgeKind.INSTANCE_SCENE,
                        weight=round(min(1.0, w_cov * 10.0), 6),
                        attrs={"area_share": w_cov}))
        # pairwise same-class support (scale arm; SCALE_K is a registered
        # threshold — single source: graph/config.py)
        for other in list(g.by_kind(NodeKind.CANDIDATE)):
            if other.id == cid:
                continue
            a1 = g.nodes[cid].payload["area_px"]
            a2 = other.payload["area_px"]
            ratio = max(a1, a2) / max(min(a1, a2), 1.0)
            w = max(0.0, 1.0 - ratio / THRESHOLDS.scale_k)
            g.add_edge(Edge(src=cid, dst=other.id,
                            kind=EdgeKind.SAME_CLASS_SUPPORT,
                            weight=round(w, 6),
                            attrs={"scale_ratio": round(ratio, 4)}))
        # union coverage mask
        m = mask_util_decode(cand["rle"]).astype(bool)
        coverage = m if coverage is None else (coverage | m)

    if coverage is not None and image_size:
        from pycocotools import mask as mask_util

        rle = mask_util.encode(
            np_asfortran(coverage.astype("uint8")))
        g.attrs["coverage_rle"] = {
            "size": list(rle["size"]),
            "counts": rle["counts"].decode("utf-8"),
        }
        g.attrs["coverage_ratio"] = round(
            float(coverage.sum()) / img_area, 6)
    return g


def np_asfortran(arr):
    import numpy as np

    return np.asfortranarray(arr)


# ---------------------------------------------------------------------------
# full build (S3.2): structural layer + all six edge families
# ---------------------------------------------------------------------------

def crop_pil(image, bbox, margin=0.1):
    """PIL crop with a small relative margin; degenerate bboxes are clamped
    to >= 1 pixel inside the image (RLE toBbox can yield w/h = 0)."""
    if bbox is None:
        return image
    x, y, w, h = bbox
    mx, my = w * margin, h * margin
    x1 = min(max(0, int(x - mx)), image.width - 1)
    y1 = min(max(0, int(y - my)), image.height - 1)
    x2 = min(image.width, max(x1 + 1, int(x + w + mx)))
    y2 = min(image.height, max(y1 + 1, int(y + h + my)))
    return image.crop((x1, y1, x2, y2))


def build_full_graph(
    image_id: int,
    class_name: str,
    candidates: List[dict],
    image=None,
    image_size: Optional[tuple] = None,
    other_candidates: Optional[Dict[str, List[dict]]] = None,
    text_sim: Optional[Dict[str, float]] = None,
    prior: Optional[Dict] = None,
    provider=None,
    scene_phrases: Optional[Sequence[str]] = None,
    feature_prefix: str = "",
):
    """Full per-(image, class) graph: structural v0 + six edge families.

    provider: object with encode_images_cached / encode_texts_cached /
    crop_text_cos-like helpers; None -> only geometry/text-free edges.
    feature_prefix scopes provider cache keys to THIS image (candidate ids
    repeat across images; without a prefix the provider cache would serve
    another image's crop features).
    """
    import numpy as np

    from graph.config import SCENE_PHRASES, THRESHOLDS, norm_text
    from graph.edges import (band_evidence, build_competition_edges,
                             build_context_edges, build_scene_edges,
                             build_support_edges, build_text_instance_edges,
                             build_unknown_edges)

    g = build_graph(image_id, class_name, candidates, image_size=image_size)

    # §2 text-instance
    if provider is not None and image is not None:
        crop_items = []
        for cand in g.by_kind(NodeKind.CANDIDATE):
            cp = crop_pil(image, cand.payload.get("bbox"))
            crop_items.append((feature_prefix + cand.id, cp))
        crop_feats = provider.encode_images_cached(crop_items)
        txt_feats = provider.encode_texts_cached([class_name])
        clip_cos = {cid: float(np.dot(crop_feats[cid], txt_feats[class_name]))
                    for cid in crop_feats}
        build_text_instance_edges(g, clip_cos)
    else:
        build_text_instance_edges(
            g, {c.id: 0.0 for c in g.by_kind(NodeKind.CANDIDATE)})

    # §3 support (appearance arm upgrades the scale-arm edges)
    if provider is not None and image is not None:
        def crop_cos(a, b):
            fa, fb = crop_feats.get(a), crop_feats.get(b)
            if fa is None or fb is None:
                return None
            return float(np.dot(fa, fb))
        build_support_edges(g, crop_cos)
    else:
        build_support_edges(g, None)

    # §4 competition
    if other_candidates and text_sim:
        from graph.edges import competing_classes

        rivals = competing_classes(class_name, text_sim)
        build_competition_edges(g, rivals, other_candidates)

    # §5 context
    if prior and other_candidates:
        prior_row = prior.get("classes", {}).get(class_name)
        prior_vocab = set(prior.get("vocabulary", []))
        evidence = band_evidence(g, other_candidates)
        build_context_edges(g, prior_row, prior_vocab, evidence)

    # §6 scene
    if provider is not None and image is not None:
        phrases = SCENE_PHRASES if scene_phrases is None else scene_phrases
        if phrases:
            txt = provider.encode_texts_cached(phrases)
            gkey = feature_prefix + "__global__"
            img_feat = provider.encode_images_cached([(gkey, image)])[gkey]
            scene_scores = {p: float(np.dot(img_feat, txt[p])) for p in phrases}
            crop_image_cos = {
                c.id: float(np.dot(crop_feats[c.id], img_feat))
                for c in g.by_kind(NodeKind.CANDIDATE) if c.id in crop_feats}
            build_scene_edges(g, scene_scores, crop_image_cos)

    # §7 unknown
    build_unknown_edges(g)
    return g
