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

from graph.schema import (Edge, EdgeKind, EvidenceGraph, Node, NodeKind)

# scale-ratio proximity: weight = 1 - min(1, ratio_gap / ratio_ref). The
# numeric reference is a v0 heuristic; it is re-registered (or kept) when
# S3.2 freezes thresholds.
SCALE_RATIO_REF = 3.0


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
        # pairwise same-class support (scale arm)
        for other in list(g.by_kind(NodeKind.CANDIDATE)):
            if other.id == cid:
                continue
            a1 = g.nodes[cid].payload["area_px"]
            a2 = other.payload["area_px"]
            ratio = max(a1, a2) / max(min(a1, a2), 1.0)
            w = max(0.0, 1.0 - ratio / SCALE_RATIO_REF)
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
