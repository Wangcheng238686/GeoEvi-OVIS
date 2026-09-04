"""S3.5 ablation pipeline (docs/graph-thresholds.md §12).

All four variants consume the SAME P0 candidate pool (predictions.jsonl,
raw score >= 0.05) — SAM3 is NOT re-run:

  A0  official-threshold baseline: score >= 0.5 + standard greedy mask NMS
  A1  edge-free control: full pool + norm_text scoring + standard NMS
  A2  full graph: all edges + message passing + merge + risk modulation
  A3  A2 minus one component (-competition / -context / -scene / -risk)

Per image: candidate masks are decoded ONCE and CLIP crop features are
computed ONCE (shared across classes and across A2/A3). Per rival class,
only the top-10 score candidates enter competition/context (registered).

Run:  source sam3_ovis/scripts/load_environment.sh
      "$PYTHON" sam3_ovis/graph/ablation.py --dataset NWPU --limit 20   # pilot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

MAINLINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAINLINE))

import numpy as np  # noqa: E402

from graph.config import MP, THRESHOLDS, norm_text  # noqa: E402

RIVAL_TOPK = 10          # registered §12
A0_THRESHOLD = 0.5       # registered §12 (official default)
NMS_IOU = 0.5
BOOT_SEED = 20260903


# ---------------------------------------------------------------------------
# shared pool loading + standard NMS
# ---------------------------------------------------------------------------

def load_pool(path: Path) -> Dict[int, list]:
    pool = defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pool[d["image_id"]].append({
                "cat_id": int(d["category_id"]),
                "cat_name": d["category_name"],
                "score": float(d["score"]),
                "rle": d["segmentation"],
            })
    return pool


def _rle_arr(rle: dict) -> np.ndarray:
    from pycocotools import mask as mask_util

    r = dict(rle)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode("utf-8")
    return mask_util.decode(r).astype(bool)


def _rle_bbox(rle: dict) -> list:
    from pycocotools import mask as mask_util

    r = dict(rle)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode("utf-8")
    x, y, w, h = mask_util.toBbox(r)
    return [float(x), float(y), float(w), float(h)]


def standard_nms(dets: List[dict], iou_thr: float = NMS_IOU) -> List[dict]:
    """Greedy score-ordered class-wise mask NMS. dets: same class."""
    order = sorted(dets, key=lambda d: -d["score"])
    kept, kept_masks = [], []
    for d in order:
        m = _rle_arr(d["rle"])
        suppressed = False
        for km in kept_masks:
            inter = int((m & km).sum())
            if inter == 0:
                continue
            if inter / float((m | km).sum()) >= iou_thr:
                suppressed = True
                break
        if not suppressed:
            kept.append(d)
            kept_masks.append(m)
    return kept


# ---------------------------------------------------------------------------
# per-image shared computation
# ---------------------------------------------------------------------------

class ImageContext:
    """Decodes every candidate mask once and computes CLIP crop features
    once; shared across classes and across A2/A3 variants."""

    def __init__(self, items, image, provider):
        self.image = image
        self.masks = {}
        self.bboxes = {}
        self.feats = {}
        crop_items = []
        for i, it in enumerate(items):
            key = f"c{i}"
            it["key"] = key
            self.masks[key] = _rle_arr(it["rle"])
            self.bboxes[key] = _rle_bbox(it["rle"])
            crop_items.append((key, _crop(image, self.bboxes[key])))
        if provider is not None:
            feats = provider.encode_images_cached(crop_items)
            self.feats.update(feats)

    def arr(self, it):
        return self.masks[it["key"]]


def _crop(image, bbox, margin=0.1):
    x, y, w, h = bbox
    mx, my = w * margin, h * margin
    return image.crop((max(0, int(x - mx)), max(0, int(y - my)),
                       min(image.width, int(x + w + mx)),
                       min(image.height, int(y + h + my))))


def cap_rivals(items, own_cat_id, topk=RIVAL_TOPK):
    """Per rival class, keep top-k score candidates (registered §12)."""
    by_cat = defaultdict(list)
    for it in items:
        if it["cat_id"] != own_cat_id:
            by_cat[it["cat_name"]].append(it)
    return {name: sorted(v, key=lambda d: -d["score"])[:topk]
            for name, v in by_cat.items()}


# ---------------------------------------------------------------------------
# variant pipelines
# ---------------------------------------------------------------------------

def run_A0(pool_img, cat_id):
    dets = [d for d in pool_img if d["cat_id"] == cat_id
            and d["score"] >= A0_THRESHOLD]
    return [{"score": d["score"], "rle": d["rle"]}
            for d in standard_nms(dets)]


def run_A1(pool_img, cat_id):
    dets = [d for d in pool_img if d["cat_id"] == cat_id]
    for d in dets:
        d["score"] = norm_text(d["score"])
    return [{"score": d["score"], "rle": d["rle"]}
            for d in standard_nms(dets)]


def run_graph_variant(pool_img, cat_id, ctx: ImageContext, provider,
                      prior, use_competition=True, use_context=True,
                      use_scene=True, use_risk=True):
    """A2 when all flags True; A3 variants flip one flag off. Graph edges are
    built once per (image, class); component switches act on message passing
    and edge filtering."""
    from graph.builder import build_full_graph
    from graph.completeness import compute_completeness
    from graph.edges import competing_classes
    from graph.message_passing import (merge_duplicate_clusters,
                                       propagate_reliability, rank_candidates)

    own = [d for d in pool_img if d["cat_id"] == cat_id]
    if not own:
        return [], {}
    rivals = cap_rivals(pool_img, cat_id)
    text_sim = {}
    if provider is not None and rivals:
        names = list(rivals.keys())
        sims = provider.encode_texts_cached([own[0]["cat_name"]] + names)
        q = sims[own[0]["cat_name"]]
        text_sim = {n: float(np.dot(q, sims[n])) for n in names}

    # monkey-free component switching: pass filtered structures
    other_candidates = {k: v for k, v in rivals.items()} \
        if use_competition or use_context else {}
    graph_text_sim = text_sim if use_competition else {}

    g = build_full_graph(
        image_id=None, class_name=own[0]["cat_name"], candidates=[
            {"id": d["key"], "score": d["score"], "rle": d["rle"],
             "bbox": ctx.bboxes[d["key"]]} for d in own],
        image=ctx.image, image_size=(ctx.image.height, ctx.image.width),
        other_candidates=other_candidates, text_sim=graph_text_sim,
        prior=prior, provider=provider,
        scene_phrases=None if use_scene else [])

    # ID mapping: builder prefixes candidate ids with "cand:"
    rel = {f"cand:{k}": v for k, v in
           propagate_reliability_flags(g, use_competition, use_context,
                                       use_scene).items()}
    merged = merge_duplicate_clusters(g, rel)
    if not use_risk:
        g.attrs["completeness"] = {"risk_score": 0.0}
        for d in merged:
            d["final_score"] = d["merged_score"]
        out = sorted(merged, key=lambda d: -d["final_score"])
    else:
        out, _ = rank_candidates(g, merged, rel)
    return [{"score": d["final_score"], "rle": d["mask_rle"]} for d in out]


def propagate_reliability_flags(g, use_competition, use_context, use_scene):
    """propagate_reliability with per-component switches (A3)."""
    import graph.message_passing as mp

    saved = (MP.alpha_compete, MP.alpha_context, MP.alpha_scene)
    obj = MP
    from dataclasses import replace
    obj = replace(obj, alpha_compete=MP.alpha_compete if use_competition else 0.0,
                  alpha_context=MP.alpha_context if use_context else 0.0,
                  alpha_scene=MP.alpha_scene if use_scene else 0.0)
    # temporarily swap module constants (frozen dataclass instance)
    orig = mp.MP
    mp.MP = obj
    try:
        return mp.propagate_reliability(g)
    finally:
        mp.MP = orig
