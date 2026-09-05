"""S3.5 ablation pipeline (docs/graph-thresholds.md §12).

All four variants consume the SAME P0 candidate pool (predictions.jsonl,
raw score >= 0.05) — SAM3 is NOT re-run:

  A0  official-threshold baseline: score >= 0.5 + standard greedy mask NMS
  A1  edge-free control: full pool + norm_text scoring + standard NMS
  A2  full graph: all edges + message passing + merge + risk modulation
  A3  A2 minus one component: -competition / -context / -scene / -risk
      (implemented as MP/rank parameter switches on the SAME graph — the
      graph is built once per (image, class))

Per image: candidate masks are decoded once and CLIP crop features are
computed once (provider-level cache keyed by candidate index; shared across
classes and variants). Rival candidates for competition/context are capped
at per-class top-10 by score (registered §12).

Run:  source sam3_ovis/scripts/load_environment.sh
      "$PYTHON" sam3_ovis/graph/run_ablation.py --dataset NWPU --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

MAINLINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAINLINE))

import numpy as np  # noqa: E402

from graph.config import MP, THRESHOLDS, norm_text
from graph.schema import NodeKind
from graph.message_passing import (_mask_arr, build_duplicate_clusters,
                                   pairwise_iou_matrix, score_clusters)  # noqa: E402

RIVAL_TOPK = 10          # registered §12
A0_THRESHOLD = 0.5       # registered §12 (official default)
NMS_IOU = 0.5
BOOT_SEED = 20260903

VARIANTS = ["A0", "A1", "A2",
            "A3-competition", "A3-context", "A3-scene", "A3-risk"]


# ---------------------------------------------------------------------------
# pool / masks / NMS
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
    """Greedy score-ordered mask NMS within one class."""
    order = sorted(dets, key=lambda d: -d["score"])
    kept, kept_masks = [], []
    for d in order:
        m = _rle_arr(d["rle"])
        dup = False
        for km in kept_masks:
            inter = int((m & km).sum())
            if inter and inter / float((m | km).sum()) >= iou_thr:
                dup = True
                break
        if not dup:
            kept.append(d)
            kept_masks.append(m)
    return kept


# ---------------------------------------------------------------------------
# per-image shared context (masks + crops decoded/encoded once)
# ---------------------------------------------------------------------------

class ImageContext:
    def __init__(self, items, image, provider):
        from graph.provider import RemoteCLIPProvider  # noqa: F401 (type only)

        self.image = image
        self.masks, self.bboxes = {}, {}
        crop_items = []
        for i, it in enumerate(items):
            key = f"c{i}"
            it["key"] = key
            self.masks[key] = _rle_arr(it["rle"])
            self.bboxes[key] = _rle_bbox(it["rle"])
            if provider is not None:
                x, y, w, h = self.bboxes[key]
                mx, my = w * 0.1, h * 0.1
                x1 = min(max(0, int(x - mx)), image.width - 1)
                y1 = min(max(0, int(y - my)), image.height - 1)
                x2 = min(image.width, max(x1 + 1, int(x + w + mx)))
                y2 = min(image.height, max(y1 + 1, int(y + h + my)))
                crop_items.append((key, image.crop((x1, y1, x2, y2))))
        if provider is not None:
            self.feats = provider.encode_images_cached(crop_items)

    def crop_cos(self, key_a: str, key_b: str):
        fa, fb = self.feats.get(key_a), self.feats.get(key_b)
        if fa is None or fb is None:
            return None
        return float(np.dot(fa, fb))


def cap_rivals(items, own_cat_id, topk=RIVAL_TOPK):
    by_cat = defaultdict(list)
    for it in items:
        if it["cat_id"] != own_cat_id:
            it.setdefault("id", it.get("key", f"r{len(by_cat)}"))
            by_cat[it["cat_name"]].append(it)
    return {name: sorted(v, key=lambda d: -d["score"])[:topk]
            for name, v in by_cat.items()}


# ---------------------------------------------------------------------------
# graph variant runner (graph built ONCE per (image, class))
# ---------------------------------------------------------------------------

def run_graph_variants(pool_img, cat_id, ctx: ImageContext, provider, prior):
    """Returns {variant: [detections]} for A2 and the four A3 variants."""
    from graph.builder import build_full_graph
    from graph.completeness import compute_completeness
    from graph.edges import competing_classes
    from graph.message_passing import (merge_duplicate_clusters,
                                       propagate_reliability, rank_candidates)

    own = [d for d in pool_img if d["cat_id"] == cat_id]
    if not own:
        return {}
    class_name = own[0]["cat_name"]
    rivals = cap_rivals(pool_img, cat_id)
    text_sim = {}
    if provider is not None and rivals:
        names = list(rivals.keys())
        sims = provider.encode_texts_cached([class_name] + names)
        q = sims[class_name]
        text_sim = {n: float(np.dot(q, sims[n])) for n in names}

    g = build_full_graph(
        image_id=None, class_name=class_name, candidates=[
            {"id": d["key"], "score": d["score"], "rle": d["rle"],
             "bbox": ctx.bboxes[d["key"]]} for d in own],
        image=ctx.image, image_size=(ctx.image.height, ctx.image.width),
        other_candidates=rivals, text_sim=text_sim,
        prior=prior, provider=provider)

    compute_completeness(g)          # populates g.attrs["completeness"]
    rel = propagate_reliability(g)   # keys already carry the cand: prefix

    # clusters depend only on support edges + mask IoU — identical across
    # variants; the IoU matrix is computed once per (image, class)
    own = [c for c in g.by_kind(NodeKind.CANDIDATE)
           if not c.payload.get("rival_class")
           and float(c.payload.get("score", 0.0)) >= THRESHOLDS.pool_floor]
    own_ids = [c.id for c in own]
    masks = {c.id: _mask_arr(c.payload["rle"]).astype(bool) for c in own}
    order = own_ids
    if order:
        mat = np.stack([masks[i] for i in order])
        from graph.message_passing import pairwise_iou_matrix
        iou_mat = pairwise_iou_matrix(mat)
        iou_lut = {(a, b): float(iou_mat[i, j])
                   for i, a in enumerate(order)
                   for j, b in enumerate(order) if j > i}
    else:
        iou_lut = {}

    def iou_fn(a, b):
        pair = (a, b) if a < b else (b, a)
        return iou_lut.get(pair, 0.0)

    clusters = build_duplicate_clusters(g, iou_fn)

    out, meta = rank_candidates(g, score_clusters(g, clusters, rel), rel)
    results = {"A2": [{"score": d["final_score"], "rle": d["mask_rle"]}
                      for d in out]}
    # A3: same graph + same clusters, component switched off via MP weights
    switches = {
        "A3-competition": {"alpha_compete": 0.0},
        "A3-context": {"alpha_context": 0.0},
        "A3-scene": {"alpha_scene": 0.0},
    }
    for variant, sw in switches.items():
        mp2 = replace(MP, **sw)
        saved = sys.modules["graph.message_passing"].MP
        sys.modules["graph.message_passing"].MP = mp2
        try:
            r2 = propagate_reliability(g)
        finally:
            sys.modules["graph.message_passing"].MP = saved
        ranked2, _ = rank_candidates(g, score_clusters(g, clusters, r2), r2)
        results[variant] = [{"score": d["final_score"], "rle": d["mask_rle"]}
                            for d in ranked2]

    # A3-risk: identical reliability, no risk modulation
    no_risk = []
    for d in score_clusters(g, clusters, rel):
        d2 = dict(d)
        d2["final_score"] = d2["merged_score"]
        no_risk.append(d2)
    results["A3-risk"] = [{"score": d["final_score"], "rle": d["mask_rle"]}
                          for d in sorted(no_risk,
                                          key=lambda d: -d["final_score"])]
    return results


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="pilot: cap number of images")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from pycocotools.coco import COCO

    from data.score_datasets import DATASETS
    from graph.provider import RemoteCLIPProvider

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    spec = DATASETS[args.dataset]
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    log_root = Path(os.environ["SAM3_OVIS_LOG_ROOT"])
    gt = COCO(str(root / spec.val_json))
    cat_ids = gt.getCatIds()
    pool = load_pool(log_root / "zeroshot" / args.dataset / "predictions.jsonl")

    img_ids = [i for i in gt.getImgIds() if i in pool]
    if args.limit:
        img_ids = img_ids[: args.limit]

    out_dir = log_root / "ablation" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    # resume: per-variant done-sets so a killed run never duplicates or
    # loses an image (variants are deterministic given the pool)
    done = {v: set() for v in VARIANTS}
    for v in VARIANTS:
        p = out_dir / f"{v}.jsonl"
        if p.exists():
            with open(p) as f:
                for line in f:
                    try:
                        done[v].add(json.loads(line)["image_id"])
                    except json.JSONDecodeError:
                        break
    todo_ids = [i for i in img_ids
                if any(i not in done[v] for v in VARIANTS)]
    print(f"[{args.dataset}] {len(img_ids)} imgs, "
          f"{len(img_ids) - len(todo_ids)} already done, "
          f"{len(todo_ids)} to (re)process", flush=True)

    files = {v: open(out_dir / f"{v}.jsonl", "a") for v in VARIANTS}
    error_log = out_dir / "error_images.jsonl"

    provider = RemoteCLIPProvider()
    prior_path = root / "priors" / "cooccurrence_iSAID.json"
    prior = json.load(open(prior_path)) if prior_path.exists() else None

    t0 = time.time()
    n_new = 0
    for k, img_id in enumerate(todo_ids):
        info = gt.loadImgs(img_id)[0]
        try:
            image = Image.open(root / spec.images_dir / info["file_name"]) \
                .convert("RGB")
            items = pool[img_id]
            from graph.edges import clear_rle_cache

            clear_rle_cache()  # per-image decode cache (id-keyed)
            ctx = ImageContext(items, image, provider)

            for cat_id in cat_ids:
                # A0 / A1: pool post-processing (no graph)
                variants_out = {
                    "A0": [{"score": d["score"], "rle": d["rle"]}
                           for d in standard_nms(
                               [d for d in items if d["cat_id"] == cat_id
                                and d["score"] >= A0_THRESHOLD])],
                    "A1": [{"score": norm_text(d["score"]), "rle": d["rle"]}
                           for d in standard_nms(
                               [d for d in items if d["cat_id"] == cat_id])],
                }
                graph_out = run_graph_variants(items, cat_id, ctx,
                                               provider, prior)
                variants_out.update(graph_out)
                cat_name = gt.loadCats(cat_id)[0]["name"]
                for variant, dets in variants_out.items():
                    if img_id in done[variant]:
                        continue
                    for d in dets:
                        files[variant].write(json.dumps({
                            "image_id": img_id, "category_id": int(cat_id),
                            "category_name": cat_name,
                            "score": d["score"], "segmentation": d["rle"],
                        }) + "\n")
        except torch.cuda.OutOfMemoryError:
            from graph.edges import clear_rle_cache

            clear_rle_cache()
            torch.cuda.empty_cache()
            with open(error_log, "a") as fo:
                fo.write(json.dumps({"image_id": img_id,
                                     "error": "cuda_oom"}) + "\n")
            print(f"  OOM skipped img {img_id}", flush=True)
            n_new += 1
            continue
        except Exception as e:  # noqa: BLE001
            from graph.edges import clear_rle_cache

            clear_rle_cache()
            torch.cuda.empty_cache()
            with open(error_log, "a") as fo:
                fo.write(json.dumps({"image_id": img_id,
                                     "error": repr(e)[:200]}) + "\n")
            print(f"  error skipped img {img_id}: {repr(e)[:120]}", flush=True)
            n_new += 1
            continue
        for v in VARIANTS:
            done[v].add(img_id)
        n_new += 1
        if n_new % 25 == 0:
            for f in files.values():
                f.flush()
            el = time.time() - t0
            print(f"  {k + 1}/{len(todo_ids)} imgs, {el / n_new:.2f}s/img, "
                  f"ETA {el / (n_new / len(todo_ids)) - el:.0f}s", flush=True)
            torch.cuda.empty_cache()
    for f in files.values():
        f.close()
    print(f"[{args.dataset}] ablation done: {n_new} imgs in "
          f"{time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
