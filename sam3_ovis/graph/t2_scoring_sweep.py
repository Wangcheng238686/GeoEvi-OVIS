"""v3.0 T2 Phase 2: offline scoring sweep over the candidate cache.

Reads the candidate cache (candidates.jsonl) and for each (beta, delta,
lambda) in the registered budget, computes B0/B1/B2/B3 scores per candidate,
writes per-variant predictions with mask RLE, and evaluates COCO segm mAP.

Selection protocol: best (beta, delta, lambda) by dev mAP among B2 variants;
then frozen and confirmed on test sets (T3).

Run:  source sam3_ovis/scripts/load_environment.sh
      "$PYTHON" sam3_ovis/graph/t2_scoring_sweep.py
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402
from graph.scoring_v3 import ScoringConfig, score_candidates
from graph.config import norm_text  # noqa: E402

BETAS = [0.0, 0.25, 0.5]
DELTAS = [0.0, 0.05]
LAMBDAS = [0.0, 1.0, 2.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    root = Path(os.environ.get("SAM3_OVIS_DATA_ROOT",
                               "/data/wangcheng/dataset/SCORE"))
    log_root = Path(os.environ.get("SAM3_OVIS_LOG_ROOT", "/tmp"))
    cache_dir = log_root / "t2_cache"
    spec = DATASETS["iSAID"]
    gt = COCO(str(root / spec.val_json))
    manifest = json.load(open(cache_dir / "manifest.json"))
    dev_ids = set(manifest["image_ids"])
    name_to_id = {c["name"]: c["id"] for c in gt.loadCats(gt.getCatIds())}

    sub = COCO()
    sub.dataset = {
        "images": [im for im in gt.dataset["images"] if im["id"] in dev_ids],
        "annotations": [a for a in gt.dataset["annotations"]
                        if a["image_id"] in dev_ids],
        "categories": gt.dataset["categories"],
    }
    sub.createIndex()
    print(f"[eval] GT: {len(sub.dataset['annotations'])} anns, "
          f"{len(sub.dataset['images'])} imgs", flush=True)

    # load candidate cache
    cands = []
    with open(cache_dir / "candidates.jsonl") as f:
        for line in f:
            if line.strip():
                cands.append(json.loads(line))
    print(f"[cache] {len(cands)} candidates loaded", flush=True)

    # group by image for spatial rival lookup
    by_img = defaultdict(list)
    for c in cands:
        by_img[c["img_id"]].append(c)

    # spatial adjacency: rivals = intersect(spatial_rivals keys, different cat)
    rival_spatial = {}   # cand_key -> max z over spatial rival classes
    rival_nospatial = {}  # cand_key -> max z over ALL other classes
    for c in cands:
        key = f"{c['img_id']}/{c['cat_name']}"
        z_own_val = c["z"][c["cat_name"]]
        # nospatial: max over ALL other classes
        ns = max((v for cn, v in c["z"].items() if cn != c["cat_name"]),
                 default=None)
        rival_nospatial[key] = ns
        # spatial: max over spatial rival candidates' classes
        sr = c.get("spatial_rivals", [])
        spatial_max = None
        for rk in sr:
            # find the rival candidate to get its class
            parts = rk.split("/")
            if len(parts) < 2:
                continue
            img_r, idx_s = parts[0], int(parts[1])
            if img_r != c["img_id"]:
                continue
        rival_spatial[key] = spatial_max  # computed in second pass below

    # second pass: resolve spatial rival z values (need cache lookup by key)
    key_to_cand = {}
    for c in cands:
        for i, key in enumerate(c.get("_keys", [])):
            pass  # keys are set in the cache gen; use cache records directly

    # Build lookup: candidate records have implicit key = img_id/index
    # Reconstruct: spatial_rivals contains keys like "img{id}/{idx}"
    for ci, c in enumerate(cands):
        pass  # all data is in c already

    # Actually, spatial_rivals in the cache stores keys of OTHER candidates.
    # We can resolve them because candidates.jsonl has all records.
    for c in cands:
        for sr_key in c.get("spatial_rivals", []):
            pass  # already stored; just resolve z values in scoring

    # scoring: for each candidate, the spatial rival z values can be
    # resolved by looking up the rival candidates' z vectors
    cand_by_key = {}
    for c in cands:
        if "spatial_rivals" in c:
            pass
    # Simplify: for each candidate, compute spatial rival max z from
    # its spatial_rivals list, looking up rival candidate's z vector
    key_z = {}
    idx = 0
    for c in cands:
        c["_ck"] = f"img{c['img_id']}/{idx}"
        idx += 1
    # z_rival_max from spatial adjacency:
    for c in cands:
        for rk in c.get("spatial_rivals", []):
            pass
    print("[warn] spatial rival resolution needs cache key mapping; "
          "falling back to inline computation", flush=True)

    # Simpler approach: recompute spatial rival z from the cached records
    # by looking up candidates in the same image with IoU >= 0.5 via the
    # spatial_rivals key.
    # For now: just recompute z_spatial from spatial_rivals class names
    # using the z vector of this candidate (no cross-candidate lookup needed
    # since z_i(y) = cos(crop_i, text_y) for ANY class y on THIS crop).
    # z_spatial_max = max over spatial rival CLASSES of z[cn].

    # group spatial_rivals keys to get rival candidate indices
    for c in cands:
        spatial_classes = set()
        for rk in c.get("spatial_rivals", []):
            # rk = "img{img_id}/{idx}" -> look up that candidate's cat_name
            parts = rk.split("/")
            if len(parts) >= 2:
                rimg, ridx = parts[0], int(parts[1])
                # find in same image
                for c2 in cands:
                    if c2["img_id"] == int(rimg.replace("img", "")):
                        if ridx < len(cands) and False:
                            pass
        # Not worth the complexity. The z vector already has ALL class
        # cosines; the spatial restriction just limits WHICH classes count
        # as rivals. We can resolve by finding spatial rival CANDIDATES and
        # their classes.

    # OK, the spatial_rivals list has candidate keys like "img1401/3".
    # The rival candidate's class can be looked up from candidates.jsonl
    # by matching img_id and index. But this requires index tracking.

    # SIMPLEST CORRECT APPROACH: re-read the cache in order; the implicit
    # key for record i in image X is "img{X}/{i}" (insertion order).
    # Build img_id -> list of records (in order).
    by_img = defaultdict(list)
    for c in cands:
        by_img[c["img_id"]].append(c)
    for img_id in by_img:
        for i, c in enumerate(by_img[img_id]):
            c["_idx"] = i

    # resolve spatial rival classes
    for c in cands:
        for rk in c.get("spatial_rivals", []):
            parts = rk.split("/")
            rimg = int(parts[0].replace("img", ""))
            ridx = int(parts[1])
            for c2 in by_img.get(rimg, []):
                if c2.get("_idx") == ridx:
                    c.setdefault("spatial_rival_classes", set()).add(
                        c2["cat_name"])
                    break

    # compute z_spatial_max
    for c in cands:
        src = c.get("spatial_rival_classes", set())
        z_own_name = c["cat_name"]
        candidates_z = [c["z"][cn] for cn in src
                        if cn != z_own_name and cn in c["z"]]
        c["z_spatial_max"] = max(candidates_z) if candidates_z else None

    print("[scoring] spatial rival resolution done", flush=True)

    # score all (beta, delta, lambda) combinations
    out_dir = Path(args.out_dir or (log_root / "t2_sweep"))
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_results = {}
    for beta in BETAS:
        for delta in DELTAS:
            for lam in LAMBDAS:
                cfg = ScoringConfig(beta=beta, delta=delta, lam=lam)
                vname = f"B2_b{beta}_d{delta}_l{lam}"
                dets = []
                for c in cands:
                    u = (1 - beta) * norm_text(c["raw_score"]) \
                        + beta * (c["z"][c["cat_name"]] + 1) / 2
                    if c["z_spatial_max"] is not None:
                        d_i = max(0.0, c["z_spatial_max"]
                                  - c["z"][c["cat_name"]] - delta)
                        u = u * float(np.exp(-lam * d_i))
                    dets.append({"image_id": c["img_id"],
                                 "category_id": name_to_id[c["cat_name"]],
                                 "score": round(u, 6),
                                 "segmentation": c["mask_rle"]})
                if not dets:
                    continue
                dt = sub.loadRes(dets)
                ev = COCOeval(sub, dt, iouType="segm")
                ev.evaluate(); ev.accumulate(); ev.summarize()
                mAP = round(ev.stats[0], 4)
                AP50 = round(ev.stats[1], 4)
                grid_results[vname] = {"mAP": mAP, "AP50": AP50}
                print(f"  {vname:30s} mAP={mAP:.4f} AP50={AP50:.4f}",
                      flush=True)

    # B0/B1 baselines
    for bname, beta in [("B0", 0.0), ("B1_b0.5", 0.5)]:
        dets = []
        for c in cands:
            b = norm_text(c["raw_score"])
            zc = (c["z"][c["cat_name"]] + 1) / 2
            u = (1 - beta) * b + beta * zc
            dets.append({"image_id": c["img_id"],
                         "category_id": name_to_id[c["cat_name"]],
                         "score": round(u, 6),
                         "segmentation": c["mask_rle"]})
        dt = sub.loadRes(dets)
        ev = COCOeval(sub, dt, iouType="segm")
        ev.evaluate(); ev.accumulate(); ev.summarize()
        grid_results[bname] = {"mAP": round(ev.stats[0], 4),
                               "AP50": round(ev.stats[1], 4)}
        print(f"  {bname:30s} mAP={grid_results[bname]['mAP']:.4f} "
              f"AP50={grid_results[bname]['AP50']:.4f}", flush=True)

    # summary
    best_b2 = max((v["mAP"] for k, v in grid_results.items()
                   if k.startswith("B2_")), default=0)
    best_key = max((k for k in grid_results if k.startswith("B2_")),
                   key=lambda k: grid_results[k]["mAP"])
    b0 = grid_results.get("B0", {}).get("mAP", 0)
    b1 = grid_results.get("B1_b0.5", {}).get("mAP", 0)
    print(f"\n=== T2 dev summary ===")
    print(f"B0 = {b0:.4f} | B1 = {b1:.4f} | best B2 = {best_b2:.4f}")
    print(f"B2 - B0 = {best_b2 - b0:+.4f} | B2 - B1 = {best_b2 - b1:+.4f}")
    print(f"best grid point: {best_key}")

    summary = {"grid": grid_results, "best_b2": best_b2, "B0": b0, "B1": b1}
    (out_dir / "sweep_results.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {out_dir / 'sweep_results.json'}", flush=True)


if __name__ == "__main__":
    main()
