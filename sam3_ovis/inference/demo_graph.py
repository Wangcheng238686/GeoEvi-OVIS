"""S3.4 end-to-end single-image demo: real predictions -> graph -> edges ->
completeness -> two-round message passing -> duplicate merge -> risk-modulated
ranking. Uses the real RemoteCLIP provider on GPU.

Run:  source sam3_ovis/scripts/load_environment.sh
      "$PYTHON" sam3_ovis/inference/demo_graph.py [--dataset NWPU] [--min-cands 15]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

MAINLINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAINLINE))

import numpy as np  # noqa: E402


def pick_image(log_root: Path, dataset: str, min_cands: int,
               gt=None, min_gt: int = 0):
    """Pick the (image_id, category_id) with the most candidates >= min.
    With min_gt>0, require the pair to have GT instances (positive showcase)."""
    counts = defaultdict(int)
    with open(log_root / dataset / "predictions.jsonl") as f:
        for line in f:
            d = json.loads(line)
            counts[(d["image_id"], d["category_id"])] += 1
    for (img_id, cat_id), n in sorted(counts.items(), key=lambda x: -x[1]):
        if n < min_cands:
            continue
        if gt is not None and min_gt > 0:
            if len(gt.getAnnIds(imgIds=[img_id], catIds=[cat_id])) < min_gt:
                continue
        return img_id, cat_id, n
    raise SystemExit("no (image, class) pair satisfying the constraints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="NWPU")
    ap.add_argument("--min-cands", type=int, default=15)
    ap.add_argument("--min-gt", type=int, default=0,
                    help="require >= min_gt GT instances of the class "
                         "(positive showcase); 0 = pure FP showcase")
    ap.add_argument("--top-k-rivals", type=int, default=3)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_util

    from graph.builder import build_full_graph
    from graph.completeness import compute_completeness
    from graph.message_passing import (merge_duplicate_clusters,
                                       propagate_reliability, rank_candidates)
    from graph.provider import RemoteCLIPProvider
    from data.score_datasets import DATASETS

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    spec = DATASETS[args.dataset]
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    log_root = Path(os.environ["SAM3_OVIS_LOG_ROOT"]) / "zeroshot"
    gt = COCO(str(root / spec.val_json))
    cat_names = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}

    img_id, cat_id, n_cand = pick_image(log_root, args.dataset,
                                        args.min_cands, gt=gt,
                                        min_gt=args.min_gt)
    info = gt.loadImgs(img_id)[0]
    query = cat_names[cat_id]
    print(f"demo image: {info['file_name']} (id={img_id}), class={query!r}, "
          f"{n_cand} candidates")

    cands = []
    with open(log_root / args.dataset / "predictions.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["image_id"] == img_id and d["category_id"] == cat_id:
                cands.append({"id": str(len(cands)), "score": d["score"],
                              "rle": d["segmentation"],
                              "bbox": [v for v in _rle_bbox(d["segmentation"])],
                              })
    image = Image.open(root / spec.images_dir / info["file_name"]).convert("RGB")

    # rival classes: other classes with candidates on the same image
    other = defaultdict(list)
    with open(log_root / args.dataset / "predictions.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if (d["image_id"] == img_id and d["category_id"] != cat_id
                    and len(other[d["category_name"]]) < args.top_k_rivals):
                other[d["category_name"]].append(
                    {"id": f"{d['category_name']}:{len(other[d['category_name']])}",
                     "score": d["score"], "rle": d["segmentation"]})
    rival_names = list(other.keys())

    # RemoteCLIP class-name similarities for rival selection
    provider = RemoteCLIPProvider()
    if rival_names:
        sims = provider.encode_texts_cached([query] + rival_names)
        q = sims[query]
        text_sim = {r: float(np.dot(q, sims[r])) for r in rival_names}
    else:
        text_sim = {}

    prior_path = root / "priors" / "cooccurrence_iSAID.json"
    prior = json.load(open(prior_path)) if prior_path.exists() else None

    g = build_full_graph(
        image_id=img_id, class_name=query, candidates=cands, image=image,
        image_size=(image.height, image.width),
        other_candidates=dict(other), text_sim=text_sim,
        prior=prior, provider=provider)

    compute_completeness(g)
    rel = propagate_reliability(g)
    merged = merge_duplicate_clusters(g, rel)
    ranked, meta = rank_candidates(g, merged, rel)

    # GT match status per merged detection
    gt_anns = gt.loadAnns(gt.getAnnIds(imgIds=[img_id]))
    gt_rles = []
    for a in gt_anns:
        seg = a["segmentation"]
        objs = seg if isinstance(seg[0], list) else [seg]
        gt_rles.append((a["category_id"],
                        mask_util.merge(mask_util.frPyObjects(
                            objs, info["height"], info["width"]))))
    for d in ranked:
        r = dict(d["mask_rle"])
        r["counts"] = r["counts"].encode("utf-8")
        best = 0.0
        for cid, gr in gt_rles:
            iou = float(mask_util.iou([r], [gr], [0])[0][0])
            if cid == cat_id and iou > best:
                best = iou
        d["gt_iou"] = round(best, 4)

    comp = g.attrs["completeness"]
    print(f"\ncompleteness: ECR={comp['explained_coverage_rate']:.3f} "
          f"URD={comp['unexplained_response_density']:.3f} "
          f"CCR={comp['conflict_candidate_rate']:.3f} "
          f"R={comp['risk_score']:.3f} "
          f"alarm={meta['incompleteness_alarm']}")
    print(f"clusters: {meta['n_clusters']} (from {meta['n_pool_candidates']} "
          f"candidates)")
    print("\ntop 10 after graph inference:")
    print(f"{'final':>7s} {'rel':>6s} {'merged':>7s} {'gtIoU':>6s}  cand")
    for d in ranked[:10]:
        print(f"{d['final_score']:7.4f} {d['reliability']:6.3f} "
              f"{d['merged_score']:7.4f} {d['gt_iou']:6.3f}  {d['cand_id']}")

    # save overlay
    out_dir = Path(os.environ["SAM3_OVIS_LOG_ROOT"]) / "graph_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.array(image).copy()
    for i, d in enumerate(ranked[:10]):
        r = dict(d["mask_rle"])
        r["counts"] = r["counts"].encode("utf-8")
        m = mask_util.decode(r).astype(bool)
        color = [(255, 60, 60), (60, 220, 60), (60, 120, 255)][i % 3]
        arr[m] = (0.5 * arr[m] + 0.5 * np.array(color)).astype(np.uint8)
    Image.fromarray(arr).save(out_dir / f"demo_{img_id}_{query}.png")

    summary = {"image_id": img_id, "class": query,
               "completeness": comp, "meta": meta,
               "ranked": [{k: v for k, v in d.items() if k != "mask_rle"}
                          for d in ranked]}
    (out_dir / f"demo_{img_id}_{query}.json").write_text(json.dumps(summary, indent=1))
    print(f"\nartifacts -> {out_dir}")


def _rle_bbox(rle):
    from pycocotools import mask as mask_util

    r = dict(rle)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode("utf-8")
    x, y, w, h = mask_util.toBbox(r)
    return [float(x), float(y), float(w), float(h)]


if __name__ == "__main__":
    main()
