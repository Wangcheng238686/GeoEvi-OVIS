"""S2.2 zero-shot text-only inference harness (P0 baseline).

For every (image, class) pair of a registered dataset: SAM3 text-only concept
segmentation -> candidates (mask RLE + score) -> JSONL predictions consumable
by sam3_ovis.eval.score_eval. Per-image backbone features are set once and
reused across that image's classes (set_text_prompt replaces the text).

Everything affecting numbers is fixed by notes/reports/p0-gate-criteria.md:
confidence_threshold=0.05 pool, verbatim registry class names, bf16 autocast,
processor resolution 1008 (whole-image resize baseline).

Run:
  source sam3_ovis/scripts/load_environment.sh
  "$PYTHON" sam3_ovis/inference/zeroshot_text.py --dataset NWPU [--max-images 20]

Outputs under $SAM3_OVIS_LOG_ROOT/zeroshot/<DATASET>/:
  predictions.jsonl  one candidate per line (image_id, category_id, score,
                     segmentation RLE, category name)
  run_meta.json      config, per-image/per-class timing, candidate counts
Resume-safe: image_ids already present in predictions.jsonl are skipped.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

MAINLINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAINLINE))

from pycocotools import mask as mask_util  # noqa: E402

THRESHOLD = 0.05  # registered in p0-gate-criteria.md; do not change here


def rle_encode(mask_tensor) -> dict:
    """bool [1,H,W] (or [H,W]) cuda tensor -> COCO RLE dict (utf8 counts)."""
    m = mask_tensor
    if m.dim() == 3:
        m = m[0]
    arr = m.detach().cpu().numpy().astype("uint8", copy=False)
    rle = mask_util.encode(np.asfortranarray(arr))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


import numpy as np  # noqa: E402  (after sys.path fix; used by rle_encode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="registry key: NWPU / SOTA / FAST / ...")
    ap.add_argument("--max-images", type=int, default=None,
                    help="pilot/debug cap; full runs omit this")
    ap.add_argument("--gpu-memory-frac", type=float, default=None,
                    help="optional torch.cuda.set_per_process_memory_fraction")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from pycocotools.coco import COCO

    from data.score_datasets import DATASETS
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    # official usage: global bf16 autocast (see sam3-api-notes.md §0)
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if args.gpu_memory_frac:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_frac)

    spec = DATASETS[args.dataset]
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    gt = COCO(str(root / spec.val_json))
    cat_ids = gt.getCatIds()
    cat_names = {c["id"]: c["name"] for c in gt.loadCats(cat_ids)}
    img_ids = gt.getImgIds()
    if args.max_images:
        img_ids = img_ids[: args.max_images]

    out_dir = Path(os.environ["SAM3_OVIS_LOG_ROOT"]) / "zeroshot" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    meta_path = out_dir / "run_meta.json"

    done_imgs = set()
    if pred_path.exists():
        with open(pred_path) as f:
            for line in f:
                try:
                    done_imgs.add(json.loads(line)["image_id"])
                except json.JSONDecodeError:
                    break  # truncated tail line from a killed run
    print(f"[{args.dataset}] {len(img_ids)} images, {len(cat_ids)} classes, "
          f"{len(done_imgs)} already done -> {pred_path}")

    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=THRESHOLD)

    stats = {"set_image_s": 0.0, "grounding_s": 0.0, "rle_s": 0.0,
             "n_pairs": 0, "n_candidates": 0,
             "per_class_candidates": defaultdict(int),
             "per_image_candidates": {}}
    t_start = time.time()
    n_new_imgs = 0

    with open(pred_path, "a") as fout:
        for k, img_id in enumerate(img_ids):
            if img_id in done_imgs:
                continue
            info = gt.loadImgs(img_id)[0]
            img_path = root / spec.images_dir / info["file_name"]
            image = Image.open(img_path).convert("RGB")

            t0 = time.time()
            state = processor.set_image(image)
            t1 = time.time()
            stats["set_image_s"] += t1 - t0

            img_cand = 0
            for cat_id in cat_ids:
                state = processor.set_text_prompt(
                    state=state, prompt=cat_names[cat_id])
                scores = state["scores"]
                n = int(scores.shape[0]) if scores is not None else 0
                stats["n_pairs"] += 1
                if n == 0:
                    continue
                stats["n_candidates"] += n
                stats["per_class_candidates"][cat_names[cat_id]] += n
                img_cand += n
                t2 = time.time()
                stats["grounding_s"] += t2 - t1
                masks = state["masks"]
                for j in range(n):
                    t3 = time.time()
                    rle = rle_encode(masks[j])
                    stats["rle_s"] += time.time() - t3
                    fout.write(json.dumps({
                        "image_id": img_id,
                        "category_id": int(cat_id),
                        "category_name": cat_names[cat_id],
                        "score": round(float(scores[j]), 6),
                        "segmentation": rle,
                    }) + "\n")
                t1 = time.time()
            stats["per_image_candidates"][str(img_id)] = img_cand
            n_new_imgs += 1
            if n_new_imgs % 20 == 0:
                fout.flush()
                el = time.time() - t_start
                done_frac = (k + 1) / len(img_ids)
                print(f"  {k + 1}/{len(img_ids)} imgs, "
                      f"{stats['n_candidates']} cands, "
                      f"{el / max(n_new_imgs, 1):.1f}s/img, "
                      f"ETA {el / max(done_frac, 1e-9) - el:.0f}s")

    total_s = time.time() - t_start   # with-block exit has flushed & closed
    meta = {
        "dataset": args.dataset,
        "threshold": THRESHOLD,
        "max_images": args.max_images,
        "images_in_run": len(img_ids),
        "images_already_done": len(done_imgs),
        "images_processed": n_new_imgs,
        "total_s": round(total_s, 1),
        "avg_set_image_s": round(stats["set_image_s"] / max(n_new_imgs, 1), 3),
        "avg_grounding_s_per_pair": round(
            stats["grounding_s"] / max(stats["n_pairs"], 1), 4),
        "n_pairs": stats["n_pairs"],
        "n_candidates": stats["n_candidates"],
        "per_class_candidates": dict(stats["per_class_candidates"]),
        "per_image_candidates": stats["per_image_candidates"],
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": os.environ["SAM3_CKPT"],
    }
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"[{args.dataset}] done: {n_new_imgs} imgs, "
          f"{stats['n_candidates']} candidates, {total_s:.0f}s -> {meta_path}")


if __name__ == "__main__":
    main()
