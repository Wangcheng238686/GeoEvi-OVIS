"""v3.0 T2 Phase 1: candidate cache generation (one-time GPU pass).

Per dev image: SAM3 text-only (15 classes) -> pool floor -> class-wise NMS
-> per-candidate CLIP features vs ALL 15 class texts -> spatial adjacency.

Output: candidates.jsonl with FULL per-candidate data:
  {img_id, cat_id, cat_name, raw_score, mask_rle, bbox,
   z: {class_name: cosine} for ALL 15 classes,
   spatial_rivals: [candidate keys with IoU >= 0.5]}

Phase 2 (offline scoring) reads this cache — no GPU needed for parameter
grid search.

Run:  source sam3_ovis/scripts/load_environment.sh
      "$PYTHON" sam3_ovis/graph/t2_candidate_cache.py [--n-images 300]
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402
from graph.provider import RemoteCLIPProvider  # noqa: E402

SEED = 20260905
NMS_IOU = 0.5
POOL_FLOOR = 0.05
CROP_MARGIN = 0.1
RIVAL_IOU = 0.5


def _tight_bbox(m):
    ys, xs = np.any(m, axis=1), np.any(m, axis=0)
    if not ys.any():
        return None
    return (int(np.where(xs)[0][0]), int(np.where(ys)[0][0]),
            int(np.where(xs)[0][-1]) + 1, int(np.where(ys)[0][-1]) + 1)


def _iou_cropped(m1, b1, m2, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = int((m1[iy1:iy2, ix1:ix2] & m2[iy1:iy2, ix1:ix2]).sum())
    if inter == 0:
        return 0.0
    return inter / float(int(m1.sum()) + int(m2.sum()) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=300)
    ap.add_argument("--n-source-groups", type=int, default=15)
    args = ap.parse_args()

    from graph.provider import RemoteCLIPProvider
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    spec = DATASETS["iSAID"]
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    log_root = Path(os.environ["SAM3_OVIS_LOG_ROOT"])
    gt = COCO(str(root / spec.val_json))
    cat_ids = gt.getCatIds()
    cat_names = {c["id"]: c["name"] for c in gt.loadCats(cat_ids)}
    name_list = [cat_names[c] for c in cat_ids]

    # source-image-grouped dev sample
    rng = np.random.default_rng(SEED)
    prefixes = defaultdict(list)
    for im in gt.dataset["images"]:
        pref = im["file_name"].replace(".png", "").split("_")[0]
        prefixes[pref].append(im["id"])
    all_prefs = sorted(prefixes.keys())
    chosen = sorted(rng.choice(all_prefs,
                               size=min(args.n_source_groups, len(all_prefs)),
                               replace=False).tolist())
    dev_ids = sorted(i for p in chosen for i in prefixes[p])
    dev_ids = dev_ids[: args.n_images]

    out_dir = log_root / "t2_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    import subprocess
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2])).decode().strip()
    except Exception:
        sha = "unknown"
    manifest = {"seed": SEED, "source_groups": chosen,
                "n_images": len(dev_ids), "image_ids": dev_ids,
                "code_sha": sha}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[cache] dev: {len(dev_ids)} imgs, code_sha={sha[:12]}", flush=True)

    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=POOL_FLOOR)
    provider = RemoteCLIPProvider()
    txt_feats = provider.encode_texts_cached(name_list)

    cache_path = out_dir / "candidates.jsonl"
    done_path = out_dir / "done_images.jsonl"
    done_ids = set()
    if done_path.exists():
        with open(done_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["img_id"])
                except (json.JSONDecodeError, KeyError):
                    break
    todo = [i for i in dev_ids if i not in done_ids]
    print(f"[cache] {len(todo)} to process ({len(done_ids)} already done)",
          flush=True)

    cache_f = open(cache_path, "a")
    done_f = open(done_path, "a")
    t0 = time.time()
    for k, img_id in enumerate(todo):
        info = gt.loadImgs(img_id)[0]
        image = Image.open(root / spec.images_dir / info["file_name"]) \
            .convert("RGB")
        state = processor.set_image(image)

        # per class: SAM3 grounding -> pool floor filter
        raw = []
        for cid in cat_ids:
            state = processor.set_text_prompt(state=state,
                                              prompt=cat_names[cid])
            sc = state["scores"]
            for j in range(int(sc.shape[0]) if sc is not None else 0):
                if float(sc[j]) < POOL_FLOOR:
                    continue
                m = state["masks"][j, 0].detach().cpu().numpy().astype(bool)
                b = _tight_bbox(m)
                if b is None:
                    continue
                raw.append({"cat_id": cid, "cat_name": cat_names[cid],
                            "score": float(sc[j]), "mask": m, "bbox": b})

        # class-wise NMS (per class, raw-score order)
        nmsd = []
        for cid in sorted({d["cat_id"] for d in raw}):
            cls = sorted([d for d in raw if d["cat_id"] == cid],
                         key=lambda d: -d["score"])
            for d in cls:
                m, b = d["mask"], d["bbox"]
                dup = any(_iou_cropped(m, b, km["mask"], km["bbox"])
                          >= NMS_IOU for km in nmsd
                          if km["cat_id"] == cid)
                if not dup:
                    d["key"] = f"img{img_id}/{len(nmsd)}"
                    nmsd.append(d)

        # CLIP features (bbox + 10% margin, image-scoped cache keys)
        provider.clear_image_cache()
        crop_items = []
        for i, d in enumerate(nmsd):
            x1, y1, x2, y2 = d["bbox"]
            mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
            cx1 = max(0, int(x1 - mx))
            cy1 = max(0, int(y1 - my))
            cx2 = min(image.width, int(x2 + mx))
            cy2 = min(image.height, int(y2 + my))
            crop_items.append((f"img{img_id}/{i}",
                               image.crop((cx1, cy1, cx2, cy2))))
        feats = provider.encode_images_cached(crop_items)

        # spatial adjacency
        spatial_adj = defaultdict(list)
        for i in range(len(nmsd)):
            bi = _tight_bbox(nmsd[i]["mask"])
            if bi is None:
                continue
            for j in range(i + 1, len(nmsd)):
                bj = _tight_bbox(nmsd[j]["mask"])
                if bj is None:
                    continue
                if _iou_cropped(nmsd[i]["mask"], bi,
                                nmsd[j]["mask"], bj) >= RIVAL_IOU:
                    ki = f"img{img_id}/{i}"
                    kj = f"img{img_id}/{j}"
                    spatial_adj[ki].append(kj)
                    spatial_adj[kj].append(ki)

        # write cache records
        for i, d in enumerate(nmsd):
            f = feats[f"img{img_id}/{i}"]
            z_vec = {cn: round(float(np.dot(f, txt_feats[cn])), 6)
                     for cn in name_list}
            m_np = d["mask"]
            rle = mask_util.encode(np.asfortranarray(m_np.astype(np.uint8)))
            cache_f.write(json.dumps({
                "img_id": img_id,
                "cat_id": d["cat_id"],
                "cat_name": d["cat_name"],
                "raw_score": d["score"],
                "mask_rle": {"size": list(rle["size"]),
                             "counts": rle["counts"].decode("utf-8")},
                "bbox": list(d["bbox"]),
                "z": z_vec,
                "spatial_rivals": spatial_adj.get(f"img{img_id}/{i}", []),
            }) + "\n")

        done_f.write(json.dumps({"img_id": img_id}) + "\n")
        cache_f.flush()
        done_f.flush()
        torch.cuda.empty_cache()

        if (k + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {k + 1}/{len(todo)} imgs, {el / (k + 1):.1f}s/img",
                  flush=True)

    cache_f.close()
    done_f.close()
    el = time.time() - t0
    print(f"[cache] done: {len(todo)} imgs in {el:.0f}s -> {cache_path}",
          flush=True)


if __name__ == "__main__":
    main()
