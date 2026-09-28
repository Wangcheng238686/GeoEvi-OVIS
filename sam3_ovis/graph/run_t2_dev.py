"""v3.0 T2 dev runner: end-to-end B0-B3 scoring sweep on an iSAID
source-image-grouped dev sample.

Protocol (implementation-v3.0.md T2, registered before running):
- Dev sample: iSAID train source-image groups (PXXXX prefix), seed 20260905,
  manifest written BEFORE any inference.
- Candidates: SAM3 text-only, 15-class vocab, pool floor 0.05, class-wise
  NMS (per class, raw-score order, IoU 0.5).
- Features: RemoteCLIP crop embeddings, bbox + 10% margin.
- Scoring: B0 (SAM3 monotone), B1 (independent CLIP+SAM3), B2 (spatial
  relative penalty), B3 (nospatial relative penalty). Grid: beta {0,.25,.5}
  x delta {0,.05} x lambda {0,1,2}.
- Output: per-variant predictions.jsonl + manifest.
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
from pycocotools.coco import COCO  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402
from graph.config import norm_text  # noqa: E402
from graph.provider import RemoteCLIPProvider  # noqa: E402

SEED = 20260905
NMS_IOU = 0.5
POOL_FLOOR = 0.05
CROP_MARGIN = 0.1
RIVAL_IOU = 0.5
BETAS = [0.0, 0.25, 0.5]
DELTAS = [0.0, 0.05]
LAMBDAS = [0.0, 1.0, 2.0]


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
    ap.add_argument("--out-dir", default=None)
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

    # 1) source-image-grouped dev sample
    rng = np.random.default_rng(SEED)
    prefixes = defaultdict(list)
    for im in gt.dataset["images"]:
        pref = im["file_name"].replace(".png", "").split("_")[0]
        prefixes[pref].append(im["id"])
    all_prefs = sorted(prefixes.keys())
    chosen = sorted(rng.choice(all_prefs, size=min(15, len(all_prefs)),
                               replace=False).tolist())
    dev_ids = sorted(i for p in chosen for i in prefixes[p])
    dev_ids = dev_ids[: args.n_images]
    out_dir = Path(args.out_dir or (log_root / "t2_dev"))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "source_groups": chosen,
                "n_images": len(dev_ids), "image_ids": dev_ids}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[t2] dev: {len(dev_ids)} imgs from {len(chosen)} source groups",
          flush=True)

    # 2) SAM3 inference + class-wise NMS + CLIP features
    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=POOL_FLOOR)
    provider = RemoteCLIPProvider()
    txt_feats = provider.encode_texts_cached(name_list)

    cands = []
    t0 = time.time()
    for k, img_id in enumerate(dev_ids):
        info = gt.loadImgs(img_id)[0]
        image = Image.open(root / spec.images_dir / info["file_name"]) \
            .convert("RGB")
        state = processor.set_image(image)

        # per class: SAM3 grounding -> pool-floor filter -> class-wise NMS
        kept, kept_m = [], []
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
                kept.append({"cat_id": cid, "cat_name": cat_names[cid],
                             "score": float(sc[j]), "mask": m, "bbox": b})
        # class-wise greedy NMS
        nmsd, nmsd_m = [], []
        for cid in sorted({d["cat_id"] for d in kept}):
            cls = sorted([d for d in kept if d["cat_id"] == cid],
                         key=lambda d: -d["score"])
            for d in cls:
                m, b = d["mask"], d["bbox"]
                dup = any(_iou_cropped(m, b, km["mask"], km["bbox"])
                          >= NMS_IOU for km in nmsd_m
                          if km["cat_id"] == cid)
                if not dup:
                    nmsd.append(d)
                    nmsd_m.append({"mask": m, "bbox": b,
                                   "cat_id": cid, "cat_name": d["cat_name"],
                                   "score": d["score"]})
        kept = nmsd

        # CLIP features (bbox+10% margin, image-scoped cache keys)
        provider.clear_image_cache()
        crop_items = []
        for i, d in enumerate(kept):
            key = f"{img_id}/{i}"
            d["key"] = key
            x1, y1, x2, y2 = d["bbox"]
            mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
            cx1 = max(0, int(x1 - mx))
            cy1 = max(0, int(y1 - my))
            cx2 = min(image.width, int(x2 + mx))
            cy2 = min(image.height, int(y2 + my))
            crop_items.append((key, image.crop((cx1, cy1, cx2, cy2))))
        feats = provider.encode_images_cached(crop_items)

        # per candidate: z_own + z_spatial_max + z_nospatial_max
        for i, d in enumerate(kept):
            f = feats[d["key"]]
            z_own = float(np.dot(f, txt_feats[d["cat_name"]]))
            z_nospatial = None
            spatial_rival_z = None
            for d2 in kept:
                if d2 is d or d2["cat_name"] == d["cat_name"]:
                    continue
                iou = _iou_cropped(d["mask"], d["bbox"],
                                   d2["mask"], d2["bbox"])
                if iou < RIVAL_IOU:
                    continue
                z2 = float(np.dot(f, txt_feats[d2["cat_name"]]))
                if spatial_rival_z is None or z2 > spatial_rival_z:
                    spatial_rival_z = z2
            for cn in name_list:
                if cn == d["cat_name"]:
                    continue
                z = float(np.dot(f, txt_feats[cn]))
                if z_nospatial is None or z > z_nospatial:
                    z_nospatial = z
            cands.append({
                "img_id": img_id, "cat_name": d["cat_name"],
                "raw_score": d["score"],
                "z_own": z_own,
                "z_spatial_max": spatial_rival_z,
                "z_nospatial_max": z_nospatial,
            })
        if (k + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {k + 1}/{len(dev_ids)} imgs, {el / (k + 1):.1f}s/img",
                  flush=True)

    print(f"[t2] inference done: {len(cands)} candidates", flush=True)

    # 3) B0-B3 scoring per candidate
    for c in cands:
        b = norm_text(c["raw_score"])
        zc = (c["z_own"] + 1) / 2
        u = 0.5 * b + 0.5 * zc
        scores = {"B0": b, "B1_b0.5": u}
        for beta in BETAS:
            uu = (1 - beta) * b + beta * zc
            scores[f"B1_b{beta}"] = uu
        for beta in BETAS:
            for delta in DELTAS:
                for lam in LAMBDAS:
                    if c["z_spatial_max"] is not None:
                        d_i = max(0.0, c["z_spatial_max"] - z_own - delta)
                        uu = u * float(np.exp(-lam * d_i))
                    else:
                        uu = u
                    scores[f"B2_b{beta}_d{delta}_l{lam}"] = uu
        for beta in BETAS:
            for delta in DELTAS:
                for lam in LAMBDAS:
                    if c["z_nospatial_max"] is not None:
                        d_i = max(0.0, c["z_nospatial_max"] - z_own - delta)
                        uu = u * float(np.exp(-lam * d_i))
                    else:
                        uu = u
                    scores[f"B3_b{beta}_d{delta}_l{lam}"] = uu
        c["scores"] = scores

    # 4) write per-variant predictions
    variants = sorted({k for c in cands for k in c["scores"]})
    for variant in variants:
        path = out_dir / f"predictions_{variant}.jsonl"
        with open(path, "w") as fout:
            for c in cands:
                fout.write(json.dumps({
                    "image_id": c["img_id"],
                    "category_name": c["cat_name"],
                    "score": round(c["scores"][variant], 6),
                }) + "\n")
    print(f"[t2] predictions -> {out_dir} ({len(variants)} variants)",
          flush=True)


if __name__ == "__main__":
    main()
