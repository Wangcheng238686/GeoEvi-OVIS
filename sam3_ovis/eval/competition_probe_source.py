"""T1 supplement: competition-mechanism trade-off probe on SOURCE-DOMAIN
real candidates (iSAID dev sample), per the project-leader directive
"在源域开发集的真实候选上，检查局部竞争机制能压低多少错类误检、会误伤
多少真目标，再决定进入 T2".

Mechanism (design-v3.0 §3, same formula as B2):
    rivals C_i = kept candidates of OTHER classes with mask IoU >= 0.5
    z_i(y)     = RemoteCLIP cos(crop_i, text y)
    d_i        = max(0, max_{y in C_i} z_i(y) - z_i(y_i) - delta)
    v_i        = u_i * exp(-lambda * d_i),   u_i = norm_text(b_i)

Measured vs iSAID GT (labels used ONLY for evaluation):
    wrong-class candidate (overlaps other-class GT >= 0.5):
        suppressed if v_i < u_i            -> "压低错类" (desired)
    TP candidate (overlaps same-class GT):
        suppressed if v_i < u_i            -> "误伤真目标" (cost)
Both fractions are reported per (delta, lambda) point; the no-rival case is
excluded from both denominators (mechanism is a no-op there by design).

Run:  source sam3_ovis/scripts/load_environment.sh
      "$PYTHON" sam3_ovis/eval/competition_probe_source.py [--n-images 300]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402
from graph.config import THRESHOLDS, norm_text  # noqa: E402

SEED = 20260905
NMS_IOU = 0.5
RIVAL_IOU = 0.5


def _arr(rle):
    r = dict(rle)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode("utf-8")
    return mask_util.decode(r).astype(bool)


def class_nms(dets, iou_thr=NMS_IOU):
    """dets carry a decoded bool 'mask' (SAM3 output in this probe)."""
    order = sorted(dets, key=lambda d: -d["score"])
    kept, kept_m = [], []
    for d in order:
        m = d["mask"]
        dup = False
        for km in kept_m:
            inter = int((m & km).sum())
            if inter and inter / float((m | km).sum()) >= iou_thr:
                dup = True
                break
        if not dup:
            kept.append(d)
            kept_m.append(m)
    return kept, kept_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=300)
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.0, 0.05])
    ap.add_argument("--lambdas", nargs="+", type=float,
                    default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from graph.provider import RemoteCLIPProvider
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    cpu_mode = not torch.cuda.is_available()
    if not cpu_mode:
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    spec = DATASETS["iSAID"]
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    gt = COCO(str(root / spec.val_json))
    cat_ids = gt.getCatIds()
    cat_names = {c["id"]: c["name"] for c in gt.loadCats(cat_ids)}

    rng = np.random.default_rng(SEED)
    img_ids = sorted(int(i) for i in gt.getImgIds() if i in
                     set(gt.getAnnIds() and [a["image_id"]
                         for a in gt.dataset["annotations"]]))
    sample = sorted(rng.choice(img_ids, size=min(args.n_images, len(img_ids)),
                               replace=False).tolist())
    manifest = {"seed": SEED, "n": len(sample), "image_ids": sample}
    Path(os.environ.get("SAM3_OVIS_LOG_ROOT", "/tmp")) \
        .mkdir(parents=True, exist_ok=True)
    Path(os.environ.get("SAM3_OVIS_LOG_ROOT", "/tmp"), "probe_manifest.json") \
        .write_text(json.dumps(manifest, indent=1))
    print(f"sample: {len(sample)} iSAID images (seed {SEED})")

    all_names = sorted({c["name"] for c in gt.loadCats(cat_ids)})

    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=THRESHOLDS.pool_floor)
    provider = RemoteCLIPProvider(device="cpu" if cpu_mode else None)
    txt = provider.encode_texts_cached(all_names)

    # per-candidate records
    out_json = Path(args.out or (Path(os.environ.get("SAM3_OVIS_LOG_ROOT", "/tmp"))
                                 / "competition_probe_iSAID.json"))
    recs_path = out_json.parent / "competition_probe_iSAID_records.jsonl"
    done_imgs = set()
    if recs_path.exists():
        with open(recs_path) as f:
            for line in f:
                try:
                    done_imgs.add(json.loads(line)["img"])
                except (json.JSONDecodeError, KeyError):
                    break

    recs_f = open(recs_path, "a")
    recs = []
    for k, img_id in enumerate(sample):
        if img_id in done_imgs:
            continue
        info = gt.loadImgs(img_id)[0]
        image = Image.open(root / spec.images_dir / info["file_name"]) \
            .convert("RGB")
        state = processor.set_image(image)

        kept, kept_m = [], []
        for cid in cat_ids:
            state = processor.set_text_prompt(state=state,
                                              prompt=cat_names[cid])
            sc = state["scores"]
            for j in range(int(sc.shape[0]) if sc is not None else 0):
                det = {"cat_id": cid, "cat_name": cat_names[cid],
                       "score": float(sc[j]),
                       "rle": None}
                m = state["masks"][j, 0].detach().cpu().numpy().astype(bool)
                det["mask"] = m
                kept.append(det)
        kept, kept_m = class_nms(kept)

        # rival structure among kept (cross-class, IoU >= 0.5)
        bb = [None] * len(kept)
        for i, m in enumerate(kept_m):
            ys, xs = np.any(m, axis=1), np.any(m, axis=0)
            if ys.any():
                bb[i] = (int(np.where(xs)[0][0]), int(np.where(ys)[0][0]),
                         int(np.where(xs)[0][-1]) + 1,
                         int(np.where(ys)[0][-1]) + 1)

        provider.clear_image_cache()
        crop_keys, crop_imgs = [], []
        for i, (d, m) in enumerate(zip(kept, kept_m)):
            key = f"img{img_id}/{i}"
            d["key"] = key
            crop_keys.append(key)
            crop_imgs.append(image)
        feats = provider.encode_images_cached(list(zip(crop_keys, crop_imgs)))

        # GT match classification
        gt_list = gt.loadAnns(gt.getAnnIds(imgIds=[img_id]))
        gt_m = []
        for a in gt_list:
            seg = a["segmentation"]
            if isinstance(seg, dict):
                r = dict(seg)
                if isinstance(r["counts"], str):
                    r["counts"] = r["counts"].encode("utf-8")
                gt_m.append((a["category_id"],
                             mask_util.decode(r).astype(bool)))
            else:
                objs = seg if isinstance(seg[0], list) else [seg]
                gt_m.append((a["category_id"], _arr({
                    "size": [info["height"], info["width"]],
                    "counts": mask_util.frPyObjects(
                        objs, info["height"], info["width"])})))

        for i, (d, m) in enumerate(zip(kept, kept_m)):
            best_same = best_cross = 0.0
            for cid, gm in gt_m:
                inter = int((m & gm).sum())
                if inter == 0:
                    continue
                iou = inter / float((m | gm).sum())
                if d["cat_id"] == cid:
                    best_same = max(best_same, iou)
                else:
                    best_cross = max(best_cross, iou)
            kind = ("tp" if best_same >= 0.5
                    else "wrong" if best_cross >= 0.5 else None)
            if kind is None:
                continue
            # distinct rival classes among overlapping kept candidates
            rival_classes = set()
            for j, (d2, m2) in enumerate(zip(kept, kept_m)):
                if d2["cat_id"] == d["cat_id"] or j == i:
                    continue
                if bb[i] is None or bb[j] is None:
                    continue
                if (bb[i][0] + bb[i][2] <= bb[j][0]
                        or bb[j][0] + bb[j][2] <= bb[i][0]
                        or bb[i][1] + bb[i][3] <= bb[j][1]
                        or bb[j][1] + bb[j][3] <= bb[i][1]):
                    continue
                inter = int((m & m2).sum())
                if inter and inter / float((m | m2).sum()) >= RIVAL_IOU:
                    rival_classes.add(d2["cat_name"])
            f = feats[d["key"]]
            z_own = float(np.dot(f, txt[d["cat_name"]]))
            z_rival_max = max((float(np.dot(f, txt[rc]))
                               for rc in rival_classes), default=None)
            rec = {"kind": kind, "u": norm_text(d["score"]),
                   "cat": d["cat_name"], "key": d["key"],
                   "img": img_id,
                   "z_own": z_own, "z_rival_max": z_rival_max,
                   "n_rival_classes": len(rival_classes)}
            recs.append(rec)
            recs_f.write(json.dumps(rec) + "\n")

        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(sample)} imgs, {len(recs)} scored recs",
                  flush=True)
        torch.cuda.empty_cache()

    recs_f.flush()
    recs_f.close()
    recs = []
    with open(recs_path) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))

    table = {}
    for delta in args.deltas:
        for lam in args.lambdas:
            s = defaultdict(lambda: [0, 0])  # kind -> [suppressed, total]
            for r in recs:
                if r["z_rival_max"] is None:
                    continue  # no rivals: mechanism is a no-op by design
                d_i = max(0.0, r["z_rival_max"] - r["z_own"] - delta)
                v = r["u"] * float(np.exp(-lam * d_i))
                s[r["kind"]][1] += 1
                s[r["kind"]][0] += int(v < r["u"])
            table[f"d{delta}_l{lam}"] = {
                "wrong_suppressed": s["wrong"][0],
                "wrong_total": s["wrong"][1],
                "tp_suppressed": s["tp"][0],
                "tp_total": s["tp"][1],
                "wrong_suppress_rate": round(
                    s["wrong"][0] / max(s["wrong"][1], 1), 4),
                "tp_suppress_rate": round(
                    s["tp"][0] / max(s["tp"][1], 1), 4),
            }
            print(f"delta={delta} lambda={lam}: "
                  f"错类压低 {s['wrong'][0]}/{s['wrong'][1]}"
                  f"={table[f'd{delta}_l{lam}']['wrong_suppress_rate']:.3f} | "
                  f"误伤TP {s['tp'][0]}/{s['tp'][1]}"
                  f"={table[f'd{delta}_l{lam}']['tp_suppress_rate']:.3f}",
                  flush=True)

    out = out_json
    out.write_text(json.dumps({"n_images": len(sample), "n_recs": len(recs),
                               "table": table,
                               "note": "rival z uses the rival class TEXT "
                                       "evaluated on this crop (design §3)"},
                              indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
