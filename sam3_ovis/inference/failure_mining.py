"""S2.4 failure mining: categorize and visualize failure cases.

Registered categories (implementation-v2 S2.4):
  miss    漏检   - GT instance with no same-class candidate at IoU>=0.5
  false   误检   - candidate unmatched to any same-class GT (greedy 1-1)
  maskq   mask质量 - matched pair with IoU in [0.5, 0.75)  (对但形状差)
  dup     重复   - candidate matched to a GT that an earlier (higher-score)
                   candidate already claimed (multi-candidate per instance)

For each requested dataset: stratified sample up to --per-class N cases per
category, render crop visualizations (image + GT contour + pred contour) into
notes/reports/figures/<dataset>/<category>/, and write a counts table json.

Run after S2.2 predictions exist:
  source sam3_ovis/scripts/load_environment.sh
  "$PYTHON" sam3_ovis/inference/failure_mining.py --dataset NWPU
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
from pycocotools import mask as mask_util  # noqa: E402

HIGH_IOU = 0.75  # matched pairs below this count as mask-quality failures


def ann_rle(ann, h, w):
    seg = ann["segmentation"]
    if isinstance(seg, dict):
        rle = dict(seg)
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return rle
    objs = seg if isinstance(seg[0], list) else [seg]
    return mask_util.merge(mask_util.frPyObjects(objs, h, w))


def mine_image(gt_anns, dts, h, w):
    """gt_anns: list of ann dicts; dts: list of (score, rle). Returns
    categorized cases with coordinates for cropping."""
    cases = {"miss": [], "false": [], "maskq": [], "dup": []}
    claimed = {}
    for score, rle in sorted(dts, key=lambda x: -x[0]):
        best_iou, best_g = 0.0, None
        for gi, ga in enumerate(gt_anns):
            iou = float(mask_util.iou([rle], [ga["rle"]], [0])[0][0])
            if iou > best_iou:
                best_iou, best_g = iou, gi
        if best_iou >= 0.5 and best_g is not None:
            if best_g in claimed:
                cases["dup"].append((score, rle, best_iou, claimed[best_g]))
            else:
                claimed[best_g] = (score, rle)
                if best_iou < HIGH_IOU:
                    cases["maskq"].append((score, rle, best_iou, best_g))
        else:
            cases["false"].append((score, rle, best_iou, best_g))
    for gi, ga in enumerate(gt_anns):
        if gi not in claimed:
            cases["miss"].append(ga)
    return cases


def render(image, gt_mask, pred_mask, out_path):
    import cv2

    img = np.array(image).copy()
    if gt_mask is not None:
        gm = mask_util.decode(gt_mask) if isinstance(gt_mask, dict) else gt_mask
        contours, _ = cv2.findContours(
            gm.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 220, 0), 2)   # GT = green
    if pred_mask is not None:
        pm = (mask_util.decode(pred_mask) if isinstance(pred_mask, dict)
              else pred_mask)
        contours, _ = cv2.findContours(
            pm.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 2)   # pred = red
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--per-class", type=int, default=12,
                    help="max visualization crops per failure category")
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    from PIL import Image
    from pycocotools.coco import COCO

    from data.score_datasets import DATASETS

    spec = DATASETS[args.dataset]
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    gt = COCO(str(root / spec.val_json))
    h_w = {i: (im["height"], im["width"]) for i, im in gt.imgs.items()}
    id2name = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}

    dts_by = defaultdict(list)
    with open(Path(os.environ["SAM3_OVIS_LOG_ROOT"]) / "zeroshot"
              / args.dataset / "predictions.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rle = dict(d["segmentation"])
            rle["counts"] = rle["counts"].encode("utf-8")
            dts_by[d["image_id"]].append((d["score"], rle, d["category_id"]))

    img_ids = gt.getImgIds()
    if args.max_images:
        img_ids = img_ids[: args.max_images]

    counts = defaultdict(int)
    saved = defaultdict(int)
    fig_root = (Path(os.environ["SAM3_OVIS_MAINLINE_ROOT"]).parent
                / "notes" / "reports" / "figures" / args.dataset)

    for img_id in img_ids:
        info = gt.loadImgs(img_id)[0]
        h, w = h_w[img_id]
        by_cat_gt = defaultdict(list)
        for ann in gt.loadAnns(gt.getAnnIds(imgIds=[img_id])):
            by_cat_gt[ann["category_id"]].append(
                {"rle": ann_rle(ann, h, w), "ann": ann})
        for cat_id, gts in by_cat_gt.items():
            dts = [(s, r) for s, r, c in dts_by.get(img_id, [])
                   if c == cat_id]
            cases = mine_image(gts, dts, h, w)
            for cat_key in ("miss", "false", "maskq", "dup"):
                counts[cat_key] += len(cases[cat_key])
                if saved[cat_key] >= args.per_class or not cases[cat_key]:
                    continue
                image = Image.open(
                    root / spec.images_dir / info["file_name"]).convert("RGB")
                for case in cases[cat_key][: args.per_class - saved[cat_key]]:
                    saved[cat_key] += 1
                    sub = fig_root / cat_key
                    sub.mkdir(parents=True, exist_ok=True)
                    out = sub / f"{img_id}_{cat_id}_{saved[cat_key]}.png"
                    if cat_key == "miss":
                        gm = case["rle"]
                        render(image, gm, None, out)
                    else:
                        _, rle, iou, _ = case
                        render(image, None, rle, out)

    table = {"dataset": args.dataset, "images_scanned": len(img_ids),
             "counts": dict(counts), "figures_saved": dict(saved),
             "note": "counts over scanned images; figures capped at "
                     f"{args.per_class}/category"}
    out = (Path(os.environ["SAM3_OVIS_LOG_ROOT"]) / "zeroshot"
           / f"{args.dataset}_failures.json")
    out.write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
