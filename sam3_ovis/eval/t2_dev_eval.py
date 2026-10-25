"""T2 dev evaluation: per-variant greedy IoU-0.5 matching (precision/recall/F1)."""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pycocotools.coco import COCO

from data.score_datasets import DATASETS


def main():
    root = Path(os.environ.get("SAM3_OVIS_DATA_ROOT", "/data/wangcheng/dataset/SCORE"))
    spec = DATASETS["iSAID"]
    gt = COCO(str(root / spec.val_json))
    manifest = json.load(open(
        str(Path(os.environ.get("SAM3_OVIS_LOG_ROOT", "/tmp")) / "t2_dev/manifest.json")))
    dev_ids = set(manifest["image_ids"])

    sub = COCO()
    sub.dataset = {
        "images": [im for im in gt.dataset["images"] if im["id"] in dev_ids],
        "annotations": [a for a in gt.dataset["annotations"]
                        if a["image_id"] in dev_ids],
        "categories": gt.dataset["categories"],
    }
    sub.createIndex()
    total_gt = len(sub.dataset["annotations"])

    t2_dir = Path(os.environ.get("SAM3_OVIS_LOG_ROOT", "/tmp")) / "t2_dev"
    variants = {}
    for p in sorted(t2_dir.glob("predictions_*.jsonl")):
        vname = p.stem.replace("predictions_", "")
        dets = []
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["image_id"] not in dev_ids:
                continue
            cid = next((c["id"] for c in sub.dataset["categories"]
                        if c["name"] == r["category_name"]), None)
            if cid is not None:
                dets.append({"image_id": r["image_id"],
                             "category_id": cid, "score": r["score"]})
        variants[vname] = dets

    print(f"GT annotations: {total_gt}")
    print(f"{'variant':24s} {'TP':>6s} {'FP':>7s} {'prec':>7s} {'recall':>7s} {'F1':>7s}")

    for vname in sorted(variants.keys()):
        dets = variants[vname]
        by_pair = defaultdict(lambda: defaultdict(list))
        for d in dets:
            by_pair[d["image_id"]][d["category_id"]].append(d)

        tp = fp = 0
        for img_id in sorted(dev_ids):
            for cat_id in sorted({d["category_id"] for d in dets
                                  if d["image_id"] == img_id}):
                gt_anns = sub.loadAnns(sub.getAnnIds(imgIds=[img_id],
                                                     catIds=[cat_id]))
                gt_boxes = [a["bbox"] for a in gt_anns]
                dt = sorted(by_pair[img_id].get(cat_id, []),
                            key=lambda d: -d["score"])
                matched = [False] * len(gt_boxes)
                for d in dt:
                    bb = d["bbox"]
                    best, bi = 0.0, -1
                    for j, gb in enumerate(gt_boxes):
                        if matched[j]:
                            continue
                        ix1 = max(bb[0], gb[0])
                        iy1 = max(bb[1], gb[1])
                        ix2 = min(bb[0] + bb[2], gb[0] + gb[2])
                        iy2 = min(bb[1] + bb[3], gb[1] + gb[3])
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        union = (bb[2] * bb[3]) + (gb[2] * gb[3]) - inter
                        if union <= 0:
                            continue
                        iou = inter / union
                        if iou > best:
                            best, bi = iou, j
                    if best >= 0.5 and bi >= 0:
                        matched[bi] = True
                        tp += 1
                    else:
                        fp += 1
        prec = tp / max(tp + fp, 1)
        rec = tp / max(total_gt, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"{vname:24s} TP={tp:5d} FP={fp:6d} P={prec:.3f} R={rec:.3f} "
              f"F1={f1:.3f}")


if __name__ == "__main__":
    main()
