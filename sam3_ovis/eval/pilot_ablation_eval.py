"""S3.5 pilot evaluation on a 20-image NWPU subset (early signal only)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_ablation_eval import (gt_masks_by_image, paired_bootstrap,  # noqa: E402
                                    per_image_maps, load_jsonl)
from eval.score_eval import (compute_diagnostics, evaluate_dataset,  # noqa: E402
                             load_gt_coco)

root = Path("/data/wangcheng/dataset/SCORE")
log = Path("/home/wangcheng2021/project/GeoEvi-OVIS/sam3_ovis/logs")
gt = load_gt_coco(str(root / "NWPU/val/instances_val.json"))
img_ids = gt.getImgIds()[:20]
keep = set(img_ids)

from pycocotools.coco import COCO  # noqa: E402

sub = COCO()
sub.dataset = {
    "images": [im for im in gt.dataset["images"] if im["id"] in keep],
    "annotations": [a for a in gt.dataset["annotations"]
                    if a["image_id"] in keep],
    "categories": gt.dataset["categories"],
}
sub.createIndex()
gt_masks = gt_masks_by_image(sub)

ap_by, res = {}, {}
for v in ["A0", "A1", "A2",
          "A3-competition", "A3-context", "A3-scene", "A3-risk"]:
    preds = [p for p in load_jsonl(log / "ablation/NWPU" / f"{v}.jsonl")
             if p["image_id"] in keep]
    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)
    s, _ = evaluate_dataset(sub, preds)
    d = compute_diagnostics(sub, preds)
    ap_by[v] = per_image_maps(gt_masks, preds_by_img)
    pi = sum(ap_by[v].values()) / max(len(ap_by[v]), 1)
    res[v] = pi
    print(f"{v:14s} mAP={s['segm/mAP']:.4f} AP50={s['segm/mAP_50']:.4f} "
          f"G1={d['recall_proxy_class_avg']:.4f} per-imgAP={pi:.4f}")

bs = paired_bootstrap(ap_by, img_ids)
for pair, st in bs.items():
    if st["mean_delta"] is not None:
        print(f"{pair}: per-image AP delta {st['mean_delta']:+.4f} "
              f"(95% CI {st['ci95'][0]:+.4f}..{st['ci95'][1]:+.4f}, "
              f"n={st['n_images']})")
