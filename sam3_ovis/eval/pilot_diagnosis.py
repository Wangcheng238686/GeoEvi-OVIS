"""S3.5 pilot diagnosis: why does A2 underperform A1 on the 20-image subset?

Checks the OR-merge inflation hypothesis: clusters with many members get
inflated merged scores regardless of GT overlap, pushing FP clusters above
true singletons.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

from eval.run_ablation_eval import gt_masks_by_image, load_jsonl  # noqa: E402
from eval.score_eval import load_gt_coco  # noqa: E402

root = Path("/data/wangcheng/dataset/SCORE")
ab = Path("/home/wangcheng2021/project/GeoEvi-OVIS/sam3_ovis/logs/ablation/NWPU")
gt = gt_masks_by_image(load_gt_coco(str(root / "NWPU/val/instances_val.json")))
img_ids = set(range(1, 100))  # unused; keep linter calm


def best_iou(dets, gts, topn=10):
    if not gts:
        return None
    best = 0.0
    for d in sorted(dets, key=lambda x: -x["score"])[:topn]:
        r = dict(d["segmentation"])
        r["counts"] = r["counts"].encode()
        m = mask_util.decode(r).astype(bool)
        for _, gm in gts:
            inter = int((m & gm).sum())
            if inter:
                best = max(best, inter / float((m | gm).sum()))
    return best


def main():
    for v in ["A0", "A1", "A2", "A3-competition"]:
        preds = [p for p in load_jsonl(ab / f"{v}.jsonl")
                 if p["image_id"] in gt]
        by_pair = defaultdict(list)
        for p in preds:
            by_pair[(p["image_id"], p["category_id"])].append(p)
        pos_pairs = top10_hit = 0
        hi_tot = hi_hit = 0
        member_inflation = []
        for (iid, cid), dets in by_pair.items():
            gts = gt.get(iid, [])
            gts = [(c, m) for c, m in gts if c == cid]
            bi = best_iou(dets, gts)
            if bi is not None:
                pos_pairs += 1
                if bi >= 0.5:
                    top10_hit += 1
            for d in dets:
                if d["score"] >= 0.9:
                    hi_tot += 1
                    bi_one = best_iou([d], gts, topn=1)
                    if bi_one is not None and bi_one >= 0.5:
                        hi_hit += 1
        print(f"{v:14s} 有GT对={pos_pairs}  top10命中率={top10_hit}/{pos_pairs}"
              f"={top10_hit/max(pos_pairs,1):.2f}  "
              f"score>=0.9检测: {hi_hit}/{hi_tot}"
              f"={hi_hit/max(hi_tot,1)*100:.0f}% 命中GT")


if __name__ == "__main__":
    main()
