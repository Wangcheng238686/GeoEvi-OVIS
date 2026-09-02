"""Self-test for sam3_ovis/eval/score_eval.py with forged predictions (S1.4).

Builds a tiny synthetic COCO GT (2 images, 2 classes), then checks:
1. exact-copy predictions  -> mAP ~ 1.0, recall/precision proxies ~ 1.0,
   conditional IoU ~ 1.0;
2. degraded predictions    -> each diagnostic moves by the predicted amount:
   - dropping one cat-1 GT instance lowers cat-1 recall proxy to 0.5,
   - adding an off-target cat-2 false detection lowers cat-2 precision proxy,
   - shifting one cat-1 mask halves its IoU -> conditional IoU < 1.

Run:  $PYTHON sam3_ovis/tests/test_score_eval.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.score_eval import run_score_eval  # noqa: E402

H = W = 64


def square_rle(x, y, w, h):
    m = np.zeros((H, W), dtype=np.uint8)
    m[y:y + h, x:x + w] = 1
    rle = mask_util.encode(np.asfortranarray(m))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def build_gt(tmp: Path) -> str:
    cats = [{"id": 1, "name": "plane"}, {"id": 2, "name": "ship"}]
    images = [
        {"id": 1, "height": H, "width": W, "file_name": "a.png"},
        {"id": 2, "height": H, "width": W, "file_name": "b.png"},
    ]
    rects = [
        (1, 1, 0, 0, 20, 20),   # img1 cat1
        (1, 1, 30, 30, 20, 20),  # img1 cat1
        (1, 2, 10, 40, 15, 15),  # img1 cat2
        (2, 1, 5, 5, 30, 30),    # img2 cat1
    ]
    annotations = []
    for i, (img_id, cat_id, x, y, w, h) in enumerate(rects, start=1):
        annotations.append({
            "id": i, "image_id": img_id, "category_id": cat_id,
            "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
            "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
        })
    gt = {"images": images, "annotations": annotations, "categories": cats}
    gt_path = tmp / "gt.json"
    gt_path.write_text(json.dumps(gt))
    return str(gt_path)


def dt(img_id, cat_id, score, rle):
    return {"image_id": img_id, "category_id": cat_id, "score": score,
            "segmentation": rle}


def approx(a, b, tol=1e-6):
    return a is not None and abs(a - b) <= tol


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sam3_ovis_eval_test_"))
    gt_path = build_gt(tmp)

    # --- case 1: exact copies -------------------------------------------
    perfect = [
        dt(1, 1, 0.9, square_rle(0, 0, 20, 20)),
        dt(1, 1, 0.8, square_rle(30, 30, 20, 20)),
        dt(1, 2, 0.7, square_rle(10, 40, 15, 15)),
        dt(2, 1, 0.6, square_rle(5, 5, 30, 30)),
    ]
    res = run_score_eval(gt_path, predictions=perfect)
    s = res["summary"]
    assert approx(s["segm/mAP"], 1.0, 1e-6), f"perfect mAP != 1: {s['segm/mAP']}"
    d = res["diagnostics"]
    assert approx(d["recall_proxy_class_avg"], 1.0), d["recall_proxy_class_avg"]
    assert approx(d["precision_proxy_class_avg"], 1.0), d["precision_proxy_class_avg"]
    assert approx(d["conditional_iou_class_avg"], 1.0), d["conditional_iou_class_avg"]

    # --- case 2: degraded -------------------------------------------------
    degraded = [
        # img1 cat1: keep only the first instance (one miss)
        dt(1, 1, 0.9, square_rle(0, 0, 20, 20)),
        # img1 cat1 duplicate of first instance (false positive under 1-1 match)
        dt(1, 1, 0.85, square_rle(0, 0, 20, 20)),
        # img2 cat1 shifted mask (IoU 24*30 / 1080 = 2/3 with GT, >= 0.5)
        dt(2, 1, 0.6, square_rle(11, 5, 30, 30)),
        # img1 cat2 detection at empty location (pure false positive)
        dt(1, 2, 0.5, square_rle(0, 0, 10, 10)),
    ]
    res2 = run_score_eval(gt_path, predictions=degraded)
    d2 = res2["diagnostics"]
    pc = d2["per_class"]

    # cat1: 3 GT, hits = img1 first + img2 shifted (IoU 24/36 = 2/3)
    assert pc["plane"]["n_gt"] == 3 and pc["plane"]["n_dt"] == 3, pc["plane"]
    expect_iou = (24 * 30) / (900 + 900 - 24 * 30)
    assert abs(pc["plane"]["conditional_iou"] - (1.0 + expect_iou) / 2) < 1e-6, (
        pc["plane"]["conditional_iou"], expect_iou)
    assert approx(pc["ship"]["precision_proxy@0.50"], 0.0), pc["ship"]
    assert approx(pc["ship"]["recall_proxy@0.50"], 0.0), pc["ship"]
    # class-averaged recall: cat1 = 2/3, cat2 = 0/1
    assert abs(d2["recall_proxy_class_avg"] - (2 / 3 + 0.0) / 2) < 1e-6, (
        d2["recall_proxy_class_avg"])
    # class-averaged precision: cat1 = 2/3 (dup fp), cat2 = 0/1
    assert abs(d2["precision_proxy_class_avg"] - (2 / 3 + 0.0) / 2) < 1e-6, (
        d2["precision_proxy_class_avg"])
    # degraded mAP strictly below perfect
    assert res2["summary"]["segm/mAP"] < 0.9, res2["summary"]["segm/mAP"]

    print("score_eval self-test PASSED")
    print("perfect mAP:", res["summary"]["segm/mAP"],
          "| degraded mAP:", res2["summary"]["segm/mAP"])
    print("degraded diagnostics:", {k: v for k, v in d2.items() if k != "per_class"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
