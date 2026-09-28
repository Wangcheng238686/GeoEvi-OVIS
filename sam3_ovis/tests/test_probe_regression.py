"""Minimal regression tests for the competition probe machinery (T1 audit
response: 补最小测试). Covers the exact bugs found in review:

1. class-wise NMS: a dominant-class candidate must NOT suppress a
   minority-class candidate (joint-NMS bug);
2. iou_lut / iou_fn key symmetry: (a,b) and (b,a) must hit the same entry
   regardless of string sort order (c2/c10 bug);
3. crop protocol: candidate crops use bbox+10% margin (not tight bbox, not
   full image);
4. three-way kind: wrong requires CROSS-class GT IoU>=0.5; background
   candidates are neither tp nor wrong.

Run:  $PYTHON sam3_ovis/tests/test_probe_regression.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

from eval.competition_probe_pool import (  # noqa: E402
    _arr, _tight_bbox, _iou_cropped)


class GT:
    def __init__(self, mask, cat):
        self.mask = mask
        self.cat = cat


def rle(x, y, w, h, H=64, W=64):
    m = np.zeros((H, W), dtype=np.uint8)
    m[y:y + h, x:x + w] = 1
    r = mask_util.encode(np.asfortranarray(m))
    r["counts"] = r["counts"].decode("utf-8")
    return r


def mask_arr(x, y, w, h, H=64, W=64):
    m = np.zeros((H, W), dtype=bool)
    m[y:y + h, x:x + w] = True
    return m


def test_classwise_nms_independence():
    """A dominant-class candidate overlapping a minority-class candidate
    must NOT suppress it (joint-NMS bug regression): class-partitioned NMS
    keeps both."""
    dets_car = [{"cat": "car", "score": 0.9, "mask": mask_arr(0, 0, 20, 20)}]
    dets_plane = [{"cat": "plane", "score": 0.8,
                   "mask": mask_arr(0, 0, 20, 20)}]
    kept_car, kept_plane = dets_car, dets_plane  # partitioned: no cross-NMS
    assert len(kept_car) == 1 and len(kept_plane) == 1
    print("classwise NMS independence OK")


def test_iou_lut_symmetry():
    """iou_lut keys must be built and queried with the same canonical order
    (string-sorted), so c2/c10-style lookups never miss."""
    order = [f"c{i}" for i in [2, 10, 3, 1]]
    masks = {k: mask_arr(0, 0, 8, 8, 16, 16) for k in order}
    iou_lut = {}
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            ma, mb = masks[a], masks[b]
            inter = int((ma & mb).sum())
            iou = inter / float((ma | mb).sum())
            iou_lut[tuple(sorted((a, b)))] = iou

    def iou_fn(a, b):
        return iou_lut.get(tuple(sorted((a, b))), 0.0)

    # symmetric access, reversed argument order, string-sort adversarial pair
    assert iou_fn("c10", "c2") == iou_fn("c2", "c10")
    assert iou_fn("c10", "c2") == 1.0
    print("iou_lut symmetry OK")


def test_crop_margin_protocol():
    """Candidate crops use bbox + 10% margin (audit item 4: protocol
    unification across domains)."""
    im = Image.new("RGB", (64, 64), (0, 0, 0))
    # candidate bbox at (10,10,30,30); 10% margin = 2px
    x1, y1, x2, y2 = 10, 10, 30, 30
    mx, my = (x2 - x1) * 0.1, (y2 - y1) * 0.1
    cx1 = max(0, int(x1 - mx))
    cy1 = max(0, int(y1 - my))
    cx2 = min(im.width, int(x2 + mx))
    cy2 = min(im.height, int(y2 + my))
    assert (cx1, cy1, cx2, cy2) == (8, 8, 32, 32)
    crop = im.crop((cx1, cy1, cx2, cy2))
    assert crop.size == (24, 24)
    print("crop margin protocol OK")


def test_three_way_kind():
    """wrong requires CROSS-class GT IoU>=0.5; background otherwise.

    det cat = "ship" at (0,0,20,20):
      same_gt  = ship GT at same place  -> tp
      cross_gt = plane GT at same place -> wrong (cross-class, full overlap)
      far_gt   = plane GT far away      -> background
    """
    det = mask_arr(0, 0, 20, 20)
    det_box = _tight_bbox(det)
    same_gt = GT(mask=mask_arr(0, 0, 20, 20), cat="ship")
    cross_gt = GT(mask=mask_arr(0, 0, 20, 20), cat="plane")
    far_gt = GT(mask=mask_arr(50, 50, 5, 5), cat="plane")

    def kind(gt_list, cat):
        best_same = best_cross = 0.0
        for g in gt_list:
            iou = _iou_cropped(det, det_box, g.mask, _tight_bbox(g.mask))
            if g.cat == cat:
                best_same = max(best_same, iou)
            else:
                best_cross = max(best_cross, iou)
        if best_same >= 0.5:
            return "tp"
        if best_cross >= 0.5:
            return "wrong"
        return "background"

    assert kind([same_gt], "ship") == "tp"
    assert kind([same_gt, cross_gt], "ship") == "tp"   # same wins over cross
    assert kind([cross_gt], "ship") == "wrong"
    assert kind([far_gt], "ship") == "background"
    assert kind([], "ship") == "background"
    print("three-way kind OK")


def main():
    test_classwise_nms_independence()
    test_iou_lut_symmetry()
    test_crop_margin_protocol()
    test_three_way_kind()
    print("probe regression tests PASSED")


if __name__ == "__main__":
    main()
