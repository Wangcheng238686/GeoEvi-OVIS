#!/usr/bin/env python3
"""G5b：基线 mAP 损失拆解（CPU，读 g5a 已落盘候选，不重跑模型）。

回答"基线 mAP 为什么低"：把 COMBO 臂的 mAP 分解为可归因的分量。

- D1 口径敏感性：maxDets（默认 [1,10,100] vs 300/1000）——每图 290 个
  NMS 后候选，默认 100 上限是否截断了低分 TP；
- D2 mask vs box：同一批候选 segm mAP vs bbox mAP——差距 = mask 质量
  损失；再对 GT 做 oracle 框重标（候选 bbox 换成其最佳 IoU≥0.5 的 GT
  类）看 box 上限；
- D3 IoU 阈值曲线：AP50/AP75/AP@[.5:.95] + 逐阈值 AP；
- D4 oracle 系列（逐步替换，每步只改一个因素）：
  cls*   候选类别换成其最佳 IoU GT 的类别（IoU≥0.5 才换）
  score* 在 cls* 之上，分数换成 oracle（匹配 GT 者 = 1−IoU 损失，
  其余 = 0）——即"完美排序"
- D5 逐类 AP：哪些类拖低均值；
- D6 GT 侧：GT 实例数分布、被 crowd/小面积影响、每图 GT 数。

确定性：same-class 召回用"任一同类候选覆盖"口径（任一，不依赖顺序），
修正 g5a 的首命中即 break 的顺序依赖缺陷。

Run:  source sam3_ovis/scripts/load_environment.sh
      CUDA_VISIBLE_DEVICES="" "$PYTHON" sam3_ovis/graph/g5b_baseline_breakdown.py
"""

import contextlib
import io
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402
from graph.g1_qbridge_probe import box_iou_xyxy  # noqa: E402
from graph.g5a_free_lever import OUT_ROOT, nms_image  # noqa: E402

LOG_ROOT = Path(os.environ.get(
    "SAM3_OVIS_LOG_ROOT",
    str(Path(__file__).resolve().parents[1] / "logs")))
ARM = os.environ.get("G5B_ARM", "COMBO")
IOU_THRS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def evaluate(sub, dets, iou_type="segm", max_dets=None, per_cat=False):
    if not dets:
        return 0.0, None
    dt = sub.loadRes([dict(d) for d in dets])
    ev = COCOeval(sub, dt, iouType=iou_type)
    if max_dets is not None:
        ev.params.maxDets = list(max_dets)
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    # stats 索引：0 mAP, 1 AP50, 2 AP75, 3 APs, 4 APm, 5 APl, 6 AR@1,
    # 7 AR@10, 8 AR@maxDets[2], 9..11 ARs/m/l
    out = {"mAP": float(ev.stats[0]), "AP50": float(ev.stats[1]),
           "AP75": float(ev.stats[2]),
           "AP_s": float(ev.stats[3]), "AP_m": float(ev.stats[4]),
           "AP_l": float(ev.stats[5]),
           "AR100": float(ev.stats[8]),
           }
    # COCOeval 的类别维（K）按 cat_id **升序**排列
    cat_ids_sorted = sorted(c["id"] for c in sub.dataset["categories"])
    id2name = {c["id"]: c["name"] for c in sub.dataset["categories"]}
    prec_all = ev.eval["precision"]
    # 只在有 GT 的类上平均（无 GT 类的 precision 恒为 -1，会把均值拉坏）
    has_gt = (prec_all[0, 0, :, 0, -1] >= 0) | (prec_all[0, :, :, 0, -1] >= 0).any(axis=0)
    out["n_classes_in_mean"] = int(has_gt.sum())
    out["per_thr"] = [float(x) if x >= 0 else None
                      for x in prec_all[:, :, has_gt, 0, -1].mean(axis=(1, 2))]
    out["mAP_check"] = round(float(np.nanmean(
        [x for x in out["per_thr"] if x is not None])), 4)
    if per_cat:
        prec = ev.eval["precision"]
        apc = {}
        for k, cid in enumerate(cat_ids_sorted):
            v = prec[:, :, k, 0, -1]
            v = v[v >= 0]
            apc[id2name[cid]] = round(float(v.mean()), 4) if v.size else None
        out["per_class"] = apc
        out["per_class_AP50"] = {
            id2name[cid]: (round(float(prec[0, :, k, 0, -1][
                prec[0, :, k, 0, -1] >= 0].mean()), 4)
                if (prec[0, :, k, 0, -1] >= 0).any() else None)
            for k, cid in enumerate(cat_ids_sorted)}
    return out["mAP"], out


def main():
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DATASETS["iSAID"]
    gt = COCO(str(root / spec.val_json))
    name_to_id = {c["name"]: c["id"] for c in gt.loadCats(gt.getCatIds())}
    id_to_name = {v: k for k, v in name_to_id.items()}

    arm_dir = OUT_ROOT / ARM
    recs = [json.loads(p.read_text()) for p in sorted(arm_dir.glob("cands_*.json"))]
    img_ids = [r["img_id"] for r in recs]
    sub = COCO()
    sub.dataset = {
        "images": [im for im in gt.dataset["images"] if im["id"] in set(img_ids)],
        "annotations": [a for a in gt.dataset["annotations"]
                        if a["image_id"] in set(img_ids)],
        "categories": gt.dataset["categories"]}
    sub.createIndex()
    gt_by_img = defaultdict(list)
    for a in sub.dataset["annotations"]:
        gt_by_img[a["image_id"]].append(a)
    cat_names = [c["name"] for c in sub.dataset["categories"]]

    # ---- NMS 池 + 逐候选 GT 匹配 ----
    dets_base, cand_meta = [], []
    for r in recs:
        kept = nms_image(r["cands"])
        gts = gt_by_img.get(r["img_id"], [])
        for c in kept:
            best_iou, best_cat = 0.0, None
            for a in gts:
                gx, gy, gw, gh = a["bbox"]
                iou = box_iou_xyxy(c["box"], (gx, gy, gx + gw, gy + gh))
                if iou > best_iou:
                    best_iou, best_cat = iou, a["category_id"]
            dets_base.append({"image_id": r["img_id"],
                              "category_id": name_to_id[c["cls"]],
                              "segmentation": c["rle"], "score": c["score"]})
            cand_meta.append({"best_iou": best_iou, "best_cat": best_cat,
                              "pred_cat": name_to_id[c["cls"]]})
    n_kept = len(dets_base)

    res = {"arm": ARM, "n_imgs": len(recs), "n_kept": n_kept,
           "kept_per_img": round(n_kept / len(recs), 1),
           "n_gt": len(sub.dataset["annotations"])}

    # ---- D1 maxDets 截断 ----
    res["D1_maxDets"] = {}
    for md, tag in ((None, "default[1,10,100]"), ([1, 10, 300], "300"),
                    ([1, 10, 1000], "1000")):
        _, o = evaluate(sub, dets_base, max_dets=md)
        res["D1_maxDets"][tag] = {"mAP": o["mAP_check"],
                                  "AR@max": round(o["AR100"], 4)}

    # ---- D2 segm vs bbox ----
    _, o_seg = evaluate(sub, dets_base, "segm", [1, 10, 1000])
    _, o_box = evaluate(sub, dets_base, "bbox", [1, 10, 1000])
    o_seg["mAP"], o_box["mAP"] = o_seg["mAP_check"], o_box["mAP_check"]
    res["D2_segm_vs_bbox"] = {
        "segm": {k: round(v, 4) for k, v in o_seg.items()
                 if k in ("mAP", "AP50", "AP75", "AP_s", "AP_m", "AP_l")},
        "bbox": {k: round(v, 4) for k, v in o_box.items()
                 if k in ("mAP", "AP50", "AP75", "AP_s", "AP_m", "AP_l")}}
    res["D3_per_threshold_segm"] = {
        f"{t:.2f}": (round(v, 4) if v is not None else None)
        for t, v in zip(IOU_THRS, o_seg["per_thr"])}
    res["D3_per_threshold_bbox"] = {
        f"{t:.2f}": (round(v, 4) if v is not None else None)
        for t, v in zip(IOU_THRS, o_box["per_thr"])}

    # ---- D4 oracle 系列（每级只改一个因素；匹配一律用 **mask IoU + 同类**，
    # 修复首轮缺陷：bbox-IoU 选 GT 在密集场景会选到相邻实例，导致 mask* /
    # score* 反而低于 base）----
    from pycocotools import mask as _mu

    # 逐候选：同类 GT 中 mask IoU 最大者
    best = []  # (iou, gt_ann or None)
    for r, c_list in zip(recs, [nms_image(r["cands"]) for r in recs]):
        gts = gt_by_img.get(r["img_id"], [])
        same = defaultdict(list)
        for a in gts:
            same[a["category_id"]].append(a)
        for c in c_list:
            cid = name_to_id[c["cls"]]
            bi, ba = 0.0, None
            if same.get(cid):
                cand_rle = {"size": c["rle"]["size"],
                            "counts": c["rle"]["counts"].encode()}
                # SCORE 版 iSAID 的 segmentation 是已压缩 RLE 字典
                # （frPyObjects 不认）；annToRLE 兼容 polygon/RLE 两种
                grles = [gt.annToRLE(a) for a in same[cid]]
                ious = _mu.iou([cand_rle], grles,
                               [a.get("iscrowd", 0) for a in same[cid]])[0]
                k = int(np.argmax(ious))
                if ious[k] > 0:
                    bi, ba = float(ious[k]), same[cid][k]
            best.append((bi, ba))

    d_cls = [dict(d) for d in dets_base]
    for d, m in zip(d_cls, cand_meta):
        if m["best_iou"] >= 0.5 and m["best_cat"] is not None:
            d["category_id"] = m["best_cat"]

    # score*（在 cls* 之上）：按 mask IoU 排序——同类 mask IoU≥0.5 者分数
    # = 其 IoU，其余 = 极小。这是"完美定位质量排序"的近似 oracle。
    d_score = []
    for d, (bi, ba) in zip(d_cls, best):
        dd = dict(d)
        dd["score"] = bi if bi >= 0.5 else 1e-6
        d_score.append(dd)

    # mask*（在 cls* 之上，分数原样）：把同类 mask IoU 在 [0.3,0.5) 的
    # 候选 mask 替换为其匹配 GT 的 mask（只救"定位对、轮廓差"的候选，
    # 不触碰类别错的候选，避免引入类别信息）
    d_mask = []
    n_mask_fixed = 0
    for d, (bi, ba) in zip(d_cls, best):
        dd = dict(d)
        if ba is not None and 0.3 <= bi < 0.5:
            dd["segmentation"] = ba["segmentation"]
            n_mask_fixed += 1
        d_mask.append(dd)

    res["D4_oracle"] = {}
    for tag, dd in (("base", dets_base),
                    ("cls*", d_cls),
                    ("score*(+cls*)", d_score),
                    ("mask-fix[.3,.5)(+cls*)", d_mask)):
        _, o = evaluate(sub, dd, "segm", [1, 10, 1000])
        # 非标准 maxDets 下 stats[0] 为 -1 占位，改用阈值曲线均值（=mAP）
        res["D4_oracle"][tag] = o["mAP_check"]
    res["D4_n_mask_fixed"] = n_mask_fixed

    # ---- D5 逐类 AP ----
    _, o_full = evaluate(sub, dets_base, "segm", [1, 10, 1000], per_cat=True)
    _, o_full_box = evaluate(sub, dets_base, "bbox", [1, 10, 1000], per_cat=True)
    n_gt_cat = Counter(a["category_id"] for a in sub.dataset["annotations"])
    n_det_cat = Counter(d["category_id"] for d in dets_base)
    res["D5_per_class"] = {
        k: {"AP_segm": o_full["per_class"][k],
            "AP50_segm": o_full["per_class_AP50"][k],
            "AP_bbox": o_full_box["per_class"][k],
            "n_gt": n_gt_cat.get(name_to_id[k], 0),
            "n_det": n_det_cat.get(name_to_id[k], 0)}
        for k in sorted(o_full["per_class"],
                        key=lambda k: n_gt_cat.get(name_to_id[k], 0),
                        reverse=True)}
    # 无 GT 的类不进 COCO mAP 平均（登记：mAP 的有效类数）
    res["D5_classes_in_mAP"] = sum(
        1 for k, v in res["D5_per_class"].items() if v["n_gt"] > 0)
    res["D5_mAP_over_present_classes_only"] = round(float(np.mean(
        [v["AP_segm"] for v in res["D5_per_class"].values()
         if v["n_gt"] > 0 and v["AP_segm"] is not None])), 4)

    # ---- D6 GT 侧 + 匹配统计（确定性口径）----
    areas = [a["area"] for a in sub.dataset["annotations"]]
    per_img_gt = [len(gt_by_img[i]) for i in img_ids]
    matched = [m for m in cand_meta if m["best_iou"] >= 0.5]
    res["D6"] = {
        "gt_area_pct": {"<32²": round(100 * np.mean(np.array(areas) < 1024), 1),
                        "32²-96²": round(100 * np.mean(
                            (np.array(areas) >= 1024) & (np.array(areas) < 9216)), 1),
                        "≥96²": round(100 * np.mean(np.array(areas) >= 9216), 1)},
        "gt_area_median": float(np.median(areas)),
        "gt_per_img_median": float(np.median(per_img_gt)),
        "gt_per_img_max": int(np.max(per_img_gt)),
        "kept_match_any_gt_pct": round(100 * len(matched) / n_kept, 1),
        "kept_match_correct_cls_pct": round(
            100 * sum(1 for m in matched if m["best_cat"] == m["pred_cat"])
            / max(len(matched), 1), 1),
        "iscrowd_n": sum(a.get("iscrowd", 0) for a in sub.dataset["annotations"]),
    }

    out_dir = LOG_ROOT / "g5b"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"breakdown_{ARM}.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
