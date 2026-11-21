#!/usr/bin/env python3
"""G4b 管线协议审计（CPU 部分）。报告：
notes/reports/2026-11-22-pipeline-protocol-audit.md。

复算 G4a §C 的"45% 召回墙"是否为 bbox-IoU≥0.5 口径伪影：
1. 三口径召回：bbox-IoU≥0.5（G4a 原口径）/ mask-IoU≥0.3 / GT 中心点命中
   （GT bbox 中心落在候选 mask 内）；另附 mask-IoU≥0.1 与任像素重叠诊断。
2. raw 池 vs 类内 NMS 池（G4a 的召回在 NMS 池上算）。
3. GT 面积分桶 / 逐类分解；漏检 GT 的最佳候选分数分布（floor 敏感性）。
4. FP/图在 bbox 口径 vs mask 口径下的差异。
5. bootstrap（图级重采样，10000 次）给全部百分点 95% CI；
   A0 (NMS 池) 全局 mAP 与逐图 AP 均值的 CI；maxDets=100 截断核查。

Run:  source sam3_ovis/scripts/load_environment.sh
      "$PYTHON" sam3_ovis/graph/g4b_pipeline_audit.py
"""

import argparse
import contextlib
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402
from graph.g1_qbridge_probe import box_iou_xyxy

CACHE_ROOT = Path("/data/wangcheng/geoevi-g2a-cache")
CAND_DIR = CACHE_ROOT / "cands"
LOG_ROOT = Path(os.environ.get(
    "SAM3_OVIS_LOG_ROOT",
    str(Path(__file__).resolve().parents[1] / "logs")))
OUT_DIR = LOG_ROOT / "g4b"
NMS_IOU = 0.5
BOOT_N = 10000
SEED = 20261122


def load_cands():
    man_ids = {int(k) for k in json.load(
        open(CACHE_ROOT / "manifest.json"))["images"]}
    out = []
    for p in sorted(CAND_DIR.glob("cands_*.json")):
        rec = json.loads(p.read_text())
        if rec["img_id"] in man_ids:
            out.append(rec)
    return out


def nms_image(cands):
    """与 g4a_audit.nms_image 完全一致（类内、bbox-IoU<0.5 保留）。"""
    kept = []
    for cls in {c["cls"] for c in cands}:
        group = sorted([c for c in cands if c["cls"] == cls],
                       key=lambda c: -c["score"])
        for c in group:
            if all(box_iou_xyxy(c["box"], k["box"]) < NMS_IOU
                   for k in kept if k["cls"] == cls):
                kept.append(c)
    return kept


def area_bucket(a):
    if a < 64:
        return "<64"
    if a < 256:
        return "64-256"
    if a < 1024:
        return "256-1024"
    return ">=1024"


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DATASETS["iSAID"]           # train json —— 与 g2a/g3b/g4a 相同口径
    gt = COCO(str(root / spec.val_json))
    id_to_name = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
    name_to_id = {v: k for k, v in id_to_name.items()}

    recs = [r for r in load_cands() if r["split"] == "val"]
    val_ids = [r["img_id"] for r in recs]
    gt_by_img = defaultdict(list)
    for a in gt.dataset["annotations"]:
        if a["image_id"] in set(val_ids):
            gt_by_img[a["image_id"]].append(a)
    n_gt_total = sum(len(v) for v in gt_by_img.values())
    print(f"[g4b-cpu] val imgs={len(recs)} gt={n_gt_total}", flush=True)

    per_gt = []                        # 逐 GT 记录（两池各口径）
    fp_per_img = {"raw": defaultdict(int), "nms": defaultdict(int)}
    pool_sizes = {"raw": [], "nms": []}
    nms_dets = []
    img_size_counter = defaultdict(int)

    for r in recs:
        info = gt.loadImgs(r["img_id"])[0]
        H, W = info["height"], info["width"]
        img_size_counter[(W, H)] += 1
        gts = gt_by_img[r["img_id"]]
        cands = r["cands"]
        C, G = len(cands), len(gts)
        gt_rles = [mask_util.merge(mask_util.frPyObjects(a["segmentation"],
                                                         H, W))
                   for a in gts]
        cand_rles = [{"size": c["rle"]["size"],
                      "counts": c["rle"]["counts"].encode()}
                     for c in cands]
        gt_boxes = [(a["bbox"][0], a["bbox"][1],
                     a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3])
                    for a in gts]

        # C×G 矩阵：bbox-IoU / mask-IoU / 中心命中 / 任像素重叠
        bbox_mat = np.zeros((C, G), np.float32)
        for i, c in enumerate(cands):
            for j, gb in enumerate(gt_boxes):
                bbox_mat[i, j] = box_iou_xyxy(c["box"], gb)
        miou_mat = (mask_util.iou(cand_rles, gt_rles, [0] * G)
                    if G else np.zeros((C, 0), np.float64))
        cen_mat = np.zeros((C, G), dtype=bool)
        pix_mat = np.zeros((C, G), dtype=bool)
        gt_masks = []
        for j, a in enumerate(gts):
            gx, gy, gw, gh = [int(t) for t in a["bbox"]]
            gm = mask_util.decode(gt_rles[j]).astype(bool)[
                gy:gy + gh, gx:gx + gw]
            gt_masks.append(gm)
        # 候选 mask 只解码一次
        for i, cr in enumerate(cand_rles):
            cm = mask_util.decode(cr).astype(bool)
            for j, a in enumerate(gts):
                gx, gy, gw, gh = [int(t) for t in a["bbox"]]
                cx = min(W - 1, int(gx + gw / 2))
                cy = min(H - 1, int(gy + gh / 2))
                if cm[cy, cx]:
                    cen_mat[i, j] = True
                if (cm[gy:gy + gh, gx:gx + gw] & gt_masks[j]).any():
                    pix_mat[i, j] = True

        # NMS 行索引（与 nms_image 同逻辑，作用在索引上）
        kept_rows = []
        for cls in {c["cls"] for c in cands}:
            grp = sorted([i for i, c in enumerate(cands) if c["cls"] == cls],
                         key=lambda i: -cands[i]["score"])
            for i in grp:
                if all(box_iou_xyxy(cands[i]["box"], cands[k]["box"])
                       < NMS_IOU for k in kept_rows
                       if cands[k]["cls"] == cls):
                    kept_rows.append(i)
        pools = {"raw": np.arange(C), "nms": np.array(kept_rows)}
        pool_sizes["raw"].append(C)
        pool_sizes["nms"].append(len(kept_rows))
        for i in kept_rows:
            nms_dets.append({"image_id": r["img_id"],
                             "category_id": name_to_id[cands[i]["cls"]],
                             "segmentation": cands[i]["rle"],
                             "score": cands[i]["score"]})

        for pname, rows in pools.items():
            # FP：候选在该口径下不与任何 GT 匹配
            fp_per_img[pname]["bbox50"] += int((bbox_mat[rows].max(axis=1)
                                                < 0.5).sum()) if G else len(rows)
            fp_per_img[pname]["mask30"] += int((miou_mat[rows].max(axis=1)
                                                < 0.3).sum()) if G else len(rows)
            fp_per_img[pname]["center"] += int((~cen_mat[rows].any(axis=1))
                                               .sum()) if G else len(rows)
            fp_per_img[pname]["pixel"] += int((~pix_mat[rows].any(axis=1))
                                              .sum()) if G else len(rows)
        for j, a in enumerate(gts):
            rec_gt = {"img": r["img_id"], "cat": a["category_id"],
                      "area": a["area"], "bucket": area_bucket(a["area"])}
            gname = id_to_name[a["category_id"]]
            for pname, rows in pools.items():
                bi = bbox_mat[rows, j]
                mi = miou_mat[rows, j]
                bb = int(np.argmax(bi)) if len(rows) else -1
                bm = int(np.argmax(mi)) if len(rows) else -1
                cen_rows = np.where(cen_mat[rows, j])[0]
                best_cen = int(cen_rows[np.argmax(
                    [cands[rows[t]]["score"] for t in cen_rows])]
                ) if len(cen_rows) else -1
                # exists 语义（稳定）：同分类别存在达到阈值的候选。
                # argmax 语义（g4a 原口径）在多类重复检测的 IoU 并列下
                # 依赖 NMS 类集合遍历序，跨 PYTHONHASHSEED 不稳定。
                cls_arr = np.array([cands[i]["cls"] for i in rows]) \
                    if len(rows) else np.array([], dtype=object)
                same_mask = cls_arr == gname
                rec_gt[pname] = {
                    "bbox_iou": float(bi[bb]) if len(rows) else 0.0,
                    "bbox_cls": cands[rows[bb]]["cls"]
                    if len(rows) else None,
                    "bbox_score": cands[rows[bb]]["score"]
                    if len(rows) else None,
                    "bbox_same_exists":
                        bool((bi[same_mask] >= 0.5).any()),
                    "mask_iou": float(mi[bm]) if len(rows) else 0.0,
                    "mask_cls": cands[rows[bm]]["cls"] if len(rows) else None,
                    "mask_score": cands[rows[bm]]["score"]
                    if len(rows) else None,
                    "mask_same_exists":
                        bool((mi[same_mask] >= 0.3).any()),
                    "center": bool(cen_mat[rows, j].any()),
                    "center_cls": cands[rows[best_cen]]["cls"]
                    if best_cen >= 0 else None,
                    "center_score": cands[rows[best_cen]]["score"]
                    if best_cen >= 0 else None,
                    "center_same_exists":
                        bool(cen_mat[rows[same_mask], j].any())
                        if len(rows) else False,
                    "pixel": bool(pix_mat[rows, j].any())}
            per_gt.append(rec_gt)

    # ---------------- 汇总 ----------------
    def pool_metrics(pool):
        n = len(per_gt)
        out = {}
        for crit in ("bbox50", "mask30", "mask10", "center", "pixel"):
            for kind in ("any", "same", "same_exists", "same_argmax"):
                k = 0
                for g in per_gt:
                    p = g[pool]
                    if crit == "bbox50":
                        hit = p["bbox_iou"] >= 0.5
                        cls = p["bbox_cls"]
                        ex = p["bbox_same_exists"]
                    elif crit in ("mask30", "mask10"):
                        th = 0.3 if crit == "mask30" else 0.1
                        hit = p["mask_iou"] >= th
                        cls = p["mask_cls"]
                        ex = p["mask_same_exists"]
                    elif crit == "center":
                        hit, cls = p["center"], p["center_cls"]
                        ex = p["center_same_exists"]
                    else:
                        hit, cls, ex = p["pixel"], None, False
                    if kind == "any":
                        ok = hit
                    elif kind == "same_exists":
                        ok = hit and ex
                    elif kind == "same_argmax":
                        ok = hit and cls == id_to_name[g["cat"]]
                    else:
                        ok = False
                    if ok:
                        k += 1
                out[f"{crit}_{kind}"] = round(100 * k / n, 2)
        return out

    areas = [g["area"] for g in per_gt]
    results = {
        "setup": {
            "n_val_imgs": len(recs),
            "n_gt": len(per_gt),
            "gt_area_median": float(np.median(areas)),
            "gt_area_p25_p75": [float(np.percentile(areas, q))
                                for q in (25, 75)],
            "img_sizes_top": {f"{w}x{h}": c for (w, h), c in
                              sorted(img_size_counter.items(),
                                     key=lambda kv: -kv[1])[:6]},
            "non_square_imgs": sum(c for (w, h), c in
                                   img_size_counter.items() if w != h)},
        "pool_per_img": {
            "raw_mean": round(float(np.mean(pool_sizes["raw"])), 1),
            "nms_mean": round(float(np.mean(pool_sizes["nms"])), 1),
            "nms_median": float(np.median(pool_sizes["nms"])),
            "nms_max": int(np.max(pool_sizes["nms"])),
            "imgs_nms_gt100": int(sum(1 for x in pool_sizes["nms"]
                                      if x > 100))},
        "recall_pct": {"raw": pool_metrics("raw"),
                       "nms": pool_metrics("nms")},
        "fp_per_img": {p: {k: round(v / len(recs), 1)
                           for k, v in fp_per_img[p].items()}
                       for p in ("raw", "nms")},
    }

    by_area = defaultdict(lambda: defaultdict(int))
    for g in per_gt:
        p = g["nms"]
        by_area[g["bucket"]]["n"] += 1
        by_area[g["bucket"]]["bbox50"] += p["bbox_iou"] >= 0.5
        by_area[g["bucket"]]["mask30"] += p["mask_iou"] >= 0.3
        by_area[g["bucket"]]["center"] += p["center"]
    results["recall_by_area_nms"] = {
        b: {"n": v["n"],
            "bbox50_any": round(100 * v["bbox50"] / v["n"], 2),
            "mask30_any": round(100 * v["mask30"] / v["n"], 2),
            "center_any": round(100 * v["center"] / v["n"], 2)}
        for b, v in sorted(by_area.items())}

    by_cls = defaultdict(lambda: defaultdict(int))
    for g in per_gt:
        cn = id_to_name[g["cat"]]
        p = g["nms"]
        by_cls[cn]["n"] += 1
        by_cls[cn]["bbox50"] += p["bbox_iou"] >= 0.5
        by_cls[cn]["mask30"] += p["mask_iou"] >= 0.3
        by_cls[cn]["center"] += p["center"]
        by_cls[cn]["bbox50_same_exists"] += (p["bbox_iou"] >= 0.5 and
                                             p["bbox_same_exists"])
        by_cls[cn]["mask30_same_exists"] += (p["mask_iou"] >= 0.3 and
                                             p["mask_same_exists"])
    results["recall_by_class_nms"] = {
        cn: {"n": v["n"],
             "bbox50_any": round(100 * v["bbox50"] / v["n"], 1),
             "bbox50_same_exists": round(100 * v["bbox50_same_exists"]
                                         / v["n"], 1),
             "mask30_any": round(100 * v["mask30"] / v["n"], 1),
             "mask30_same_exists": round(100 * v["mask30_same_exists"]
                                         / v["n"], 1),
             "center_any": round(100 * v["center"] / v["n"], 1)}
        for cn, v in sorted(by_cls.items(), key=lambda kv: -kv[1]["n"])}

    missed = [g for g in per_gt if g["nms"]["bbox_iou"] < 0.5]
    rescue = {"n_missed_bbox50": len(missed)}
    rescue["hit_mask30"] = sum(1 for g in missed
                               if g["nms"]["mask_iou"] >= 0.3)
    rescue["hit_center"] = sum(1 for g in missed if g["nms"]["center"])
    rescue["hit_pixel"] = sum(1 for g in missed if g["nms"]["pixel"])
    sc = [g["nms"]["mask_score"] for g in missed
          if g["nms"]["mask_iou"] >= 0.3]
    rescue["rescued_by_mask30_score_hist"] = {
        "<0.1": sum(1 for s in sc if s < 0.1),
        "0.1-0.3": sum(1 for s in sc if 0.1 <= s < 0.3),
        ">=0.3": sum(1 for s in sc if s >= 0.3)}
    allmiss = [g for g in missed if g["nms"]["mask_iou"] < 0.3
               and not g["nms"]["center"]]
    mi = [g["nms"]["mask_iou"] for g in allmiss]
    rescue["n_missed_all_criteria"] = len(allmiss)
    rescue["best_maskiou_hist_of_allmissed"] = {
        "0.1-0.3": sum(1 for x in mi if 0.1 <= x < 0.3),
        "0.01-0.1": sum(1 for x in mi if 0.01 <= x < 0.1),
        "<0.01": sum(1 for x in mi if x < 0.01)}
    results["rescue_nms"] = rescue

    # ---------------- mAP 复算（A0 NMS 池）+ maxDets 核查 ----------------
    val_set = set(val_ids)
    sub = COCO()
    sub.dataset = {
        "images": [im for im in gt.dataset["images"] if im["id"] in val_set],
        "annotations": [a for a in gt.dataset["annotations"]
                        if a["image_id"] in val_set],
        "categories": gt.dataset["categories"]}
    sub.createIndex()
    dt = sub.loadRes([dict(d) for d in nms_dets])
    ev = COCOeval(sub, dt, iouType="segm")
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    ap100 = float(ev.stats[0])
    # maxDets 截断核查（summarize 的 _summarize 是闭包，直接取 precision 数组）
    ev.params.maxDets = [1, 100, 1000]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    prec = ev.eval["precision"]                    # [T,R,K,A,M]
    ap1000 = float(np.mean(prec[..., 2][prec[..., 2] > -1]))
    results["ap_A0_nms"] = {"maxDets100": round(ap100, 4),
                            "maxDets1000": round(ap1000, 4)}

    # oracle 换类 mAP 复算（g4a 同法：NMS 候选换成最佳 bbox-IoU GT 的类）
    dets_orc = []
    for r in recs:
        gts = gt_by_img[r["img_id"]]
        for c in nms_image(r["cands"]):
            best_iou, best_cat = 0.0, None
            for a in gts:
                gx, gy, gw, gh = a["bbox"]
                iou = box_iou_xyxy(c["box"], (gx, gy, gx + gw, gy + gh))
                if iou > best_iou:
                    best_iou, best_cat = iou, a["category_id"]
            cat = best_cat if (best_iou >= 0.5 and best_cat is not None) \
                else name_to_id[c["cls"]]
            dets_orc.append({"image_id": r["img_id"],
                             "category_id": cat,
                             "segmentation": c["rle"],
                             "score": c["score"]})
    dt_orc = sub.loadRes([dict(d) for d in dets_orc])
    ev_orc = COCOeval(sub, dt_orc, iouType="segm")
    with contextlib.redirect_stdout(io.StringIO()):
        ev_orc.evaluate(); ev_orc.accumulate(); ev_orc.summarize()
    results["ap_A0_nms"]["oracle_reclass_maxDets100"] = round(
        float(ev_orc.stats[0]), 4)

    per_ap = {}
    for img_id in val_ids:
        s = COCO()
        s.dataset = {
            "images": [im for im in sub.dataset["images"]
                       if im["id"] == img_id],
            "annotations": [a for a in sub.dataset["annotations"]
                            if a["image_id"] == img_id],
            "categories": sub.dataset["categories"]}
        s.createIndex()
        d = [x for x in nms_dets if x["image_id"] == img_id]
        if d:
            dti = s.loadRes([dict(x) for x in d])
            ev = COCOeval(s, dti, iouType="segm")
            with contextlib.redirect_stdout(io.StringIO()):
                ev.evaluate(); ev.accumulate(); ev.summarize()
            per_ap[img_id] = float(ev.stats[0])
        else:
            per_ap[img_id] = 0.0

    # ---------------- bootstrap（图级重采样） ----------------
    results["bootstrap"] = {}
    by_img_gt = defaultdict(list)
    for g in per_gt:
        by_img_gt[g["img"]].append(g)
    rng = np.random.default_rng(SEED)

    def hit_vec(pool, crit, kind):
        out = []
        for i in val_ids:
            k = 0
            for g in by_img_gt[i]:
                p = g[pool]
                if crit == "bbox50":
                    hit = p["bbox_iou"] >= 0.5
                    ex = p["bbox_same_exists"]
                elif crit == "mask30":
                    hit = p["mask_iou"] >= 0.3
                    ex = p["mask_same_exists"]
                else:
                    hit = p["center"]
                    ex = p["center_same_exists"]
                if hit and (kind == "any" or ex):
                    k += 1
            out.append(k)
        return np.array(out)

    gt_vec = np.array([len(by_img_gt[i]) for i in val_ids])
    for crit in ("bbox50", "mask30", "center"):
        for kind in ("any", "same_exists"):
            h = hit_vec("nms", crit, kind)
            boots = np.empty(BOOT_N)
            for t in range(BOOT_N):
                idx = rng.integers(0, len(val_ids), len(val_ids))
                boots[t] = h[idx].sum() / max(gt_vec[idx].sum(), 1)
            results["bootstrap"][f"nms_{crit}_{kind}"] = {
                "point_pct": round(100 * h.sum() / gt_vec.sum(), 2),
                "ci95_pct": [round(100 * float(np.percentile(boots, 2.5)), 2),
                             round(100 * float(np.percentile(boots, 97.5)), 2)]}

    ap_vec = np.array([per_ap[i] for i in val_ids])
    boots = np.empty(BOOT_N)
    for t in range(BOOT_N):
        idx = rng.integers(0, len(val_ids), len(val_ids))
        boots[t] = ap_vec[idx].mean()
    results["bootstrap"]["A0_nms_meanAP"] = {
        "point_pct": round(100 * float(ap_vec.mean()), 2),
        "ci95_pct": [round(100 * float(np.percentile(boots, 2.5)), 2),
                     round(100 * float(np.percentile(boots, 97.5)), 2)]}
    results["ap_A0_nms"]["per_img_AP_mean_pct"] = round(
        100 * float(ap_vec.mean()), 2)

    fp_img_list = []
    # 逐图 FP 重取（bbox50，NMS 池）
    for r in recs:
        gts = gt_by_img[r["img_id"]]
        kept = nms_image(r["cands"])
        n = sum(1 for c in kept if not any(box_iou_xyxy(
            c["box"], (a["bbox"][0], a["bbox"][1],
                       a["bbox"][0] + a["bbox"][2],
                       a["bbox"][1] + a["bbox"][3])) >= 0.5
            for a in gts))
        fp_img_list.append(n)
    fp_vec = np.array(fp_img_list, dtype=float)
    boots = np.empty(BOOT_N)
    for t in range(BOOT_N):
        idx = rng.integers(0, len(val_ids), len(val_ids))
        boots[t] = fp_vec[idx].mean()
    results["bootstrap"]["fp_per_img_bbox50_nms"] = {
        "point": round(float(fp_vec.mean()), 1),
        "ci95": [round(float(np.percentile(boots, 2.5)), 1),
                 round(float(np.percentile(boots, 97.5)), 1)]}

    (OUT_DIR / "pipeline_audit_cpu.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False))
    print(json.dumps({k: results[k] for k in (
        "recall_pct", "fp_per_img", "recall_by_area_nms", "rescue_nms",
        "ap_A0_nms")}, indent=1, ensure_ascii=False))
    print(f"-> {OUT_DIR / 'pipeline_audit_cpu.json'}", flush=True)


if __name__ == "__main__":
    main()
