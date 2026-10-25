#!/usr/bin/env python3
"""v4-S0 C0 统计分析：CPU 离线阶段（读 v4_s0_c0_run.py 的 candidates.jsonl）。

对应 implementation-v4.0 S0 第 3 条（design-v4.0 §5 的准入统计）。本脚本不碰
GPU、不 import torch；融合与统计全部离线做，保证 GPU 阶段逻辑最小。

口径登记（2026-10-25 冻结于任何结果之前；详见 notes/reports/v4-s0-c0-run-plan.md）：
- GT：DOTA labelTxt OBB polygon -> pycocotools RLE（全局 [H,W]）；外接轴对齐
  bbox = 8 坐标 min/max。difficult 实例整体剔除（不进 GT；对其的预测按普通
  FP 计——已登记为口径注意项，可能低估 AP）。
- C0 融合：global 坐标下逐类 mask NMS（严格按 cat_name 分组，绝不跨类合并；
  IoU>=0.5 抑制，score 降序贪心）。
- 命中（b）：GT 与任一同类融合预测 mask IoU>=0.5。多重匹配不互斥（e 需要
  重复计数）。
- 漏检但邻窗可见（c）：融合后未命中 GT 中，存在任一融合前候选同类别且
  mask IoU>=0.5；按该候选 bbox_full_in_tile 与否分账（full=观察未被 tile
  边界截断；贴图像边缘豁免，定义见 v4_s0_c0_run.py）。
- oracle 正框恢复上限（d）：c 的上界口径，双分母：/未命中 GT（可恢复率）
  与 /全部 GT（对整体实例空间的绝对余量，即 mAP headroom 口径）。
- 重复实例率（e）：同一 GT 被 >=2 个融合预测匹配的比例（分母 = 至少被 1 个
  预测匹配的 GT）。边界截断子集（classify 的 cross-x/cross-y/corner）单列。
- 边界分类与 tile 锚点逐字复用 v4_s0_recon_dota.py（tile_anchors/classify），
  保证与已批准的摸底报告同口径。

bbox 口径：本脚本内部一律 xyxy；COCO ann 用 toBbox 产出的 xywh。

Run（无需 source 环境，但需 numpy/PIL/pycocotools）:
  "$PYTHON" sam3_ovis/eval/v4_s0_c0_analyze.py            # 默认 run 目录
  "$PYTHON" sam3_ovis/eval/v4_s0_c0_analyze.py --self-test  # 纯 CPU 自测
输出：stdout 摘要 + <log_root>/v4_s0_c0_stats.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

from v4_s0_recon_dota import classify, tile_anchors  # noqa: E402  冻结口径
from v4_s0_c0_run import DOTA15_CLASSES, parse_label_txt  # noqa: E402

MATCH_IOU = 0.5   # GT<->预测命中阈值
NMS_IOU = 0.5     # 类内 NMS 抑制阈值
BOUNDARY_CLASSES = ("cross-x", "cross-y", "corner")


def default_log_root():
    env = os.environ.get("SAM3_OVIS_LOG_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "logs"


def rle_from_record(rec):
    """jsonl 里的 mask_rle（counts 为 str）-> pycocotools RLE（counts 为 bytes）。"""
    return {"size": rec["size"], "counts": rec["counts"].encode("utf-8")}


def poly_bbox(poly):
    """OBB 8 坐标 -> 外接轴对齐 xyxy。"""
    return (min(poly[0::2]), min(poly[1::2]),
            max(poly[0::2]), max(poly[1::2]))


def gt_rle(poly, H, W):
    """OBB polygon -> 全局二值 RLE。"""
    rles = mask_util.frPyObjects([list(map(float, poly))], int(H), int(W))
    return mask_util.merge(rles)


def mask_iou(rle_a, rle_b):
    return float(mask_util.iou([rle_a], [rle_b], [0])[0][0])


def fuse_image(cands, nms_iou=NMS_IOU):
    """全图 C0 融合：严格逐类（按 cat_name 分组）mask NMS，score 降序贪心。

    返回保持原有字段、附 iou_rle 的列表（fusion 不改坐标）。绝不跨类抑制。
    """
    kept = []
    for cls in sorted({c["cat_name"] for c in cands}):
        group = sorted([c for c in cands if c["cat_name"] == cls],
                       key=lambda c: -c["score"])
        for c in group:
            if all(mask_iou(c["iou_rle"], k["iou_rle"]) < nms_iou
                   for k in kept if k["cat_name"] == cls):
                kept.append(c)
    return kept


def _ap_from_precision(prec):
    v = prec[prec >= 0]
    return float(v.mean()) if v.size else float("nan")


def self_test():
    """CPU 自测：>=6 组断言（OBB->RLE、逐类 NMS、匹配口径、边界分类、往返）。"""
    n_checks = 0
    H, W = 200, 300

    # 1. OBB polygon -> RLE：面积与外接框
    poly = [50.0, 40.0, 150.0, 40.0, 150.0, 90.0, 50.0, 90.0]  # 100x50 矩形
    rle = gt_rle(poly, H, W)
    assert int(mask_util.area(rle)) == 100 * 50, \
        f"OBB RLE 面积应 5000，得到 {int(mask_util.area(rle))}"
    assert poly_bbox(poly) == (50.0, 40.0, 150.0, 90.0), "外接框 min/max 错误"
    n_checks += 1

    # 2. RLE str<->bytes 往返（jsonl 落盘格式）
    m = np.zeros((H, W), np.uint8)
    m[10:40, 20:80] = 1
    r0 = mask_util.encode(np.asfortranarray(m))
    rec = {"size": list(r0["size"]), "counts": r0["counts"].decode("utf-8")}
    assert int(mask_util.area(rle_from_record(rec))) == 30 * 60, \
        "str counts -> bytes RLE 往返失败"
    n_checks += 1

    # 3. 逐类 NMS：同类高 IoU 抑制；跨类同位置不抑制（历史失效模式回归）
    m_a = np.zeros((H, W), np.uint8)
    m_a[20:60, 20:60] = 1
    m_b = np.zeros((H, W), np.uint8)   # 与 A IoU = 400/2800（构造已知）
    m_b[40:80, 40:80] = 1
    m_c = np.zeros((H, W), np.uint8)   # 与 A IoU = 0
    m_c[150:170, 250:270] = 1
    r = [mask_util.encode(np.asfortranarray(x)) for x in (m_a, m_b, m_c)]
    iou_ab = mask_iou(r[0], r[1])      # 20x20 / (1600+1600-400) = 400/2800
    assert abs(iou_ab - 400.0 / 2800.0) < 1e-9, f"合成 IoU 错误: {iou_ab}"
    m_b_full = np.zeros((H, W), np.uint8)  # 与 A 完全重合
    m_b_full[20:60, 20:60] = 1
    r_full = mask_util.encode(np.asfortranarray(m_b_full))
    assert mask_iou(r[0], r_full) == 1.0, "完全重合 IoU 应为 1"
    cands = [
        {"cat_name": "plane", "score": 0.9, "iou_rle": r[0]},
        {"cat_name": "plane", "score": 0.8, "iou_rle": r_full},   # 同类 IoU=1 -> 抑制
        {"cat_name": "ship", "score": 0.7, "iou_rle": r_full},    # 跨类同位置 -> 保留
        {"cat_name": "plane", "score": 0.6, "iou_rle": r[2]},     # 同类 IoU=0 -> 保留
    ]
    fused = fuse_image(cands)
    got = sorted((c["cat_name"], round(c["score"], 2)) for c in fused)
    assert got == [("plane", 0.6), ("plane", 0.9), ("ship", 0.7)], \
        f"逐类 NMS 结果错误: {got}"
    n_checks += 1

    # 4. 匹配口径：同一 GT 被 2 个预测匹配（无互斥），另一 GT 未命中
    gt1 = mask_util.encode(np.asfortranarray(m_a))
    half1 = np.zeros((H, W), np.uint8)
    half1[20:60, 20:40] = 1
    half2 = np.zeros((H, W), np.uint8)
    half2[20:60, 40:60] = 1
    r_h1 = mask_util.encode(np.asfortranarray(half1))
    r_h2 = mask_util.encode(np.asfortranarray(half2))
    # 每半与 GT IoU = 800/1600 = 0.5 -> 都 >= MATCH_IOU 命中
    assert mask_iou(gt1, r_h1) == 0.5 and mask_iou(gt1, r_h2) == 0.5
    # 两半互相 IoU = 0 -> NMS 不会去掉任何一个（融合后重复保留）
    assert mask_iou(r_h1, r_h2) == 0.0
    fused2 = fuse_image([{"cat_name": "plane", "score": 0.9, "iou_rle": r_h1},
                         {"cat_name": "plane", "score": 0.8, "iou_rle": r_h2}])
    n_match = sum(1 for p in fused2
                  if p["cat_name"] == "plane" and mask_iou(gt1, p["iou_rle"]) >= 0.5)
    assert n_match == 2, f"截断两半应对同一 GT 双匹配，得到 {n_match}（e 的口径）"
    n_checks += 1

    # 5. 边界分类（复用摸底脚本逻辑）：跨界/角落/inside。
    #    构造要点：只跨内部线还不够——小框仍完整落在 tile(0,0) 内会被判
    #    inside，必须同时不被任何单 tile 包含。
    #    注意 H=1848 时唯一 y 锚点 824 == H-tile（末 tile 吸附），不算内部线，
    #    故 corner/cross-y 用 H=2900（内部 y 线 824/1648 存在）。
    W2, H2, TILE, STRIDE = 2000, 1848, 1024, 824
    ax, ay = tile_anchors(W2, TILE, STRIDE), tile_anchors(H2, TILE, STRIDE)
    assert ax == [0, 824, 976] and ay == [0, 824], f"锚点错误: {ax} {ay}"
    mid = 824  # 内部竖网格线（lx = [824]，976 为吸附边不算）
    assert classify((mid - 24, 10, mid + 276, 100), W2, H2, TILE,
                    STRIDE) == "cross-x", "跨内部 x 线且无单 tile 包含"
    assert classify((0, 10, 100, 100), W2, H2, TILE, STRIDE) == "inside"
    H3 = 2900
    ay3 = tile_anchors(H3, TILE, STRIDE)
    assert ay3 == [0, 824, 1648, 1876], f"H=2900 锚点错误: {ay3}"
    assert classify((mid - 24, mid - 24, mid + 276, mid + 276), W2, H3,
                    TILE, STRIDE) == "corner"
    # 只跨 y 内部线（x 完整在首个 tile 内）= cross-y
    assert classify((0, 0, 100, 1100), W2, H3, TILE, STRIDE) == "cross-y"
    n_checks += 1

    # 6. 融合前可见性判定（c 口径）：未命中 GT 但融合前候选 IoU>=0.5 且类别对
    gt2 = mask_util.encode(np.asfortranarray(m_c))
    vis = {"cat_name": "plane", "score": 0.051, "iou_rle": r[2],
           "bbox_full_in_tile": False}
    vis_wrong_cls = {"cat_name": "ship", "score": 0.9, "iou_rle": r[2],
                     "bbox_full_in_tile": True}
    visible = [c for c in (vis, vis_wrong_cls)
               if c["cat_name"] == "plane" and mask_iou(gt2, c["iou_rle"]) >= 0.5]
    assert visible == [vis], "c 口径必须类别对 + IoU>=0.5"
    assert visible[0]["bbox_full_in_tile"] is False  # 分账字段可区分
    n_checks += 1

    # 7. COCOeval 集成：完美匹配 AP50=1；零预测路径可用（stats=-1 不崩）
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO()
    coco.dataset = {
        "images": [{"id": 1, "width": W, "height": H}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1, "segmentation": r[0],
            "area": float(mask_util.area(r[0])), "iscrowd": 0,
            "bbox": [float(v) for v in mask_util.toBbox(r[0])],
        }],
        "categories": [{"id": 1, "name": "plane"}],
    }
    coco.createIndex()
    dt = coco.loadRes([{"image_id": 1, "category_id": 1,
                        "segmentation": r[0], "score": 0.9}])
    ev = COCOeval(coco, dt, iouType="segm")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    assert abs(float(ev.stats[1]) - 1.0) < 1e-6, \
        f"完美匹配 AP50 应为 1，得到 {ev.stats[1]}"
    assert ev.eval["precision"].shape[:2] == (10, 101), \
        "COCOeval precision T/R 维度不符"
    # 注意：本机 pycocotools 的 loadRes([]) 会 IndexError，主流程已对
    # 全空预测特判（零预测图通过 images 列表保留在分母）。
    n_checks += 1

    print(f"[self-test] v4_s0_c0_analyze: {n_checks} 组断言全部通过")
    return n_checks


def load_run(run_dir):
    """读 manifest/done/candidates。崩溃续跑的半图候选块按 done 记录截尾丢弃。"""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    done = {}
    with open(run_dir / "done_images.jsonl") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                break
            done[rec["img"]] = rec
    cands_by_img = defaultdict(list)
    with open(run_dir / "candidates.jsonl") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                break
            cands_by_img[rec["img"]].append(rec)
    n_discarded = 0
    for name, rec_d in done.items():
        have = cands_by_img.get(name, [])
        want = int(rec_d.get("n_candidates", len(have)))
        if len(have) > want:  # 该图曾半途崩溃后重跑：只保留最后一个完整块
            n_discarded += len(have) - want
            cands_by_img[name] = have[len(have) - want:]
        elif len(have) < want:
            raise RuntimeError(f"{name}: 候选数 {len(have)} < done 记录 {want}，"
                               f"文件损坏，请检查 {run_dir / 'candidates.jsonl'}")
    return manifest, done, dict(cands_by_img), n_discarded


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="默认 <SAM3_OVIS_LOG_ROOT>/v4_s0_c0")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None,
                    help="默认 <SAM3_OVIS_LOG_ROOT>/v4_s0_c0_stats.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0
    stride = args.tile - args.overlap
    run_dir = args.run_dir or (default_log_root() / "v4_s0_c0")
    out_path = args.out or (default_log_root() / "v4_s0_c0_stats.json")

    manifest, done, cands_by_img, n_discarded = load_run(run_dir)
    if manifest["tile"] != args.tile or manifest["overlap"] != args.overlap:
        raise SystemExit(f"tile/overlap 与 manifest 不符: "
                         f"{manifest['tile']}/{manifest['overlap']}")
    eval_names = sorted(set(done) & set(manifest["images"]))
    print(f"[analyze] run_dir={run_dir}\n[analyze] images: done={len(done)} "
          f"in-manifest={len(manifest['images'])} evaluated={len(eval_names)} "
          f"(discarded partial cand blocks: {n_discarded})", flush=True)

    cat_ids = {name: i + 1 for i, name in enumerate(DOTA15_CLASSES)}
    coco_images, coco_gt_anns, coco_dets = [], [], []
    ann_id = 0
    agg = {  # b/c/d/e 聚合
        "n_gt": 0, "n_gt_difficult": 0, "n_hit": 0,
        "n_missed": 0, "n_visible": 0, "n_visible_full": 0,
        "n_visible_trunc_only": 0,
        "n_gt_matched": 0, "n_gt_multi": 0,
        "bnd": defaultdict(lambda: {"n": 0, "hit": 0, "matched": 0, "multi": 0}),
        "per_class": defaultdict(lambda: {
            "n_gt": 0, "hit": 0, "missed": 0, "visible": 0, "visible_full": 0}),
        "n_raw_cand": 0, "n_fused": 0, "n_zero_pred_images": 0,
    }

    for name in eval_names:
        W, H = done[name]["W"], done[name]["H"]
        img_id = len(coco_images)
        coco_images.append({"id": img_id, "width": W, "height": H, "file_name": name})

        raw = cands_by_img.get(name, [])
        for c in raw:
            c["iou_rle"] = rle_from_record(c["mask_rle"])
        agg["n_raw_cand"] += len(raw)
        fused = fuse_image(raw)
        agg["n_fused"] += len(fused)
        if not fused:  # 零预测图保留在 mAP 分母（implementation-v4.0 测试要求）
            agg["n_zero_pred_images"] += 1
        for c in fused:
            coco_dets.append({
                "image_id": img_id, "category_id": cat_ids[c["cat_name"]],
                "segmentation": c["iou_rle"], "score": c["score"]})

        gts = []
        for it in parse_label_txt(manifest["images"][name]["label_path"]):
            if it["difficult"]:
                agg["n_gt_difficult"] += 1
                continue
            rle = gt_rle(it["poly"], H, W)
            gts.append({
                "cat": it["name"], "rle": rle,
                "bbox": poly_bbox(it["poly"]),
                "bclass": classify(poly_bbox(it["poly"]), W, H,
                                   args.tile, stride),
            })
        for g in gts:
            same_cls = [p for p in fused if p["cat_name"] == g["cat"]]
            matched = [p for p in same_cls
                       if mask_iou(g["rle"], p["iou_rle"]) >= MATCH_IOU]
            hit = len(matched) >= 1
            agg["n_gt"] += 1
            agg["per_class"][g["cat"]]["n_gt"] += 1
            b = agg["bnd"][g["bclass"]]
            b["n"] += 1
            if hit:
                agg["n_hit"] += 1
                agg["per_class"][g["cat"]]["hit"] += 1
                b["hit"] += 1
            else:
                agg["n_missed"] += 1
                agg["per_class"][g["cat"]]["missed"] += 1
                # c：融合前（raw）同类别候选 IoU>=0.5 = 邻窗可见
                vis = [c for c in raw if c["cat_name"] == g["cat"]
                       and mask_iou(g["rle"], c["iou_rle"]) >= MATCH_IOU]
                if vis:
                    agg["n_visible"] += 1
                    agg["per_class"][g["cat"]]["visible"] += 1
                    if any(c["bbox_full_in_tile"] for c in vis):
                        agg["n_visible_full"] += 1
                        agg["per_class"][g["cat"]]["visible_full"] += 1
                    else:
                        agg["n_visible_trunc_only"] += 1
            if matched:
                agg["n_gt_matched"] += 1
                b["matched"] += 1
            if len(matched) >= 2:  # e：同一 GT 多预测匹配（无互斥口径）
                agg["n_gt_multi"] += 1
                b["multi"] += 1
            # COCO GT ann（xywh bbox 由 toBbox 生成）
            ann_id += 1
            coco_gt_anns.append({
                "id": ann_id, "image_id": img_id,
                "category_id": cat_ids[g["cat"]], "segmentation": g["rle"],
                "area": float(mask_util.area(g["rle"])), "iscrowd": 0,
                "bbox": [float(v) for v in mask_util.toBbox(g["rle"])],
            })

    # ---- a: 全图 segm mAP/AP50（COCOeval；零预测图保留在分母） ----
    # 本机 pycocotools 的 loadRes([]) 会 IndexError，全空预测特判为 0。
    ap_cls = {c: float("nan") for c in DOTA15_CLASSES}
    mAP = AP50 = AP75 = 0.0
    if coco_gt_anns and coco_dets:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": coco_images, "annotations": coco_gt_anns,
            "categories": [{"id": i, "name": n}
                           for n, i in sorted(cat_ids.items(),
                                              key=lambda kv: kv[1])],
        }
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(coco_dets)
        ev = COCOeval(coco_gt, coco_dt, iouType="segm")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        mAP, AP50, AP75 = (float(ev.stats[0]), float(ev.stats[1]),
                           float(ev.stats[2]))
        prec = ev.eval["precision"]
        name_by_id = {i: n for n, i in cat_ids.items()}
        for k in range(prec.shape[2]):
            ap_cls[name_by_id[k + 1]] = round(_ap_from_precision(prec[:, :, k, 0, 2]), 4)

    def pct(num, den):
        return round(100.0 * num / den, 2) if den else None

    bnd_detail = {k: {"n_gt": v["n"], "hit_rate_pct": pct(v["hit"], v["n"]),
                      "n_gt_matched": v["matched"], "n_gt_multi_matched": v["multi"],
                      "dup_rate_pct": pct(v["multi"], v["matched"])}
                  for k, v in sorted(agg["bnd"].items())}
    stats = {
        "protocol": {
            "tile": args.tile, "overlap": args.overlap, "stride": stride,
            "match_iou": MATCH_IOU, "nms_iou": NMS_IOU,
            "nms_scope": "per-class only (cat_name), global coords",
            "pool_floor": manifest["pool_floor"], "seed": manifest["seed"],
            "difficult": "excluded from GT; predictions on them count as FP",
            "match_rule": "any same-class fused pred with mask IoU>=0.5, "
                          "non-exclusive (duplicates counted)",
            "bbox_full_in_tile": "tile-local tight bbox touches no interior "
                                 "tile border; image-border sides exempt",
            "run_code_sha": manifest.get("code_sha"),
        },
        "data": {
            "n_images_evaluated": len(eval_names), "n_images_done": len(done),
            "n_gt_nondifficult": agg["n_gt"],
            "n_gt_difficult_excluded": agg["n_gt_difficult"],
            "n_candidates_raw": agg["n_raw_cand"],
            "n_candidates_after_fusion": agg["n_fused"],
            "n_images_zero_fused_pred": agg["n_zero_pred_images"],
            "n_partial_candidate_blocks_discarded": n_discarded,
        },
        "a_coco_segm": {"mAP": round(mAP, 4), "AP50": round(AP50, 4),
                        "AP75": round(AP75, 4),
                        "per_class_AP50": {c: ap_cls[c]
                                           for c in sorted(ap_cls)}},
        "b_hit": {"n_gt": agg["n_gt"], "n_hit": agg["n_hit"],
                  "recall_pct": pct(agg["n_hit"], agg["n_gt"])},
        "c_missed_visible": {
            "n_missed": agg["n_missed"],
            "visible_any": agg["n_visible"],
            "visible_any_pct_of_missed": pct(agg["n_visible"], agg["n_missed"]),
            "visible_with_full_in_tile_cand": agg["n_visible_full"],
            "visible_with_full_in_tile_cand_pct_of_missed":
                pct(agg["n_visible_full"], agg["n_missed"]),
            "visible_truncated_only": agg["n_visible_trunc_only"],
            "visible_truncated_only_pct_of_missed":
                pct(agg["n_visible_trunc_only"], agg["n_missed"]),
        },
        "d_oracle_recovery": {
            "note": "upper bound for positive-box recovery from a neighbor "
                    "window observation; same numerator as c",
            "over_missed_gt_pct": pct(agg["n_visible"], agg["n_missed"]),
            "over_all_gt_pct": pct(agg["n_visible"], agg["n_gt"]),
            "over_all_gt_strict_full_tile_pct":
                pct(agg["n_visible_full"], agg["n_gt"]),
        },
        "e_duplicates": {
            "n_gt_matched": agg["n_gt_matched"],
            "n_gt_multi_matched": agg["n_gt_multi"],
            "dup_rate_pct": pct(agg["n_gt_multi"], agg["n_gt_matched"]),
            "boundary_subset_cross_or_corner": {
                k: v for k, v in bnd_detail.items() if k in BOUNDARY_CLASSES},
            "by_boundary_class": bnd_detail,
        },
        "per_class_gt": {
            c: {"n_gt": v["n_gt"], "hit_pct": pct(v["hit"], v["n_gt"]),
                "missed": v["missed"],
                "missed_visible_pct": pct(v["visible"], v["missed"]),
                "missed_visible_full_tile_pct": pct(v["visible_full"],
                                                    v["missed"])}
            for c, v in sorted(agg["per_class"].items())},
        "generated": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path.write_text(json.dumps(stats, indent=1, ensure_ascii=False))

    print("\n===== v4-S0 C0 stats (frozen protocol, see run-plan) =====")
    print(f"images={stats['data']['n_images_evaluated']} "
          f"GT(non-diff)={agg['n_gt']} difficult-excluded={agg['n_gt_difficult']} "
          f"raw-cands={agg['n_raw_cand']} fused={agg['n_fused']}")
    print(f"[a] segm mAP={stats['a_coco_segm']['mAP']} "
          f"AP50={stats['a_coco_segm']['AP50']} AP75={stats['a_coco_segm']['AP75']}")
    print(f"[b] recall={stats['b_hit']['recall_pct']}% "
          f"({agg['n_hit']}/{agg['n_gt']})")
    c = stats["c_missed_visible"]
    print(f"[c] missed={c['n_missed']}; visible-in-some-tile "
          f"{c['visible_any_pct_of_missed']}% | with full-in-tile cand "
          f"{c['visible_with_full_in_tile_cand_pct_of_missed']}% | "
          f"truncated-only {c['visible_truncated_only_pct_of_missed']}%")
    d = stats["d_oracle_recovery"]
    print(f"[d] oracle: /missed {d['over_missed_gt_pct']}% | /all-GT "
          f"{d['over_all_gt_pct']}% (strict full-tile {d['over_all_gt_strict_full_tile_pct']}%)")
    e = stats["e_duplicates"]
    print(f"[e] dup rate={e['dup_rate_pct']}% "
          f"({e['n_gt_multi_matched']}/{e['n_gt_matched']}); "
          f"boundary(cross/corner) detail: {e['boundary_subset_cross_or_corner']}")
    print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
