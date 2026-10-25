#!/usr/bin/env python3
"""v4-S0 数据摸底：DOTA-v1.5 原始大图 + 固定二维 tiling 模拟统计。

协议登记（数字产生前冻结，2026-10-25，与 v4_s0_recon_isaid.py 同协议）：
- 数据：/data/wangcheng/dataset/DOTA train/val 原图（未切分）+ labelTxt-v1.5
  （每行 = 一个实例：8 坐标旋转框 + 类别 + difficult，全局坐标）。
  实例外接矩形 = 8 坐标的 min/max；实例身份 = (图, 行号)，可恢复。
- tiling 草案 A：tile=1024, overlap=200；草案 B：overlap=128。两案并列报告。
- 分类同 iSAID 脚本：inside / cross-x / cross-y / corner / larger-than-tile，
  bbox 严格包含口径，贴图像边缘不算截断。
- difficult=1 实例单列计数，不进主比例（DOTA 惯例）。

输出：stdout 摘要 + sam3_ovis/logs/v4_s0_recon_dota.json。
"""

import json
import os
import sys
from collections import Counter

from PIL import Image

DATA_ROOT = "/data/wangcheng/dataset/DOTA"
SPLITS = {
    "train": "train/images/part*/images",
    "val": "val/images/part*/images",
}
LABELS = {
    "train": "train/labelTxt-v1.5/DOTA-v1.5_train",
    "val": "val/labelTxt-v1.5/DOTA-v1.5_val",
}
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "v4_s0_recon_dota.json")

TILE = 1024
OVERLAPS = {"A200": 200, "B128": 128}


def tile_anchors(size, tile, stride):
    if size <= tile:
        return [0]
    a = list(range(0, size - tile + 1, stride))
    if a[-1] != size - tile:
        a.append(size - tile)
    return a


def classify(bbox, W, H, tile, stride):
    x1, y1, x2, y2 = bbox
    ax = tile_anchors(W, tile, stride)
    ay = tile_anchors(H, tile, stride)
    for tx in ax:
        if tx <= x1 and x2 <= tx + tile:
            for ty in ay:
                if ty <= y1 and y2 <= ty + tile:
                    return "inside"
    lx = [L for L in ax[1:] if L != W - tile]
    ly = [L for L in ay[1:] if L != H - tile]
    cx = any(x1 < L < x2 for L in lx)
    cy = any(y1 < L < y2 for L in ly)
    if cx and cy:
        return "corner"
    if cx:
        return "cross-x"
    if cy:
        return "cross-y"
    return "larger-than-tile"


def main():
    import glob

    result = {"tile": TILE, "version": "DOTA-v1.5", "protocols": {}, "splits": {}}
    for split in SPLITS:
        imgs = {}
        for p in glob.glob(os.path.join(DATA_ROOT, SPLITS[split], "*.png")):
            imgs[os.path.basename(p)[:-4]] = p  # 去 .png，与标签键一致
        label_dir = os.path.join(DATA_ROOT, LABELS[split])
        labels = {}
        for p in glob.glob(os.path.join(label_dir, "*.txt")):
            labels[os.path.basename(p)[:-4]] = p

        sizes = {}
        for name, p in imgs.items():
            with Image.open(p) as im:
                sizes[name] = im.size  # (W, H)

        ws = sorted(w for w, _ in sizes.values())
        hs = sorted(h for _, h in sizes.values())
        stat = {
            "n_images": len(imgs),
            "n_labeled_images": len(labels),
            "W_min_med_max": [ws[0], ws[len(ws) // 2], ws[-1]],
            "H_min_med_max": [hs[0], hs[len(hs) // 2], hs[-1]],
            "n_images_gt_tile": sum(1 for w, h in sizes.values() if w > TILE or h > TILE),
            "n_instances": 0,
            "n_difficult": 0,
            "area_quantiles": None,
            "protocols": {},
            "per_cat": {},
        }

        areas = []
        ntile_med = []
        for name in sorted(sizes):
            W, H = sizes[name]
            ax = tile_anchors(W, TILE, TILE - 200)
            ay = tile_anchors(H, TILE, TILE - 200)
            ntile_med.append(len(ax) * len(ay))
        ntile_med.sort()
        stat["tiles_per_image_median"] = ntile_med[len(ntile_med) // 2]

        matched = set(imgs) & set(labels)
        for name in sorted(matched):
            W, H = sizes[name]
            ax_all = tile_anchors(W, TILE, TILE - 200)
            ay_all = tile_anchors(H, TILE, TILE - 200)
            W, H = sizes[name]
            with open(labels[name]) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 9 or parts[0] in ("imagesource", "gsd"):
                        continue
                    try:
                        cs = list(map(float, parts[:8]))
                    except ValueError:
                        continue
                    cls_name, diff = parts[8], int(parts[9]) if len(parts) > 9 else 0
                    x1, x2 = min(cs[0::2]), max(cs[0::2])
                    y1, y2 = min(cs[1::2]), max(cs[1::2])
                    area = (x2 - x1) * (y2 - y1)
                    stat["n_instances"] += 1
                    if diff:
                        stat["n_difficult"] += 1
                        continue
                    areas.append(area)
                    pc = stat["per_cat"].setdefault(cls_name, Counter())
                    # 多 tile 可见性：bbox 与多少个 tile 区域相交（overlap 区实例 ≥2）
                    vtile = 0
                    for tx in ax_all:
                        if tx < x2 and x1 < tx + TILE:
                            for ty in ay_all:
                                if ty < y2 and y1 < ty + TILE:
                                    vtile += 1
                    pc["vtile_hist"] = pc.get("vtile_hist", Counter())
                    pc["vtile_hist"][min(vtile, 4)] += 1
                    for pname, ov in OVERLAPS.items():
                        k = classify((x1, y1, x2, y2), W, H, TILE, TILE - ov)
                        pc[pname] = pc.get(pname, Counter())
                        pc[pname][k] += 1

        areas.sort()
        n = len(areas)
        stat["area_quantiles"] = {
            "p05": areas[int(0.05 * n)], "p25": areas[int(0.25 * n)],
            "p50": areas[int(0.50 * n)], "p75": areas[int(0.75 * n)],
            "p95": areas[int(0.95 * n)],
        }

        for pname, ov in OVERLAPS.items():
            tot = sum(sum(pc[pname].values()) for pc in stat["per_cat"].values())
            cnt = Counter()
            for pc in stat["per_cat"].values():
                cnt.update(pc[pname])
            vhist = Counter()
            for pc in stat["per_cat"].values():
                vhist.update(pc.get("vtile_hist", Counter()))
            vtot = sum(vhist.values())
            stat["protocols"][pname] = {
                "overlap": ov,
                "evaluated_instances": tot,
                "counts": dict(cnt),
                "inside_pct": round(100.0 * cnt["inside"] / tot, 2),
                "cross_pct": round(100.0 * (cnt["cross-x"] + cnt["cross-y"]) / tot, 2),
                "corner_pct": round(100.0 * cnt["corner"] / tot, 2),
                "larger_pct": round(100.0 * cnt["larger-than-tile"] / tot, 2),
                "vtile_hist_pct": {
                    str(k): round(100.0 * v / vtot, 2) for k, v in sorted(vhist.items())
                },
                "per_cat_inside_pct": {
                    c: round(100.0 * pc[pname]["inside"] / sum(pc[pname].values()), 2)
                    for c, pc in sorted(stat["per_cat"].items())
                },
            }

        stat["per_cat_instances"] = {
            c: sum(pc["A200"].values()) for c, pc in stat["per_cat"].items()
        }
        result["splits"][split] = stat
        print(f"[{split}] images={stat['n_images']} labeled={stat['n_labeled_images']} "
              f"matched={len(matched)} inst={stat['n_instances']} "
              f"(difficult {stat['n_difficult']}) "
              f"W={stat['W_min_med_max']} H={stat['H_min_med_max']} "
              f"tiles/img(med)={stat['tiles_per_image_median']}")
        for pname in OVERLAPS:
            p = stat["protocols"][pname]
            print(f"  [{pname}] inside={p['inside_pct']}% cross={p['cross_pct']}% "
                  f"corner={p['corner_pct']}% larger={p['larger_pct']}%")

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"saved -> {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    sys.exit(main())
