#!/usr/bin/env python3
"""v4-S0 数据摸底：iSAID 原始大图 + 固定二维 tiling 模拟统计。

协议登记（数字产生前冻结，2026-10-25）：
- 数据：/data/wangcheng/dataset/iSAID train/val，COCO JSON 标注（全局坐标）。
- tiling 草案 A：tile=1024, overlap=200（stride=824）；
  tiling 草案 B：tile=1024, overlap=128（stride=896）。
  两案并列报告，仅供 S0 协议选择；正式 manifest 在任何 SAM3 结果运行前另行冻结。
- 实例归类用 bbox（严格包含：tile_x0 <= x1 且 x2 <= tile_x0+TILE）；
  mask 可能略超出 bbox，统计解释时保留该不确定性。
- 分类：
  * inside-1-tile：存在某 tile 完整包含该实例 bbox；
  * cross-x / cross-y：无 tile 完整包含，且 bbox 跨越 ≥1 条 x 网格线 / y 网格线；
  * corner：x、y 两方向都跨。
- 网格线 = 相邻 tile 锚点差（stride）整数倍位置，且剔除贴边（首尾 tile 与图像边缘
  重合的线不算跨界——实例贴图像边不算被截断）。
- 抽样：全量 train+val 的 bbox 统计（JSON 已含尺寸，无需读图）。

输出：stdout 摘要 + sam3_ovis/logs/v4_s0_recon_isaid.json。
"""

import json
import os
import sys
from collections import Counter

DATA_ROOT = "/data/wangcheng/dataset/iSAID"
SPLITS = {"train": "train/Annotations/iSAID_train.json", "val": "val/Annotations/iSAID_val.json"}
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "v4_s0_recon_isaid.json")

TILE = 1024
OVERLAPS = {"A200": 200, "B128": 128}


def tile_anchors(size: int, tile: int, stride: int):
    """Tile 左上角锚点序列，末 tile 吸附到右/下边缘保证全覆盖。"""
    if size <= tile:
        return [0]
    a = list(range(0, size - tile + 1, stride))
    if a[-1] != size - tile:
        a.append(size - tile)
    return a


def classify(bbox, W, H, tile, stride):
    """返回 'inside' / 'cross-x' / 'cross-y' / 'corner'。"""
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + w, y + h
    ax = tile_anchors(W, tile, stride)
    ay = tile_anchors(H, tile, stride)
    for tx in ax:
        if tx <= x1 and x2 <= tx + tile:
            for ty in ay:
                if ty <= y1 and y2 <= ty + tile:
                    return "inside"
    # 网格线：stride 整数倍处，且不与图像边缘重合（贴图边不算截断）
    def crosses(v1, v2, lines):
        return any(v1 < L < v2 for L in lines)

    lx = [L for L in ax[1:] if L != W - tile]  # 首锚点 0 无左线；尾锚点对应右缘线
    ly = [L for L in ay[1:] if L != H - tile]
    cx = crosses(x1, x2, lx)
    cy = crosses(y1, y2, ly)
    if cx and cy:
        return "corner"
    if cx:
        return "cross-x"
    if cy:
        return "cross-y"
    # 无 tile 完整包含、也不跨内网格线：实例比 tile 大
    return "larger-than-tile"


def main():
    result = {"tile": TILE, "protocols": {}, "splits": {}}
    for split, rel in SPLITS.items():
        path = os.path.join(DATA_ROOT, rel)
        with open(path) as f:
            d = json.load(f)
        imgs = {im["id"]: im for im in d["images"]}
        cats = {c["id"]: c["name"] for c in d["categories"]}

        img_dir = os.path.join(DATA_ROOT, split, "images")
        missing = [im for im in d["images"] if "width" not in im]
        if missing:
            from PIL import Image  # train JSON 无尺寸字段，读 PNG header 补齐

            for im in missing:
                with Image.open(os.path.join(img_dir, im["file_name"])) as ig:
                    im["width"], im["height"] = ig.size
        sizes = [(im["width"], im["height"]) for im in d["images"]]
        ws = sorted(s[0] for s in sizes)
        hs = sorted(s[1] for s in sizes)

        split_stat = {
            "n_images": len(imgs),
            "n_instances": len(d["annotations"]),
            "n_categories": len(cats),
            "W_min_med_max": [ws[0], ws[len(ws) // 2], ws[-1]],
            "H_min_med_max": [hs[0], hs[len(hs) // 2], hs[-1]],
            "n_images_gt_tile": sum(1 for w, h in sizes if w > TILE or h > TILE),
            "tiles_per_image_median": None,
            "area_quantiles": None,
            "per_cat": {},
        }

        areas = sorted(a["area"] for a in d["annotations"])
        n = len(areas)
        split_stat["area_quantiles"] = {
            "p05": areas[int(0.05 * n)], "p25": areas[int(0.25 * n)],
            "p50": areas[int(0.50 * n)], "p75": areas[int(0.75 * n)],
            "p95": areas[int(0.95 * n)],
        }

        ntile = []
        for w, h in sizes:
            ax = tile_anchors(w, TILE, TILE - 200)
            ay = tile_anchors(h, TILE, TILE - 200)
            ntile.append(len(ax) * len(ay))
        ntile.sort()
        split_stat["tiles_per_image_median"] = ntile[len(ntile) // 2]

        for pname, ov in OVERLAPS.items():
            stride = TILE - ov
            cls = Counter()
            per_cat = {}
            for a in d["annotations"]:
                im = imgs[a["image_id"]]
                k = classify(a["bbox"], im["width"], im["height"], TILE, stride)
                cls[k] += 1
                c = cats[a["category_id"]]
                pc = per_cat.setdefault(c, Counter())
                pc[k] += 1
            tot = sum(cls.values())
            split_stat["protocols"] = split_stat.get("protocols", {})
            split_stat["protocols"][pname] = {
                "overlap": ov,
                "counts": dict(cls),
                "inside_pct": round(100.0 * cls["inside"] / tot, 2),
                "cross_pct": round(100.0 * (cls["cross-x"] + cls["cross-y"]) / tot, 2),
                "corner_pct": round(100.0 * cls["corner"] / tot, 2),
                "larger_pct": round(100.0 * cls["larger-than-tile"] / tot, 2),
                "per_cat_inside_pct": {
                    c: round(100.0 * pc["inside"] / sum(pc.values()), 2)
                    for c, pc in sorted(per_cat.items())
                },
            }
        # 类别覆盖
        cat_count = Counter(cats[a["category_id"]] for a in d["annotations"])
        split_stat["per_cat_instances"] = dict(cat_count.most_common())
        result["splits"][split] = split_stat
        print(f"[{split}] images={split_stat['n_images']} "
              f"inst={split_stat['n_instances']} "
              f"W(med)={split_stat['W_min_med_max'][1]} H(med)={split_stat['H_min_med_max'][1]} "
              f"gt-tile images={split_stat['n_images_gt_tile']} "
              f"tiles/img(med)={split_stat['tiles_per_image_median']}")
        for pname in OVERLAPS:
            p = split_stat["protocols"][pname]
            print(f"  [{pname}] inside={p['inside_pct']}% cross={p['cross_pct']}% "
                  f"corner={p['corner_pct']}% larger={p['larger_pct']}%")

    os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"saved -> {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    sys.exit(main())
