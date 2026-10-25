#!/usr/bin/env python3
"""v4-S0 C0 统计 runner：GPU 推理阶段（design-v4.0 §5 / implementation-v4.0 S0 第 3 条）。

C0 = 独立二维 tiles + 固定全图融合。本脚本只做 GPU 部分：对按 seed 抽样的
DOTA-v1.5 val 原图，用与 sam3_ovis/eval/v4_s0_recon_dota.py 完全一致的
tile_anchors（含末 tile 吸附边缘）切窗，逐 tile 对 16 类跑 SAM3 text-only
（pool floor 0.05），把每个融合前候选按全局坐标落盘。

协议登记（2026-10-25，任何结果产生前冻结；详见 notes/reports/v4-s0-c0-run-plan.md）：
- 抽样：有标签 val 图，seed=20261025；优先抽含 >=1 个非 difficult 实例的图，
  不足再从其余有标签图补（均按名字排序后抽样，确定性）。名单在推理前写入 manifest。
- prompt：DOTA-v1.5 原始类名（连字符形式，如 "small-vehicle"），不做改写。
- 候选 = (tile-local mask, score)，score > pool floor；tight bbox 由 mask 重算；
  global = local + tile_origin（mask 贴到 [H,W] 画布后重新计算 tight bbox）。
- bbox_full_in_tile：tile-local tight bbox 不接触任何"内部分割边界"的 tile 边
  （贴图像边缘的 tile 边豁免——贴图像边缘不算截断，与摸底报告口径一致）。
  语义 = 该观察未被 tile 边界截断，可作 TG-CWP 可靠 anchor 候选。
- 本脚本不做融合、不做 NMS（离线在 v4_s0_c0_analyze.py 做，避免 GPU 阶段逻辑
  复杂化）。逐图 done ledger + flush；每图后 torch.cuda.empty_cache()。
- 显存纪律：tile 串行、同一时刻只有一份 image embedding；bf16 autocast、
  resolution=1008 下预计峰值 ~6 GB。

bbox 口径：全部 xyxy（pixel、含端点前开区间由 mask 决定）；COCO ann 才转 xywh。

输出：<log_root>/v4_s0_c0/{manifest.json, candidates.jsonl, done_images.jsonl}
断点续跑：同参数重跑，done ledger 中的图自动跳过。

Run:  source sam3_ovis/scripts/load_environment.sh
      CUDA_VISIBLE_DEVICES=0 "$PYTHON" sam3_ovis/eval/v4_s0_c0_run.py --n-images 100
自测（纯 CPU、不 import torch）："$PYTHON" sam3_ovis/eval/v4_s0_c0_run.py --self-test
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

from v4_s0_recon_dota import tile_anchors  # noqa: E402  冻结的锚点逻辑，逐字照抄

DOTA_ROOT = "/data/wangcheng/dataset/DOTA"
IMAGE_GLOB = "val/images/part*/images/*.png"
LABEL_DIR = "val/labelTxt-v1.5/DOTA-v1.5_val"

# DOTA-v1.5 十六类（与摸底结果 v4_s0_recon_dota.json 的 per_cat 键集合核对一致）
DOTA15_CLASSES = (
    "baseball-diamond", "basketball-court", "bridge", "container-crane",
    "ground-track-field", "harbor", "helicopter", "large-vehicle", "plane",
    "roundabout", "ship", "small-vehicle", "soccer-ball-field",
    "storage-tank", "swimming-pool", "tennis-court",
)


def default_log_root():
    env = os.environ.get("SAM3_OVIS_LOG_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "logs"


def parse_label_txt(path):
    """DOTA labelTxt -> [({poly: [x1,y1,...,x4,y4], name: str, difficult: int})]。

    跳过头两行 imagesource/gsd（按内容识别，不按行号，容错无头文件）。
    与 v4_s0_recon_dota.py 的解析规则一致；v4_s0_c0_analyze.py 复用本函数。
    """
    out = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9 or parts[0] in ("imagesource", "gsd"):
                continue
            try:
                poly = [float(v) for v in parts[:8]]
            except ValueError:
                continue
            difficult = int(parts[9]) if len(parts) > 9 else 0
            out.append({"poly": poly, "name": parts[8], "difficult": difficult})
    return out


def tight_bbox(m):
    """bool mask -> (x1, y1, x2, y2)（xyxy，x2/y2 为最后一个非零像素 +1）。"""
    ys, xs = np.any(m, axis=1), np.any(m, axis=0)
    if not ys.any():
        return None
    xs_i, ys_i = np.where(xs)[0], np.where(ys)[0]
    return (int(xs_i[0]), int(ys_i[0]), int(xs_i[-1]) + 1, int(ys_i[-1]) + 1)


def local_bbox_full_in_tile(bbox_local, tx, ty, tw, th, W, H):
    """tile-local tight bbox 是否未接触 tile 的内部分割边（贴图像边缘的边豁免）。

    bbox_local: (x1, y1, x2, y2) tile-local；(tx, ty) tile 全局原点；
    (tw, th) 实际 crop 尺寸（图小于 tile 时 < tile）；(W, H) 原图尺寸。
    """
    bx1, by1, bx2, by2 = bbox_local
    left_ok = bx1 > 0 or tx == 0            # tile 左边即图像左边
    top_ok = by1 > 0 or ty == 0
    right_ok = bx2 < tw or tx + tw == W     # tile 右边即图像右边
    bottom_ok = by2 < th or ty + th == H
    return bool(left_ok and top_ok and right_ok and bottom_ok)


def paste_global(canvas, mask_local, tx, ty):
    """把 tile-local bool mask 贴到全局画布。调用方负责用 clear_tile 复位。"""
    th, tw = mask_local.shape
    canvas[ty:ty + th, tx:tx + tw] = mask_local
    return th, tw


def clear_tile(canvas, tx, ty, th, tw):
    canvas[ty:ty + th, tx:tx + tw] = 0


def self_test():
    """CPU 自测：tile_anchors 边界行为 + global<->local 坐标往返 + RLE/IoU。

    返回断言总数（脚本要求 >= 6）；任何一条失败直接 AssertionError。
    """
    n_checks = 0

    # --- tile_anchors：原点、覆盖、末 tile 吸附 ---
    assert tile_anchors(800, 1024, 824) == [0], "图 <= tile 时唯一锚点是 0"
    n_checks += 1
    a = tile_anchors(2000, 1024, 824)
    assert a == [0, 824, 976] and a[-1] == 2000 - 1024, \
        f"末 tile 必须吸附边缘: {a}"
    n_checks += 1
    for size in (1024, 1025, 1848, 1849, 5000, 12029):  # 边界: 恰好整除/余1
        ax = tile_anchors(size, 1024, 824)
        assert ax[0] == 0 and ax[-1] + 1024 == size and ax == sorted(set(ax)), \
            f"锚点必须覆盖且末锚点吸附: size={size}, anchors={ax}"
        steps = [b - c for b, c in zip(ax[1:], ax[:-1])]
        assert all(s <= 824 for s in steps), f"步长超上限: size={size}, {steps}"
    n_checks += 1

    # --- global <-> local 往返（跨界实例：mask 贴着 tile 右/下边） ---
    W, H, TILE = 2000, 1500, 1024
    tx, ty = 824, 0
    mask_local = np.zeros((TILE, TILE), bool)
    mask_local[500:520, 1000:1024] = True          # 跨 tile 右边界
    canvas = np.zeros((H, W), np.uint8)
    paste_global(canvas, mask_local, tx, ty)
    assert canvas[500:520, 1824:1848].all() and canvas.sum() == 20 * 24, \
        "全局贴图位置错误（原点位移）"
    n_checks += 1
    bbox_g = tight_bbox(canvas)
    bbox_l = tight_bbox(mask_local)
    assert bbox_g == (bbox_l[0] + tx, bbox_l[1] + ty,
                      bbox_l[2] + tx, bbox_l[3] + ty), \
        f"global bbox 必须等于 local bbox 平移: {bbox_g} vs {bbox_l}"
    n_checks += 1
    round_trip = canvas[ty:ty + TILE, tx:tx + TILE].astype(bool)
    assert np.array_equal(round_trip, mask_local), "global 裁回 local 必须无损"
    n_checks += 1

    # --- full_in_tile 判定 ---
    f = local_bbox_full_in_tile
    assert f((10, 10, 100, 100), 824, 824, 1024, 1024, W, H) is True, \
        "tile 内部 bbox 应判 full"
    assert f((0, 10, 100, 100), 824, 824, 1024, 1024, W, H) is False, \
        "接触内部左边界应判截断"
    assert f((0, 10, 100, 100), 0, 824, 1024, 1024, W, H) is True, \
        "贴图像左边缘的 tile 左边应豁免"
    assert f((10, 10, 1024, 100), 824, 824, 1024, 1024, W, H) is False, \
        "接触 tile 右边界应判截断"
    assert f((10, 10, 1024, 100), 976, 824, 1024, 1024, 2000, H) is True, \
        "贴图像右边缘的末 tile 右边应豁免"
    n_checks += 1

    # --- pycocotools：RLE round-trip + 合成 mask IoU ---
    m = np.zeros((50, 60), np.uint8)
    m[10:30, 20:45] = 1                            # 面积 20*25=500
    rle = mask_util.encode(np.asfortranarray(m))
    back = mask_util.decode(rle)
    assert np.array_equal(back, m) and int(mask_util.area(rle)) == 500, \
        "RLE 编解码往返必须无损"
    n_checks += 1
    m2 = np.zeros((50, 60), np.uint8)
    m2[20:30, 20:45] = 1                            # 面积 250，交 250 并 500
    rle2 = mask_util.encode(np.asfortranarray(m2))
    iou = float(mask_util.iou([rle], [rle2], [0])[0][0])
    assert abs(iou - 0.5) < 1e-9, f"合成 IoU 应为 0.5，得到 {iou}"
    n_checks += 1
    # 与分析端 OBB polygon GT 的互通：同一 [H,W] 画布上，贴入 tile 的全局
    # mask RLE 与 polygon RLE 完全重合时 IoU 必须为 1（跨尺寸 RLE 的 IoU 是
    # pycocotools C 接口的未定义行为，双方必须同尺寸——这里锁定该约定）。
    GH, GW = 100, 120
    gcanvas = np.zeros((GH, GW), np.uint8)
    gmask = np.zeros((30, 45), bool)                # tile-local (th=30, tw=45)
    gmask[10:30, 20:45] = True                      # 20x25=500 px
    paste_global(gcanvas, gmask, 55, 40)            # tile 原点 (tx=55, ty=40)
    rle_g = mask_util.encode(np.asfortranarray(gcanvas))
    # 贴入后的全局矩形：行 50..69、列 75..99
    poly = [75.0, 50.0, 100.0, 50.0, 100.0, 70.0, 75.0, 70.0]
    rle_p = mask_util.merge(
        mask_util.frPyObjects([poly], GH, GW))
    iou_g = float(mask_util.iou([rle_g], [rle_p], [0])[0][0])
    assert iou_g == 1.0, \
        f"全局 mask RLE 与重合 OBB polygon RLE IoU 应为 1，得到 {iou_g}"
    assert int(mask_util.area(rle_g)) == 500, \
        f"贴入全局画布后的面积应保持不变(20x25)，得到 {mask_util.area(rle_g)}"
    n_checks += 1

    print(f"[self-test] v4_s0_c0_run: {n_checks} 组断言全部通过")
    return n_checks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--pool-floor", type=float, default=0.05)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="默认 <SAM3_OVIS_LOG_ROOT>/v4_s0_c0")
    ap.add_argument("--self-test", action="store_true",
                    help="只跑 CPU 自测（不 import torch、不占 GPU）")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    if args.overlap >= args.tile:
        ap.error("--overlap 必须 < --tile")
    stride = args.tile - args.overlap
    run_dir = args.run_dir or (default_log_root() / "v4_s0_c0")
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 数据索引与抽样（确定性，先于任何推理） ----------
    img_paths = {}
    for p in glob.glob(os.path.join(DOTA_ROOT, IMAGE_GLOB)):
        img_paths[os.path.basename(p)[:-4]] = p
    label_paths = {}
    for p in glob.glob(os.path.join(DOTA_ROOT, LABEL_DIR, "*.txt")):
        label_paths[os.path.basename(p)[:-4]] = p
    labeled = {n: label_paths[n] for n in sorted(set(img_paths) & set(label_paths))}
    if not labeled:
        print(f"[error] 未找到有标签图: {DOTA_ROOT}", file=sys.stderr)
        return 2

    with_inst, wo_inst = [], []
    info = {}
    for name, lpath in labeled.items():
        insts = parse_label_txt(lpath)
        n_nd = sum(1 for it in insts if not it["difficult"])
        with Image.open(img_paths[name]) as im:  # 只读头，不解码
            W, H = im.size
        info[name] = {"W": W, "H": H, "n_instances": len(insts),
                      "n_nondifficult": n_nd, "label_path": lpath,
                      "image_path": img_paths[name]}
        (with_inst if n_nd > 0 else wo_inst).append(name)

    rng = np.random.default_rng(args.seed)
    n_pref = min(args.n_images, len(with_inst))
    chosen = list(rng.choice(np.array(with_inst, dtype=object), size=n_pref,
                             replace=False))
    if len(chosen) < args.n_images and wo_inst:
        n_fill = min(args.n_images - len(chosen), len(wo_inst))
        chosen += list(rng.choice(np.array(wo_inst, dtype=object), size=n_fill,
                                  replace=False))
    chosen = sorted(chosen)

    try:
        import subprocess
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2])).decode().strip()
    except Exception:
        sha = "unknown"
    manifest = {
        "seed": args.seed, "tile": args.tile, "overlap": args.overlap,
        "stride": stride, "pool_floor": args.pool_floor,
        "classes": list(DOTA15_CLASSES),
        "prompt_policy": "raw DOTA-v1.5 class names",
        "n_images_requested": args.n_images, "n_images_chosen": len(chosen),
        "images": {n: info[n] for n in chosen},
        "code_sha": sha, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": __file__,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[run] {len(chosen)} images sampled (preferred pool {len(with_inst)}, "
          f"fill {len(chosen) - n_pref}) -> {run_dir / 'manifest.json'}", flush=True)

    # ---------- 2. 断点续跑 ledger ----------
    cache_path = run_dir / "candidates.jsonl"
    done_path = run_dir / "done_images.jsonl"
    done = set()
    if done_path.exists():
        with open(done_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["img"])
                except (json.JSONDecodeError, KeyError):
                    break  # 半行（崩溃时）：该图重跑
    todo = [n for n in chosen if n not in done]
    print(f"[run] {len(todo)} to process ({len(done)} already done)", flush=True)
    if not todo:
        return 0

    # ---------- 3. SAM3（延迟 import：--self-test 路径无需 torch/GPU） ----------
    ckpt = os.environ.get("SAM3_CKPT")
    if not ckpt:
        print("[error] 未设置 SAM3_CKPT；先 source sam3_ovis/scripts/load_environment.sh",
              file=sys.stderr)
        return 2
    import torch  # noqa: WPS433
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
    from sam3.model_builder import build_sam3_image_model  # noqa: E402

    model = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=args.pool_floor)

    cand_f = open(cache_path, "a")
    done_f = open(done_path, "a")
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for k, name in enumerate(todo):
            ts = time.time()
            W, H = info[name]["W"], info[name]["H"]
            ax = tile_anchors(W, args.tile, stride)
            ay = tile_anchors(H, args.tile, stride)
            canvas = np.zeros((H, W), np.uint8)  # 全局画布：逐候选复用
            n_cand = 0
            with Image.open(info[name]["image_path"]) as im:
                for ty in ay:
                    for tx in ax:
                        # 按图像边界裁实际尺寸：小于 tile 的图 PIL 会把超界
                        # crop 零填充到请求尺寸，mask(1024²) 贴不进小画布；
                        # 直接给 SAM3 实际尺寸（t2 管线同口径，mask=输入尺寸）
                        crop = im.crop((tx, ty,
                                        min(tx + args.tile, W),
                                        min(ty + args.tile, H)))
                        tw, th = crop.size  # 图 <= tile 时 < tile
                        state = processor.set_image(crop.convert("RGB"))
                        for cls in DOTA15_CLASSES:
                            state = processor.set_text_prompt(
                                state=state, prompt=cls)
                            sc = state["scores"]
                            for j in range(int(sc.shape[0])
                                           if sc is not None else 0):
                                mask_l = state["masks"][j, 0].detach().cpu() \
                                              .numpy().astype(bool)
                                if mask_l.shape != (th, tw):
                                    raise RuntimeError(
                                        f"mask 尺寸 {mask_l.shape} != crop "
                                        f"{(th, tw)}（tile {tx},{ty}）")
                                paste_global(canvas, mask_l, tx, ty)
                                bbox_g = tight_bbox(canvas)  # global 重算
                                if bbox_g is None:  # 空 mask（理论不可达）
                                    clear_tile(canvas, tx, ty, th, tw)
                                    continue
                                bbox_l = (bbox_g[0] - tx, bbox_g[1] - ty,
                                          bbox_g[2] - tx, bbox_g[3] - ty)
                                rle = mask_util.encode(
                                    np.asfortranarray(canvas))
                                clear_tile(canvas, tx, ty, th, tw)
                                cand_f.write(json.dumps({
                                    "img": name, "W": W, "H": H,
                                    "tile_origin": [int(tx), int(ty)],
                                    "tile_size": [int(tw), int(th)],
                                    "cat_name": cls,
                                    "score": round(float(sc[j]), 6),
                                    "mask_rle": {
                                        "size": [int(rle["size"][0]),
                                                 int(rle["size"][1])],
                                        "counts": rle["counts"].decode("utf-8"),
                                    },
                                    "bbox_global": [int(v) for v in bbox_g],
                                    "bbox_local": [int(v) for v in bbox_l],
                                    "bbox_full_in_tile":
                                        local_bbox_full_in_tile(
                                            bbox_l, tx, ty, tw, th, W, H),
                                }) + "\n")
                                n_cand += 1
                        del state  # 释放该 tile 的 backbone 输出
            done_f.write(json.dumps({
                "img": name, "W": W, "H": H, "n_candidates": n_cand,
                "n_tiles": len(ax) * len(ay),
                "elapsed_s": round(time.time() - ts, 1)}) + "\n")
            cand_f.flush()
            done_f.flush()
            os.fsync(cand_f.fileno())
            os.fsync(done_f.fileno())
            torch.cuda.empty_cache()
            if (k + 1) % 10 == 0 or k + 1 == len(todo):
                el = time.time() - t0
                print(f"  {k + 1}/{len(todo)} imgs, {el / (k + 1):.1f}s/img, "
                      f"{n_cand} cands last img", flush=True)
    cand_f.close()
    done_f.close()
    print(f"[run] done: {len(todo)} imgs in {(time.time() - t0) / 60:.1f} min "
          f"-> {cache_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
