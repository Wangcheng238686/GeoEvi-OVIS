#!/usr/bin/env python3
"""G2a 缓存段：冻结主干特征/查询/目标与 bank 候选的逐图落盘（协议
notes/reports/2026-11-16-g2a-protocol.md §3）。

全 bf16（权重 1.6GB + 激活 ~1GB，峰值预算 ≤3.5GB）。逐图 .pt 落盘，
已存在即跳过（断点续跑）。缓存根：/data/wangcheng/geoevi-g2a-cache
（磁盘预算登记 300MB/30 图）。

Run:  source sam3_ovis/scripts/load_environment.sh
      CUDA_VISIBLE_DEVICES=<freest> "$PYTHON" sam3_ovis/graph/g2a_cache.py \
          [--n-images 30] [--per-group 5]
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402
from graph.g1_qbridge_probe import box_iou_xyxy, grounding_once, pick_level

SEED = 20261116
POOL_FLOOR = 0.05
TARGET_MIN_SCORE = 0.3
BANK_MIN_SCORE = 0.15  # bank 候选单独门槛（0.3 下非目标预测过少，结果前修订）
BANK_TOPN = 4
GT_MATCH_IOU = 0.5
CACHE_ROOT = Path("/data/wangcheng/geoevi-g2a-cache")
LOG_ROOT = Path(os.environ.get(
    "SAM3_OVIS_LOG_ROOT",
    str(Path(__file__).resolve().parents[1] / "logs")))


def crop_mask(mask: np.ndarray):
    """bool mask -> (crop, x1, y1) 裁剪存储。"""
    ys, xs = np.any(mask, axis=1), np.any(mask, axis=0)
    if not ys.any():
        return None, 0, 0
    y1, y2 = np.where(ys)[0][[0, -1]]
    x1, x2 = np.where(xs)[0][[0, -1]]
    return mask[y1:y2 + 1, x1:x2 + 1], int(x1), int(y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=30)
    ap.add_argument("--per-group", type=int, default=5)
    args = ap.parse_args()
    torch.manual_seed(SEED)

    device = "cuda"
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DATASETS["iSAID"]
    gt = COCO(str(root / spec.val_json))
    splits = json.load(open(LOG_ROOT / "g0_splits" / "manifest.json"))
    dev_groups = splits["splits"]["dev"]["groups"]
    # 组级 70/30 train/val（协议 §3）；交错迭代保证两 split 同时填充
    n_tr = int(len(dev_groups) * 0.7)
    tr_groups = [g for i, g in enumerate(dev_groups) if i < n_tr]
    va_groups = [g for i, g in enumerate(dev_groups) if i >= n_tr]
    group_split = {g: ("train" if i < n_tr else "val")
                   for i, g in enumerate(dev_groups)}
    order = []
    for i in range(max(len(tr_groups), len(va_groups))):
        if i < len(tr_groups):
            order.append(tr_groups[i])
        if i < len(va_groups):
            order.append(va_groups[i])

    cat_ids = gt.getCatIds()
    cat_names = [gt.loadCats([c])[0]["name"] for c in cat_ids]

    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False,
        enable_inst_interactivity=True)
    # 权重 fp32 + autocast bf16：vendored 代码内有 fp32 常量（几何 prompt 等），
    # 整模型 bf16 会 dtype 碰撞；autocast 自动处理混精度。峰值 ~4.3GB，
    # 只在余量 >=5GB 的卡上运行（协议 §3 预算登记的实测替代）。
    model = model.to(device=device).eval()
    processor = Sam3Processor(model, confidence_threshold=POOL_FLOOR,
                              device=device)

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    (CACHE_ROOT / "group_split.json").write_text(json.dumps(group_split,
                                                            indent=1))
    manifest = {"seed": SEED, "dtype": "bfloat16", "images": {}}

    n_done, n_groups_used = 0, 0
    anns_by_img = {}
    for a in gt.dataset["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    imgs_by_group = {}
    for im in gt.dataset["images"]:
        imgs_by_group.setdefault(
            im["file_name"].replace(".png", "").split("_")[0], []).append(im)

    for g in order:
        if n_done >= args.n_images:
            break
        got = 0
        for im in sorted(imgs_by_group.get(g, []), key=lambda x: x["id"]):
            if got >= args.per_group or n_done >= args.n_images:
                break
            if im["id"] not in anns_by_img:
                continue
            out_path = CACHE_ROOT / f"img_{im['id']}.pt"
            if out_path.exists():
                got += 1
                continue
            W, H = im["width"], im["height"]
            image = Image.open(root / spec.images_dir /
                               im["file_name"]).convert("RGB")
            with torch.no_grad(), torch.autocast("cuda",
                                                 dtype=torch.bfloat16):
                img_bf = processor.transform(
                    v2.functional.to_image(image)).unsqueeze(0).to(device)
                state = {"backbone_out": model.backbone.forward_image(img_bf),
                         "original_height": H, "original_width": W}
                first_pass = []
                bank_pool = []
                for cname in cat_names:
                    state = processor.set_text_prompt(prompt=cname,
                                                      state=state)
                    sc = state["scores"]
                    for j in range(int(sc.shape[0])
                                   if sc is not None else 0):
                        s = float(sc[j])
                        if s < TARGET_MIN_SCORE:
                            if s >= BANK_MIN_SCORE:
                                bank_pool.append({
                                    "cls": cname, "score": s,
                                    "mask": state["masks"][j, 0]
                                    .cpu().numpy(),
                                    "box": state["boxes"][j].tolist()})
                            continue
                        first_pass.append({
                            "cls": cname,
                            "box": state["boxes"][j].tolist(),
                            "mask": state["masks"][j, 0].cpu().numpy(),
                            "score": s})
                if not first_pass:
                    continue
                first_pass.sort(key=lambda t: -t["score"])
                targets = first_pass[:16]

                # 15 文本 queries/prompt/boxes（原生 grounding，no_grad）
                feat_bchw, pos, grid = pick_level(model, state["backbone_out"])
                per_text = {}
                for cname in cat_names:
                    r = grounding_once(model, state["backbone_out"],
                                       processor.find_stage, cname, device)
                    b = r["boxes"]
                    boxes_px = torch.stack([
                        torch.clamp(b[:, 0] - b[:, 2] / 2, min=0),
                        torch.clamp(b[:, 1] - b[:, 3] / 2, min=0),
                        torch.clamp(b[:, 0] + b[:, 2] / 2, max=1),
                        torch.clamp(b[:, 1] + b[:, 3] / 2, max=1)], dim=1) \
                        * torch.tensor([W, H, W, H], device=device)
                    per_text[cname] = {
                        "queries": r["queries"][0].float(),
                        "prompt": r["prompt"].float(),
                        "prompt_mask": r["prompt_mask"],
                        "boxes_px": boxes_px}

                # GT 匹配（标签 = 最佳 IoU>=0.5 的 GT 类名）
                gt_rles = []
                for a in anns_by_img[im["id"]]:
                    rles = mask_util.frPyObjects(
                        a["segmentation"], H, W)
                    gt_rles.append((a["category_id"],
                                    mask_util.merge(rles)))

            t_out, b_out = [], []
            for t in targets:
                t_rle = mask_util.encode(
                    np.asfortranarray(t["mask"].astype(np.uint8)))
                best_iou, best_cat = 0.0, None
                for cid, grle in gt_rles:
                    iou = float(mask_util.iou([t_rle], [grle], [0])[0][0])
                    if iou > best_iou:
                        best_iou, best_cat = iou, cid
                crop, x1, y1 = crop_mask(t["mask"])
                if crop is None:
                    continue
                t_out.append({
                    "cls": t["cls"], "score": t["score"], "box": t["box"],
                    "mask_crop": torch.from_numpy(crop), "x1": x1, "y1": y1,
                    "mask_hw": (H, W),
                    "gt_label": (gt.loadCats([best_cat])[0]["name"]
                                 if best_iou >= GT_MATCH_IOU else None),
                    "gt_iou": round(best_iou, 4),
                    "pred_eq_gt": (gt.loadCats([best_cat])[0]["name"]
                                   == t["cls"]
                                   if best_iou >= GT_MATCH_IOU else None)})
            for c in sorted(bank_pool + first_pass,
                            key=lambda t: -t["score"]):
                if len(b_out) >= BANK_TOPN:
                    break
                if all(box_iou_xyxy(c["box"], t["box"]) < 0.5
                       for t in targets):
                    crop, x1, y1 = crop_mask(c["mask"])
                    if crop is None:
                        continue
                    b_out.append({"cls": c["cls"], "score": c["score"],
                                  "mask_crop": torch.from_numpy(crop),
                                  "x1": x1, "y1": y1, "mask_hw": (H, W)})
            if not t_out or not b_out:
                continue

            rec = {
                "img_id": im["id"], "file": im["file_name"],
                "group": g, "split": group_split[g],
                "grid": list(grid),
                # canvas：[1,C,h,w] -> [HW,C]（flatten 在 4-D 上才生效）
                "canvas": feat_bchw.flatten(2)[0].t().float(),
                "pos": pos[:, 0].float(),                        # [HW,C]
                "targets": t_out, "bank": b_out, "texts": per_text,
            }
            for k in ("canvas", "pos"):
                rec[k] = rec[k].to(torch.bfloat16)
            for cname in per_text:
                per_text[cname]["queries"] = per_text[cname]["queries"] \
                    .to(torch.bfloat16)
                per_text[cname]["prompt"] = per_text[cname]["prompt"] \
                    .to(torch.bfloat16)
            torch.save(rec, out_path)
            manifest["images"][str(im["id"])] = {
                "file": im["file_name"], "group": g, "split": group_split[g],
                "n_targets": len(t_out), "n_bank": len(b_out)}
            got += 1
            n_done += 1
            n_groups_used = len({v["group"]
                                 for v in manifest["images"].values()})
            print(f"[cache] {n_done}/{args.n_images} {im['file_name']} "
                  f"({g},{group_split[g]}) targets={len(t_out)} "
                  f"bank={len(b_out)}", flush=True)
            torch.cuda.empty_cache()

    (CACHE_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=1))
    splits_count = {}
    for v in manifest["images"].values():
        splits_count[v["split"]] = splits_count.get(v["split"], 0) + 1
    print(f"[cache] done: {n_done} imgs across {n_groups_used} groups, "
          f"split={splits_count} -> {CACHE_ROOT}", flush=True)


if __name__ == "__main__":
    main()
