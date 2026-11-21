#!/usr/bin/env python3
"""G4b 管线协议审计（GPU 探针）。报告：
notes/reports/2026-11-22-pipeline-protocol-audit.md。

探针（全部单卡短跑，GPU0，流程照抄 g2a_cache.py / g3b_map_gate.py：
冻结 SAM3、逐类文本枚举、pool floor 过滤；登记见报告 §审计 trail）：
A floor      —— pool floor 0.05→0.01，5 张漏检最多的 val 图；逐候选解耦
                记录 cls-only sigmoid / presence sigmoid / fused（presence
                耦合核查），新低分带候选是否覆盖原漏检 GT。
B resolution —— 800（原图，恒等 resize）/ 1008（管线现状）/ 1260，2 张
                高漏检图 + 1 张非方形图（731×1836，长边被压）。
C prompt     —— 类名原样（Small_Vehicle，含下划线/大小写）vs 自然化
                （small vehicle），2 张图。
D merged     —— 15 类合并为单条 ". " 分隔提示 vs 逐类，1 张图。

Run:  source sam3_ovis/scripts/load_environment.sh
      CUDA_VISIBLE_DEVICES=0 "$PYTHON" sam3_ovis/graph/g4b_gpu_probes.py \
          [--phase all|floor|resolution|prompt|merged]
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
from sam3.model import box_ops  # noqa: E402
from sam3.model.data_misc import interpolate  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402
from graph.g1_qbridge_probe import box_iou_xyxy

CACHE_ROOT = Path("/data/wangcheng/geoevi-g2a-cache")
LOG_ROOT = Path(os.environ.get(
    "SAM3_OVIS_LOG_ROOT",
    str(Path(__file__).resolve().parents[1] / "logs")))
OUT_DIR = LOG_ROOT / "g4b"
FLOOR_IMGS = [23520, 24193, 23966, 23434, 23519]
RES_IMGS = [23966, 23520, 22602]          # 22602 = P2177.png 731×1836
PROMPT_IMGS = [23962, 23517]
MERGED_IMGS = [23966]
SEED = 20261122


def _load_eval_ctx():
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DATASETS["iSAID"]
    gt = COCO(str(root / spec.val_json))
    cat_ids = gt.getCatIds()
    cat_names = [gt.loadCats([c])[0]["name"] for c in cat_ids]
    return gt, spec, cat_names


def _open_image(gt, spec, img_id):
    info = gt.loadImgs(img_id)[0]
    image = Image.open(Path(os.environ["SAM3_OVIS_DATA_ROOT"])
                       / spec.images_dir / info["file_name"]).convert("RGB")
    return image, info


def _gt_rles(gt, img_id, H, W):
    anns = [a for a in gt.dataset["annotations"]
            if a["image_id"] == img_id]
    out = []
    for a in anns:
        rles = mask_util.frPyObjects(a["segmentation"], H, W)
        out.append((a, mask_util.merge(rles)))
    return out


def _tight_box(m):
    ys, xs = np.any(m, axis=1), np.any(m, axis=0)
    if not ys.any():
        return None
    return [int(np.where(xs)[0][0]), int(np.where(ys)[0][0]),
            int(np.where(xs)[0][-1]) + 1, int(np.where(ys)[0][-1]) + 1]


def enumerate_full(model, processor, image, H, W, names, floor,
                   device="cuda", chunk=64):
    """逐类枚举（g3b phase_cand 口径），但阈值可调，且解耦记录
    cls-only / presence / fused。返回候选 dict 列表（box 为原图坐标）。
    与 Sam3Processor._forward_grounding 逐步对齐。"""
    img_t = processor.transform(
        v2.functional.to_image(image)).unsqueeze(0).to(device)
    out = []
    with torch.inference_mode():        # vitdet 融合核要求（同 processor）
        backbone_out = model.backbone.forward_image(img_t)
        for cname in names:
            text_outputs = model.backbone.forward_text([cname],
                                                       device=device)
            # 逐类浅拷贝 backbone_out（与 processor 每次 set_text_prompt
            # 的隔离语义对齐；枚举结果与 g3b 缓存逐候选一致，已验证）。
            bb = dict(backbone_out)
            bb.update(text_outputs)
            geo = model._get_dummy_prompt()
            res = model.forward_grounding(
                backbone_out=bb, find_input=processor.find_stage,
                geometric_prompt=geo, find_target=None)
            cls_p = res["pred_logits"].sigmoid().reshape(-1)
            pres_all = res["presence_logit_dec"].sigmoid().reshape(-1)
            # presence 是逐提示（类）标量门控：同一类的全部候选同乘
            pres = pres_all[:1].expand(cls_p.shape[0]) \
                if pres_all.numel() == 1 else pres_all
            fused = cls_p * pres
            if not getattr(enumerate_full, "_shapes_logged", False):
                enumerate_full._shapes_logged = True
                print(f"[shapes] pred_logits={tuple(res['pred_logits'].shape)}"
                      f" presence={tuple(res['presence_logit_dec'].shape)}"
                      f" pred_boxes={tuple(res['pred_boxes'].shape)}"
                      f" pred_masks={tuple(res['pred_masks'].shape)}",
                      flush=True)
            boxes_c = res["pred_boxes"].reshape(-1, 4)
            masks_l = res["pred_masks"].reshape(
                -1, *res["pred_masks"].shape[-2:])
            keep = torch.where(fused > floor)[0]
            if keep.numel() == 0:
                continue
            boxes_xy = box_ops.box_cxcywh_to_xyxy(boxes_c[keep]) \
                * torch.tensor([W, H, W, H], device=device)
            for s in range(0, keep.numel(), chunk):
                idx = keep[s:s + chunk]
                m = interpolate(masks_l[idx].unsqueeze(1), (H, W),
                                mode="bilinear",
                                align_corners=False).sigmoid() > 0.5
                m_np = m[:, 0].cpu().numpy().astype(bool)
                for t in range(m_np.shape[0]):
                    if not m_np[t].any():
                        continue
                    g = int(idx[t])
                    box = _tight_box(m_np[t])
                    rle = mask_util.encode(np.asfortranarray(
                        m_np[t].astype(np.uint8)))
                    out.append({
                        "cls": cname,
                        "score": float(fused[g]),
                        "cls_only": float(cls_p[g]),
                        "presence": float(pres[g]),
                        "box": [float(x) for x in boxes_xy[t].tolist()],
                        "box_tight": box,
                        "rle": {"size": list(rle["size"]),
                                "counts": rle["counts"].decode("utf-8")}})
    return out


def simple_enum(model, processor, image, H, W, names, floor, device="cuda"):
    """processor 直通枚举（resolution/prompt/merged 用；不逐候选解耦）。"""
    cands = enumerate_full(model, processor, image, H, W, names, floor,
                           device)
    return cands


def eval_recall(cands, gt_list, ious=None):
    """三口径 any-class 召回 + same_exists；cands 用 tight bbox。"""
    H, W = gt_list[0][1]["size"] if gt_list else (0, 0)
    gt_rles = [g[1] for g in gt_list]
    crles = [{"size": c["rle"]["size"],
              "counts": c["rle"]["counts"].encode()} for c in cands]
    n_gt = len(gt_list)
    res = {"n_cands": len(cands), "n_gt": n_gt}
    if n_gt == 0 or not cands:
        for k in ("bbox50", "mask30", "center"):
            res[k] = 0.0
            res[k + "_same_exists"] = 0.0
        return res
    miou = mask_util.iou(crles, gt_rles, [0] * n_gt)
    dec = [mask_util.decode(cr).astype(bool) for cr in crles]
    cls_arr = np.array([c["cls"] for c in cands], dtype=object)
    for crit in ("bbox50", "mask30", "center"):
        hit = same = 0
        for j, (a, grle) in enumerate(gt_list):
            gname = a["category_name"]
            if crit == "bbox50":
                gx, gy, gw, gh = a["bbox"]
                gb = (gx, gy, gx + gw, gy + gh)
                col = np.array([box_iou_xyxy(c["box_tight"], gb)
                                for c in cands])
            elif crit == "mask30":
                col = miou[:, j]
            else:
                gx, gy, gw, gh = a["bbox"]
                cx = min(W - 1, int(gx + gw / 2))
                cy = min(H - 1, int(gy + gh / 2))
                col = np.array([d[cy, cx] for d in dec])
            th = 0.5 if crit == "bbox50" else (0.3 if crit == "mask30"
                                               else 0.5)
            m = col >= th if crit != "center" else col.astype(bool)
            if m.any():
                hit += 1
                if (cls_arr[m] == gname).any():
                    same += 1
        res[crit] = round(100 * hit / n_gt, 2)
        res[crit + "_same_exists"] = round(100 * same / n_gt, 2)
    return res


def _gt_rles_named(gt, img_id, H, W):
    name_by_id = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
    anns = [a for a in gt.dataset["annotations"]
            if a["image_id"] == img_id]
    out = []
    for a in anns:
        rles = mask_util.frPyObjects(a["segmentation"], H, W)
        a = dict(a)
        a["category_name"] = name_by_id[a["category_id"]]
        out.append((a, mask_util.merge(rles)))
    return out


def phase_floor(model, processor, gt, spec, results):
    cat_names = [gt.loadCats([c])[0]["name"]
                 for c in gt.getCatIds()]
    out = {}
    for img_id in FLOOR_IMGS:
        image, info = _open_image(gt, spec, img_id)
        H, W = info["height"], info["width"]
        cands01 = enumerate_full(model, processor, image, H, W,
                                 cat_names, 0.01)
        gts = _gt_rles_named(gt, img_id, H, W)
        # 与缓存 0.05 池对齐的子池
        cands05 = [c for c in cands01 if c["score"] >= 0.05]
        new = [c for c in cands01 if c["score"] < 0.05]
        r05 = eval_recall(cands05, gts)
        r01 = eval_recall(cands01, gts)
        # 原 0.05 漏检 GT（bbox50 与全口径两组）在 0.01 池下被哪些新候选覆盖
        rescue = {"n_missed_bbox50_at05": 0, "rescued_bbox50_by_new": 0,
                  "n_missed_all_at05": 0, "rescued_all_by_new": 0}
        crles_new = [{"size": c["rle"]["size"],
                      "counts": c["rle"]["counts"].encode()} for c in new]
        dec_new = [mask_util.decode(cr).astype(bool) for cr in crles_new]
        cls_new = np.array([c["cls"] for c in new], dtype=object)
        crles05 = [{"size": c["rle"]["size"],
                    "counts": c["rle"]["counts"].encode()}
                   for c in cands05]
        dec05 = [mask_util.decode(cr).astype(bool) for cr in crles05]
        for a, grle in gts:
            gx, gy, gw, gh = a["bbox"]
            gb = (gx, gy, gx + gw, gy + gh)
            cx = min(W - 1, int(gx + gw / 2))
            cy = min(H - 1, int(gy + gh / 2))
            hb = any(box_iou_xyxy(c["box_tight"], gb) >= 0.5
                     for c in cands05)
            hm = any(mask_util.iou([cr], [grle], [0])[0][0] >= 0.3
                     for cr in crles05)
            hc = any(d[cy, cx] for d in dec05)
            if not hb:
                rescue["n_missed_bbox50_at05"] += 1
                nb = any(box_iou_xyxy(c["box_tight"], gb) >= 0.5
                         for c in new)
                if nb:
                    rescue["rescued_bbox50_by_new"] += 1
            if not (hb or hm or hc):
                rescue["n_missed_all_at05"] += 1
                nb = any(box_iou_xyxy(c["box_tight"], gb) >= 0.5
                         for c in new)
                nm = any(mask_util.iou([cr], [grle], [0])[0][0] >= 0.3
                         for cr in crles_new)
                nc = any(d[cy, cx] for d in dec_new)
                if nb or nm or nc:
                    rescue["rescued_all_by_new"] += 1
        # presence 耦合：新低分候选里 cls-only 高但 presence 低的比例
        if new:
            hi_cls = [c for c in new if c["cls_only"] >= 0.5]
            pres_med = float(np.median([c["presence"] for c in new]))
            fused_med = float(np.median([c["score"] for c in new]))
            cls_med = float(np.median([c["cls_only"] for c in new]))
        else:
            hi_cls, pres_med, fused_med, cls_med = [], None, None, None
        out[str(img_id)] = {
            "n_cands01": len(cands01), "n_cands05": len(cands05),
            "n_new_band": len(new),
            "recall05": r05, "recall01": r01, "rescue": rescue,
            "new_band_stats": {
                "median_cls_only": cls_med, "median_presence": pres_med,
                "median_fused": fused_med,
                "n_cls_only>=0.5": len(hi_cls),
                "frac_cls_only>=0.5": round(len(hi_cls) / len(new), 3)
                if new else None},
            "score_hist_new": {
                "0.01-0.02": sum(1 for c in new if c["score"] < 0.02),
                "0.02-0.03": sum(1 for c in new if 0.02 <= c["score"] < 0.03),
                "0.03-0.05": sum(1 for c in new if 0.03 <= c["score"] < 0.05)}}
        print(f"[floor] {img_id}: cands01={len(cands01)} "
              f"05={len(cands05)} new={len(new)} "
              f"rec_bbox50 {r05['bbox50']}->{r01['bbox50']} "
              f"rec_center {r05['center']}->{r01['center']}", flush=True)
        torch.cuda.empty_cache()
    results["floor"] = out


def set_grid(model, g, device="cuda"):
    """把全局注意力块（window_size==0）的 rope freqs 重建为 g×g 网格。
    窗口块的 freqs 固定 24×24（与网格无关，不重建）；rel-pos / abs-pos
    在 forward 内插值，自适应。"""
    trunk = model.backbone.vision_backbone.trunk
    n = 0
    for blk in trunk.blocks:
        attn = getattr(blk, "attn", None)
        if (attn is not None and getattr(attn, "use_rope", False)
                and attn.freqs_cis is not None
                and getattr(blk, "window_size", 0) == 0):
            attn.input_size = (g, g)
            attn._setup_rope_freqs()
            attn.freqs_cis = attn.freqs_cis.to(device)
            if getattr(attn, "use_rope_real", False):
                attn.freqs_cis_real = attn.freqs_cis_real.to(device)
                attn.freqs_cis_imag = attn.freqs_cis_imag.to(device)
            n += 1
    return n


def phase_resolution(model, gt, spec, results, device="cuda"):
    """窗口注意力要求 patch 网格可被 24 整除（resolution=336 的倍数），
    800/1260 不合法。改测三个条件：
    - resize1008：管线现状（800 图上采样 ×1.26）；
    - resize1344：上采样 ×1.68（小目标有效分辨率 ×1.78 面积）；
    - pad1008   ：800 原生分辨率 + replicate 填充到 1008（内容不缩放，
                   小目标有效分辨率 = 原生，比 resize1008 低）。
    若召回随有效分辨率单调，则证明小目标召回受分辨率支配。"""
    from sam3.model.sam3_image_processor import Sam3Processor
    cat_names = [gt.loadCats([c])[0]["name"]
                 for c in gt.getCatIds()]
    out = {}
    for img_id in RES_IMGS[:2]:
        image, info = _open_image(gt, spec, img_id)
        H, W = info["height"], info["width"]
        gts = _gt_rles_named(gt, img_id, H, W)
        out[str(img_id)] = {"W": W, "H": H, "n_gt": len(gts)}
        for tag, res in (("resize1008", 1008), ("resize1344", 1344)):
            if res != 1008:
                n_re = set_grid(model, res // 14, device)
                print(f"[res] rebuilt rope for {n_re} global blocks "
                      f"grid={res // 14}", flush=True)
            proc = Sam3Processor(model, resolution=res, device=device)
            cands = enumerate_full(model, proc, image, H, W, cat_names,
                                   0.05, device)
            r = eval_recall(cands, gts)
            out[str(img_id)][tag] = r
            print(f"[res] {img_id} {tag}: n={r['n_cands']} "
                  f"bbox50={r['bbox50']} mask30={r['mask30']} "
                  f"center={r['center']}", flush=True)
            if res != 1008:
                set_grid(model, 72, device)   # 还原，供后续条件/探针使用
            torch.cuda.empty_cache()
        # pad1008：内容不缩放，GT 平移到画布坐标
        pad = (1008 - W) // 2, (1008 - H) // 2     # px, py
        canvas = Image.new("RGB", (1008, 1008))
        canvas.paste(image, (pad[0], pad[1]))
        gts_pad = []
        for a, rle in gts:
            m = mask_util.decode(rle).astype(bool)
            mp = np.pad(m, ((pad[1], pad[1]), (pad[0], pad[0])),
                        constant_values=False)
            a2 = dict(a)
            gx, gy, gw, gh = a["bbox"]
            a2["bbox"] = [gx + pad[0], gy + pad[1], gw, gh]
            gts_pad.append((a2, mask_util.encode(
                np.asfortranarray(mp.astype(np.uint8)))))
        proc = Sam3Processor(model, resolution=1008, device=device)
        cands = enumerate_full(model, proc, canvas, 1008, 1008, cat_names,
                               0.05, device)
        for c in cands:      # 候选坐标回到原图系
            c["box_tight"] = [c["box_tight"][0] - pad[0],
                              c["box_tight"][1] - pad[1],
                              c["box_tight"][2] - pad[0],
                              c["box_tight"][3] - pad[1]]
            m = mask_util.decode({"size": c["rle"]["size"],
                                  "counts": c["rle"]["counts"]
                                  .encode()}).astype(bool)
            m2 = m[pad[1]:pad[1] + H, pad[0]:pad[0] + W]
            rle2 = mask_util.encode(np.asfortranarray(m2.astype(np.uint8)))
            c["rle"] = {"size": list(rle2["size"]),
                        "counts": rle2["counts"].decode("utf-8")}
        r = eval_recall(cands, gts)
        out[str(img_id)]["pad1008"] = r
        print(f"[res] {img_id} pad1008: n={r['n_cands']} "
              f"bbox50={r['bbox50']} mask30={r['mask30']} "
              f"center={r['center']}", flush=True)
        torch.cuda.empty_cache()
    results["resolution"] = out


def phase_prompt(model, processor, gt, spec, results):
    cat_names = [gt.loadCats([c])[0]["name"]
                 for c in gt.getCatIds()]
    nat = [n.lower().replace("_", " ") for n in cat_names]
    out = {}
    for img_id in PROMPT_IMGS:
        image, info = _open_image(gt, spec, img_id)
        H, W = info["height"], info["width"]
        gts = _gt_rles_named(gt, img_id, H, W)
        row = {}
        for tag, names in (("orig", cat_names), ("natural", nat)):
            cands = enumerate_full(model, processor, image, H, W, names,
                                   0.05)
            # natural 分支候选的 cls 是自然化名——匹配 GT 类名时也自然化
            gts_n = [(dict(a, category_name=a["category_name"].lower()
                           .replace("_", " ")), rle)
                     for a, rle in gts]
            r = eval_recall(cands, gts if tag == "orig" else gts_n)
            row[tag] = r
            row[tag + "_median_score"] = float(np.median(
                [c["score"] for c in cands])) if cands else None
            print(f"[prompt] {img_id} {tag}: n={r['n_cands']} "
                  f"bbox50={r['bbox50']} center={r['center']} "
                  f"same_exists={r.get('bbox50_same_exists')}",
                  flush=True)
        out[str(img_id)] = row
        torch.cuda.empty_cache()
    results["prompt"] = out


def phase_merged(model, processor, gt, spec, results):
    cat_names = [gt.loadCats([c])[0]["name"]
                 for c in gt.getCatIds()]
    out = {}
    merged = ". ".join(cat_names) + "."
    for img_id in MERGED_IMGS:
        image, info = _open_image(gt, spec, img_id)
        H, W = info["height"], info["width"]
        gts = _gt_rles_named(gt, img_id, H, W)
        cands = enumerate_full(model, processor, image, H, W, [merged],
                               0.05)
        # 合并提示无逐类标签：只评 any-class 召回（类名视作单一标签）
        gts_any = [(dict(a, category_name="__merged__"), rle)
                   for a, rle in gts]
        r = eval_recall(cands, gts_any)
        out[str(img_id)] = {"n_cands": len(cands), "bbox50_any": r["bbox50"],
                            "mask30_any": r["mask30"],
                            "center_any": r["center"],
                            "median_score": float(np.median(
                                [c["score"] for c in cands]))
                            if cands else None}
        print(f"[merged] {img_id}: {out[str(img_id)]}", flush=True)
    results["merged"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "floor", "resolution", "prompt",
                             "merged"])
    args = ap.parse_args()
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "gpu_probes.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    device = "cuda"
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    gt, spec, _ = _load_eval_ctx()
    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False,
        enable_inst_interactivity=False).to(device).eval()
    processor = Sam3Processor(model, confidence_threshold=0.05,
                              device=device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        if args.phase in ("all", "floor"):
            phase_floor(model, processor, gt, spec, results)
            out_path.write_text(json.dumps(results, indent=1,
                                           ensure_ascii=False))
        if args.phase in ("all", "resolution"):
            phase_resolution(model, gt, spec, results, device)
            out_path.write_text(json.dumps(results, indent=1,
                                           ensure_ascii=False))
        if args.phase in ("all", "prompt"):
            phase_prompt(model, processor, gt, spec, results)
            out_path.write_text(json.dumps(results, indent=1,
                                           ensure_ascii=False))
        if args.phase in ("all", "merged"):
            phase_merged(model, processor, gt, spec, results)

    out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
