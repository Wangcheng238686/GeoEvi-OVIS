#!/usr/bin/env python3
"""G1 Q-bridge 类别路径探针（协议 notes/reports/2026-11-05-g1-qbridge-protocol.md）。

M1 分数变化幅度 | M2 非纯尺度（跨文本 CV+argmax 翻转）| M3 本图 vs 异图 bank
可分性（AUC）| M4 梯度连通 | M5 K 曲线 | M6 空库 bit-exact。

零 vendored 修改：backbone.forward_image/forward_text、forward_grounding、
inst_interactive_predictor.model.{maskmem_backbone,transformer.encoder}、
model.dot_prod_scoring 裸调用；prompt/prompt_mask 取自
out["prev_encoder_out"]["encoder_out"]（原生 scoring 所用同一批张量）。

Run:  source sam3_ovis/scripts/load_environment.sh
      CUDA_VISIBLE_DEVICES=3 "$PYTHON" sam3_ovis/graph/g1_qbridge_probe.py \
          [--n-images 12] [--mode smoke|full]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402

SEED = 20261105
POOL_FLOOR = 0.05
TARGET_MIN_SCORE = 0.3  # 2026-11-05 修订：0.5 在 iSAID 文本提示下滤空（协议 §3）
TARGET_MAX_N = 16
CANDIDATE_SCAN = 60     # 扫描前 60 张 dev 切片凑满 n_images 张有目标图
K_STATS = 2
QUERY_GRID_TARGET = 63
HARD_LOGIT = 10.0
OUT_NAME = "g1_qbridge"


def log_root():
    return Path(os.environ["SAM3_OVIS_LOG_ROOT"])


def grounding_once(model, backbone_out, find_stage, text, device, grad=False):
    """单文本原生 forward_grounding；返回 out（含 queries/boxes/prompt）。"""
    bo = dict(backbone_out)
    bo.update(model.backbone.forward_text([text], device=device))
    geo = model._get_dummy_prompt()
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = model.forward_grounding(
            backbone_out=bo, find_input=find_stage,
            geometric_prompt=geo, find_target=None)
    enc = out["prev_encoder_out"]["encoder_out"]
    return {"queries": out["queries"], "boxes": out["pred_boxes"][0],
            "prompt": enc["prompt_before_enc"],
            "prompt_mask": enc["prompt_mask"]}


def pick_level(model, backbone_out):
    n_lvl = model.num_feature_levels
    feats = backbone_out["backbone_fpn"][-n_lvl:]
    poss = backbone_out["vision_pos_enc"][-n_lvl:]
    best, best_d, best_pos = None, 10 ** 9, None
    for f, p in zip(feats, poss):
        d = abs(f.shape[-2] - QUERY_GRID_TARGET)
        if d < best_d:
            best, best_d, best_pos = f, d, p
    f = best[0:1]
    pos = best_pos[0:1].flatten(2).permute(2, 0, 1)
    return f, pos, tuple(f.shape[-2:])


def build_records(writer, feat_bchw, masks_hw, grid_hw, grad=False):
    recs = []
    h, w = grid_hw
    for m in masks_hw:
        mm = torch.nn.functional.interpolate(
            m[None, None].float(), size=(h, w), mode="nearest")  # [1,1,h,w]
        logits = torch.where(mm > 0.5, HARD_LOGIT, -HARD_LOGIT)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = writer(feat_bchw, logits, skip_mask_sigmoid=False)
            rec = out["vision_features"]
            if rec.shape[-2:] != (h, w):
                rec = torch.nn.functional.interpolate(
                    rec, size=(h, w), mode="bilinear", align_corners=False)
        recs.append(rec[0])
    return torch.stack(recs)


def reader_delta(reader, canvas, pos, recs, grid_hw):
    h, w = grid_hw
    hw = h * w
    K = recs.shape[0]
    p_real = recs.flatten(2).permute(2, 0, 1).reshape(K * hw, 1, 64)
    p_null = torch.zeros_like(p_real)

    def run(p):
        return reader(
            src=[canvas], src_pos=[pos], src_key_padding_mask=[None],
            prompt=p, prompt_pos=torch.zeros_like(p),
            feat_sizes=[(h, w)], num_obj_ptr_tokens=0)["memory"]

    with torch.no_grad():
        return run(p_real.detach()) - run(p_null)


def maskpool(delta, mask_hw, grid_hw):
    h, w = grid_hw
    m = torch.nn.functional.interpolate(
        mask_hw[None, None].float(), size=(h, w), mode="nearest")[0, 0]
    d = delta.view(h, w, 1, -1)                      # [h,w,1,C]
    w4 = m[:, :, None, None]                          # [h,w,1,1] 显式 4-D
    area = m.sum()
    if float(area) < 1:
        return delta.new_zeros(delta.shape[-1])
    return ((d * w4).sum(dim=(0, 1))[0] / area)       # [C]


def score_queries(model, queries, prompt, prompt_mask, j):
    s = model.dot_prod_scoring(
        queries.unsqueeze(0), prompt, prompt_mask)
    return s[0, 0, j, 0]


def box_iou_xyxy(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda"
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    out_dir = log_root() / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DATASETS["iSAID"]
    gt = COCO(str(root / spec.val_json))
    splits = json.load(open(log_root() / "g0_splits" / "manifest.json"))
    img_anns = set(a["image_id"] for a in gt.dataset["annotations"])
    imgs = {im["id"]: im for im in gt.dataset["images"]}
    picked = []
    for g in splits["splits"]["dev"]["groups"]:
        picked.extend(sorted(
            i for i in imgs
            if imgs[i]["file_name"].startswith(g + "_") and i in img_anns))
        if len(picked) >= max(args.n_images * 5, CANDIDATE_SCAN):
            break
    picked = picked[:CANDIDATE_SCAN]
    cat_ids = gt.getCatIds()
    cat_names = [gt.loadCats([c])[0]["name"] for c in cat_ids]
    print(f"[g1] {len(picked)} dev images x {len(cat_names)} texts", flush=True)

    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False,
        enable_inst_interactivity=True)  # 装载 tracker writer/reader 权重
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = Sam3Processor(model, confidence_threshold=POOL_FLOOR,
                              device=device)
    trk = model.inst_interactive_predictor.model
    writer, reader = trk.maskmem_backbone, trk.transformer.encoder
    bridge = torch.nn.Linear(256, 256, bias=False).to(device)
    torch.nn.init.orthogonal_(bridge.weight)

    all_records, m5_curve = [], []
    foreign_recs = None  # 上一图的 bank（M3 异图对照）
    t_run0 = time.time()

    done_imgs = 0
    for k_img, img_id in enumerate(picked):
        if done_imgs >= args.n_images:
            break
        t0 = time.time()
        info = gt.loadImgs(img_id)[0]
        image = Image.open(root / spec.images_dir /
                           info["file_name"]).convert("RGB")

        # ---- 首遍（推理态，真实预测建库）----
        state = processor.set_image(image)
        first_pass = []
        for cid, cname in zip(cat_ids, cat_names):
            state = processor.set_text_prompt(prompt=cname, state=state)
            sc = state["scores"]
            for j in range(int(sc.shape[0]) if sc is not None else 0):
                s = float(sc[j])
                if s >= TARGET_MIN_SCORE:
                    first_pass.append(
                        {"cls": cname, "box": state["boxes"][j].tolist(),
                         "mask": state["masks"][j, 0], "score": s})
        first_pass.sort(key=lambda t: -t["score"])
        targets = first_pass[:TARGET_MAX_N]
        if args.mode == "smoke":
            targets = targets[:2]
        if not targets:
            continue

        # ---- 画布（fp32，统计段无需梯度；vitdet 融合核要求 no_grad）----
        img_t = processor.transform(
            v2.functional.to_image(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            backbone_out = model.backbone.forward_image(img_t)
        feat_bchw, pos, grid = pick_level(model, backbone_out)
        canvas = feat_bchw.flatten(2).permute(2, 0, 1).detach()

        # bank：与任何目标 bbox IoU<0.5 的最高分预测（排除目标自身/重复）
        bank_masks = []
        for c in first_pass:
            if all(box_iou_xyxy(c["box"], t["box"]) < 0.5 for t in targets):
                bank_masks.append(c["mask"])
                if len(bank_masks) >= K_STATS:
                    break
        if not bank_masks:
            continue
        recs_own = build_records(writer, feat_bchw.detach(),
                                 bank_masks, grid)
        delta_own = reader_delta(reader, canvas, pos, recs_own, grid)
        delta_for = (reader_delta(reader, canvas, pos,
                                  foreign_recs.to(device), grid)
                     if foreign_recs is not None else None)
        foreign_recs = recs_own.detach().cpu()

        # ---- M5 K 曲线 ----
        for K in (1, 2, 4, 8):
            if K > len(first_pass):
                continue
            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            _ = reader_delta(reader, canvas, pos,
                             build_records(writer, feat_bchw.detach(),
                                           [c["mask"] for c in first_pass[:K]],
                                           grid), grid)
            torch.cuda.synchronize()
            m5_curve.append({
                "K": K,
                "peak_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                "ms": round((time.time() - t1) * 1e3, 1)})

        # ---- 每文本一次前向，缓存；逐目标匹配+重打分 ----
        per_text = {}
        with torch.no_grad():
            for cname in cat_names:
                r = grounding_once(model, backbone_out,
                                   processor.find_stage, cname, device)
                b = r["boxes"]
                boxes_px = torch.stack([
                    torch.clamp(b[:, 0] - b[:, 2] / 2, min=0),
                    torch.clamp(b[:, 1] - b[:, 3] / 2, min=0),
                    torch.clamp(b[:, 0] + b[:, 2] / 2, max=1),
                    torch.clamp(b[:, 1] + b[:, 3] / 2, max=1)], dim=1) \
                    * torch.tensor([info["width"], info["height"],
                                    info["width"], info["height"]],
                                   device=device)
                per_text[cname] = {**r, "boxes_px": boxes_px}

        img_rec = {"img_id": img_id, "file": info["file_name"],
                   "n_targets": len(targets), "grid": list(grid),
                   "targets": []}
        for t in targets:
            trec = {"cls": t["cls"], "score": t["score"], "texts": {},
                    "m6_bitexact": None}
            pool_own = maskpool(delta_own, t["mask"], grid)
            if args.mode == "smoke" and not all_records and not trec["texts"]:
                print(f"[dbg] grid={grid} delta={tuple(delta_own.shape)} "
                      f"pool_own={tuple(pool_own.shape)} "
                      f"canvas={tuple(canvas.shape)} "
                      f"recs={tuple(recs_own.shape)}", flush=True)
            pool_for = (maskpool(delta_for, t["mask"], grid)
                        if delta_for is not None else None)
            dq_own = bridge(pool_own)
            dq_for = bridge(pool_for) if pool_for is not None else None
            for cname in cat_names:
                r = per_text[cname]
                best_j, best_iou = -1, 0.0
                for j in range(r["boxes_px"].shape[0]):
                    iou = box_iou_xyxy(t["box"],
                                       r["boxes_px"][j].tolist())
                    if iou > best_iou:
                        best_j, best_iou = j, iou
                if best_iou < 0.5:
                    trec["texts"][cname] = {"matched": False}
                    continue
                q = r["queries"]
                s_base = score_queries(model, q, r["prompt"],
                                       r["prompt_mask"], best_j)
                q_own = q.clone()
                q_own[0, best_j] = q_own[0, best_j] + dq_own
                s_own = score_queries(model, q_own, r["prompt"],
                                      r["prompt_mask"], best_j)
                row = {"matched": True, "iou": round(best_iou, 4),
                       "s_base": round(float(s_base), 6),
                       "s_own": round(float(s_own), 6),
                       "d_own": round(float(s_own - s_base), 6)}
                if dq_for is not None:
                    q_for = q.clone()
                    q_for[0, best_j] = q_for[0, best_j] + dq_for
                    s_for = score_queries(model, q_for, r["prompt"],
                                          r["prompt_mask"], best_j)
                    row["s_for"] = round(float(s_for), 6)
                    row["d_for"] = round(float(s_for - s_base), 6)
                if cname == t["cls"] and trec["m6_bitexact"] is None:
                    q_null = q.clone()
                    q_null[0, best_j] = q_null[0, best_j] + bridge(
                        torch.zeros_like(pool_own))
                    s_null = score_queries(model, q_null, r["prompt"],
                                           r["prompt_mask"], best_j)
                    trec["m6_bitexact"] = bool(
                        torch.equal(s_null, s_base))
                trec["texts"][cname] = row
            img_rec["targets"].append(trec)

        # ---- M4 梯度连通（自身类文本，全图通路，单目标）----
        # vitdet Mlp 前向是无梯度融合核；此处运行时替换为数学等价标准算子
        # （仅本段生效，结束即还原；登记为探针适配，不改 vendored 文件）
        import sam3.model.vitdet as _vd
        _orig_mlp_fwd = _vd.Mlp.forward

        def _std_mlp_fwd(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop1(x)
            x = self.norm(x)
            x = self.fc2(x)
            x = self.drop2(x)
            return x

        _vd.Mlp.forward = _std_mlp_fwd
        t0m4 = targets[0]
        for p in list(bridge.parameters()) + list(reader.parameters()) \
                + list(writer.parameters()) + list(model.backbone.parameters()):
            p.requires_grad_(True)
        backbone_out_g = model.backbone.forward_image(img_t)
        feat_g, pos_g, grid_g = pick_level(model, backbone_out_g)
        canvas_g = feat_g.flatten(2).permute(2, 0, 1)
        h, w = grid_g
        hw = h * w
        recs_g = build_records(writer, feat_g, bank_masks, grid_g, grad=True)
        p_real = recs_g.flatten(2).permute(2, 0, 1).reshape(
            recs_g.shape[0] * hw, 1, 64)
        out_real = reader(
            src=[canvas_g], src_pos=[pos_g], src_key_padding_mask=[None],
            prompt=p_real, prompt_pos=torch.zeros_like(p_real),
            feat_sizes=[(h, w)], num_obj_ptr_tokens=0)["memory"]
        out_null = reader(
            src=[canvas_g], src_pos=[pos_g], src_key_padding_mask=[None],
            prompt=torch.zeros_like(p_real),
            prompt_pos=torch.zeros_like(p_real),
            feat_sizes=[(h, w)], num_obj_ptr_tokens=0)["memory"]
        delta_g = out_real - out_null
        r_g = grounding_once(model, backbone_out_g, processor.find_stage,
                             t0m4["cls"], device, grad=True)
        pool_g = maskpool(delta_g, t0m4["mask"], grid_g)
        qs = r_g["queries"]
        best_j, best_iou = -1, 0.0
        with torch.no_grad():
            bpx = r_g["boxes"]
            boxes_px = torch.stack([
                torch.clamp(bpx[:, 0] - bpx[:, 2] / 2, min=0),
                torch.clamp(bpx[:, 1] - bpx[:, 3] / 2, min=0),
                torch.clamp(bpx[:, 0] + bpx[:, 2] / 2, max=1),
                torch.clamp(bpx[:, 1] + bpx[:, 3] / 2, max=1)], dim=1) \
                * torch.tensor([info["width"], info["height"],
                                info["width"], info["height"]],
                               device=device)
            for j in range(boxes_px.shape[0]):
                iou = box_iou_xyxy(t0m4["box"], boxes_px[j].tolist())
                if iou > best_iou:
                    best_j, best_iou = j, iou
        qs_mod = qs.clone()
        qs_mod[0, best_j] = qs_mod[0, best_j] + bridge(pool_g)
        loss = score_queries(model, qs_mod, r_g["prompt"],
                             r_g["prompt_mask"], best_j)
        loss.backward()

        def gn(mod):
            return round(float(sum(
                (p.grad.norm() ** 2).item() for p in mod.parameters()
                if p.grad is not None) ** 0.5), 6)

        img_rec["m4_grads"] = {"bridge": gn(bridge), "reader": gn(reader),
                               "writer": gn(writer),
                               "backbone": gn(model.backbone)}
        for p in model.parameters():
            p.requires_grad_(False)
        for p in bridge.parameters():
            p.requires_grad_(True)
        _vd.Mlp.forward = _orig_mlp_fwd  # 还原融合前向

        img_rec["elapsed_s"] = round(time.time() - t0, 1)
        all_records.append(img_rec)
        done_imgs += 1
        with open(out_dir / "records.jsonl", "a") as f:
            f.write(json.dumps(img_rec) + "\n")
        print(f"[g1] {k_img + 1}/{len(picked)} {info['file_name']}: "
              f"{len(targets)} targets m4={img_rec['m4_grads']} "
              f"{img_rec['elapsed_s']}s", flush=True)
        torch.cuda.empty_cache()

    summary = summarize(all_records, m5_curve)
    summary["total_min"] = round((time.time() - t_run0) / 60, 1)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"-> {out_dir / 'summary.json'}", flush=True)


def summarize(recs, m5):
    d_own, d_for, cvs, flips, m6 = [], [], [], [], []
    m5_agg = {}
    for e in m5:
        m5_agg.setdefault(e["K"], []).append((e["peak_mb"], e["ms"]))
    for r in recs:
        for t in r["targets"]:
            rows = [v for v in t["texts"].values() if v.get("matched")]
            if not rows:
                continue
            for v in rows:
                d_own.append(v["d_own"])
                if "d_for" in v:
                    d_for.append(v["d_for"])
            if len(rows) >= 8:
                arr = np.array([v["d_own"] for v in rows])
                cvs.append(float(arr.std() / (abs(arr.mean()) + 1e-9)))
                amax_b = max(rows, key=lambda v: v["s_base"])
                amax_o = max(rows, key=lambda v: v["s_own"])
                flips.append(amax_b != amax_o)
            if t["m6_bitexact"] is not None:
                m6.append(t["m6_bitexact"])

    def auc(pos, neg):
        if not pos or not neg:
            return None
        pos, neg = np.abs(np.array(pos)), np.abs(np.array(neg))
        order = np.concatenate([pos, neg])
        ranks = np.argsort(np.argsort(order)) + 1
        rp = ranks[:len(pos)].sum()
        return round(float((rp - len(pos) * (len(pos) + 1) / 2)
                           / (len(pos) * len(neg))), 4)

    d = np.abs(np.array(d_own)) if d_own else np.array([0.0])
    return {
        "n_images": len(recs),
        "n_targets": sum(len(r["targets"]) for r in recs),
        "M1": {"median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
               "frac_gt_1e-3": float((d > 1e-3).mean()), "n": len(d)},
        "M2": {"mean_cv": float(np.mean(cvs)) if cvs else None,
               "argmax_flip_rate": float(np.mean(flips)) if flips else None,
               "n_targets_eval": len(cvs)},
        "M3": {"auc_own_vs_foreign": auc(d_own, d_for),
               "n_own": len(d_own), "n_foreign": len(d_for),
               "median_abs_own": float(np.median(np.abs(d_own)))
               if d_own else None,
               "median_abs_foreign": float(np.median(np.abs(d_for)))
               if d_for else None},
        "M5": {str(k): {"peak_mb_med": float(np.median([a for a, _ in v])),
                        "ms_med": float(np.median([b for _, b in v]))}
               for k, v in sorted(m5_agg.items())},
        "M6": {"frac_bitexact": float(np.mean(m6)) if m6 else None,
               "n": len(m6)},
    }


if __name__ == "__main__":
    main()
