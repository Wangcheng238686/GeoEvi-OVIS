#!/usr/bin/env python3
"""Neck 失配敏感度探针（协议 notes/reports/2026-11-16-neck-mismatch-sensitivity.md）。

问题：G1 用 detector FPN 特征造记忆喂给预训练于 tracker neck 分布的
writer/reader（neck 失配，own-vs-foreign AUC 仅 0.609）。本探针对比
"detector FPN 造记忆（arm=det，G1 现状）" vs "sam2_backbone 造记忆
（arm=trk，tracker 原生分布）"：

D1 分布差（同图同区域）：
  - 逐通道区域内均值/方差差（|Δμ|/σ_det、方差比）、全图逐通道统计、
    通道均值向量互余弦、逐位置模长、区域池化模长比；
  - 互余弦矩阵（det 区域 × trk 区域，对角 vs 非对角）；
  - 正交 Procrustes（中心化、无尺度）：对齐前后相对残差、对齐后成对余弦、
    奇异值谱 top-k 能量；逐图与全图拼接两种；补充 det vs trk 高分辨率级
    （144×144 avg-pool 到 72²）；
  - 记忆记录差：同 mask 造出的 det-record vs trk-record 的余弦与模长比。
S2 记忆敏感性（规则与 G1 相同：dev 前 12 张有目标图、目标 score≥0.3 上限 16、
  bank K=2 与全部目标 IoU<0.5、跨文本 IoU≥0.5 匹配、foreign=上一图同 arm
  bank、空库=零 prompt）：每 arm 重算 M1（|Δ| 中位/p90/超 1e-3）、M2（CV、
  argmax 翻转）、M3（own vs foreign AUC）；主统计 ΔAUC=AUC_trk−AUC_det，
  逐图聚类 bootstrap 95% CI；双 bridge（seed 20261105 主 / 20261116 稳健）。
V3 判定（预登记）：AUC_trk≥0.7 ⇒ neck 来源是 0.609 主要解释；
  ΔAUC≥0.02 ⇒ 部分解释；|ΔAUC|<0.02 且 CI 含 0 ⇒ 不足以解释。

机械复用：import g1_qbridge_probe（禁改原文件），build_records/reader_delta/
maskpool/score_queries/grounding_once/pick_level/box_iou_xyxy 原样调用。
sam2_backbone_out 可得性已源码确认：enable_inst_interactivity=True →
sam2_backbone_out["backbone_fpn"][-1] = 72×72×256（与 detector 画布同网格），
无需代理。reader src 恒为 detector 画布（成对控制，Q-bridge 语义不变）。
不做 M4/M5（与 neck 问题无关，省显存）；每图 backbone 只前向一次
（复用 set_image 的 backbone_out，含双颈输出）。

精度：--precision auto（空余≥10.5GB→fp32；≥7GB→bf16 混合：backbone/
grounding 在 bf16 autocast，writer/reader/打分/统计 fp32，见协议 §3 标注）。

Run:  source sam3_ovis/scripts/load_environment.sh
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      CUDA_VISIBLE_DEVICES=<gpu> "$PYTHON" sam3_ovis/graph/neck_mismatch_probe.py \
          [--n-images 12] [--mode smoke|full] [--precision auto|fp32|bf16]
CPU 自测（不加载模型）：... neck_mismatch_probe.py --self-test
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
import torch.nn.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

import graph.g1_qbridge_probe as g1  # noqa: E402  (机械复用，禁改原文件)
from data.score_datasets import DATASETS  # noqa: E402

SEED = 20261105
SEED_BRIDGE = {"b1": 20261105, "b2": 20261116}
ARMS = ("det", "trk")
OUT_NAME = "neck_mismatch"
FP32_FREE_MB = 10500  # auto：≥此值用 fp32；≥7000 用 bf16 混合
BF16_FREE_MB = 7000


def log_root():
    return Path(os.environ["SAM3_OVIS_LOG_ROOT"])


# ---------------------------------------------------------------- D1 统计


def mask_on_grid(mask_hw, grid_hw):
    h, w = grid_hw
    return TF.interpolate(mask_hw[None, None].float(), size=(h, w),
                          mode="nearest")[0, 0]


def pooled_vec(feat_chw, m_grid):
    """区域内逐通道均值。feat [C,H,W]，m [H,W]；面积<1 返回零向量。"""
    area = float(m_grid.sum())
    if area < 1.0:
        return feat_chw.new_zeros(feat_chw.shape[0])
    return (feat_chw * m_grid).sum(dim=(1, 2)) / area


def region_chan_stats(feat_chw, m_grid):
    """区域内逐通道均值/方差（mask 加权）。返回 (mu [C], var [C])。"""
    area = float(m_grid.sum())
    if area < 1.0:
        z = feat_chw.new_zeros(feat_chw.shape[0])
        return z, z.clone()
    mu = (feat_chw * m_grid).sum(dim=(1, 2)) / area
    var = (feat_chw.pow(2) * m_grid).sum(dim=(1, 2)) / area - mu.pow(2)
    return mu, var.clamp_min(0.0)


def procrustes(A, B):
    """中心化正交 Procrustes：min_Q ||AcQ − Bc||_F。A/B [R,C] fp32。"""
    eps = 1e-8
    Ac = A - A.mean(0, keepdim=True)
    Bc = B - B.mean(0, keepdim=True)
    nb = float(Bc.norm()) + eps
    res_before = float((Ac - Bc).norm() / nb)
    U, S, Vh = torch.linalg.svd(Ac.T @ Bc)
    Q = U @ Vh
    D = Ac @ Q - Bc
    res_after = float(D.norm() / nb)
    a = Ac @ Q
    cos = TF.cosine_similarity(a.flatten(1), Bc.flatten(1), dim=1)
    s2 = S.pow(2)
    cum = torch.cumsum(s2 / (s2.sum() + eps), 0)
    return {
        "res_before": round(res_before, 4),
        "res_after": round(res_after, 4),
        "cos_after_mean": round(float(cos.mean()), 4),
        "sv_energy_cum": {str(k): round(float(cum[min(k, len(cum)) - 1]), 4)
                          for k in (4, 16, 64)},
        "R": int(A.shape[0]),
    }


def cos_matrix_diag_offdiag(A, B):
    """A/B [R,C] 行归一化互余弦矩阵：对角均值 vs 非对角均值/最大。"""
    An = TF.normalize(A, dim=1)
    Bn = TF.normalize(B, dim=1)
    M = An @ Bn.T
    R = M.shape[0]
    diag = float(M.diagonal().mean())
    if R < 2:
        return {"diag_mean": round(diag, 4), "offdiag_mean": None,
                "offdiag_max": None}
    off = M[~torch.eye(R, dtype=torch.bool, device=M.device)]
    return {"diag_mean": round(diag, 4),
            "offdiag_mean": round(float(off.mean()), 4),
            "offdiag_max": round(float(off.max()), 4)}


def dist_stats_pair(fdet_chw, ftrk_chw, ftrkhr_chw, region_masks, grid_hw):
    """D1 主函数：det 72² vs trk 72²（+trk 144²→72² 补充）。全部 fp32。
    返回 (stats, Ad, At, Ath)：Ad/At/Ath 为区域池化矩阵 [R,C]。"""
    eps = 1e-6
    out = {"n_regions": len(region_masks), "grid": list(grid_hw)}
    # 全图逐通道统计 + 逐位置模长
    for tag, f in (("det", fdet_chw), ("trk", ftrk_chw)):
        mu, var = f.mean(dim=(1, 2)), f.var(dim=(1, 2), unbiased=False)
        out[f"map_{tag}"] = {
            "chan_mu_absmed": round(float(mu.abs().median()), 4),
            "chan_var_med": round(float(var.median()), 4),
            "pos_norm_mean": round(float(f.norm(dim=0).mean()), 4),
            "pos_norm_p90": round(float(
                torch.quantile(f.norm(dim=0).flatten(), 0.9)), 4)}
    out["map_chan_mu_cos"] = round(float(TF.cosine_similarity(
        fdet_chw.mean(dim=(1, 2))[None], ftrk_chw.mean(dim=(1, 2))[None])), 4)
    out["map_pos_norm_ratio"] = round(float(
        ftrk_chw.norm(dim=0).mean() / (fdet_chw.norm(dim=0).mean() + eps)), 4)

    # 区域级
    Ad, At, Ath = [], [], []
    dmu_sd, var_ratio, nd, nt = [], [], [], []
    for m in region_masks:
        mg = mask_on_grid(m, grid_hw)
        Ad.append(pooled_vec(fdet_chw, mg))
        At.append(pooled_vec(ftrk_chw, mg))
        if ftrkhr_chw is not None:
            Ath.append(pooled_vec(ftrkhr_chw, mg))
        mud, vard = region_chan_stats(fdet_chw, mg)
        mut, vart = region_chan_stats(ftrk_chw, mg)
        dmu_sd.append((mud - mut).abs() / (vard.sqrt() + eps))
        var_ratio.append(vart / (vard + eps))
        nd.append(float(mud.norm()))
        nt.append(float(mut.norm()))
    Ad, At = torch.stack(Ad), torch.stack(At)
    dmu_sd, var_ratio = torch.stack(dmu_sd), torch.stack(var_ratio)
    out["region"] = {
        "chan_absdmu_over_sd_med": round(float(dmu_sd.median()), 4),
        "chan_absdmu_over_sd_p90": round(float(
            torch.quantile(dmu_sd.flatten(), 0.9)), 4),
        "chan_var_ratio_med": round(float(var_ratio.median()), 4),
        "pooled_norm_ratio": round(float(np.mean(nt) / (np.mean(nd) + eps)), 4),
        "pooled_cos": cos_matrix_diag_offdiag(Ad, At),
    }
    out["region"]["procrustes_det_trk"] = procrustes(Ad, At)
    if ftrkhr_chw is not None and len(Ath) == len(Ad):
        Ath = torch.stack(Ath)
        out["region"]["pooled_cos_det_trkhr"] = cos_matrix_diag_offdiag(Ad, Ath)
        out["region"]["procrustes_det_trkhr"] = procrustes(Ad, Ath)
    else:
        Ath = None
    return out, Ad, At, Ath


# ---------------------------------------------------------------- AUC/汇总


def auc(pos, neg):
    """G1 同式 rank AUC（|·| 后合并排序）。"""
    if not pos or not neg:
        return None
    pos, neg = np.abs(np.array(pos)), np.abs(np.array(neg))
    order = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(order)) + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def m1_stats(vals):
    d = np.abs(np.array(vals)) if vals else np.array([0.0])
    return {"median": round(float(np.median(d)), 6),
            "p90": round(float(np.percentile(d, 90)), 6),
            "frac_gt_1e-3": round(float((d > 1e-3).mean()), 4), "n": len(d)}


def per_image_pairs(rec, arm, bridge):
    own, forr = [], []
    for t in rec["targets"]:
        for v in t["texts"].values():
            if not v.get("matched") or arm not in v.get("d", {}):
                continue
            dd = v["d"][arm]
            if bridge in dd and dd[bridge] is not None:
                own.append(dd[bridge])
            fd = v.get("d_for", {}).get(arm, {})
            if bridge in fd and fd[bridge] is not None:
                forr.append(fd[bridge])
    return own, forr


def summarize(recs, bridge="b1"):
    out = {"n_images": len(recs),
           "n_targets": sum(len(r["targets"]) for r in recs),
           "bridge": bridge, "arms": {}}
    for arm in ARMS:
        d_own, d_for, cvs, flips = [], [], [], []
        for r in recs:
            for t in r["targets"]:
                rows = [v for v in t["texts"].values()
                        if v.get("matched") and arm in v.get("d", {})
                        and bridge in v["d"][arm]]
                if not rows:
                    continue
                for v in rows:
                    d_own.append(v["d"][arm][bridge])
                    f = v.get("d_for", {}).get(arm, {}).get(bridge)
                    if f is not None:
                        d_for.append(f)
                if len(rows) >= 8:
                    arr = np.array([v["d"][arm][bridge] for v in rows])
                    cvs.append(float(arr.std() / (abs(arr.mean()) + 1e-9)))
                    amax_b = max(rows, key=lambda v: v["s_base"])
                    amax_o = max(rows, key=lambda v: v["s"][arm][bridge])
                    flips.append(amax_b != amax_o)
        out["arms"][arm] = {
            "M1": m1_stats(d_own),
            "M2": {"mean_cv": round(float(np.mean(cvs)), 4) if cvs else None,
                   "argmax_flip_rate": float(np.mean(flips))
                   if flips else None, "n_targets_eval": len(cvs)},
            "M3": {"auc_own_vs_foreign": auc(d_own, d_for),
                   "n_own": len(d_own), "n_foreign": len(d_for),
                   "median_abs_own": round(float(np.median(np.abs(d_own))), 6)
                   if d_own else None,
                   "median_abs_foreign": round(
                       float(np.median(np.abs(d_for))), 6)
                   if d_for else None}}
    a_det = out["arms"]["det"]["M3"]["auc_own_vs_foreign"]
    a_trk = out["arms"]["trk"]["M3"]["auc_own_vs_foreign"]
    if a_det is not None and a_trk is not None:
        out["delta_auc_trk_minus_det"] = round(a_trk - a_det, 4)
    # 成对 |Δ| 比较（同 (目标,文本)）
    rat, sgn = [], []
    for r in recs:
        for t in r["targets"]:
            for v in t["texts"].values():
                if not v.get("matched") or "det" not in v.get("d", {}):
                    continue
                dd = v["d"]
                if bridge in dd["det"] and bridge in dd.get("trk", {}):
                    a, b = abs(dd["det"][bridge]), abs(dd["trk"][bridge])
                    if a > 1e-9:
                        rat.append(b / a)
                    sgn.append(b > a)
    out["paired"] = {
        "abs_ratio_median": round(float(np.median(rat)), 4) if rat else None,
        "frac_trk_larger": round(float(np.mean(sgn)), 4) if sgn else None,
        "n": len(sgn)}
    out["bootstrap"] = bootstrap_dauc(recs, bridge=bridge)
    out["verdict"] = verdict(out)
    return out


def bootstrap_dauc(recs, bridge="b1", n=1000, seed=SEED):
    """逐图聚类 bootstrap：重采样图，ΔAUC 分布与 95% CI。"""
    rng = np.random.default_rng(seed)
    pics = []
    for r in recs:
        od, fd = per_image_pairs(r, "det", bridge)
        ot, ft = per_image_pairs(r, "trk", bridge)
        if od and fd and ot and ft:
            pics.append((od, fd, ot, ft))
    if len(pics) < 3:
        return {"n_images_usable": len(pics), "ci95": None}
    diffs, a_ds, a_ts = [], [], []
    for _ in range(n):
        idx = rng.integers(0, len(pics), len(pics))
        od = [x for i in idx for x in pics[i][0]]
        fd = [x for i in idx for x in pics[i][1]]
        ot = [x for i in idx for x in pics[i][2]]
        ft = [x for i in idx for x in pics[i][3]]
        ad, at = auc(od, fd), auc(ot, ft)
        if ad is None or at is None:
            continue
        diffs.append(at - ad)
        a_ds.append(ad)
        a_ts.append(at)
    if not diffs:
        return {"n_images_usable": len(pics), "ci95": None}
    d = np.array(diffs)
    return {"n_images_usable": len(pics), "n_boot": len(diffs),
            "ci95": [round(float(np.percentile(d, 2.5)), 4),
                     round(float(np.percentile(d, 97.5)), 4)],
            "frac_diff_gt0": round(float((d > 0).mean()), 4),
            "auc_det_boot_med": round(float(np.median(a_ds)), 4),
            "auc_trk_boot_med": round(float(np.median(a_ts)), 4)}


def verdict(summary):
    """预登记判定（协议 §2）。"""
    a_det = summary["arms"]["det"]["M3"]["auc_own_vs_foreign"]
    a_trk = summary["arms"]["trk"]["M3"]["auc_own_vs_foreign"]
    if a_det is None or a_trk is None:
        return "insufficient_data"
    dauc = a_trk - a_det
    ci = (summary.get("bootstrap") or {}).get("ci95") or [None, None]
    ci_excl_0 = ci[0] is not None and (ci[0] > 0 or ci[1] < 0)
    if a_trk >= 0.7 and dauc >= 0.02:
        return "neck_source_main_explanation"
    if dauc >= 0.02 and ci_excl_0:
        return "partial_explanation"
    return "not_sufficient_to_explain_0.609"


# ---------------------------------------------------------------- 主流程


def pick_precision(arg):
    if arg != "auto":
        return arg, 0
    free_mb = torch.cuda.mem_get_info()[0] / 2 ** 20
    if free_mb >= FP32_FREE_MB:
        return "fp32", free_mb
    return "bf16", free_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--precision", choices=["auto", "fp32", "bf16"],
                    default="auto")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda"
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    precision, free_mb = pick_precision(args.precision)
    print(f"[neck] precision={precision} (free={free_mb:.0f}MiB, "
          f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')})",
          flush=True)
    ac = torch.autocast("cuda", dtype=torch.bfloat16,
                        enabled=(precision == "bf16"))

    out_dir = log_root() / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DATASETS["iSAID"]
    gt = COCO(str(root / spec.val_json))
    splits = json.load(open(log_root() / "g0_splits" / "manifest.json"))
    img_anns = set(a["image_id"] for a in gt.dataset["annotations"])
    imgs = {im["id"]: im for im in gt.dataset["images"]}
    picked = []
    for grp in splits["splits"]["dev"]["groups"]:
        picked.extend(sorted(
            i for i in imgs
            if imgs[i]["file_name"].startswith(grp + "_") and i in img_anns))
        if len(picked) >= max(args.n_images * 5, g1.CANDIDATE_SCAN):
            break
    picked = picked[: g1.CANDIDATE_SCAN]
    cat_ids = gt.getCatIds()
    cat_names = [gt.loadCats([c])[0]["name"] for c in cat_ids]
    print(f"[neck] {len(picked)} dev images x {len(cat_names)} texts",
          flush=True)

    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False,
        enable_inst_interactivity=True)  # 同 G1：装载 tracker writer/reader
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = Sam3Processor(model, confidence_threshold=g1.POOL_FLOOR,
                              device=device)
    trk_mod = model.inst_interactive_predictor.model
    writer, reader = trk_mod.maskmem_backbone, trk_mod.transformer.encoder

    bridges = {}
    for bk, sd in SEED_BRIDGE.items():
        torch.manual_seed(sd)  # 立即前置，保证可复现
        b = torch.nn.Linear(256, 256, bias=False).to(device)
        torch.nn.init.orthogonal_(b.weight)
        bridges[bk] = b
    torch.manual_seed(SEED)

    all_records, glob_A, glob_B, glob_Bhr = [], [], [], []
    foreign_recs = {a: None for a in ARMS}
    rec_stats_all = []
    t_run0 = time.time()
    done = 0

    for k_img, img_id in enumerate(picked):
        if done >= args.n_images:
            break
        t0 = time.time()
        info = gt.loadImgs(img_id)[0]
        image = Image.open(root / spec.images_dir /
                           info["file_name"]).convert("RGB")

        # ---- 首遍（推理态，与 G1 逐条一致；bf16 时在 autocast 下）----
        with ac:
            state = processor.set_image(image)
        first_pass = []
        for cid, cname in zip(cat_ids, cat_names):
            with ac:
                state = processor.set_text_prompt(prompt=cname, state=state)
            sc = state["scores"]
            for j in range(int(sc.shape[0]) if sc is not None else 0):
                s = float(sc[j])
                if s >= g1.TARGET_MIN_SCORE:
                    first_pass.append(
                        {"cls": cname, "box": state["boxes"][j].tolist(),
                         "mask": state["masks"][j, 0], "score": s})
        first_pass.sort(key=lambda x: -x["score"])
        targets = first_pass[: g1.TARGET_MAX_N]
        if args.mode == "smoke":
            targets = targets[:2]
        if not targets:
            continue

        bank_masks = []
        for c in first_pass:
            if all(g1.box_iou_xyxy(c["box"], t["box"]) < 0.5 for t in targets):
                bank_masks.append(c["mask"])
                if len(bank_masks) >= g1.K_STATS:
                    break
        if not bank_masks:
            continue

        # ---- 双颈特征（复用 set_image 的 backbone_out，单次前向）----
        bo = state["backbone_out"]
        feat_det, pos, grid = g1.pick_level(model, bo)
        feat_det = feat_det.detach().float()
        canvas = feat_det.flatten(2).permute(2, 0, 1).detach()
        pos_f = pos.detach().float()
        s2 = bo["sam2_backbone_out"]
        feat_trk = s2["backbone_fpn"][-1][0:1].detach().float()
        assert tuple(feat_trk.shape[-2:]) == tuple(grid), (
            f"sam2 fpn[-1] {tuple(feat_trk.shape)} != detector grid {grid}")
        assert feat_trk.shape[1] == feat_det.shape[1], "通道数不一致"
        feat_trk_hr = s2["backbone_fpn"][-2][0:1].detach().float()
        if tuple(feat_trk_hr.shape[-2:]) != tuple(grid):
            feat_trk_hr = TF.adaptive_avg_pool2d(feat_trk_hr, grid)

        # ---- 两 arm 记忆（fp32，writer 不 autocast）----
        feats = {"det": feat_det, "trk": feat_trk}
        with torch.no_grad():
            recs_own = {arm: g1.build_records(
                writer, feats[arm], bank_masks, grid) for arm in ARMS}

            # ---- D1 分布差（fp32）----
            regions = [t["mask"] for t in targets] + bank_masks
            d1, Ad, At, Ath = dist_stats_pair(
                feat_det[0], feat_trk[0], feat_trk_hr[0], regions, grid)
            glob_A.append(Ad)
            glob_B.append(At)
            glob_Bhr.append(Ath)
            # 记忆记录差：同 mask 双源 record 的余弦/模长比
            cos = TF.cosine_similarity(
                recs_own["det"].flatten(2), recs_own["trk"].flatten(2), dim=1)
            d1["record_det_vs_trk"] = {
                "cos_mean": round(float(cos.mean()), 4),
                "cos_min": round(float(cos.min()), 4),
                "norm_ratio": round(float(
                    recs_own["trk"].norm()
                    / (recs_own["det"].norm() + 1e-8)), 4)}
            rec_stats_all.append(d1["record_det_vs_trk"])

            # ---- Δ画布（fp32，reader 不 autocast）----
            delta_own, delta_for = {}, {}
            for arm in ARMS:
                delta_own[arm] = g1.reader_delta(
                    reader, canvas, pos_f, recs_own[arm], grid)
                if foreign_recs[arm] is not None:
                    delta_for[arm] = g1.reader_delta(
                        reader, canvas, pos_f,
                        foreign_recs[arm].to(device), grid)
            for arm in ARMS:
                foreign_recs[arm] = recs_own[arm].detach().cpu()

        # ---- 每文本一次前向（共享，两 arm 复用）----
        per_text = {}
        with torch.no_grad():
            for cname in cat_names:
                with ac:
                    r = g1.grounding_once(model, bo, processor.find_stage,
                                          cname, device)
                b = r["boxes"]
                boxes_px = torch.stack([
                    torch.clamp(b[:, 0] - b[:, 2] / 2, min=0),
                    torch.clamp(b[:, 1] - b[:, 3] / 2, min=0),
                    torch.clamp(b[:, 0] + b[:, 2] / 2, max=1),
                    torch.clamp(b[:, 1] + b[:, 3] / 2, max=1)], dim=1) \
                    * torch.tensor([info["width"], info["height"],
                                    info["width"], info["height"]],
                                   device=device)
                per_text[cname] = {
                    "queries": r["queries"].detach().float(),
                    "boxes_px": boxes_px,
                    "prompt": r["prompt"].detach().float(),
                    "prompt_mask": r["prompt_mask"]}

        img_rec = {"img_id": img_id, "file": info["file_name"],
                   "grid": list(grid), "n_targets": len(targets),
                   "d1": d1, "targets": []}
        m6_done = False
        for t in targets:
            trec = {"cls": t["cls"], "score": t["score"], "texts": {}}
            pools = {arm: {bk: bridges[bk](g1.maskpool(delta_own[arm],
                                                       t["mask"], grid))
                           for bk in bridges} for arm in ARMS}
            pools_for = {arm: ({bk: bridges[bk](g1.maskpool(
                delta_for[arm], t["mask"], grid)) for bk in bridges}
                if arm in delta_for else None) for arm in ARMS}
            for cname in cat_names:
                r = per_text[cname]
                best_j, best_iou = -1, 0.0
                for j in range(r["boxes_px"].shape[0]):
                    iou = g1.box_iou_xyxy(t["box"], r["boxes_px"][j].tolist())
                    if iou > best_iou:
                        best_j, best_iou = j, iou
                if best_iou < 0.5:
                    trec["texts"][cname] = {"matched": False}
                    continue
                q = r["queries"]
                s_base = g1.score_queries(model, q, r["prompt"],
                                          r["prompt_mask"], best_j)
                row = {"matched": True, "iou": round(best_iou, 4),
                       "s_base": round(float(s_base), 6),
                       "d": {}, "s": {}, "d_for": {}, "s_for": {}}
                for arm in ARMS:
                    for bk, br in bridges.items():
                        q_a = q.clone()
                        q_a[0, best_j] = q_a[0, best_j] + pools[arm][bk]
                        s_a = g1.score_queries(model, q_a, r["prompt"],
                                               r["prompt_mask"], best_j)
                        row["d"][arm] = row["d"].get(arm, {})
                        row["s"][arm] = row["s"].get(arm, {})
                        row["d"][arm][bk] = round(float(s_a - s_base), 6)
                        row["s"][arm][bk] = round(float(s_a), 6)
                        pf = pools_for[arm]
                        if pf is not None:
                            q_f = q.clone()
                            q_f[0, best_j] = q_f[0, best_j] + pf[bk]
                            s_f = g1.score_queries(model, q_f, r["prompt"],
                                                   r["prompt_mask"], best_j)
                            row["d_for"][arm] = row["d_for"].get(arm, {})
                            row["s_for"][arm] = row["s_for"].get(arm, {})
                            row["d_for"][arm][bk] = round(
                                float(s_f - s_base), 6)
                            row["s_for"][arm][bk] = round(float(s_f), 6)
                if not m6_done and cname == t["cls"]:
                    q_null = q.clone()
                    q_null[0, best_j] = q_null[0, best_j] + bridges["b1"](
                        torch.zeros_like(pools["det"]["b1"]))
                    s_null = g1.score_queries(model, q_null, r["prompt"],
                                              r["prompt_mask"], best_j)
                    trec["m6_bitexact"] = bool(
                        torch.equal(s_null, s_base))
                    m6_done = True
                trec["texts"][cname] = row
            img_rec["targets"].append(trec)

        img_rec["elapsed_s"] = round(time.time() - t0, 1)
        all_records.append(img_rec)
        done += 1
        with open(out_dir / "records.jsonl", "a") as f:
            f.write(json.dumps(img_rec) + "\n")
        print(f"[neck] {k_img + 1}/{len(picked)} {info['file_name']}: "
              f"{len(targets)} targets d1.cos={d1['region']['pooled_cos']} "
              f"rec={d1['record_det_vs_trk']} {img_rec['elapsed_s']}s",
              flush=True)
        del state, bo, per_text, delta_own, delta_for, recs_own
        torch.cuda.empty_cache()

    # ---- 汇总 ----
    summary = {"precision": precision,
               "free_mb_at_launch": round(free_mb),
               "bf16_annotation": (
                   "bf16 混合：backbone/首遍/grounding 在 bf16 autocast；"
                   "writer/reader/打分/统计 fp32（协议 §3）"
                   if precision == "bf16" else None),
               "m6_bitexact": [t.get("m6_bitexact")
                               for r in all_records
                               for t in r["targets"]
                               if t.get("m6_bitexact") is not None]}
    for bk in bridges:
        summary[f"summary_{bk}"] = summarize(all_records, bridge=bk)
    with torch.no_grad():
        A_all = torch.cat(glob_A).cpu()
        B_all = torch.cat(glob_B).cpu()
        hrs = [b for b in glob_Bhr if b is not None]
        Bhr_all = torch.cat(hrs).cpu() if hrs else None
    summary["D1_global"] = {
        "procrustes_det_trk": procrustes(A_all, B_all),
        "procrustes_det_trkhr": (procrustes(A_all, Bhr_all)
                                 if Bhr_all is not None else None),
        "pooled_cos_global": cos_matrix_diag_offdiag(A_all, B_all),
        "record_det_vs_trk_med": {
            k: round(float(np.median([r[k] for r in rec_stats_all])), 4)
            for k in rec_stats_all[0]} if rec_stats_all else None,
        "per_image": [{"img_id": r["img_id"], "d1": r["d1"]}
                      for r in all_records]}
    summary["total_min"] = round((time.time() - t_run0) / 60, 1)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    (out_dir / "dist_stats.json").write_text(json.dumps(
        {"per_image": summary["D1_global"]["per_image"],
         "global": {k: v for k, v in summary["D1_global"].items()
                    if k != "per_image"}}, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "D1_global"}, indent=1, ensure_ascii=False))
    print(f"-> {out_dir / 'summary.json'}", flush=True)


# ---------------------------------------------------------------- CPU 自测


def self_test():
    """CPU 合成数据自测：统计函数/Procrustes/AUC/bootstrap/汇总。不加载模型。"""
    torch.manual_seed(0)
    ok = []

    def check(name, cond):
        ok.append((name, bool(cond)))
        print(f"[selftest] {'PASS' if cond else 'FAIL'} {name}", flush=True)

    C, H, W, R = 32, 8, 8, 5
    fdet = torch.randn(C, H, W)
    # 恒等：trk=det → dmu/sd=0、Procrustes 残差≈0、对角余弦=1
    masks = [torch.zeros(H, W) for _ in range(R)]
    for i, m in enumerate(masks):
        m[i * 1:(i + 1) * 1 + 2, :] = 1.0
    d1, Ad, At, _ = dist_stats_pair(fdet, fdet.clone(), None, masks, (H, W))
    check("identity_chan_dmu", d1["region"]["chan_absdmu_over_sd_med"] < 1e-4)
    check("identity_procrustes",
          d1["region"]["procrustes_det_trk"]["res_after"] < 1e-3)
    check("identity_cos_diag",
          abs(d1["region"]["pooled_cos"]["diag_mean"] - 1.0) < 1e-3)
    # 旋转+噪声：trk = det 经随机正交 Q 逐位变换 + 小噪声
    Q0 = torch.linalg.qr(torch.randn(C, C))[0]
    ftrk = torch.einsum("cd,dhw->chw", Q0, fdet) + 0.05 * torch.randn(C, H, W)
    d2, _, _, _ = dist_stats_pair(fdet, ftrk, None, masks, (H, W))
    pr = d2["region"]["procrustes_det_trk"]
    check("rot_res_after_lt_before", pr["res_after"] < pr["res_before"])
    check("rot_cos_after_high", pr["cos_after_mean"] > 0.9)
    check("rot_sv_energy_concentrated", pr["sv_energy_cum"]["16"] > 0.9)
    # 无关特征：残差高、对角余弦低
    d3, _, _, _ = dist_stats_pair(fdet, torch.randn(C, H, W), None, masks,
                                  (H, W))
    check("unrelated_diag_low",
          d3["region"]["pooled_cos"]["diag_mean"] < 0.6)

    # AUC
    check("auc_perfect", auc([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]) == 1.0)
    check("auc_half", abs(auc([1.0, 4.0], [2.0, 3.0]) - 0.5) < 1e-9)

    # bootstrap/verdict：合成 records——trk arm 可分、det arm 不可分
    rng = np.random.default_rng(1)
    recs = []
    for i in range(6):
        ttexts = {}
        for c in range(10):
            d_det = float(0.08 + 0.03 * rng.standard_normal())
            d_trk = float(0.25 + 0.04 * rng.standard_normal())
            f_det = float(0.08 + 0.03 * rng.standard_normal())
            f_trk = float(0.08 + 0.04 * rng.standard_normal())
            ttexts[f"c{c}"] = {"matched": True, "s_base": 1.0,
                               "d": {"det": {"b1": d_det},
                                     "trk": {"b1": d_trk}},
                               "s": {"det": {"b1": 1.1}, "trk": {"b1": 1.2}},
                               "d_for": {"det": {"b1": f_det},
                                         "trk": {"b1": f_trk}}}
        recs.append({"targets": [{"cls": "x", "texts": ttexts}]})
    s = summarize(recs)
    check("syn_dauc_positive", s["delta_auc_trk_minus_det"] > 0.3)
    check("syn_verdict_main", s["verdict"] == "neck_source_main_explanation")
    check("syn_ci_pos", s["bootstrap"]["ci95"][0] > 0)
    # 同 arm → CI 含 0、判 not_sufficient
    recs2 = json.loads(json.dumps(recs))
    for r in recs2:
        for t in r["targets"]:
            for v in t["texts"].values():
                v["d"]["trk"] = dict(v["d"]["det"])
                v["d_for"]["trk"] = dict(v["d_for"]["det"])
    s2 = summarize(recs2)
    check("syn_identical_ci0", s2["bootstrap"]["ci95"][0] <= 0
          <= s2["bootstrap"]["ci95"][1])
    check("syn_identical_verdict",
          s2["verdict"] == "not_sufficient_to_explain_0.609")

    n_fail = sum(1 for _, c in ok if not c)
    print(f"[selftest] {len(ok) - n_fail}/{len(ok)} passed", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    main()
