#!/usr/bin/env python3
"""R2 零训练串扰基线探针（设计登记，冻结于任何统计产出之前）。

临时编号 R2-XTALK-20261116。问题（routes 文档 §4 R2 前置验证）：**把"已知
类别 c_n 的邻居实例记忆"读入目标 j 的 query 后，目标的 15 文本分数是否被
拉向邻居类方向？** 即零训练原生 reader+bridge 路径是否存在身份混淆（串扰）。
这是 R2（角色区分读取）立项的门控证据；不声称训练后行为，不选架构参数。

## 信息路径（与 G1 同构，全部原生模块，零 vendored 修改；机械 import 自
g1_qbridge_probe，不修改该文件）
首遍 Sam3Processor 15 类 text-only（pool floor 0.05，目标门槛 score≥0.3，
Top-16）→ 强邻居 n = first_pass 中与 j 异类且（mask IoU≥0.5 或 bbox
IoU≥0.2）的最高分预测 → writer 单条记录（n 的 mask，本图画布）→ reader
Δ画布 = reader(真记忆) − reader(零记忆) → maskpool@j 自身 mask → bridge
（Linear 256→256 无偏置，正交初始化，seed 值 20261105 与 G1 同；正交抽样前
重播 manual_seed 保证桥权重可复现）→ q'_j = q_j + bridge(δ)，
dot_prod_scoring 重算 15 文本分数（外部重算口径，与 G1 同：绕开 presence
耦合；clamp ±12 生效于测得分数）。

## 精度口径（2026-11-16 修订，先于任何统计产出；烟雾阶段发现）
事实：`build_sam3_image_model(enable_inst_interactivity=True)` 构建时经
`sam3_tracking_predictor.py:51` 进入**全局 bf16 autocast 且永不退出**
（"keep using for the entire model process"）；G1 的运行即处于该漏出状态
——其 records 中 Δ 全部落在 bf16 网格（如 0.078125=10×2^-7），192 目标
15 文本 Δ 的中位仅 7 个不同值。仓库既有登记：demo_text.py:34、
validate_api.py:87（官方用法=全局 bf16 autocast，fp32-only 会被
perflib.fused.addmm_act 的无条件 bf16 输出打断）。
本探针修订为混合精度并固定（不再提供 fp32 选项）：
- 特征路径（首遍/画布/writer/reader/grounding 前向）：运行在与 G1 完全
  相同的全局 bf16 autocast 下（若构建未漏出则显式进入同款 autocast，
  demo_text 同款机制），保证路径口径与 G1 一致；
- **打分/bridge/maskpool/文本方向：autocast(enabled=False) 的 fp32 岛**，
  岛内输入先 cast fp32——消除 Δ 的 bf16 量化（G1 口径中位 7/15 个不同
  Δ 值的并列会直接稀释分位类度量）。

## bank 条件（每目标三臂，K=1 单条记忆，同一目标同批匹配 query 上配对）
- neighbor：n 的单条记忆（处理臂）；
- self：j 自身 mask 的单条记忆（内容对照：若路径有类保真应拉向 c_j，
  在固定方向 d 下应显负号）；
- foreign：上一处理图 top 目标自身 mask 的单条记录（sham 对照：内容与
  本图无关；首图无此臂，登记）。

## 预注册度量与判读门（结果产生前冻结）
设 Δs_t^b = 注入后 − 基线（b∈三臂；t 遍历该目标的匹配文本，匹配 = 逐文本
bbox IoU≥0.5 取最优 query，与 G1 同）。文本打分方向
p_t = prompt_proj(meanpool(prompt_mlp(prompt_t)))（dot_prod_scoring 原生
前向；hs_proj 为线性，Δs_t = scale·⟨hs_proj·dq, p_t⟩，偏置自消）。
- P1 pull-to-neighbor 分位（主指标）：Δs_{c_n}^{neighbor} 在该目标匹配文本
  Δs 中的分位 =（严格小于数 + 0.5×并列数）/ n_matched；0.5 = 无拉动；
- P2 pull 强度：Δs_{c_n}^b − mean_t(Δs_t^b)，三臂同槽位（都在 c_n 文本上）
  配对差：neighbor−self、neighbor−foreign；
- P3 方向性：Pearson corr_t(Δs_t^b, g_t)，g_t = ⟨unit(p_t), unit(d)⟩，
  d = p_{c_n} − p_{c_j}，需 ≥5 个匹配文本；
- P4 argmax 移向邻居类比例（neighbor 臂）：基线 argmax≠c_n 且注入后
  argmax==c_n；伴随登记三臂任意翻转率（G1 已观测 0）；
- 伴随观察：|Δs| 幅度（接 G1 M1）、self 臂在 c_j 槽位分位（类保真反向
  对照）、邻居 (mask IoU, bbox IoU, 分数差) 描述统计、首遍/首图与 G1
  records 的可对照性。

判定门（不等式，先于结果登记）：
- 串扰显著（零训练即有身份混淆 → R2 有存在空间）：P1 中位分位 ≥0.60 且
  P2 两配对差（neighbor−self、neighbor−foreign）簇 bootstrap 95% CI 下界
  均 >0 且 P3 neighbor 臂中位 Pearson >0；
- 串扰不显著（零训练基线不支持 R2 前提）：P1 中位分位 ∈[0.45,0.55] 或
  任一 P2 配对差 CI 含 0 → "身份混淆须训练后重测，零训练下不成立"；
- 其余情形如实报告为弱串扰（中间带）。

显著性方法：以图为聚类单位的配对簇 bootstrap（重采样图 id，10,000 次，
seed 20261116，百分位 95% CI；两尾 p = 2·min(P(≤0), P(≥0))）。选择口径：
无模型选择（冻结诊断），全部统计基于 last-run 全量记录。

## 样本与执行纪律
g0_splits dev 组（manifest 顺序）扫描前 60 候选取前 12 张有 score≥0.3
目标图（复用 G1 选择逻辑，seed 20261105）；smoke = 首个存在强邻居目标
的图、该图前 2 目标（无邻居图无法验证全路径，跳过；仅 smoke 生效）。
GPU：只用 CUDA_VISIBLE_DEVICES=1；启动前 mem_get_info 校验空余 ≥6GB 否则
退出码 3（不挤占）；PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 由
环境传入；估计峰值 ≤6GB，实测峰值落盘。不变量：bridge(0) 注入后分数与
基线 bit-exact（每目标首个匹配文本校验）。首遍预测与逐文本匹配存在 GPU
非确定性核常规抖动（与 G1 同口径，登记）。

## 输出
logs/r2_crosstalk/{records.jsonl, summary.json}（smoke 写 *_smoke.*；全量
运行先截断 records.jsonl 防追加污染）。本脚本不改任何现有文件、不 git commit。

Run:  source sam3_ovis/scripts/load_environment.sh
      CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PYTHON" sam3_ovis/graph/r2_crosstalk_probe.py --mode smoke
      （smoke 通过后）同上 --mode full [--n-images 12]
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402
from g1_qbridge_probe import (  # noqa: E402 复用 G1 机械，不修改该文件
    CANDIDATE_SCAN, POOL_FLOOR, SEED, TARGET_MAX_N, TARGET_MIN_SCORE,
    box_iou_xyxy, build_records, grounding_once, log_root, maskpool,
    pick_level, reader_delta, score_queries)

SEED_BRIDGE = 20261105   # 与 G1 同 seed 值；正交抽样前重播，桥权重可复现
BOOT_N = 10000
BOOT_SEED = 20261116
MIN_FREE_GB = 6.0
NEIGH_MASK_IOU = 0.5
NEIGH_BOX_IOU = 0.2
MIN_MATCH_FOR_CORR = 5
BANKS = ("neighbor", "self", "foreign")
OUT_NAME = "r2_crosstalk"


@contextmanager
def fp32_island():
    """打分/桥/文本方向的 fp32 岛：显式关闭（漏出的）全局 bf16 autocast。"""
    with torch.autocast(device_type="cuda", enabled=False):
        yield


def mask_iou_np(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def text_direction(scorer, prompt, prompt_mask):
    """dot_prod_scoring 原生前向的文本方向 p_t（mlp→meanpool→prompt_proj）。"""
    with torch.no_grad(), fp32_island():
        pr = scorer.prompt_mlp(prompt)
        pooled = scorer.mean_pool_text(pr, prompt_mask)
        return scorer.prompt_proj(pooled)[0]        # [256]


def percentile_of(vals, v):
    """v 在 vals 中的分位：(严格小于数 + 0.5×并列数)/n。"""
    n = len(vals)
    if n == 0:
        return None
    less = sum(1 for x in vals if x < v)
    ties = sum(1 for x in vals if x == v)
    return (less + 0.5 * ties) / n


def find_neighbor(t, first_pass):
    """强邻居：异类 + (mask IoU≥0.5 或 bbox IoU≥0.2) 的最高分预测。"""
    best, best_key = None, -1.0
    for c in first_pass:
        if c["cls"] == t["cls"]:
            continue
        mi = mask_iou_np(t["mask"], c["mask"])
        bi = box_iou_xyxy(t["box"], c["box"])
        if mi >= NEIGH_MASK_IOU or bi >= NEIGH_BOX_IOU:
            if c["score"] > best_key:
                best = {"idx": c["_idx"], "cls": c["cls"], "score": c["score"],
                        "mask_iou": round(mi, 4), "box_iou": round(bi, 4)}
                best_key = c["score"]
    return best


def compute_metrics(trec, per_text):
    """单目标三臂度量（P1-P4 + 伴随），写回 trec["metrics"]。"""
    rows = [(cn, v) for cn, v in trec["texts"].items() if v.get("matched")]
    out = {"n_matched": len(rows)}
    c_n, c_j = trec["neighbor"]["cls"], trec["cls"]
    banks_ok = {b: all(f"d_{b}" in v for _, v in rows) for b in BANKS}
    cn_row = trec["texts"].get(c_n, {})
    cn_matched = bool(cn_row.get("matched"))
    if len(rows) >= 2:
        # P1 分位 + P2 pull 强度（三臂同在 c_n 槽位配对）
        if cn_matched:
            for b in BANKS:
                if banks_ok[b] and f"d_{b}" in cn_row:
                    d_cn = cn_row[f"d_{b}"]
                    dv = [v[f"d_{b}"] for _, v in rows]
                    out[f"pct_cn_{b}"] = percentile_of(dv, d_cn)
                    out[f"pull_{b}"] = d_cn - float(np.mean(dv))
        # 类保真反向对照：self 臂在 c_j 槽位分位
        cj_row = trec["texts"].get(c_j, {})
        if banks_ok["self"] and cj_row.get("matched") and "d_self" in cj_row:
            out["pct_cj_self"] = percentile_of(
                [v["d_self"] for _, v in rows], cj_row["d_self"])
        # P3 方向性：Pearson(Δs_t^b, g_t)，g_t=⟨unit(p_t), unit(p_cn−p_cj)⟩
        if len(rows) >= MIN_MATCH_FOR_CORR:
            p = {cn: per_text[cn]["p_dir"].numpy() for cn, _ in rows}
            d_vec = p[c_n] - p[c_j]
            dn = float(np.linalg.norm(d_vec))
            if dn > 1e-6:
                du = d_vec / dn
                order = [cn for cn, _ in rows]
                g = np.array([float(np.dot(
                    p[cn] / (np.linalg.norm(p[cn]) + 1e-9), du))
                    for cn in order])
                if g.std() > 1e-12:
                    for b in BANKS:
                        if not banks_ok[b]:
                            continue
                        dv = np.array([trec["texts"][cn][f"d_{b}"]
                                       for cn in order])
                        if dv.std() > 1e-12:
                            out[f"dir_{b}"] = float(np.corrcoef(dv, g)[0, 1])
        # P4 argmax 移向邻居 + 任意翻转
        amax_b = max(rows, key=lambda rv: rv[1]["s_base"])[0]
        for b in BANKS:
            if not banks_ok[b]:
                continue
            amax_a = max(rows, key=lambda rv: rv[1][f"s_{b}"])[0]
            out[f"argmax_flip_{b}"] = bool(amax_a != amax_b)
            out[f"argmax_to_neighbor_{b}"] = bool(
                amax_a == c_n and amax_b != c_n)
    return out


def cluster_boot(vals, img_ids, stat=np.median, n=BOOT_N, seed=BOOT_SEED):
    """以图为聚类单位的 bootstrap：返回 (点估计, CI_lo, CI_hi, 两尾 p)。"""
    vals = np.asarray(vals, float)
    uniq = list(dict.fromkeys(img_ids))
    idx = {g: np.array([i for i, gg in enumerate(img_ids) if gg == g])
           for g in uniq}
    point = float(stat(vals))
    rng = np.random.default_rng(seed)
    boots = np.empty(n)
    for k in range(n):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        sel = np.concatenate([idx[uniq[j]] for j in pick])
        boots[k] = stat(vals[sel])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_two = min(1.0, 2 * min(float((boots <= 0).mean()),
                             float((boots >= 0).mean())))
    return point, float(lo), float(hi), p_two


def summarize(recs):
    rows = []
    for r in recs:
        for t in r["targets"]:
            if t.get("skip_reason"):
                continue
            m = t.get("metrics", {})
            if t.get("neighbor") and any(
                    k in m for k in ("pct_cn_neighbor", "pull_neighbor",
                                     "dir_neighbor")):
                rows.append({"img": r["img_id"], **m})
    if not rows:
        return {"error": "no evaluable target rows"}

    def col(key):
        return [(row["img"], row[key]) for row in rows if key in row]

    def med(key):
        c = col(key)
        return {"median": float(np.median([v for _, v in c])),
                "n": len(c)} if c else None

    s = {"n_target_rows": len(rows),
         "n_images_with_rows": len({row["img"] for row in rows})}

    # 单臂聚合（P1/P2/P3 主量 + 伴随）
    for key in ("pct_cn_neighbor", "pct_cn_self", "pct_cn_foreign",
                "pull_neighbor", "pull_self", "pull_foreign",
                "dir_neighbor", "dir_self", "dir_foreign", "pct_cj_self"):
        s[key] = med(key)
    # |Δs| 幅度（接 G1 M1 口径，逐(目标,文本,臂)）
    d_all = []
    for r in recs:
        for t in r["targets"]:
            for v in t.get("texts", {}).values():
                if v.get("matched"):
                    d_all += [abs(v[f"d_{b}"]) for b in BANKS
                              if f"d_{b}" in v]
    s["abs_delta"] = {"median": float(np.median(d_all)),
                      "p90": float(np.percentile(d_all, 90)),
                      "n": len(d_all)} if d_all else None

    # P4 比例
    for b in BANKS:
        for key in (f"argmax_to_neighbor_{b}", f"argmax_flip_{b}"):
            c = col(key)
            s[key] = {"rate": float(np.mean([v for _, v in c])),
                      "n": len(c)} if c else None

    # 配对差（同目标内）：P1 分位差、P2 强度差、P3 方向差
    pairs = [("pct_cn", "neighbor", "self"), ("pct_cn", "neighbor", "foreign"),
             ("pull", "neighbor", "self"), ("pull", "neighbor", "foreign"),
             ("dir", "neighbor", "self"), ("dir", "neighbor", "foreign")]
    s["paired"] = {}
    for pref, a, b in pairs:
        c = [(row["img"], row[f"{pref}_{a}"] - row[f"{pref}_{b}"])
             for row in rows
             if f"{pref}_{a}" in row and f"{pref}_{b}" in row]
        if not c:
            s["paired"][f"{a}_minus_{b}_{pref}"] = None
            continue
        pt, lo, hi, p = cluster_boot([v for _, v in c], [g for g, _ in c])
        s["paired"][f"{a}_minus_{b}_{pref}"] = {
            "median": pt, "ci95": [lo, hi], "p_two": round(p, 4),
            "n": len(c)}

    # 邻居描述统计
    nbm = [(t["neighbor"]["mask_iou"], t["neighbor"]["box_iou"],
            t["score"] - t["neighbor"]["score"])
           for r in recs for t in r["targets"] if t.get("neighbor")]
    if nbm:
        arr = np.array(nbm)
        s["neighbor_meta"] = {
            "mask_iou_median": float(np.median(arr[:, 0])),
            "box_iou_median": float(np.median(arr[:, 1])),
            "score_gap_median": float(np.median(arr[:, 2])),
            "n": len(arr)}

    # 不变量
    be = [t["m_bitexact"] for r in recs for t in r["targets"]
          if t.get("m_bitexact") is not None]
    s["bitexact"] = {"rate": float(np.mean(be)), "n": len(be)} if be else None

    # 预注册判定门
    p1 = s["pct_cn_neighbor"]["median"] if s["pct_cn_neighbor"] else None
    d_ps = s["paired"]["neighbor_minus_self_pull"]
    d_pf = s["paired"]["neighbor_minus_foreign_pull"]
    d3 = s["dir_neighbor"]["median"] if s["dir_neighbor"] else None
    ci_pos = (d_ps and d_ps["ci95"][0] > 0 and d_pf
              and d_pf["ci95"][0] > 0)
    ci_covers0 = ((d_ps and d_ps["ci95"][0] <= 0 <= d_ps["ci95"][1])
                  or (d_pf and d_pf["ci95"][0] <= 0 <= d_pf["ci95"][1]))
    if p1 is not None and p1 >= 0.60 and ci_pos and d3 is not None and d3 > 0:
        verdict = "CROSSTALK_SIGNIFICANT"
    elif (p1 is not None and 0.45 <= p1 <= 0.55) or ci_covers0 or not (
            d_ps and d_pf):
        verdict = "CROSSTALK_NULL"
    else:
        verdict = "CROSSTALK_INTERMEDIATE"
    s["verdict"] = verdict
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda"
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"[r2] visible GPU free={free_b / 2 ** 30:.2f} GiB / "
          f"total={total_b / 2 ** 30:.2f} GiB", flush=True)
    if free_b < MIN_FREE_GB * 1024 ** 3:
        print(f"[r2] ABORT: free {free_b / 2 ** 30:.2f} GiB < {MIN_FREE_GB} "
              f"GiB 门（不挤占该卡）", flush=True)
        sys.exit(3)

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    suffix = "_smoke" if args.mode == "smoke" else ""
    out_dir = log_root() / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / f"records{suffix}.jsonl"
    sum_path = out_dir / f"summary{suffix}.json"
    if args.mode == "full":
        rec_path.write_text("")                      # 全量先截断，防追加污染

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
    n_want = 1 if args.mode == "smoke" else args.n_images
    print(f"[r2] {len(picked)} dev candidates x {len(cat_names)} texts, "
          f"want {n_want} images", flush=True)

    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False,
        enable_inst_interactivity=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    leaked_amp = torch.is_autocast_enabled()
    if not leaked_amp:
        # 保险：若环境不再漏出全局 autocast，显式进入同款（demo_text 机制）
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    print(f"[r2] global bf16 autocast active: {torch.is_autocast_enabled()} "
          f"(leaked_from_build={leaked_amp})", flush=True)
    processor = Sam3Processor(model, confidence_threshold=POOL_FLOOR,
                              device=device)
    trk = model.inst_interactive_predictor.model
    writer, reader = trk.maskmem_backbone, trk.transformer.encoder
    torch.manual_seed(SEED_BRIDGE)                   # 桥权重可复现
    bridge = torch.nn.Linear(256, 256, bias=False).to(device)
    torch.nn.init.orthogonal_(bridge.weight)

    all_records = []
    foreign_rec = None          # 上一处理图 top 目标的自身单条记录（CPU）
    t_run0 = time.time()
    done_imgs, n_tgt_no_neighbor = 0, 0

    for k_img, img_id in enumerate(picked):
        if done_imgs >= n_want:
            break
        t0 = time.time()
        info = gt.loadImgs(img_id)[0]
        image = Image.open(root / spec.images_dir /
                           info["file_name"]).convert("RGB")

        # ---- 首遍（推理态，真实预测建库；mask 即刻转 CPU 降峰值）----
        state = processor.set_image(image)
        first_pass = []
        for cid, cname in zip(cat_ids, cat_names):
            state = processor.set_text_prompt(prompt=cname, state=state)
            sc = state["scores"]
            for j in range(int(sc.shape[0]) if sc is not None else 0):
                s = float(sc[j])
                if s >= TARGET_MIN_SCORE:
                    first_pass.append({
                        "_idx": len(first_pass), "cls": cname,
                        "box": state["boxes"][j].tolist(),
                        "mask": state["masks"][j, 0].cpu().numpy(),
                        "score": s})
        first_pass.sort(key=lambda c: -c["score"])
        targets = first_pass[:TARGET_MAX_N]
        if args.mode == "smoke":
            targets = targets[:2]
        if not targets:
            del state
            torch.cuda.empty_cache()
            continue
        if args.mode == "smoke" and not any(
                find_neighbor(t, first_pass) for t in targets):
            # smoke 需要验证全路径：跳过无强邻居目标的图（登记，仅 smoke 生效）
            del state
            torch.cuda.empty_cache()
            continue
        del state                                   # 只需 processor.find_stage
        torch.cuda.empty_cache()

        # ---- 画布（统计段无需梯度；vitdet 融合核要求 autocast/no_grad）----
        img_t = processor.transform(
            v2.functional.to_image(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            backbone_out = model.backbone.forward_image(img_t)
        feat_bchw, pos, grid = pick_level(model, backbone_out)
        canvas = feat_bchw.flatten(2).permute(2, 0, 1).detach()

        # ---- 逐文本一次前向缓存（+文本打分方向 p_t；打分相关张量回 fp32）----
        per_text = {}
        with torch.no_grad():
            for cname in cat_names:
                r = grounding_once(model, backbone_out,
                                   processor.find_stage, cname, device)
                b = r["boxes"].float()
                boxes_px = torch.stack([
                    torch.clamp(b[:, 0] - b[:, 2] / 2, min=0),
                    torch.clamp(b[:, 1] - b[:, 3] / 2, min=0),
                    torch.clamp(b[:, 0] + b[:, 2] / 2, max=1),
                    torch.clamp(b[:, 1] + b[:, 3] / 2, max=1)], dim=1) \
                    * torch.tensor([info["width"], info["height"],
                                    info["width"], info["height"]],
                                   device=device)
                p_dir = text_direction(
                    model.dot_prod_scoring, r["prompt"].float(),
                    r["prompt_mask"])
                per_text[cname] = {"queries": r["queries"].float(),
                                   "prompt": r["prompt"].float(),
                                   "prompt_mask": r["prompt_mask"],
                                   "boxes_px": boxes_px,
                                   "p_dir": p_dir.detach().cpu()}

        # ---- 邻居与三臂 Δ画布（按唯一 mask 去重缓存 reader 调用）----
        neighbor_of, delta_cache = {}, {}
        for ti, t in enumerate(targets):
            nb = find_neighbor(t, first_pass)
            neighbor_of[ti] = nb
            if nb is None:
                continue
            if ("nb", nb["idx"]) not in delta_cache:
                with torch.no_grad():
                    recs = build_records(
                        writer, feat_bchw.detach(),
                        [torch.from_numpy(
                            first_pass[nb["idx"]]["mask"]).to(device)], grid)
                    delta_cache[("nb", nb["idx"])] = reader_delta(
                        reader, canvas, pos, recs, grid)
            if ("self", ti) not in delta_cache:
                with torch.no_grad():
                    recs = build_records(
                        writer, feat_bchw.detach(),
                        [torch.from_numpy(t["mask"]).to(device)], grid)
                    delta_cache[("self", ti)] = reader_delta(
                        reader, canvas, pos, recs, grid)
        delta_foreign = None
        if foreign_rec is not None:
            with torch.no_grad():
                delta_foreign = reader_delta(
                    reader, canvas, pos, foreign_rec.to(device), grid)

        # 异图臂下一候选：本图 top 目标自身单条记录（先于缓存释放构建；
        # 保持 [1,64,h,w]，与 G1 的异图 bank 同构，勿切片降维）
        with torch.no_grad():
            keep_foreign = build_records(
                writer, feat_bchw.detach(),
                [torch.from_numpy(targets[0]["mask"]).to(device)],
                grid).detach().cpu()

        img_rec = {"img_id": img_id, "file": info["file_name"],
                   "n_first_pass": len(first_pass), "n_targets": len(targets),
                   "grid": list(grid), "targets": []}

        for ti, t in enumerate(targets):
            trec = {"cls": t["cls"], "score": t["score"], "texts": {},
                    "m_bitexact": None}
            nb = neighbor_of[ti]
            if nb is None:
                trec["skip_reason"] = "no_strong_neighbor"
                n_tgt_no_neighbor += 1
                img_rec["targets"].append(trec)
                continue
            trec["neighbor"] = nb

            # 三臂 dq 与全部打分都在 fp32 岛内（Δ 去量化，登记于 docstring）
            t_mask = torch.from_numpy(t["mask"]).to(device)
            dq = {}
            with torch.no_grad(), fp32_island():
                for bank in BANKS:
                    if bank == "foreign":
                        if delta_foreign is None:
                            continue
                        delta_b = delta_foreign
                    elif bank == "neighbor":
                        delta_b = delta_cache[("nb", nb["idx"])]
                    else:
                        delta_b = delta_cache[("self", ti)]
                    pool = maskpool(delta_b, t_mask, grid)
                    dq[bank] = bridge(pool.float())

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
                    row = {"matched": True, "iou": round(best_iou, 4),
                           "s_base": round(float(s_base), 6)}
                    for bank in BANKS:
                        if bank not in dq:
                            continue
                        q_b = q.clone()
                        q_b[0, best_j] = q_b[0, best_j] + dq[bank]
                        s_b = score_queries(model, q_b, r["prompt"],
                                            r["prompt_mask"], best_j)
                        row[f"s_{bank}"] = round(float(s_b), 6)
                        row[f"d_{bank}"] = round(float(s_b - s_base), 6)
                    if trec["m_bitexact"] is None:
                        q_null = q.clone()
                        q_null[0, best_j] = q_null[0, best_j] + bridge(
                            torch.zeros_like(dq["self"]))
                        s_null = score_queries(model, q_null, r["prompt"],
                                               r["prompt_mask"], best_j)
                        trec["m_bitexact"] = bool(
                            torch.equal(s_null, s_base))
                    trec["texts"][cname] = row

            trec["metrics"] = compute_metrics(trec, per_text)
            img_rec["targets"].append(trec)

        del delta_cache, delta_foreign, per_text, backbone_out, feat_bchw, \
            canvas, pos
        torch.cuda.empty_cache()
        foreign_rec = keep_foreign

        img_rec["elapsed_s"] = round(time.time() - t0, 1)
        all_records.append(img_rec)
        done_imgs += 1
        with open(rec_path, "a") as f:
            f.write(json.dumps(img_rec) + "\n")
        n_be = sum(1 for x in img_rec["targets"]
                   if x.get("m_bitexact") is not None and x["m_bitexact"])
        n_ev = sum(1 for x in img_rec["targets"]
                   if "pct_cn_neighbor" in x.get("metrics", {}))
        peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30
        print(f"[r2] {k_img + 1}/{len(picked)} {info['file_name']}: "
              f"{len(targets)} targets, evaluable={n_ev}, bitexact={n_be}, "
              f"peak={peak_gb:.2f} GiB, {img_rec['elapsed_s']}s", flush=True)

    summary = summarize(all_records)
    summary.update({
        "exp": "R2-XTALK-20261116", "mode": args.mode,
        "precision": "hybrid: features under global bf16 autocast "
                     "(leaked_from_build=%s); scoring/bridge/p_t in fp32 "
                     "island" % leaked_amp,
        "n_images_processed": len(all_records),
        "n_targets_total": sum(len(r["targets"]) for r in all_records),
        "n_targets_no_strong_neighbor": n_tgt_no_neighbor,
        "bridge_seed": SEED_BRIDGE, "boot": {"n": BOOT_N, "seed": BOOT_SEED},
        "gpu_free_at_start_gib": round(free_b / 2 ** 30, 2),
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
        "total_min": round((time.time() - t_run0) / 60, 1),
        "gates_registered": {
            "CROSSTALK_SIGNIFICANT": "P1_med>=0.60 AND CI(pull nb-self)>0 "
                                     "AND CI(pull nb-foreign)>0 AND "
                                     "dir_neighbor_med>0",
            "CROSSTALK_NULL": "P1_med in [0.45,0.55] OR any pull-diff CI "
                              "covers 0"},
    })
    sum_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"-> {sum_path}", flush=True)


if __name__ == "__main__":
    main()
