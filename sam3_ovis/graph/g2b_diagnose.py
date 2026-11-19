#!/usr/bin/env python3
"""G2b 诊断：记忆路线失败根因分离（协议 g2b-diagnosis-protocol.md）。

D1 det/trk 特征分布差距 | D2 writer/reader 响应组合 | D3 差分病理 |
D4 线性探针（记忆内容信息量下界）。全部 no_grad + 未训练预训练权重。

Run:  source sam3_ovis/scripts/load_environment.sh
      CUDA_VISIBLE_DEVICES=0 "$PYTHON" sam3_ovis/graph/g2b_diagnose.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchvision.transforms import v2  # noqa: E402
from PIL import Image  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402

CACHE_ROOT = Path("/data/wangcheng/geoevi-g2a-cache")
LOG_ROOT = Path(os.environ.get(
    "SAM3_OVIS_LOG_ROOT",
    str(Path(__file__).resolve().parents[1] / "logs")))
OUT_DIR = LOG_ROOT / "g2b_diag"
K_REC = 2
HARD_LOGIT = 10.0


def full_mask_np(t):
    H, W = t["mask_hw"]
    m = np.zeros((H, W), bool)
    c = t["mask_crop"].numpy()
    m[t["y1"]:t["y1"] + c.shape[0], t["x1"]:t["x1"] + c.shape[1]] = c > 0
    return m


def to_grid_mask(m_np, grid):
    return F.interpolate(torch.from_numpy(m_np)[None, None].float().cuda(),
                         size=grid, mode="nearest")


def build_record(writer, feat_bchw, m_np, grid):
    mm = to_grid_mask(m_np, grid)
    logits = torch.where(mm > 0.5, HARD_LOGIT, -HARD_LOGIT)
    with torch.no_grad():
        out = writer(feat_bchw, logits, skip_mask_sigmoid=False)
        rec = out["vision_features"]
        if rec.shape[-2:] != tuple(grid):
            rec = F.interpolate(rec, size=grid, mode="bilinear",
                                align_corners=False)
    return rec[0]  # [64,h,w]


def reader_run(reader, canvas, pos, prompt, grid):
    return reader(
        src=[canvas], src_pos=[pos], src_key_padding_mask=[None],
        prompt=prompt, prompt_pos=torch.zeros_like(prompt),
        feat_sizes=[grid], num_obj_ptr_tokens=0)["memory"]


def pool(delta, m_np, grid):
    h, w = grid
    m = to_grid_mask(m_np, grid)[0, 0]
    d = delta.view(h, w, 1, -1)
    area = float(m.sum())
    if area < 1:
        return delta.new_zeros(delta.shape[-1])
    return (d * m[:, :, None, None]).sum(dim=(0, 1))[0] / area


def cos(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def main():
    torch.manual_seed(20261120)
    device = "cuda"
    from sam3.model_builder import build_sam3_image_model
    from data.score_datasets import DATASETS as DS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DS["iSAID"]

    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False,
        enable_inst_interactivity=True).to(device).eval()
    trk_mod = model.inst_interactive_predictor.model
    writer, reader = trk_mod.maskmem_backbone, trk_mod.transformer.encoder
    from sam3.model.sam3_image_processor import Sam3Processor
    processor = Sam3Processor(model, confidence_threshold=0.05,
                              device=device)

    cache = [torch.load(p, weights_only=False)
             for p in sorted(CACHE_ROOT.glob("img_*.pt"))]
    print(f"[g2b] {len(cache)} cached images", flush=True)

    d1, d2, d3 = [], [], []
    feat_targets = {"real": [], "null": [], "canvas": [], "label": [],
                    "split": []}

    for rec in cache:
        img = Image.open(root / spec.images_dir /
                         rec["file"]).convert("RGB")
        with torch.no_grad():
            img_t = processor.transform(
                v2.functional.to_image(img)).unsqueeze(0).to(device)
            bo = model.backbone.forward_image(img_t)
            # det 画布：G2a 同款（最近 63 的 FPN 层）
            n_lvl = model.num_feature_levels
            feats = bo["backbone_fpn"][-n_lvl:]
            det = min(feats, key=lambda f: abs(f.shape[-2] - 63))
            det_bchw = det[0:1].float()  # [1,C,h,w]（sam2 分支可能 bf16）
            # trk 画布：sam2 neck 的方形网格层（优先与 det 网格一致）
            sam2 = bo["sam2_backbone_out"]["backbone_fpn"]
            square = [f for f in sam2 if f.shape[-2] == f.shape[-1]]
            trk_lv = min(square, key=lambda f: abs(f.shape[-2] - 63))
            trk_bchw = trk_lv[0:1].float()
            gd = (det_bchw.shape[-2], det_bchw.shape[-1])
            gt_ = (trk_bchw.shape[-2], trk_bchw.shape[-1])
            det_pos = None  # pos 用零（诊断只需一致性，不需要绝对位置）
            trk_pos = None

            # ---- D1 分布差距 ----
            a = det_bchw.flatten(2)[0].t()          # [HW,C]
            b = F.interpolate(trk_bchw, size=gd, mode="bilinear",
                              align_corners=False).flatten(2)[0].t()
            a_n = F.normalize(a, dim=1)
            b_n = F.normalize(b, dim=1)
            d1.append({
                "file": rec["file"], "det_grid": list(gd),
                "trk_grid": list(gt_),
                "cos_pos": round(float((a_n * b_n).sum(1).mean()), 4),
                "rms_ratio": round(float(a.norm() / b.norm()), 4),
                "det_std": round(float(a.std()), 4),
                "trk_std": round(float(b.std()), 4),
                "ch_mean_corr": round(float(np.corrcoef(
                    a.mean(0).cpu().numpy(), b.mean(0).cpu().numpy())[0, 1]),
                    4)})

            # ---- D2 writer/reader 响应（四种组合） ----
            bank_masks = [full_mask_np(x) for x in rec["bank"][:K_REC]]
            if not bank_masks:
                continue
            combos = {}
            for src_name, src_feat, grid in (
                    ("det", det_bchw, gd), ("trk", trk_bchw, gt_)):
                for rec_name, rec_feat in (
                        ("det", det_bchw), ("trk", trk_bchw)):
                    recs = torch.stack([
                        build_record(writer, rec_feat, m, grid)
                        for m in bank_masks])
                    n_rec = recs.shape[0]
                    hw = grid[0] * grid[1]
                    p_real = recs.flatten(2).permute(2, 0, 1).reshape(
                        n_rec * hw, 1, 64)
                    canvas = src_feat.flatten(2).permute(2, 0, 1)
                    pos = torch.zeros_like(canvas)
                    out_r = reader_run(reader, canvas, pos, p_real, grid)
                    p_null = torch.zeros_like(p_real)
                    out_n = reader_run(reader, canvas, pos, p_null, grid)
                    delta = out_r - out_n
                    combos[f"{src_name}<-{rec_name}"] = {
                        "delta_rms": round(float(delta.pow(2).mean()
                                                    .sqrt()), 5),
                        "out_rms": round(float(out_r.pow(2).mean()
                                               .sqrt()), 5),
                        "null_out_rms": round(float(out_n.pow(2).mean()
                                                   .sqrt()), 5),
                        "record_rms": round(float(recs.pow(2).mean()
                                                  .sqrt()), 5)}
            d2.append({"file": rec["file"], "combos": combos})

            # ---- D3 差分病理 + D4 特征（det 路线，与 G2a 同款） ----
            canvas = det_bchw.flatten(2).permute(2, 0, 1)
            pos = torch.zeros_like(canvas)
            recs = torch.stack([build_record(writer, det_bchw, m, gd)
                                for m in bank_masks])
            hw = gd[0] * gd[1]
            p_real = recs.flatten(2).permute(2, 0, 1).reshape(
                recs.shape[0] * hw, 1, 64)
            out_r = reader_run(reader, canvas, pos, p_real, gd)
            out_n = reader_run(reader, canvas, pos,
                               torch.zeros_like(p_real), gd)
            for t in rec["targets"]:
                if t["gt_label"] is None:
                    continue
                m_np = full_mask_np(t)
                pr = pool(out_r, m_np, gd)
                pn = pool(out_n, m_np, gd)
                pc = pool(canvas, m_np, gd)
                d3.append({
                    "cos_rn": round(cos(pr, pn), 4),
                    "delta_ratio": round(float((pr - pn).norm()
                                                / (pr.norm() + 1e-9)), 4)})
                feat_targets["real"].append(pr.cpu().numpy())
                feat_targets["null"].append(pn.cpu().numpy())
                feat_targets["canvas"].append(pc.cpu().numpy())
                feat_targets["label"].append(t["gt_label"])
                feat_targets["split"].append(rec["split"])
        torch.cuda.empty_cache()

    # ---- D4 线性探针 ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    labels = sorted(set(feat_targets["label"]))
    y = np.array([labels.index(x) for x in feat_targets["label"]])
    sp = np.array(feat_targets["split"])
    probes = {}
    for key in ("real", "null", "canvas"):
        X = np.stack(feat_targets[key])
        sc = StandardScaler().fit(X[sp == "train"])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(
            sc.transform(X[sp == "train"]), y[sp == "train"])
        acc = float(clf.score(sc.transform(X[sp == "val"]), y[sp == "val"]))
        probes[key] = round(acc, 4)

    def agg(rows, key):
        v = [r[key] for r in rows]
        return round(float(np.mean(v)), 4)

    d2_agg = {}
    for r in d2:
        for k, v in r["combos"].items():
            d2_agg.setdefault(k, []).append(v)
    d2_summary = {k: {kk: round(float(np.mean([x[kk] for x in v])), 5)
                      for kk in v[0]} for k, v in d2_agg.items()}

    summary = {
        "n_images": len(d1),
        "D1": {kk: agg(d1, kk) for kk in
               ("cos_pos", "rms_ratio", "det_std", "trk_std",
                "ch_mean_corr")},
        "D2": d2_summary,
        "D3": {"cos_real_null": agg(d3, "cos_rn"),
               "delta_over_real": agg(d3, "delta_ratio"),
               "n_targets": len(d3)},
        "D4": {"probes_val_top1": probes,
               "real_minus_null_pt": round(100 * (probes["real"]
                                                  - probes["null"]), 2),
               "canvas_minus_null_pt": round(100 * (probes["canvas"]
                                                    - probes["null"]), 2),
               "n_train": int((sp == "train").sum()),
               "n_val": int((sp == "val").sum())},
    }
    (OUT_DIR / "diag_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    (OUT_DIR / "d1_rows.json").write_text(json.dumps(d1))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"-> {OUT_DIR / 'diag_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
