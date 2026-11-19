#!/usr/bin/env python3
"""G3a 证据源信息审计（协议 g3a-protocol.md）：各候选源相对 query 的
线性信息增量（D4 同款方法论，先证信息含量再设计架构）。

S0 query 基线 | S1 ctx-RemoteCLIP | S2 ctx-高分辨率重编码 | S4 self-重编码
（+GT 版上下文上限参考）。LR 探针，train 171 / val 122，主指标
Δtop-1 与逐目标配对符号检验。

Run:  source sam3_ovis/scripts/load_environment.sh
      CUDA_VISIBLE_DEVICES=0 "$PYTHON" sam3_ovis/graph/g3a_evidence_audit.py
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402
from graph.g1_qbridge_probe import box_iou_xyxy

CACHE_ROOT = Path("/data/wangcheng/geoevi-g2a-cache")
LOG_ROOT = Path(os.environ.get(
    "SAM3_OVIS_LOG_ROOT",
    str(Path(__file__).resolve().parents[1] / "logs")))
OUT_DIR = LOG_ROOT / "g3a"
MARGIN = 0.1
K_CTX = 2


def bbox_of(t):
    """裁剪存储 -> 全图 bbox (x1,y1,x2,y2)。"""
    h, w = t["mask_crop"].shape
    return (t["x1"], t["y1"], t["x1"] + w, t["y1"] + h)


def crop_img(image, bbox, margin=MARGIN):
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return image.crop((max(0, int(x1 - mx)), max(0, int(y1 - my)),
                       min(image.width, int(x2 + mx)),
                       min(image.height, int(y2 + my))))


def sign_p(d):
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    k = sum(1 for x in d if x > 0)
    return min(1.0, 2 * sum(math.comb(n, i)
                            for i in range(min(k, n - k) + 1)) / 2 ** n)


def main():
    torch.manual_seed(20261120)
    device = "cuda"
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from graph.provider import RemoteCLIPProvider

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DATASETS["iSAID"]

    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False,
        enable_inst_interactivity=False).to(device).eval()
    processor = Sam3Processor(model, confidence_threshold=0.05,
                              device=device)
    clip = RemoteCLIPProvider()
    cat_names = None

    rows = []  # 每目标一行
    for p in sorted(CACHE_ROOT.glob("img_*.pt")):
        rec = torch.load(p, weights_only=False)
        if cat_names is None:
            cat_names = sorted(rec["texts"].keys())
        image = Image.open(root / spec.images_dir /
                           rec["file"]).convert("RGB")
        W, H = image.size

        # GT 框（上限参考用）：面积最大的 K_CTX 个框，排除与任何目标重叠
        from pycocotools.coco import COCO
        gt_boxes = rec.get("gt_boxes")  # 缓存没有——从标注重取
        # （缓存未存全图 GT；用 targets 的 gt 匹配之外别无来源，改为
        #  用 bank+targets 框做 ctx 的 GT 版不可行——GT 版改用"全部目标
        #  框（含他目标）"作代理，登记于报告。）
        all_boxes = [bbox_of(t) for t in rec["targets"]]

        # 每图 crop 特征缓存（bbox -> feats）
        sam3_feats, clip_feats = {}, {}

        def sam3_pool(bbox):
            key = tuple(bbox)
            if key not in sam3_feats:
                crop = crop_img(image, bbox)
                with torch.no_grad(), torch.autocast("cuda",
                                                     dtype=torch.bfloat16):
                    t = processor.transform(
                        v2.functional.to_image(crop)).unsqueeze(0) \
                        .to(device)
                    bo = model.backbone.forward_image(t)
                    n_lvl = model.num_feature_levels
                    feats = bo["backbone_fpn"][-n_lvl:]
                    lv = min(feats, key=lambda f: abs(f.shape[-2] - 63))
                    lv = lv.float()
                sam3_feats[key] = lv[0].mean(dim=(1, 2)) \
                    .cpu().numpy()
            return sam3_feats[key]

        def clip_feat(bbox):
            key = tuple(bbox)
            if key not in clip_feats:
                crop = crop_img(image, bbox)
                f = clip.encode_images_cached([("c", crop)])["c"]
                clip_feats[key] = np.asarray(f, dtype=np.float32)
            return clip_feats[key]

        bank_boxes = [bbox_of(b) for b in rec["bank"][:K_CTX]]
        for t in rec["targets"]:
            if t["gt_label"] is None:
                continue
            # matched query（目标预测类文本，与 G2a 同口径）
            tr = rec["texts"][t["cls"]]
            bpx = tr["boxes_px"].float()
            best_j, best_iou = -1, 0.0
            for j in range(bpx.shape[0]):
                iou = box_iou_xyxy(t["box"], bpx[j].tolist())
                if iou > best_iou:
                    best_j, best_iou = j, iou
            if best_iou < 0.5:
                continue
            q = tr["queries"].float()[best_j].cpu().numpy()

            tb = bbox_of(t)
            self_box = tb
            ctx = [b for b in bank_boxes
                   if box_iou_xyxy(b, tb) < 0.5][:K_CTX]
            # GT 版代理：其他目标（预测）框中不与自身重叠的前 K 个
            ctx_gt = [b for b in all_boxes
                      if box_iou_xyxy(b, tb) < 0.5][:K_CTX]

            def agg(feats):
                return np.mean(feats, axis=0) if feats else None

            rows.append({
                "split": rec["split"], "label": t["gt_label"],
                "q": q,
                "S1": agg([clip_feat(b) for b in ctx]) if ctx else None,
                "S2": agg([sam3_pool(b) for b in ctx]) if ctx else None,
                "S4": sam3_pool(tb),
                "S1g": agg([clip_feat(b) for b in ctx_gt])
                if ctx_gt else None,
                "S2g": agg([sam3_pool(b) for b in ctx_gt])
                if ctx_gt else None,
            })
        clip.clear_image_cache()
        torch.cuda.empty_cache()

    # ---- 探针 ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    labels = sorted({r["label"] for r in rows})
    y = np.array([labels.index(r["label"]) for r in rows])
    sp = np.array([r["split"] for r in rows]) == "train"
    Q = np.stack([r["q"] for r in rows])

    def get(key):
        out = []
        for r in rows:
            v = r[key]
            out.append(np.zeros(1, np.float32) if v is None else v)
        widths = {x.shape[0] for x in out}
        if len(widths) > 1:  # 尺寸不齐（理论不发生）：pad 到最大
            m = max(widths)
            out = [np.pad(x, (0, m - x.shape[0])) for x in out]
        return np.stack(out)

    def probe(X):
        sc = StandardScaler().fit(X[sp])
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(
            sc.transform(X[sp]), y[sp])
        pred = clf.predict(sc.transform(X[~sp]))
        return pred

    base_pred = probe(Q)
    results = {"n": {"train": int(sp.sum()), "val": int((~sp).sum())},
               "base_top1": round(float((base_pred == y[~sp]).mean()), 4),
               "sources": {}}
    combos = {"S1": ["S1"], "S2": ["S2"], "S4": ["S4"],
              "S1+S2": ["S1", "S2"], "S1g": ["S1g"], "S2g": ["S2g"],
              "S1g+S2g": ["S1g", "S2g"]}
    yv = y[~sp]
    for name, keys in combos.items():
        X = np.concatenate([Q] + [get(k) for k in keys], axis=1)
        pred = probe(X)
        d = [int(a == yy) - int(b == yy)
             for a, b, yy in zip(pred, base_pred, yv)]
        results["sources"][name] = {
            "top1": round(float((pred == yv).mean()), 4),
            "delta_pt": round(100 * float(np.mean(d)), 2),
            "sign_p": round(sign_p(d), 4)}
        print(f"[g3a] {name}: top1={results['sources'][name]['top1']:.4f} "
              f"Δ={results['sources'][name]['delta_pt']:+.2f}pt "
              f"p={results['sources'][name]['sign_p']}", flush=True)
    print(f"[g3a] base(query only) top1={results['base_top1']}", flush=True)
    (OUT_DIR / "audit.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False))
    print(f"-> {OUT_DIR / 'audit.json'}", flush=True)


if __name__ == "__main__":
    main()
