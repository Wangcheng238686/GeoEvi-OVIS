#!/usr/bin/env python3
"""G3a 第二轮：文本锚定读出 + logit 融合（协议 §5 修订）。

仪器：① base = 缓存 query 的 DPS 分数（模型自身读出，sanity 天花板）；
② 融合 = s_dps + α·R_i，R_i = 证据源特征 vs 15 类文本嵌入余弦（α 仅在
train 网格选择）；③ MLP 探针（特征级，3 种子）；④ LR（第一轮对照）。
主指标：val top-1 增量 + 逐目标配对符号检验。合格：任一仪器 Δ≥+3pt
且 p<0.1。

Run:  source sam3_ovis/scripts/load_environment.sh
      CUDA_VISIBLE_DEVICES=0 "$PYTHON" sam3_ovis/graph/g3a_audit2.py
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
ALPHAS = [0, 1, 2, 5, 10, 25, 50, 100]


def bbox_of(t):
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
    torch.manual_seed(20261121)
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
    dps = model.dot_prod_scoring
    clip = RemoteCLIPProvider()

    rows = []
    for p in sorted(CACHE_ROOT.glob("img_*.pt")):
        rec = torch.load(p, weights_only=False)
        cat_names = sorted(rec["texts"].keys())
        txt = np.stack([clip.encode_texts_cached(cat_names)[c]
                        for c in cat_names]).astype(np.float32)
        txt = txt / np.linalg.norm(txt, axis=1, keepdims=True)
        image = Image.open(root / spec.images_dir /
                           rec["file"]).convert("RGB")
        clip_cache, sam3_cache = {}, {}

        def clip_feat(bbox):
            key = tuple(bbox)
            if key not in clip_cache:
                # provider 按键缓存：必须用唯一键，否则同图所有 crop 复用
                # 第一个特征（本轮已发生一次，见报告 trail）
                f = np.asarray(clip.encode_images_cached(
                    [(f"c{key}", crop_img(image, bbox))])[f"c{key}"],
                    dtype=np.float32)
                f = f / (np.linalg.norm(f) + 1e-9)
                clip_cache[key] = f
            return clip_cache[key]

        def sam3_pool(bbox):
            key = tuple(bbox)
            if key not in sam3_cache:
                with torch.no_grad(), torch.autocast(
                        "cuda", dtype=torch.bfloat16):
                    t = processor.transform(v2.functional.to_image(
                        crop_img(image, bbox))).unsqueeze(0).to(device)
                    bo = model.backbone.forward_image(t)
                    feats = bo["backbone_fpn"][-model.num_feature_levels:]
                    lv = min(feats,
                             key=lambda f: abs(f.shape[-2] - 63)).float()
                sam3_cache[key] = lv[0].mean(dim=(1, 2)).cpu().numpy()
            return sam3_cache[key]

        all_boxes = [bbox_of(t) for t in rec["targets"]]
        bank_boxes = [bbox_of(b) for b in rec["bank"][:K_CTX]]

        for t in rec["targets"]:
            if t["gt_label"] is None:
                continue
            tr = rec["texts"][t["cls"]]
            bpx = tr["boxes_px"].float()
            best_j, best_iou = -1, 0.0
            for j in range(bpx.shape[0]):
                iou = box_iou_xyxy(t["box"], bpx[j].tolist())
                if iou > best_iou:
                    best_j, best_iou = j, iou
            if best_iou < 0.5:
                continue
            # base DPS 分数 [15]
            s = []
            with torch.no_grad():
                for cname in cat_names:
                    c = rec["texts"][cname]
                    jj = None
                    b2 = c["boxes_px"].float()
                    bj, bi = -1, 0.0
                    for j2 in range(b2.shape[0]):
                        iou = box_iou_xyxy(t["box"], b2[j2].tolist())
                        if iou > bi:
                            bj, bi = j2, iou
                    if bi < 0.5:
                        s.append(-1e9)
                        continue
                    s.append(float(dps(c["queries"].float()
                                       .unsqueeze(0).unsqueeze(0),
                                       c["prompt"].float(),
                                       c["prompt_mask"])[0, 0, bj, 0]))
            tb = bbox_of(t)
            ctx = [b for b in bank_boxes if box_iou_xyxy(b, tb) < 0.5]
            ctx_gt = [b for b in all_boxes if box_iou_xyxy(b, tb) < 0.5]

            def clip_logits(bboxes):
                if not bboxes:
                    return None
                fs = np.mean([clip_feat(b) for b in bboxes], axis=0)
                return fs @ txt.T  # [15]

            rows.append({
                "split": rec["split"], "label": t["gt_label"],
                "cats": cat_names,
                "s_dps": np.array(s, np.float32),
                "R1": clip_logits(ctx), "R1g": clip_logits(ctx_gt),
                "R4": clip_feat(tb) @ txt.T,
                "F2": (np.mean([sam3_pool(b) for b in ctx], axis=0)
                       if ctx else None),
                "F4": sam3_pool(tb),
                "Fq": tr["queries"].float()[best_j].cpu().numpy(),
            })
        clip.clear_image_cache()
        torch.cuda.empty_cache()

    cat_names = rows[0]["cats"]
    y = np.array([cat_names.index(r["label"]) for r in rows])
    sp = np.array([r["split"] for r in rows]) == "train"
    S = np.stack([r["s_dps"] for r in rows])
    base_pred = S.argmax(1)
    base_val = base_pred[~sp] == y[~sp]
    print(f"[g3a2] n={len(rows)} (train {sp.sum()}/val {(~sp).sum()}) "
          f"base(DPS) val top1={base_val.mean():.4f}", flush=True)

    results = {"n": {"train": int(sp.sum()), "val": int((~sp).sum())},
               "base_dps_top1": round(float(base_val.mean()), 4),
               "fusion": {}, "mlp": {}, "lr": {}}
    yv = y[~sp]

    # ---- ② 文本锚定融合（α 仅 train 选择；R 与 S 同在 cat_names 序）----
    for name in ("R1", "R1g", "R4"):
        R = np.stack([r[name] if r[name] is not None
                      else np.zeros(len(cat_names), np.float32)
                      for r in rows])
        best_a, best_tr = 0, -1
        for a in ALPHAS:
            tr_acc = ((S[sp] + a * R[sp]).argmax(1)
                      == y[sp]).mean()
            if tr_acc > best_tr:
                best_a, best_tr = a, tr_acc
        pred = (S + best_a * R).argmax(1)
        d = [int(a == yy) - int(b == yy)
             for a, b, yy in zip(pred[~sp], base_pred[~sp], yv)]
        results["fusion"][name] = {
            "alpha": best_a,
            "top1": round(float((pred[~sp] == yv).mean()), 4),
            "delta_pt": round(100 * float(np.mean(d)), 2),
            "sign_p": round(sign_p(d), 4)}
        print(f"[g3a2] fusion {name}: alpha={best_a} "
              f"top1={results['fusion'][name]['top1']:.4f} "
              f"Δ={results['fusion'][name]['delta_pt']:+.2f}pt "
              f"p={results['fusion'][name]['sign_p']}", flush=True)

    # ---- ③ MLP / ④ LR 探针（特征级）----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def feats(keys):
        out = []
        for r in rows:
            parts = [r["Fq"]]
            for k in keys:
                v = r[k]
                if v is None:
                    v = np.zeros(256, np.float32)
                parts.append(v)
            out.append(np.concatenate(parts))
        return np.stack(out)

    def eval_mlp(X, seed):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(),
            torch.nn.Dropout(0.3), torch.nn.Linear(256, 15)).cuda()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt = torch.tensor(
            StandardScaler().fit(X[sp]).transform(X), dtype=torch.float32
        ).cuda()
        yt = torch.tensor(y).cuda()
        tr_idx = np.where(sp)[0]
        for ep in range(300):
            net.train()
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                net(Xt[tr_idx]), yt[tr_idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(Xt).argmax(1).cpu().numpy()
        return pred

    for name, keys in (("Fq(base)", []), ("F2", ["F2"]), ("F4", ["F4"]),
                       ("F2+F4", ["F2", "F4"])):
        X = feats(keys)
        # LR
        sc = StandardScaler().fit(X[sp])
        clf = LogisticRegression(max_iter=3000).fit(
            sc.transform(X[sp]), y[sp])
        pred_lr = clf.predict(sc.transform(X[~sp]))
        # MLP（3 种子投票）
        preds = [eval_mlp(X, sd) for sd in range(3)]
        pred_m = np.mean([p[~sp][None] == yv[None] for p in preds],
                         axis=0)
        results["lr"][name] = {
            "top1": round(float((pred_lr == yv).mean()), 4)}
        results["mlp"][name] = {"mean_acc": round(float(pred_m.mean()), 4)}
        print(f"[g3a2] LR {name}: top1={results['lr'][name]['top1']:.4f}"
              f" | MLP mean_acc={pred_m.mean():.4f}", flush=True)
    # 特征级增量 = LR(Fq+S) − LR(Fq)（同读出、同基线）
    lr_inc = {}
    for name in ("F2", "F4", "F2+F4"):
        lr_inc[name] = round(100 * (results["lr"][name]["top1"]
                                    - results["lr"]["Fq(base)"]["top1"]),
                             2)
    results["lr_increment_pt"] = lr_inc
    print(f"[g3a2] LR increment over Fq base: {lr_inc}", flush=True)

    (OUT_DIR / "audit2.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False))
    print(f"-> {OUT_DIR / 'audit2.json'}", flush=True)


if __name__ == "__main__":
    main()
