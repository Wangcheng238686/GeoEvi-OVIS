#!/usr/bin/env python3
"""G3c：冻结基座可训适配器确认（协议 2026-11-21-g3c-protocol.md）。

阶段 train（GPU）：G2a N 臂同机制训练，100 图缓存（773/401 目标），
3000 步 × 3 种子，val 401 目标 top-1 + 配对符号检验。
阶段 map（GPU）：全候选适配分数（批量线性分解：s'=clamp(scale·(hs_proj
(q)·P + hs_proj(dq)·P))），COCO segm mAP + 逐图 AP 组配对 bootstrap。

Run:  source sam3_ovis/scripts/load_environment.sh
      CUDA_VISIBLE_DEVICES=0 "$PYTHON" sam3_ovis/graph/g3c_adapter_confirm.py --phase train
      CUDA_VISIBLE_DEVICES=0 "$PYTHON" sam3_ovis/graph/g3c_adapter_confirm.py --phase map
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from graph.g2a_train import (  # noqa: E402
    K_TRAIN, full_mask, load_cache, match_query, reader_run,
)
from graph.g1_qbridge_probe import box_iou_xyxy  # noqa: E402

CACHE_ROOT = Path("/data/wangcheng/geoevi-g2a-cache")
CAND_DIR = CACHE_ROOT / "cands"
LOG_ROOT = Path(os.environ.get(
    "SAM3_OVIS_LOG_ROOT",
    str(Path(__file__).resolve().parents[1] / "logs")))
OUT_DIR = LOG_ROOT / "g3c"
LR = 1e-4
BATCH = 4
STEPS = 3000
SEEDS = 3
BOOT_N = 10000
SEED = 20261121


def sign_p(d):
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    k = sum(1 for x in d if x > 0)
    return min(1.0, 2 * sum(math.comb(n, i)
                            for i in range(min(k, n - k) + 1)) / 2 ** n)


def build_modules():
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False,
        enable_inst_interactivity=True).cuda().eval()
    trk = model.inst_interactive_predictor.model
    reader, writer, dps = (trk.transformer.encoder,
                           trk.maskmem_backbone, model.dot_prod_scoring)
    for p in list(reader.parameters()) + list(writer.parameters()):
        p.requires_grad_(True)
    for p in dps.parameters():
        p.requires_grad_(False)
    reader, writer, dps = reader.cuda(), writer.cuda(), dps.cuda().float()
    del model
    torch.cuda.empty_cache()
    return reader, writer, dps


def delta_null(reader, img):
    grid = tuple(img["grid"])
    h, w = grid
    canvas = img["canvas32"].unsqueeze(1)
    pos = img["pos32"].unsqueeze(1)
    k = min(K_TRAIN, max(len(img["bank"]), 1))
    p_null = torch.zeros(k * h * w, 1, 64, device=canvas.device)
    with torch.no_grad():
        return reader_run(reader, canvas, pos, p_null, grid)


def phase_train():
    from graph.g2a_train import maskpool
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reader, writer, dps = build_modules()
    imgs = load_cache()
    class_names = sorted(imgs[0]["texts"].keys())
    label_idx = {c: i for i, c in enumerate(class_names)}
    all_targets = {"train": [], "val": []}
    for img in imgs:
        for t in img["targets"]:
            if t["gt_label"] is None:
                continue
            match = {c: match_query(t, img["texts"][c])
                     for c in class_names}
            all_targets[img["split"]].append(
                {"img": img, "target": t, "mask": full_mask(t),
                 "match": match})
    print(f"[g3c] targets train={len(all_targets['train'])} "
          f"val={len(all_targets['val'])}", flush=True)

    out = {"seeds": []}
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        bridge = torch.nn.Linear(256, 256, bias=False).cuda()
        torch.nn.init.orthogonal_(bridge.weight)
        params = list(bridge.parameters()) + list(reader.parameters()) \
            + list(writer.parameters())
        opt = torch.optim.AdamW(params, lr=LR)
        rng = np.random.default_rng(seed)
        pool = all_targets["train"]
        t0 = time.time()
        hist = []
        for step in range(STEPS):
            sel = [pool[i] for i in rng.integers(0, len(pool), BATCH)]
            by_img = {}
            for t in sel:
                by_img.setdefault(t["img"]["img_id"],
                                  (t["img"], []))[1].append(t)
            losses = []
            for img_id, (img, ts) in by_img.items():
                # N 臂：Δ = reader(canvas, 零bank)，可训
                grid = tuple(img["grid"])
                h, w = grid
                canvas = img["canvas32"].unsqueeze(1)
                pos = img["pos32"].unsqueeze(1)
                k = min(K_TRAIN, max(len(img["bank"]), 1))
                p_null = torch.zeros(k * h * w, 1, 64,
                                     device=canvas.device)
                delta = reader_run(reader, canvas, pos, p_null, grid)
                for t in ts:
                    pv = maskpool(delta, t["mask"], grid)
                    dq = bridge(pv)
                    dev = dq.device
                    s = []
                    for cname in class_names:
                        j = t["match"][cname]
                        tr = img["texts"][cname]
                        if j < 0:
                            s.append(torch.tensor(-1e9, device=dev))
                            continue
                        q = tr["q32"].clone()
                        q[j] = q[j] + dq
                        s.append(dps(q.unsqueeze(0).unsqueeze(0),
                                     tr["p32"], tr["pm"])[0, 0, j, 0])
                    y = label_idx[t["target"]["gt_label"]]
                    losses.append(torch.nn.functional.cross_entropy(
                        torch.stack(s).unsqueeze(0),
                        torch.tensor([y], device=dev)))
            loss = torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (step + 1) % 500 == 0:
                hist.append({"step": step + 1,
                             "loss": round(float(loss), 4)})
                print(f"  [s{seed}] {step + 1}/{STEPS} "
                      f"loss={float(loss):.4f}", flush=True)
        # val 评估（目标级）
        n, base_ok, arm_ok, per = 0, 0, 0, []
        with torch.no_grad():
            deltas = {im["img_id"]: delta_null(reader, im) for im in imgs
                      if im["split"] == "val"}
            for ti, t in enumerate(all_targets["val"]):
                img = t["img"]
                grid = tuple(img["grid"])
                from graph.g2a_train import maskpool as mp
                pv = mp(deltas[img["img_id"]], t["mask"], grid)
                dq = bridge(pv)
                s_base, s_arm = [], []
                for cname in class_names:
                    tr = img["texts"][cname]
                    j = t["match"][cname]
                    if j < 0:
                        s_base.append(-1e9)
                        s_arm.append(-1e9)
                        continue
                    q = tr["q32"]
                    s_base.append(float(dps(q.unsqueeze(0).unsqueeze(0),
                                            tr["p32"], tr["pm"]
                                            )[0, 0, j, 0]))
                    qm = q.clone()
                    qm[j] = qm[j] + dq
                    s_arm.append(float(dps(qm.unsqueeze(0).unsqueeze(0),
                                           tr["p32"], tr["pm"]
                                           )[0, 0, j, 0]))
                if max(s_base) <= -1e8:
                    continue
                y = label_idx[t["target"]["gt_label"]]
                n += 1
                b_ok = int(np.argmax(s_base) == y)
                a_ok = int(np.argmax(s_arm) == y)
                base_ok += b_ok
                arm_ok += a_ok
                per.append({"id": ti, "b": b_ok, "a": a_ok})
        d = [p["a"] - p["b"] for p in per]
        rec = {"seed": seed, "n": n,
               "base_top1": round(base_ok / n, 4),
               "arm_top1": round(arm_ok / n, 4),
               "delta_pt": round(100 * float(np.mean(d)), 2),
               "sign_p": round(sign_p(d), 5),
               "hist": hist,
               "min": round((time.time() - t0) / 60, 1)}
        out["seeds"].append(rec)
        print(f"[g3c s{seed}] base={rec['base_top1']} "
              f"arm={rec['arm_top1']} Δ={rec['delta_pt']:+.2f}pt "
              f"p={rec['sign_p']}", flush=True)
        torch.save({"bridge": bridge.state_dict(),
                    "reader": reader.state_dict(),
                    "writer": writer.state_dict()},
                   OUT_DIR / f"ckpt_s{seed}.pt")
        # 重置 reader/writer 梯度累积（下一种子从预训练重开）
        from sam3.model_builder import build_sam3_image_model
        reader2, writer2, dps2 = build_modules()
        reader, writer, dps = reader2, writer2, dps2
    (OUT_DIR / "train.json").write_text(json.dumps(out, indent=1))
    print(f"-> {OUT_DIR / 'train.json'}", flush=True)


def phase_map():
    import contextlib
    import io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from data.score_datasets import DATASETS
    from graph.g2a_train import maskpool

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reader, writer, dps = build_modules()
    imgs = load_cache()
    by_id = {im["img_id"]: im for im in imgs}
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DATASETS["iSAID"]
    gt = COCO(str(root / spec.val_json))
    name_to_id = {c["name"]: c["id"]
                  for c in gt.loadCats(gt.getCatIds())}

    train_out = json.load(open(OUT_DIR / "train.json"))
    results = {"seeds": []}
    for rec in train_out["seeds"]:
        seed = rec["seed"]
        bridge = torch.nn.Linear(256, 256, bias=False).cuda()
        ck = torch.load(OUT_DIR / f"ckpt_s{seed}.pt", weights_only=True)
        bridge.load_state_dict(ck["bridge"])
        reader.load_state_dict(ck["reader"])  # N 臂训练后的 reader
        writer.load_state_dict(ck["writer"])
        # 预计算每图 delta（零库 reader，用训练后 reader）
        deltas = {}
        with torch.no_grad():
            for im in imgs:
                if im["split"] == "val":
                    deltas[im["img_id"]] = delta_null(reader, im)
        dets_base, dets_arm = [], []
        n_unmatched = 0
        n_cands = 0
        with torch.no_grad():
            for p in sorted(CAND_DIR.glob("cands_*.json")):
                crec = json.loads(p.read_text())
                img_id = crec["img_id"]
                img = by_id.get(img_id)
                if img is None or img["split"] != "val":
                    continue
                grid = tuple(img["grid"])
                delta = deltas[img_id]
                # 该图全部候选：mask 由 rle 解码
                from pycocotools import mask as mask_util
                for c in crec["cands"]:
                    n_cands += 1
                    cname = c["cls"]
                    tr = img["texts"][cname]
                    bpx = tr["boxes_px"].float()
                    best_j, best_iou = -1, 0.0
                    for j in range(bpx.shape[0]):
                        iou = box_iou_xyxy(c["box"], bpx[j].tolist())
                        if iou > best_iou:
                            best_j, best_iou = j, iou
                    s0 = c["score"]
                    dets_base.append({
                        "image_id": img_id,
                        "category_id": name_to_id[cname],
                        "segmentation": c["rle"], "score": s0})
                    if best_iou < 0.5:
                        n_unmatched += 1
                        dets_arm.append({
                            "image_id": img_id,
                            "category_id": name_to_id[cname],
                            "segmentation": c["rle"], "score": s0})
                        continue
                    # 适配分（批量线性分解 + clamp 复算）
                    m = mask_util.decode({"size": c["rle"]["size"],
                                          "counts": c["rle"]["counts"]
                                          .encode("utf-8")}).astype(bool)
                    pv = maskpool(delta, m, grid)
                    dq = bridge(pv)
                    q = tr["q32"]
                    p_eff = dps.prompt_mlp(tr["p32"]) \
                        if getattr(dps, "prompt_mlp", None) is not None \
                        else tr["p32"]
                    pooled = dps.mean_pool_text(p_eff, tr["pm"])
                    proj_p = dps.prompt_proj(pooled)[0]  # [d_proj]
                    s_base_j = (dps.hs_proj(q).matmul(proj_p)
                                * dps.scale)[best_j]
                    s_arm_j = (dps.hs_proj(q[best_j] + dq)
                               .matmul(proj_p) * dps.scale)
                    cmax = getattr(dps, "clamp_max_val", 12.0)
                    s_arm_j = float(s_arm_j.clamp(-cmax, cmax))
                    # 校验：分解与直接调用一致（首个候选）
                    if n_cands == 1:
                        direct = float(dps(q.unsqueeze(0).unsqueeze(0),
                                           tr["p32"], tr["pm"]
                                           )[0, 0, best_j, 0])
                        assert abs(float(s_base_j) - direct) < 1e-3, \
                            (float(s_base_j), direct)
                    dets_arm.append({
                        "image_id": img_id,
                        "category_id": name_to_id[cname],
                        "segmentation": c["rle"],
                        "score": s_arm_j})
        # GT 子集
        val_ids = sorted({d["image_id"] for d in dets_base})
        sub = COCO()
        sub.dataset = {
            "images": [im for im in gt.dataset["images"]
                       if im["id"] in set(val_ids)],
            "annotations": [a for a in gt.dataset["annotations"]
                            if a["image_id"] in set(val_ids)],
            "categories": gt.dataset["categories"]}
        sub.createIndex()

        def mAP(dets):
            dt = sub.loadRes([dict(d) for d in dets])
            ev = COCOeval(sub, dt, iouType="segm")
            with contextlib.redirect_stdout(io.StringIO()):
                ev.evaluate(); ev.accumulate(); ev.summarize()
            return float(ev.stats[0])

        def per_image(dets):
            per = {}
            for img_id in val_ids:
                d = [dict(x) for x in dets
                     if x["image_id"] == img_id]
                s = COCO()
                s.dataset = {
                    "images": [im for im in sub.dataset["images"]
                               if im["id"] == img_id],
                    "annotations": [a for a in sub.dataset["annotations"]
                                    if a["image_id"] == img_id],
                    "categories": sub.dataset["categories"]}
                s.createIndex()
                if d:
                    dt = s.loadRes(d)
                    ev = COCOeval(s, dt, iouType="segm")
                    with contextlib.redirect_stdout(io.StringIO()):
                        ev.evaluate(); ev.accumulate(); ev.summarize()
                    per[img_id] = float(ev.stats[0])
                else:
                    per[img_id] = 0.0
            return per

        m0, m1 = mAP(dets_base), mAP(dets_arm)
        p0, p1 = per_image(dets_base), per_image(dets_arm)
        diff = np.array([p1[i] - p0[i] for i in val_ids])
        rng = np.random.default_rng(SEED + seed)
        n = len(val_ids)
        samples = np.empty(BOOT_N)
        for t in range(BOOT_N):
            idx = rng.integers(0, n, n)
            samples[t] = diff[idx].mean()
        lo, hi = np.percentile(samples, [2.5, 97.5])
        results["seeds"].append({
            "seed": seed, "map_base": round(m0, 4), "map_arm": round(m1, 4),
            "delta_map_pt": round(100 * (m1 - m0), 2),
            "boot_ci95": [round(float(lo), 5), round(float(hi), 5)],
            "per_img_mean_diff": round(float(diff.mean()), 5),
            "n_cands": n_cands,
            "unmatched_pct": round(100 * n_unmatched / n_cands, 2)})
        print(f"[g3c-map s{seed}] mAP {m0:.4f}->{m1:.4f} "
              f"({100 * (m1 - m0):+.2f}pt) CI[{lo:+.5f},{hi:+.5f}] "
              f"unmatched={100 * n_unmatched / n_cands:.1f}%", flush=True)
    (OUT_DIR / "confirm.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False))
    print(f"-> {OUT_DIR / 'confirm.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "map"], required=True)
    args = ap.parse_args()
    if args.phase == "train":
        phase_train()
    else:
        phase_map()


if __name__ == "__main__":
    main()
