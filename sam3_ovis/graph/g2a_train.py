#!/usr/bin/env python3
"""G2a 训练段：冻结主干下的三臂记忆路径对照（协议 g2a-protocol.md §2-3）。

臂 M（real bank）/ N（null bank）/ X（foreign bank）共享缓存与可训参数结构
（bridge+reader+writer；DPS 头冻结），唯一差异为 bank 内容。K_train=2。
损失 = 目标 15 文本分数 CE（标签=GT 类名）。val 每 250 步评估。

Run:  source sam3_ovis/scripts/load_environment.sh
      CUDA_VISIBLE_DEVICES=2 "$PYTHON" sam3_ovis/graph/g2a_train.py \
          [--steps 1500] [--seeds 3] [--arms M N X]
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

from graph.g1_qbridge_probe import box_iou_xyxy

CACHE_ROOT = Path("/data/wangcheng/geoevi-g2a-cache")
LOG_ROOT = Path(os.environ.get(
    "SAM3_OVIS_LOG_ROOT",
    str(Path(__file__).resolve().parents[1] / "logs")))
LR = 1e-4
BATCH = 4
K_TRAIN = 2
EVAL_EVERY = 250
HARD_LOGIT = 10.0


def load_cache():
    imgs = []
    for p in sorted(CACHE_ROOT.glob("img_*.pt")):
        rec = torch.load(p, weights_only=False)
        rec["canvas32"] = rec["canvas"].float().cuda()      # [HW,C]
        rec["pos32"] = rec["pos"].float().cuda()            # [HW,C]
        for cname in rec["texts"]:
            t = rec["texts"][cname]
            t["q32"] = t["queries"].float().cuda()
            t["p32"] = t["prompt"].float().cuda()
            t["pm"] = t["prompt_mask"].cuda()
            t["bpx"] = t["boxes_px"].float().cuda()
        imgs.append(rec)
    return imgs


def full_mask(t):
    """裁剪存储 -> 全图 bool mask（numpy）。"""
    H, W = t["mask_hw"]
    m = np.zeros((H, W), bool)
    c = t["mask_crop"].numpy()
    m[t["y1"]:t["y1"] + c.shape[0], t["x1"]:t["x1"] + c.shape[1]] = c > 0
    return m


def match_query(t, text_rec):
    """目标 box 与该文本 boxes 的最大 IoU query 索引（>=0.5 才有效）。"""
    best_j, best_iou = -1, 0.0
    bpx = text_rec["bpx"]
    for j in range(bpx.shape[0]):
        iou = box_iou_xyxy(t["box"], bpx[j].tolist())
        if iou > best_iou:
            best_j, best_iou = j, iou
    return best_j if best_iou >= 0.5 else -1


def build_records(writer, canvas_bchw, masks, grid_hw, grad=True):
    h, w = grid_hw
    recs = []
    for m in masks:
        if isinstance(m, np.ndarray):
            m = torch.from_numpy(m)
        mm = torch.nn.functional.interpolate(
            m[None, None].float().cuda(), size=(h, w), mode="nearest")
        logits = torch.where(mm > 0.5, HARD_LOGIT, -HARD_LOGIT)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = writer(canvas_bchw, logits, skip_mask_sigmoid=False)
            rec = out["vision_features"]
            if rec.shape[-2:] != (h, w):
                rec = torch.nn.functional.interpolate(
                    rec, size=(h, w), mode="bilinear", align_corners=False)
        recs.append(rec[0])
    return torch.stack(recs)


def reader_run(reader, canvas, pos, prompt, grid_hw):
    h, w = grid_hw
    return reader(
        src=[canvas], src_pos=[pos], src_key_padding_mask=[None],
        prompt=prompt, prompt_pos=torch.zeros_like(prompt),
        feat_sizes=[(h, w)], num_obj_ptr_tokens=0)["memory"]


def maskpool(delta, mask_hw_np, grid_hw):
    h, w = grid_hw
    m = torch.nn.functional.interpolate(
        torch.from_numpy(mask_hw_np)[None, None].float().cuda(),
        size=(h, w), mode="nearest")[0, 0]
    d = delta.view(h, w, 1, -1)
    area = float(m.sum())
    if area < 1:
        return delta.new_zeros(delta.shape[-1])
    return (d * m[:, :, None, None]).sum(dim=(0, 1))[0] / area


class Arm:
    def __init__(self, name, reader, writer, bridge, imgs):
        self.name, self.reader, self.writer, self.bridge = \
            name, reader, writer, bridge
        self.imgs = imgs
        self.params = (list(bridge.parameters())
                       + list(reader.parameters())
                       + list(writer.parameters()))
        self.opt = torch.optim.AdamW(self.params, lr=LR)

    def delta_for(self, img, rng):
        """各臂注入的 Δ画布。

        M：reader(本图 bank) − reader(零 bank)；X：writer+记录来自随机异图
        （reader 仍跑本图画布）；N（2026-11-17 修复）：注入 reader(零 bank)
        输出本身——原实现 Δ=real−null 在零 bank 下恒为零，N 沦为空操作
        （0 翻转是构造使然）；修复后同模块可训、同预算，但不含记忆内容，
        即协议 §2 "收益是否只来自多了可训路径"的正确对照。
        """
        grid = tuple(img["grid"])
        h, w = grid
        canvas = img["canvas32"].unsqueeze(1)      # [HW,1,C]
        pos = img["pos32"].unsqueeze(1)
        k = min(K_TRAIN, max(len(img["bank"]), 1))
        if self.name == "N":
            p_null = torch.zeros(k * h * w, 1, 64, device=canvas.device)
            out_null = reader_run(self.reader, canvas, pos, p_null, grid)
            return out_null, out_null
        if self.name == "M":
            banks, wimg = img["bank"][:K_TRAIN], img
        else:  # X
            others = [o for o in self.imgs
                      if o["img_id"] != img["img_id"] and o["bank"]]
            wimg = others[rng.integers(len(others))]
            banks = wimg["bank"][:K_TRAIN]
        masks = [full_mask(b) for b in banks]
        canvas_bchw = wimg["canvas32"].view(h, w, -1).permute(2, 0, 1) \
            .unsqueeze(0)                                   # [1,C,h,w]
        recs = build_records(self.writer, canvas_bchw, masks, grid,
                             grad=True)
        p_real = recs.flatten(2).permute(2, 0, 1).reshape(
            recs.shape[0] * h * w, 1, 64)
        out_real = reader_run(self.reader, canvas, pos, p_real, grid)
        p_null = torch.zeros(p_real.shape, device=canvas.device)
        out_null = reader_run(self.reader, canvas, pos, p_null, grid)
        return out_real - out_null, out_null


def evaluate(arm, val_targets, class_names, label_idx):
    """逐目标评估：返回聚合指标 + 逐目标正确向量（供跨臂配对符号检验）。"""
    dps = arm.dps
    per = []  # {"id","base_ok","arm_ok"}
    with torch.no_grad():
        uniq = {t["img"]["img_id"]: t["img"] for t in val_targets}
        deltas = {}
        for img in uniq.values():
            deltas[img["img_id"]], _ = arm.delta_for(
                img, np.random.default_rng(0))
        for ti, t in enumerate(val_targets):
            img = t["img"]
            grid = tuple(img["grid"])
            pool_v = maskpool(deltas[img["img_id"]], t["mask"], grid)
            dq = arm.bridge(pool_v)
            s_base, s_arm = [], []
            for cname in sorted(class_names):
                tr = img["texts"][cname]
                j = t["match"][cname]
                if j < 0:
                    s_base.append(-1e9)
                    s_arm.append(-1e9)
                    continue
                q = tr["q32"]
                s_base.append(float(dps(q.unsqueeze(0).unsqueeze(0), tr["p32"],
                                        tr["pm"])[0, 0, j, 0]))
                qm = q.clone()
                qm[j] = qm[j] + dq
                s_arm.append(float(dps(qm.unsqueeze(0).unsqueeze(0), tr["p32"],
                                       tr["pm"])[0, 0, j, 0]))
            if max(s_base) <= -1e8:
                continue
            y = label_idx[t["target"]["gt_label"]]
            b_idx, a_idx = int(np.argmax(s_base)), int(np.argmax(s_arm))
            per.append({"id": ti, "base_ok": int(b_idx == y),
                        "arm_ok": int(a_idx == y),
                        "flip": int(a_idx != b_idx),
                        "to_correct": int(a_idx != b_idx and a_idx == y)})
    n = len(per)
    return {"n": n,
            "base_top1": sum(p["base_ok"] for p in per) / max(n, 1),
            "arm_top1": sum(p["arm_ok"] for p in per) / max(n, 1),
            "flips": sum(p["flip"] for p in per),
            "flip_to_correct": sum(p["to_correct"] for p in per),
            "per_target": per}


def sign_test_p(diff):
    """配对符号检验 p 值（双尾，忽略 0）。"""
    d = [x for x in diff if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    k = sum(1 for x in d if x > 0)
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", nargs="+", default=["M", "N", "X"])
    args = ap.parse_args()

    from sam3.model_builder import build_sam3_image_model
    # 完整模型只用于抽取 reader/writer/DPS，随后释放（峰值 ~3.4GB 瞬时）
    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False,
        enable_inst_interactivity=True).cuda().eval()
    trk = model.inst_interactive_predictor.model
    reader, writer, dps = (trk.transformer.encoder, trk.maskmem_backbone,
                           model.dot_prod_scoring)
    for p in list(reader.parameters()) + list(writer.parameters()):
        p.requires_grad_(True)
    for p in dps.parameters():
        p.requires_grad_(False)
    # 移出大模型并释放
    reader = reader.cuda()
    writer = writer.cuda()
    dps = dps.cuda().float()
    del model
    torch.cuda.empty_cache()

    imgs = load_cache()
    class_names = sorted(imgs[0]["texts"].keys())
    label_idx = {c: i for i, c in enumerate(class_names)}

    # 目标展开 + 训练期匹配（与 G1 同口径：IoU>=0.5）
    all_targets = {"train": [], "val": []}
    for img in imgs:
        for t in img["targets"]:
            if t["gt_label"] is None:
                continue
            match = {}
            for cname in class_names:
                match[cname] = match_query(t, img["texts"][cname])
            all_targets[img["split"]].append(
                {"img": img, "target": t, "mask": full_mask(t),
                 "match": match})
    print(f"[g2a] targets: train={len(all_targets['train'])} "
          f"val={len(all_targets['val'])}", flush=True)

    out = {"arms": {}, "config": {"steps": args.steps, "lr": LR,
                                  "batch": BATCH, "K": K_TRAIN}}
    for arm_name in args.arms:
        runs = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            bridge = torch.nn.Linear(256, 256, bias=False).cuda()
            torch.nn.init.orthogonal_(bridge.weight)
            arm = Arm(arm_name, reader, writer, bridge,
                      [i for i in imgs if i["split"] == "train"])
            arm.dps = dps
            rng = np.random.default_rng(seed)
            pool = all_targets["train"]
            hist = []
            t0 = time.time()
            for step in range(args.steps):
                sel = [pool[i] for i in rng.integers(0, len(pool), BATCH)]
                by_img = {}
                for t in sel:
                    by_img.setdefault(t["img"]["img_id"],
                                      (t["img"], []))[1].append(t)
                losses = []
                for img_id, (img, ts) in by_img.items():
                    delta, _ = arm.delta_for(img, rng)
                    grid = tuple(img["grid"])
                    for t in ts:
                        pool_v = maskpool(delta, t["mask"], grid)
                        dq = arm.bridge(pool_v)
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
                            s.append(dps(q.unsqueeze(0).unsqueeze(0), tr["p32"],
                                         tr["pm"])[0, 0, j, 0])
                        y = label_idx[t["target"]["gt_label"]]
                        losses.append(torch.nn.functional.cross_entropy(
                            torch.stack(s).unsqueeze(0),
                            torch.tensor([y], device=dev)))
                loss = torch.stack(losses).mean()
                arm.opt.zero_grad(set_to_none=True)
                loss.backward()
                arm.opt.step()
                if (step + 1) % EVAL_EVERY == 0:
                    ev = evaluate(arm, all_targets["val"], class_names,
                                  label_idx)
                    hist.append({"step": step + 1,
                                 "train_loss": round(float(loss), 4),
                                 **ev})
                    print(f"  [{arm_name} s{seed}] step {step + 1} "
                          f"loss={float(loss):.4f} "
                          f"val_top1={ev['arm_top1']:.3f} "
                          f"(base {ev['base_top1']:.3f})", flush=True)
            final = evaluate(arm, all_targets["val"], class_names,
                             label_idx)
            runs.append({"seed": seed, "final": final, "hist": hist,
                         "min": round((time.time() - t0) / 60, 1)})
            print(f"[{arm_name} s{seed}] final: "
                  f"base={final['base_top1']:.3f} arm={final['arm_top1']:.3f}"
                  f" flips={final['flips']} "
                  f"to_correct={final['flip_to_correct']}", flush=True)
            del arm
            torch.cuda.empty_cache()
        out["arms"][arm_name] = runs

    # 汇总 + 判读（配对按目标：同种子 M vs N 的逐目标 top1 差）
    summary = {"runs": out["arms"]}
    if "M" in out["arms"] and "N" in out["arms"]:
        per_seed_pt, paired_p = [], []
        for rm, rn in zip(out["arms"]["M"], out["arms"]["N"]):
            pm = {p["id"]: p["arm_ok"] for p in rm["final"]["per_target"]}
            pn = {p["id"]: p["arm_ok"] for p in rn["final"]["per_target"]}
            common = sorted(set(pm) & set(pn))
            d = [pm[i] - pn[i] for i in common]
            per_seed_pt.append(round(100 * float(np.mean(d)), 2))
            paired_p.append(sign_test_p(d))
        summary["M_minus_N_per_seed_pt"] = per_seed_pt
        summary["M_minus_N_paired_p"] = paired_p
        summary["M_minus_N_mean_pt"] = round(float(np.mean(per_seed_pt)), 2)
    if "X" in out["arms"] and "M" in out["arms"]:
        mx = []
        for rm, rx in zip(out["arms"]["M"], out["arms"]["X"]):
            mx.append(round(100 * (rm["final"]["arm_top1"]
                                   - rx["final"]["arm_top1"]), 2))
        summary["M_minus_X_per_seed_pt"] = mx
    (LOG_ROOT / "g2a").mkdir(exist_ok=True)
    (LOG_ROOT / "g2a" / "train_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in summary if k != "runs"},
                     indent=1))
    print(f"-> {LOG_ROOT / 'g2a' / 'train_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
