#!/usr/bin/env python3
"""R1 v2 评估：微调 ckpt 的首遍候选 → g5a 口径（117 val 组图）四要素判读。

流程：
1. 基座构建（sam3.pt）→ r1_lora.apply_lora（与训练同构）→ 加载
   /data/wangcheng/geoevi-r1/run1/checkpoints/checkpoint.pt 的 model
   state（strict=True，LoRA 键必须逐位对上）；
2. g5a.gen_arm("R1v2", COMBO 设置{space, 0.01, 1344}) 生成臂候选；
3. g5a.eval_arms 全臂评估（R1v2 自动入表）；
4. R1v2−COMBO 逐图配对 bootstrap（协议显著性口径）；COMBO 复算值与
   eval_arms 输出做奇偶校验（|Δ|<1e-6，防复算漂移）；
5. 按协议 v2 判定门打印 GO/PARTIAL/NO-GO。

Run:  source sam3_ovis/scripts/load_environment.sh
      CUDA_VISIBLE_DEVICES=0 "$PYTHON" sam3_ovis/graph/r1_v2_eval.py \
          [--ckpt /data/wangcheng/geoevi-r1/run1/checkpoints/checkpoint.pt]
"""

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402
from graph import g5a_free_lever as g5a  # noqa: E402

BOOT_N = 2000
SEED = 20261122
COMBO_CFG = {"prompt": "space", "floor": 0.01, "res": 1344}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data/wangcheng/geoevi-r1/run1/"
                                      "checkpoints/checkpoint.pt")
    args = ap.parse_args()

    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DATASETS["iSAID"]
    gt = COCO(str(root / spec.val_json))
    name_to_id = {c["name"]: c["id"]
                  for c in gt.loadCats(gt.getCatIds())}

    # ---- 1. 微调模型 ----
    sys.path.insert(0, "/home/wangcheng2021/project/GeoEvi-OVIS/sam3")
    from sam3.model_builder import build_sam3_image_model
    from r1_lora import apply_lora
    # 评估需要 mask（g5a 口径）→ seg head 必须构建；训练时 seg off，
    # ckpt 无 segmentation_head 键 → strict=False + 断言缺失键仅此一族
    # （seg head 保持基座预训练值，训练从未触碰）。
    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False,
        enable_inst_interactivity=False, enable_segmentation=True)
    apply_lora(model, r=8, alpha=16)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    res = model.load_state_dict(ck["model"], strict=False)
    assert all(k.startswith("segmentation_head.")
               for k in res.missing_keys), res.missing_keys[:5]
    assert not res.unexpected_keys, res.unexpected_keys[:5]
    print(f"[r1eval] ckpt loaded (epoch={ck.get('epoch')}, "
          f"steps={ck.get('steps')}, "
          f"seg-head keys kept from base: {len(res.missing_keys)})",
          flush=True)
    model = model.cuda().eval()

    # ---- 2. R1v2 臂候选生成 ----
    g5a.gen_arm("R1v2", COMBO_CFG, gt, spec, root, model)
    del model
    torch.cuda.empty_cache()

    # ---- 3. 全臂评估 ----
    g5a.eval_arms(gt, name_to_id)

    # ---- 4. R1v2−COMBO 配对 + 奇偶校验 ----
    arm_recs = {}
    for arm_dir in sorted(g5a.OUT_ROOT.glob("*")):
        if arm_dir.is_dir():
            arm_recs[arm_dir.name] = [json.loads(p.read_text())
                                      for p in sorted(
                                          arm_dir.glob("cands_*.json"))]
    val_ids = sorted({r["img_id"]
                      for r in arm_recs["R1v2"]})

    def dets_of(arm):
        out = []
        for r in arm_recs[arm]:
            for c in g5a.nms_image(r["cands"]):
                out.append({"image_id": r["img_id"],
                            "category_id": name_to_id[c["cls"]],
                            "segmentation": c["rle"],
                            "score": c["score"]})
        return out

    sub = COCO()
    sub.dataset = {
        "images": [im for im in gt.dataset["images"]
                   if im["id"] in set(val_ids)],
        "annotations": [a for a in gt.dataset["annotations"]
                        if a["image_id"] in set(val_ids)],
        "categories": gt.dataset["categories"]}
    sub.createIndex()

    def per_image(dets):
        out = {}
        for img_id in val_ids:
            d = [dict(x) for x in dets if x["image_id"] == img_id]
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
                out[img_id] = float(ev.stats[0])
            else:
                out[img_id] = 0.0
        return out

    p_combo = per_image(dets_of("COMBO"))
    p_r1 = per_image(dets_of("R1v2"))
    ablation = json.load(open(g5a.RES_DIR / "ablation.json"))
    m_combo_re = float(np.mean([p_combo[i] for i in val_ids]))
    # 奇偶校验：COMBO 复算均值与 eval_arms 的 mAP 一致
    assert abs(m_combo_re - ablation["COMBO"]["mAP"]) < 1e-6, \
        (m_combo_re, ablation["COMBO"]["mAP"])
    d = np.array([p_r1[i] - p_combo[i] for i in val_ids])
    rng = np.random.default_rng(SEED)
    n = len(val_ids)
    samples = np.array([d[rng.integers(0, n, n)].mean()
                        for _ in range(BOOT_N)])
    lo, hi = np.percentile(samples, [2.5, 97.5])
    pairing = {"delta_map_pt": round(100 * float(d.mean()), 2),
               "ci95_pt": [round(100 * float(lo), 2),
                           round(100 * float(hi), 2)],
               "parity_check": "COMBO per-image mean == eval_arms mAP "
                               "(<1e-6) PASSED"}
    print(f"[r1eval] R1v2-COMBO: {pairing['delta_map_pt']:+.2f}pt "
          f"CI{pairing['ci95_pt']}", flush=True)

    # ---- 5. 判定门（协议 v2 §1）----
    r1 = ablation["R1v2"]
    verdict = {
        "mAP": r1["mAP"], "gate_mAP": 0.39,
        "same_class_pct": r1["recall_same_class_pct"],
        "gate_same_class": 45,
        "fp_per_img": r1["fp_per_img"], "gate_fp_max": 350,
        "vs_COMBO": pairing}
    ok = (r1["mAP"] >= 0.39 and lo > 0
          and r1["recall_same_class_pct"] >= 45
          and r1["fp_per_img"] <= 350)
    partial = (0.30 <= r1["mAP"] < 0.39 and lo > 0)
    verdict["VERDICT"] = ("GO" if ok else
                          "PARTIAL" if partial else "NO-GO")
    out_path = g5a.RES_DIR / "r1v2_verdict.json"
    out_path.write_text(json.dumps(verdict, indent=1,
                                   ensure_ascii=False))
    print(json.dumps(verdict, indent=1, ensure_ascii=False), flush=True)
    print(f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
