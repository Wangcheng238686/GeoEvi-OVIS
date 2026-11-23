#!/usr/bin/env python3
"""S2 排序/校准线（协议 notes/reports/2026-11-24-s2-protocol.md）。

冻结 R1v2 的候选与 mask，只做分数重标：逐类阈值 τ_c（5 个封闭配置）。

温度维度已撤销（协议 §2.1）：s^(1/T) 是对所有候选的同一严格单调函数，
保持全局排序不变，对 COCO mAP 零影响——tests/test_s2_rescore.py 以
测试钉住这一事实。rescore() 仍保留 temp 形参仅为接口稳定，网格中
恒为 1.0。
三阶段物理隔离，val 不参与选参：

  --phase gen    (GPU) 生成 train 池候选（42 组 / 790 图），R1v2 与 COMBO。
  --phase select (CPU) 只读 train 池：对 15 个封闭网格配置算 segm mAP，
                        选 train mAP 最高者；此阶段不触碰 val 目录。
  --phase eval   (CPU)  对选出的单一配置，在 val 117 图上评估一次：mAP、
                        配对 bootstrap（vs 未校准 R1v2）、FP/图、候选纯度、
                        center 召回；输出判定。

重标定义（协议 §2）：
  丢弃     若 s < τ_c，τ_c = train 池该类分数的 q 分位   （逐类阈值）
  分数本身不变（温度维度已撤销，见协议 §2.1）
阈值作用于 **NMS 之前**的候选（低于阈值者丢弃，不参与 NMS，也不输出）。
NMS 规则与 g5a 完全一致（复用 g5a.nms_image，并列规则不动）。

Run:
  source sam3_ovis/scripts/load_environment.sh
  CUDA_VISIBLE_DEVICES=<free> "$PYTHON" sam3_ovis/graph/s2_rescore.py --phase gen
  CUDA_VISIBLE_DEVICES=""       "$PYTHON" sam3_ovis/graph/s2_rescore.py --phase select
  CUDA_VISIBLE_DEVICES=""       "$PYTHON" sam3_ovis/graph/s2_rescore.py --phase eval
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
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402
from graph import g5a_free_lever as g5a  # noqa: E402

TRAIN_ROOT = Path("/data/wangcheng/geoevi-r1/freeler_train")
VAL_ROOT = g5a.OUT_ROOT                      # /data/wangcheng/geoevi-r1/freeler
LOG_DIR = Path(os.environ.get(
    "SAM3_OVIS_LOG_ROOT",
    str(Path(__file__).resolve().parents[1] / "logs"))) / "s2"
COMBO_CFG = {"prompt": "space", "floor": 0.01, "res": 1344}
R1V2_CKPT = "/data/wangcheng/geoevi-r1/run1/checkpoints/checkpoint.pt"

# 封闭网格（协议 §2 + §2.1；登记后不得扩展）：5 个逐类阈值配置
QS = [0, 50, 80, 90, 95]
TEMP = 1.0   # 温度维度已撤销；保留常量仅为接口稳定，恒为 1.0
BOOT_N = 2000
SEED = 20261122
GATE_FP = 350.0
GATE_MAP_LO = -1.0   # pt；配对 CI 下界须 > -1.0


# ---------------------------------------------------------------- 数据 ----
def _gt():
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    return COCO(str(root / DATASETS["iSAID"].val_json))


def _train_img_ids(gt):
    """42 个 dev 组（非 val）下的全部图，按 id 排序。"""
    split = json.load(open(Path(
        "/data/wangcheng/geoevi-g2a-cache/group_split.json")))
    assert split and set(split.values()) == {"train", "val"}, \
        "group_split.json 语义与预期不符"
    train_groups = {g for g, s in split.items() if s == "train"}
    val_groups = {g for g, s in split.items() if s == "val"}
    id2g = {im["id"]: im["file_name"].replace(".png", "").split("_")[0]
            for im in gt.dataset["images"]}
    tr = sorted(i for i, g in id2g.items() if g in train_groups)
    va = {i for i, g in id2g.items() if g in val_groups}
    assert not (set(tr) & va), "train/val 图交集非空"
    return tr, va


def _load_recs(root, arm):
    d = root / arm
    if not d.is_dir():
        sys.exit(f"[fatal] 缺少候选目录 {d}（先跑 --phase gen）")
    return [json.loads(p.read_text())
            for p in sorted(d.glob("cands_*.json"))]


# ---------------------------------------------------------------- gen ----
def phase_gen():
    if g5a.OUT_ROOT == TRAIN_ROOT:
        sys.exit("[fatal] train/val 输出根不得相同")
    val_ids = {json.loads(p.read_text())["img_id"]
               for p in (g5a.OUT_ROOT / "COMBO").glob("cands_*.json")}
    gt = _gt()
    tr, va = _train_img_ids(gt)
    assert not (set(tr) & set(val_ids)), "train 池与已有 val 候选有交集"
    print(f"[s2-gen] train pool: {len(tr)} imgs (val pool {len(va)}; "
          f"overlap 0)", flush=True)

    # 候选生成循环在本函数内自含（不复用 g5a.gen_arm：后者按 val
    # manifest 筛图且写 g5a.OUT_ROOT；此处输出到独立的 train 目录）
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    from graph.g4b_gpu_probes import set_grid
    from PIL import Image

    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    spec = DATASETS["iSAID"]
    cat_ids = gt.getCatIds()
    cat_names = [gt.loadCats([c])[0]["name"] for c in cat_ids]

    def build(arm):
        if arm == "COMBO":
            m = build_sam3_image_model(
                checkpoint_path=os.environ["SAM3_CKPT"],
                load_from_HF=False, enable_inst_interactivity=False)
        else:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                                   / "sam3"))
            from r1_lora import apply_lora
            m = build_sam3_image_model(
                checkpoint_path=os.environ["SAM3_CKPT"],
                load_from_HF=False, enable_inst_interactivity=False,
                enable_segmentation=True)
            apply_lora(m, r=8, alpha=16)
            ck = torch_load(R1V2_CKPT)
            res = m.load_state_dict(ck["model"], strict=False)
            assert all(k.startswith("segmentation_head.")
                       for k in res.missing_keys), res.missing_keys[:5]
            assert not res.unexpected_keys, res.unexpected_keys[:5]
        return m.cuda().eval()

    for arm in ("COMBO", "R1v2"):
        out_dir = TRAIN_ROOT / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        model = build(arm)
        cfg = COMBO_CFG
        set_grid(model, cfg["res"] // 14, "cuda")
        processor = Sam3Processor(
            model, device="cuda", resolution=cfg["res"],
            confidence_threshold=cfg["floor"])
        t0 = __import__("time").time()
        n = 0
        for k, img_id in enumerate(tr):
            out_path = out_dir / f"cands_{img_id}.json"
            if out_path.exists():
                continue
            info = gt.loadImgs(img_id)[0]
            image = Image.open(root / spec.images_dir
                               / info["file_name"]).convert("RGB")
            with __import__("torch").autocast("cuda",
                                              dtype=__import__("torch")
                                              .bfloat16):
                state = processor.set_image(image)
                cands = []
                for cname in cat_names:
                    state = processor.set_text_prompt(
                        prompt=g5a.prompt_of(cname, cfg["prompt"]),
                        state=state)
                    sc = state["scores"]
                    for j in range(int(sc.shape[0])
                                   if sc is not None else 0):
                        s = float(sc[j])
                        if s < cfg["floor"]:
                            continue
                        m = state["masks"][j, 0].cpu().numpy().astype(bool)
                        if not m.any():
                            continue
                        ys = np.where(m.any(1))[0]
                        xs = np.where(m.any(0))[0]
                        from pycocotools import mask as mask_util
                        rle = mask_util.encode(np.asfortranarray(
                            m.astype(np.uint8)))
                        cands.append({
                            "cls": cname, "score": s,
                            "box": [int(xs[0]), int(ys[0]),
                                    int(xs[-1]) + 1, int(ys[-1]) + 1],
                            "rle": {"size": list(rle["size"]),
                                    "counts": rle["counts"]
                                    .decode("utf-8")}})
            out_path.write_text(json.dumps({
                "img_id": img_id, "W": image.width, "H": image.height,
                "cands": cands}))
            n += 1
            __import__("torch").cuda.empty_cache()
            if n % 50 == 0:
                print(f"[s2-gen {arm}] {k + 1}/{len(tr)} "
                      f"{__import__('time').time() - t0:.0f}s", flush=True)
        del model, processor
        __import__("torch").cuda.empty_cache()
        print(f"[s2-gen {arm}] done -> {out_dir}", flush=True)


def torch_load(p):
    import torch
    return torch.load(p, map_location="cpu", weights_only=True)


# -------------------------------------------------------------- 重标 ----
def rescore(cands, temp, q, tau):
    """返回重标后的候选列表（新 dict，不改输入）。tau: {cls: 阈值}。"""
    out = []
    for c in cands:
        t = tau.get(c["cls"])
        if t is not None and c["score"] < t:
            continue                      # 低于逐类阈值：丢弃
        s = c["score"] ** (1.0 / temp) if temp != 1.0 else c["score"]
        d = dict(c)
        d["score"] = s
        out.append(d)
    return out


def dets_of(recs, temp, q, tau, name_to_id):
    out = []
    for r in recs:
        for c in g5a.nms_image(rescore(r["cands"], temp, q, tau)):
            out.append({"image_id": r["img_id"],
                        "category_id": name_to_id[c["cls"]],
                        "segmentation": c["rle"],
                        "score": c["score"]})
    return out


def make_tau(recs, q):
    """逐类阈值 = train 池该类（NMS 前）分数的 q 分位。q=0 → 无阈值。"""
    if q == 0:
        return {}
    by = {}
    for r in recs:
        for c in r["cands"]:
            by.setdefault(c["cls"], []).append(c["score"])
    return {k: float(np.percentile(v, q)) for k, v in by.items()}


# -------------------------------------------------------------- 评估 ----
def _sub(gt, img_ids):
    s = COCO()
    ids = set(img_ids)
    s.dataset = {
        "images": [im for im in gt.dataset["images"] if im["id"] in ids],
        "annotations": [a for a in gt.dataset["annotations"]
                        if a["image_id"] in ids],
        "categories": gt.dataset["categories"]}
    s.createIndex()
    return s


def _eval(sub, dets):
    if not dets:
        return 0.0
    ev = COCOeval(sub, sub.loadRes([dict(d) for d in dets]),
                  iouType="segm")
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0])


def _per_image(gt, dets, img_ids):
    out = {}
    by = {}
    for d in dets:
        by.setdefault(d["image_id"], []).append(d)
    for i in img_ids:
        out[i] = _eval(_sub(gt, [i]), by.get(i, []))
    return out


# ------------------------------------------------------------ select ----
def phase_select():
    """只读 train 池。（本函数的函数体不得引用任何 val 候选目录；
    tests/test_s2_rescore.py 对此做静态检查。）"""
    gt = _gt()
    name_to_id = {c["name"]: c["id"] for c in gt.loadCats(gt.getCatIds())}
    recs = _load_recs(TRAIN_ROOT, "R1v2")
    img_ids = [r["img_id"] for r in recs]
    sub = _sub(gt, img_ids)
    print(f"[s2-select] train pool: {len(recs)} imgs, "
          f"{sum(len(r['cands']) for r in recs)} raw cands", flush=True)

    grid = []
    for q in QS:
        tau = make_tau(recs, q)
        grid.append({"q": q,
                     "mAP": _eval(sub, dets_of(recs, TEMP, q, tau,
                                               name_to_id)),
                     "n_tau": len(tau)})
        print(f"  q={grid[-1]['q']:2d} mAP={grid[-1]['mAP']:.4f} "
              f"(tau on {grid[-1]['n_tau']} classes)", flush=True)
    # 并列取更低的 q（更少丢弃 = 更保守的干预）
    best = max(grid, key=lambda g: (g["mAP"], -g["q"]))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "select.json").write_text(json.dumps(
        {"pool": "train (42 dev groups)", "n_imgs": len(recs), "grid": grid,
         "selected": {"q": best["q"]},
         "tie_break": "max mAP, then lower q"}, indent=1))
    print(f"[s2-select] SELECTED q={best['q']} "
          f"(train mAP {best['mAP']:.4f}) -> {LOG_DIR / 'select.json'}",
          flush=True)


# -------------------------------------------------------------- eval ----
def phase_eval():
    gt = _gt()
    name_to_id = {c["name"]: c["id"] for c in gt.loadCats(gt.getCatIds())}
    sel = json.load(open(LOG_DIR / "select.json"))["selected"]
    T, q = TEMP, sel["q"]
    tau = make_tau(_load_recs(TRAIN_ROOT, "R1v2"), q)   # 阈值仅来自 train

    val = _load_recs(VAL_ROOT, "R1v2")
    img_ids = [r["img_id"] for r in val]
    sub = _sub(gt, img_ids)
    d_base = dets_of(val, 1.0, 0, {}, name_to_id)
    d_new = dets_of(val, T, q, tau, name_to_id)
    m_base, m_new = _eval(sub, d_base), _eval(sub, d_new)
    p_base, p_new = _per_image(gt, d_base, img_ids), _per_image(
        gt, d_new, img_ids)

    d = np.array([p_new[i] - p_base[i] for i in img_ids])
    rng = np.random.default_rng(SEED)
    n = len(img_ids)
    smp = np.array([d[rng.integers(0, n, n)].mean()
                    for _ in range(BOOT_N)])
    lo, hi = np.percentile(smp, [2.5, 97.5])
    fp_new = sum(1 for r in val for _ in g5a.nms_image(
        rescore(r["cands"], T, q, tau))) / len(val)
    fp_base = sum(1 for r in val for _ in g5a.nms_image(
        rescore(r["cands"], 1.0, 0, {}))) / len(val)

    # 候选纯度与 center 召回（与 g5a S1 修复后同口径，确定性）
    id_to_name = {v: k for k, v in name_to_id.items()}
    gt_by = {}
    for a in gt.dataset["annotations"]:
        if a["image_id"] in set(img_ids):
            gt_by.setdefault(a["image_id"], []).append(a)
    from graph.g1_qbridge_probe import box_iou_xyxy
    from pycocotools import mask as mask_util

    def extras(recs_sel, temp, qq, tau_):
        """候选纯度 / center 召回 / 保留数，按给定重标参数计算。"""
        n_k = n_m = n_ms = n_ctr = n_g = 0
        for r in recs_sel:
            kept = g5a.nms_image(rescore(r["cands"], temp, qq, tau_))
            gts = gt_by.get(r["img_id"], [])
            n_g += len(gts)
            n_k += len(kept)
            for c in kept:
                bi, bc = 0.0, None
                for a in gts:
                    ax, ay, aw, ah = a["bbox"]
                    iou = box_iou_xyxy(c["box"],
                                       (ax, ay, ax + aw, ay + ah))
                    if iou > bi:
                        bi, bc = iou, a["category_id"]
                n_m += bi >= 0.5
                n_ms += (bi >= 0.5 and bc == name_to_id[c["cls"]])
            for a in gts:
                ax, ay, aw, ah = a["bbox"]
                cx, cy = int(ax + aw / 2), int(ay + ah / 2)
                n_ctr += any(mask_util.decode({
                    "size": c["rle"]["size"],
                    "counts": c["rle"]["counts"].encode()})[cy, cx]
                    for c in kept)
        return {"purity_pct": round(100 * n_ms / max(n_m, 1), 2),
                "center_recall_pct": round(100 * n_ctr / max(n_g, 1), 2),
                "kept_per_img": round(n_k / len(recs_sel), 1)}

    res = {
        "selected": {"q": q, "n_classes_with_tau": len(tau),
                     "note": "temperature dimension withdrawn "
                             "(protocol §2.1); scores not transformed"},
        "val": {
            "mAP_base": round(m_base, 4), "mAP_rescored": round(m_new, 4),
            "paired_delta_pt": round(100 * float(d.mean()), 2),
            "paired_ci95_pt": [round(100 * float(lo), 2),
                               round(100 * float(hi), 2)],
            "fp_per_img_base": round(fp_base, 1),
            "fp_per_img_rescored": round(fp_new, 1),
            "base": extras(val, 1.0, 0, {}),
            "rescored": extras(val, T, q, tau)},
    }
    # base 的伴随指标必须与 g5a 存档逐位一致（自校验：口径未漂移）
    ab = json.load(open(g5a.RES_DIR / "ablation.json"))["R1v2"]
    assert abs(res["val"]["base"]["purity_pct"]
               - ab["class_purity_pct"]) < 0.02, \
        (res["val"]["base"]["purity_pct"], ab["class_purity_pct"])
    assert res["val"]["base"]["fp_per_img"] == ab["fp_per_img"], \
        (res["val"]["base"]["fp_per_img"], ab["fp_per_img"])
    pass_ok = fp_new <= GATE_FP and lo * 100 > GATE_MAP_LO
    res["gate"] = {"fp_max": GATE_FP, "map_ci_lo_min_pt": GATE_MAP_LO,
                   "fp_pass": bool(fp_new <= GATE_FP),
                   "map_pass": bool(lo * 100 > GATE_MAP_LO),
                   "VERDICT": "PASS" if pass_ok else "FAIL"}
    (LOG_DIR / "eval.json").write_text(json.dumps(res, indent=1,
                                                  ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gen", "select", "eval"],
                    required=True)
    a = ap.parse_args()
    {"gen": phase_gen, "select": phase_select, "eval": phase_eval}[a.phase]()


if __name__ == "__main__":
    main()
