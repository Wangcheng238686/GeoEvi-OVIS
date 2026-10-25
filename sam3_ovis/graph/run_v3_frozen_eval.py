#!/usr/bin/env python3
"""v3.0 冻结参数独立验证（T2 收尾，一次运行）。

冻结参数（2026-09-28 审计后登记，不得更改）：beta=0.5, delta=0, lambda=2。
验证集：iSAID train 源图组中**未参与 dev 选择**的组（排除 t2_cache/manifest.json
登记的 15 个 dev 组），seed 20260907，--n-source-groups（默认 10）。

两个阶段：
  --phase cache  (GPU)  候选缓存：SAM3 text-only -> pool floor -> 类内 NMS
                        -> CLIP 全类 z 向量 -> 空间邻接。逐图 done ledger 断点续跑。
  --phase score  (CPU)  离线 B0/B1/B2/B3 打分（一律经 graph.scoring_v3.
                        score_candidates，禁止手写公式）+ COCO segm mAP
                        + 逐图配对 bootstrap（AP@[.50:.95]，maxDets=100）。

B3 定义（审计口径，2026-09-28）：rival 集合 = 同图其他**预测类别**
（by_img 的 cat_name 并集去掉自身类），不做空间重叠限制。
B2 rival 集合 = 空间重叠（IoU>=0.5）候选的类别。
两者都从本候选 z 向量取值（z_i(y) 定义在本 crop 上）。

判据树（预注册，见 notes/reports/v3-frozen-eval-plan.md）：
  B2 <= B1        -> 停止相对惩罚机制；
  B3 >= B2        -> 停止空间图主张（至多保留同图类别对比重排）；
  B2 > B1 且 B2 > B3（bootstrap CI 不含 0）-> 才有资格进入 T3 讨论。

Run:  source sam3_ovis/scripts/load_environment.sh
      "$PYTHON" sam3_ovis/graph/run_v3_frozen_eval.py --phase cache
      "$PYTHON" sam3_ovis/graph/run_v3_frozen_eval.py --phase score
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402
from graph.scoring_v3 import ScoringConfig, score_candidates  # noqa: E402

FROZEN = ScoringConfig(beta=0.5, delta=0.0, lam=2.0)
SEED = 20260907          # 验证集抽样 seed（登记于本文件，先于任何结果冻结）
N_IMAGES_DEFAULT = 200
N_GROUPS_DEFAULT = 10
BOOT_N = 10000
PER_IMAGE_MAXDETS = 100

OUT_DIR_NAME = "v3_frozen_eval"


def _out_root() -> Path:
    return Path(os.environ["SAM3_OVIS_LOG_ROOT"]) / OUT_DIR_NAME


# ---------------------------------------------------------------- cache ----
def phase_cache(n_groups: int, n_images: int) -> None:
    """与 t2_candidate_cache.py 同口径的候选生成，但抽样排除 dev 15 组。"""
    import torch
    from graph.provider import RemoteCLIPProvider
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    from graph.t2_candidate_cache import (
        CROP_MARGIN, NMS_IOU, POOL_FLOOR, RIVAL_IOU, SEED as DEV_SEED,
        _iou_cropped, _tight_bbox,
    )

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    spec = DATASETS["iSAID"]
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    log_root = Path(os.environ["SAM3_OVIS_LOG_ROOT"])
    gt = COCO(str(root / spec.val_json))
    cat_ids = gt.getCatIds()
    cat_names = {c["id"]: c["name"] for c in gt.loadCats(cat_ids)}
    name_list = [cat_names[c] for c in cat_ids]

    prefixes = defaultdict(list)
    for im in gt.dataset["images"]:
        pref = im["file_name"].replace(".png", "").split("_")[0]
        prefixes[pref].append(im["id"])
    all_prefs = sorted(prefixes.keys())

    dev_manifest = json.load(open(log_root / "t2_cache" / "manifest.json"))
    dev_groups = set(dev_manifest["source_groups"])
    held_out = [p for p in all_prefs if p not in dev_groups]
    # 只保留有 GT 标注的组（登记口径：零 GT 图不进验证分母）
    img_anns = set(a["image_id"] for a in gt.dataset["annotations"])
    held_out = [p for p in held_out
                if any(i in img_anns for i in prefixes[p])]

    rng = np.random.default_rng(SEED)
    chosen = sorted(rng.choice(held_out,
                               size=min(n_groups, len(held_out)),
                               replace=False).tolist())
    val_ids = []
    for p in chosen:
        val_ids.extend(sorted(i for i in prefixes[p] if i in img_anns))
    val_ids = sorted(val_ids)[:n_images]

    out_dir = _out_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    import subprocess
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2])).decode().strip()
    except Exception:
        sha = "unknown"
    manifest = {"seed": SEED, "frozen_cfg": {"beta": FROZEN.beta,
                                             "delta": FROZEN.delta,
                                             "lam": FROZEN.lam},
                "excluded_dev_groups": sorted(dev_groups),
                "source_groups": chosen, "n_images": len(val_ids),
                "image_ids": val_ids, "code_sha": sha}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[frozen] {len(val_ids)} imgs from {len(chosen)} held-out groups "
          f"(excluded {len(dev_groups)} dev groups), sha={sha[:12]}", flush=True)

    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=POOL_FLOOR)
    provider = RemoteCLIPProvider()
    txt_feats = provider.encode_texts_cached(name_list)

    cache_path = out_dir / "candidates.jsonl"
    done_path = out_dir / "done_images.jsonl"
    done_ids = set()
    if done_path.exists():
        with open(done_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["img_id"])
                except (json.JSONDecodeError, KeyError):
                    break
    todo = [i for i in val_ids if i not in done_ids]
    print(f"[frozen] {len(todo)} to process ({len(done_ids)} done)", flush=True)

    cache_f = open(cache_path, "a")
    done_f = open(done_path, "a")
    t0 = time.time()
    for k, img_id in enumerate(todo):
        info = gt.loadImgs(img_id)[0]
        image = Image.open(root / spec.images_dir / info["file_name"]) \
            .convert("RGB")
        state = processor.set_image(image)

        raw = []
        for cid in cat_ids:
            state = processor.set_text_prompt(state=state,
                                              prompt=cat_names[cid])
            sc = state["scores"]
            for j in range(int(sc.shape[0]) if sc is not None else 0):
                if float(sc[j]) < POOL_FLOOR:
                    continue
                m = state["masks"][j, 0].detach().cpu().numpy().astype(bool)
                b = _tight_bbox(m)
                if b is None:
                    continue
                raw.append({"cat_id": cid, "cat_name": cat_names[cid],
                            "score": float(sc[j]), "mask": m, "bbox": b})

        nmsd = []
        for cid in sorted({d["cat_id"] for d in raw}):
            cls = sorted([d for d in raw if d["cat_id"] == cid],
                         key=lambda d: -d["score"])
            for d in cls:
                if not any(_iou_cropped(d["mask"], d["bbox"],
                                        km["mask"], km["bbox"]) >= NMS_IOU
                           for km in nmsd if km["cat_id"] == cid):
                    nmsd.append(d)

        provider.clear_image_cache()
        crop_items = []
        for i, d in enumerate(nmsd):
            x1, y1, x2, y2 = d["bbox"]
            mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
            crop_items.append((f"img{img_id}/{i}",
                               image.crop((max(0, int(x1 - mx)),
                                           max(0, int(y1 - my)),
                                           min(image.width, int(x2 + mx)),
                                           min(image.height, int(y2 + my))))))
        feats = provider.encode_images_cached(crop_items)

        # 空间邻接（key = img{id}/{i}，与 t2_candidate_cache 相同口径）
        spatial_adj = defaultdict(list)
        for i in range(len(nmsd)):
            for j in range(i + 1, len(nmsd)):
                if _iou_cropped(nmsd[i]["mask"], nmsd[i]["bbox"],
                                nmsd[j]["mask"], nmsd[j]["bbox"]) >= RIVAL_IOU:
                    spatial_adj[f"img{img_id}/{i}"].append(
                        f"img{img_id}/{j}")
                    spatial_adj[f"img{img_id}/{j}"].append(
                        f"img{img_id}/{i}")

        by_idx = {i: d for i, d in enumerate(nmsd)}
        for i, d in enumerate(nmsd):
            f = feats[f"img{img_id}/{i}"]
            z_vec = {cn: round(float(np.dot(f, txt_feats[cn])), 6)
                     for cn in name_list}
            # 直接记录两类 rival 类集合的 z 最大值，score 阶段零歧义
            rival_keys = spatial_adj.get(f"img{img_id}/{i}", [])
            spatial_cls = {by_idx[int(rk.split('/')[1])]["cat_name"]
                           for rk in rival_keys} - {d["cat_name"]}
            pred_cls = {c2["cat_name"] for c2 in nmsd} - {d["cat_name"]}
            rle = mask_util.encode(
                np.asfortranarray(d["mask"].astype(np.uint8)))
            cache_f.write(json.dumps({
                "img_id": img_id,
                "cat_id": d["cat_id"],
                "cat_name": d["cat_name"],
                "raw_score": d["score"],
                "mask_rle": {"size": list(rle["size"]),
                             "counts": rle["counts"].decode("utf-8")},
                "bbox": list(d["bbox"]),
                "z": z_vec,
                "rival_z_spatial": (max(z_vec[cn] for cn in spatial_cls)
                                    if spatial_cls else None),
                "rival_z_pred": (max(z_vec[cn] for cn in pred_cls)
                                 if pred_cls else None),
            }) + "\n")

        done_f.write(json.dumps({"img_id": img_id}) + "\n")
        cache_f.flush()
        done_f.flush()
        torch.cuda.empty_cache()
        if (k + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {k + 1}/{len(todo)} imgs, {el / (k + 1):.1f}s/img",
                  flush=True)

    cache_f.close()
    done_f.close()
    print(f"[frozen] cache done -> {cache_path}", flush=True)


# ---------------------------------------------------------------- score ----
def _per_image_ap(gt: COCO, dets, img_ids) -> dict:
    """逐图 segm AP@[.50:.95]（maxDets=100，全部面积档）。零预测图计 0。"""
    per = {}
    for img_id in img_ids:
        sub = COCO()
        anns = [a for a in gt.dataset["annotations"]
                if a["image_id"] == img_id]
        im = [i for i in gt.dataset["images"] if i["id"] == img_id]
        sub.dataset = {"images": im, "annotations": anns,
                       "categories": gt.dataset["categories"]}
        sub.createIndex()
        d = [x for x in dets if x["image_id"] == img_id]
        dt = sub.loadRes(d) if d else None
        ev = COCOeval(sub, dt, iouType="segm")
        ev.params.maxDets = [PER_IMAGE_MAXDETS]
        ev.evaluate(); ev.accumulate(); ev.summarize()
        per[img_id] = float(ev.stats[0])
    return per


def phase_score() -> None:
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    log_root = Path(os.environ["SAM3_OVIS_LOG_ROOT"])
    out_dir = _out_root()
    spec = DATASETS["iSAID"]
    gt = COCO(str(root / spec.val_json))
    manifest = json.load(open(out_dir / "manifest.json"))
    img_ids = manifest["image_ids"]
    name_to_id = {c["name"]: c["id"] for c in gt.loadCats(gt.getCatIds())}

    cands = [json.loads(l) for l in open(out_dir / "candidates.jsonl")
             if l.strip()]
    print(f"[frozen] {len(cands)} candidates, {len(img_ids)} imgs", flush=True)
    cached_ids = {c["img_id"] for c in cands}
    missing = [i for i in img_ids if i not in cached_ids]
    if missing:
        sys.exit(f"[fatal] {len(missing)} manifest 图无缓存记录，先补 cache")

    raw_scores, z_own, z_spatial, z_pred, cand_meta = {}, {}, {}, {}, []
    for i, c in enumerate(cands):
        cid = f"c{i}"
        raw_scores[cid] = c["raw_score"]
        z_own[cid] = c["z"][c["cat_name"]]
        z_spatial[cid] = c.get("rival_z_spatial")
        z_pred[cid] = c.get("rival_z_pred")
        cand_meta.append({"cid": cid, "img_id": c["img_id"],
                          "category_id": name_to_id[c["cat_name"]],
                          "segmentation": c["mask_rle"]})

    scored = score_candidates(raw_scores, z_own, z_spatial, FROZEN,
                              z_rival_max_nospatial=z_pred)
    # 硬验收：FROZEN.lam=2 下 B2/B3 均不高于 B1；lam=0 时 B2==B1（单元测试覆盖）
    for cid in raw_scores:
        assert scored[cid]["B1"] >= scored[cid]["B2"] - 1e-9
        assert scored[cid]["B1"] >= scored[cid]["B3"] - 1e-9

    results, per_img = {}, {}
    for vname in ("B0", "B1", "B2", "B3"):
        dets = [{"image_id": m["img_id"],
                 "category_id": m["category_id"],
                 "score": scored[m["cid"]][vname],
                 "segmentation": m["segmentation"]} for m in cand_meta]
        sub = COCO()
        sub.dataset = {
            "images": [i for i in gt.dataset["images"]
                       if i["id"] in set(img_ids)],
            "annotations": [a for a in gt.dataset["annotations"]
                            if a["image_id"] in set(img_ids)],
            "categories": gt.dataset["categories"]}
        sub.createIndex()
        dt = sub.loadRes(dets) if dets else None
        ev = COCOeval(sub, dt, iouType="segm")
        ev.evaluate(); ev.accumulate(); ev.summarize()
        results[vname] = {"mAP": round(float(ev.stats[0]), 4),
                          "AP50": round(float(ev.stats[1]), 4)}
        per_img[vname] = _per_image_ap(gt, dets, img_ids)
        print(f"  {vname}  mAP={results[vname]['mAP']:.4f} "
              f"AP50={results[vname]['AP50']:.4f}", flush=True)

    # 逐图配对 bootstrap（同一图样本跨变体配对）
    rng = np.random.default_rng(SEED)
    vecs = {v: np.array([per_img[v][i] for i in img_ids])
            for v in ("B0", "B1", "B2", "B3")}
    boot = {}
    for a, b in (("B2", "B1"), ("B2", "B3"), ("B3", "B1"), ("B1", "B0"),
                 ("B2", "B0")):
        diff = vecs[a] - vecs[b]
        n = len(diff)
        samples = np.empty(BOOT_N)
        for t in range(BOOT_N):
            idx = rng.integers(0, n, n)
            samples[t] = diff[idx].mean()
        lo, hi = np.percentile(samples, [2.5, 97.5])
        boot[f"{a}-{b}"] = {
            "mean_diff": round(float(diff.mean()), 6),
            "ci95": [round(float(lo), 6), round(float(hi), 6)],
            "p_gt0": round(float((samples > 0).mean()), 4)}
        print(f"  {a}-{b}: {diff.mean():+.6f} "
              f"CI95=[{lo:+.6f},{hi:+.6f}] P(>0)={(samples > 0).mean():.3f}",
              flush=True)

    summary = {"frozen_cfg": {"beta": FROZEN.beta, "delta": FROZEN.delta,
                              "lam": FROZEN.lam},
               "n_images": len(img_ids), "results": results,
               "paired_bootstrap": boot, "seed": SEED}
    (out_dir / "frozen_results.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"-> {out_dir / 'frozen_results.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["cache", "score"], required=True)
    ap.add_argument("--n-source-groups", type=int, default=N_GROUPS_DEFAULT)
    ap.add_argument("--n-images", type=int, default=N_IMAGES_DEFAULT)
    args = ap.parse_args()
    if args.phase == "cache":
        phase_cache(args.n_source_groups, args.n_images)
    else:
        phase_score()


if __name__ == "__main__":
    main()
