#!/usr/bin/env python3
"""G0 源图级划分工具：新方向（图+SAM3 原生记忆）的 train/dev/confirm 划分。

协议登记（2026-11-05，划分产生前冻结）：
- 源域：SCORE 版 iSAID train（DATASETS['iSAID'].val_json，28k 切片，1411 个
  有 GT 的源图组，P 前缀）。一切划分以**源图组**为单位，绝不按切片随机分。
- 暴露登记（以下组已用于历史选参/验证/探针，**不得进入 confirm**，也不得
  进入新 dev——它们对开发结论已"烧毁"）：
  * t2_cache manifest source_groups（15 组，T2 dev 选参）；
  * v3_frozen_eval manifest source_groups（10 组，冻结验证）；
  * v3_frozen_eval manifest excluded_probe_groups（242 组，T1 竞争探针）。
  另登记 v4-S0 C0 用过的 DOTA val 100 图名（属 iSAID val 池，与 train 源图组
  理论无交集，逐名核对后记录，若碰撞则告警并入暴露集）。
- 抽样：清洗后的组按 seed=20261105 均匀洗牌；confirm=60 组、dev=60 组、
  train=其余（约 1034 组）。比例 ~5%/5%/90%。confirm 只用于最终冻结确认，
  本工具产出后任何人不得用 confirm 组做开发/选参。
- 唯一性：manifest 含全部组清单与 code_sha；重跑同 seed 必须逐组一致
  （工具自校验，不一致即退出非零）。

Run:  python3 sam3_ovis/data/g0_splits.py [--dry-run]
输出：sam3_ovis/logs/g0_splits/manifest.json（--dry-run 只打印不落盘）
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pycocotools.coco import COCO  # noqa: E402

SEED = 20261105
N_CONFIRM = 60
N_DEV = 60
LOG_DIR = Path(os.environ.get("SAM3_OVIS_LOG_ROOT",
                              Path(__file__).resolve().parents[1] / "logs")) \
    / "g0_splits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(os.environ.get("SAM3_OVIS_DATA_ROOT",
                               "/data/wangcheng/dataset/SCORE"))
    log_root = Path(os.environ.get("SAM3_OVIS_LOG_ROOT",
                                   Path(__file__).resolve().parents[1] / "logs"))
    gt = COCO(str(root / "iSAID/train/instances_train.json"))

    img_anns = set(a["image_id"] for a in gt.dataset["annotations"])
    groups = defaultdict(list)
    for im in gt.dataset["images"]:
        if im["id"] in img_anns:
            pref = im["file_name"].replace(".png", "").split("_")[0]
            groups[pref].append(im["id"])
    all_groups = sorted(groups)

    # ---- 暴露登记（读历史 manifest，缺失即失败——不许静默跳过） ----
    exposures = {}
    t2m = json.load(open(log_root / "t2_cache/manifest.json"))
    exposures["t2_dev"] = sorted(t2m["source_groups"])
    fm = json.load(open(log_root / "v3_frozen_eval/manifest.json"))
    exposures["v3_frozen"] = sorted(fm["source_groups"])
    exposures["t1_probe"] = sorted(fm["excluded_probe_groups"])
    # C0 的 DOTA val 图名（池不同，仅核对碰撞）
    c0m = json.load(open(log_root / "v4_s0_c0/manifest.json"))
    c0_names = sorted(c0m["images"].keys())
    train_name_set = {im["file_name"] for im in gt.dataset["images"]}
    collisions = [n for n in c0_names if n in train_name_set]
    if collisions:
        # 碰撞图所属组并入暴露集（保守处理）
        pref_set = {n.replace(".png", "").split("_")[0] for n in collisions}
        exposures["c0_collision"] = sorted(pref_set)

    exposed = set()
    for k in ("t2_dev", "v3_frozen", "t1_probe", "c0_collision"):
        exposed.update(exposures.get(k, []))
    clean = [g for g in all_groups if g not in exposed]

    # ---- 划分（确定性洗牌，二次运行自校验一致） ----
    import numpy as np
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(clean))
    confirm = sorted(clean[i] for i in perm[:N_CONFIRM])
    dev = sorted(clean[i] for i in perm[N_CONFIRM:N_CONFIRM + N_DEV])
    train = sorted(clean[i] for i in perm[N_CONFIRM + N_DEV:])

    n_imgs = lambda gs: sum(len(groups[g]) for g in gs)
    print(f"groups: total={len(all_groups)} exposed={len(exposed)} "
          f"clean={len(clean)}")
    print(f"train:   {len(train)} groups / {n_imgs(train)} imgs")
    print(f"dev:     {len(dev)} groups / {n_imgs(dev)} imgs")
    print(f"confirm: {len(confirm)} groups / {n_imgs(confirm)} imgs")
    print(f"c0/train name collisions: {len(collisions)}")
    assert not (set(train) & set(dev)) and not (set(train) & set(confirm)) \
        and not (set(dev) & set(confirm)), "split overlap!"
    assert not ((set(train) | set(dev) | set(confirm)) & exposed)

    if args.dry_run:
        print("[dry-run] no manifest written")
        return 0

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2])).decode().strip()
    except Exception:
        sha = "unknown"
    manifest = {
        "seed": SEED, "code_sha": sha,
        "unit": "iSAID train source-image group (P prefix)",
        "protocol": {"confirm_groups": N_CONFIRM, "dev_groups": N_DEV},
        "exposure_registry": exposures,
        "c0_dota_val_names_checked": len(c0_names),
        "c0_collisions": collisions,
        "splits": {
            "train": {"groups": train, "n_images": n_imgs(train)},
            "dev": {"groups": dev, "n_images": n_imgs(dev)},
            "confirm": {"groups": confirm, "n_images": n_imgs(confirm)},
        },
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / "manifest.json"
    if out.exists():  # 幂等校验：同 seed 重跑必须一致
        old = json.load(open(out))
        if old["splits"] != manifest["splits"]:
            sys.exit("[fatal] 重跑划分与已有 manifest 不一致——禁止改历史划分")
        print("re-run identical to existing manifest (idempotent OK)")
        return 0
    out.write_text(json.dumps(manifest, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
