# contrib/ — WHU 之外的数据集实验线

| 路径 | 用途 |
|---|---|
| `ablations_uecoco/` | ue_coco 上的 R1-C4-RD 自包含 wrapper 与 GPU 监视启动脚本。 |
| `resplit_ue_coco.py` | ue_coco 格网重切分（产出 `*.clean.json`，供上面 wrapper 使用）。 |
| `build_isaid_patches.py` | DOTA+iSAID → 800×800 COCO patches 一次性数据准备。 |
