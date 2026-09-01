# configs/ — MMEngine 配置与机器环境

| 文件 | 用途 |
|---|---|
| `_sam2_registry.py` | SAM2 型号 → checkpoint/Hydra YAML/neck 通道的唯一映射来源。 |
| `rsprompter_anchor_satS_v11_sam2_large_full.py` | 根配置（历史文件名）：detector/RPN/RoI 结构与优化默认值。 |
| `whu1024_baseplus_clean.py` | WHU-1024 MLP 桥接基线（`prompt_sparse_mode="point"`）。 |
| `whu1024_baseplus_explicit_coarse.py` | coarse 主配置：解析全部架构环境变量为 `cfg.model`。 |
| `environment.sh` | 本机默认环境（Python/数据/SAM2 权重/输出根目录）。 |
| `environment2.sh` | lthpc 主机版本；经 `environment.local.sh` 软链或 `PORTABLE_SAM2_ENV_FILE` 激活。 |

继承链：根配置 → clean → explicit_coarse。架构行为全部由环境变量解析进 `cfg.model`。
