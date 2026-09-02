# sam3_ovis/ — SAM3 开放词汇遥感实例分割主线（design-v2）

> 对应 `notes/design-v2.md` 与 `notes/implementation-v2.md`；与冻结的
> `geoevi_ovis/`（C5-v2 线）物理隔离，只读复用其工具，不修改原文件。

## 目录索引

| 目录 | 说明 | 详细介绍 |
|---|---|---|
| `configs/` | 机器环境与运行配置（SAM3_REPO / SAM3_CKPT / 数据根） | [README](configs/README.md) |
| `data/` | 五数据集 + 补充集的 COCO-instance 对齐与 loader | [README](data/README.md) |
| `graph/` | 同图实例—上下文证据图（P1 启用） | [README](graph/README.md) |
| `eval/` | SCORE 跨数据集评测口径 + 诊断量 | [README](eval/README.md) |
| `inference/` | SAM3 推理 harness（零样本 / 图推理 / exemplar） | [README](inference/README.md) |
| `scripts/` | 可执行入口（带 env 检查，不与 C5-v2 环境共存假设） | [README](scripts/README.md) |
| `tests/` | 冒烟框架与单测 | [README](tests/README.md) |
| `docs/` | SAM3 API 笔记、license 结论、GT 来源登记 | [README](docs/README.md) |

vendored 官方 repo 在仓库根 `sam3/`（与 `sam2/` 平级，env 变量 `SAM3_REPO`）。

## 阶段状态

- **P-1 工程与数据准备**（进行中）：S1.1 环境与权重 / S1.2 API 面验证 /
  S1.3 数据登记 / S1.4 评测管线 / S1.5 骨架。出口 M0 =
  `tests/smoke_pipeline.py` 空跑通过。
