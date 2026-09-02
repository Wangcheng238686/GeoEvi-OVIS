# GeoEvi-OVIS

单流遥感实例分割项目（WHU-1024 / SAM2 Base+）：以 SAM2 Hiera 视觉编码器 +
PAFPN 检测分支产生 proposal，经 **ECPG**（显式 coarse mask → 点/框/dense
prompt 生成器）与 **P2-BRR**（P2 边界约束残差修正器）把可解释的粗形状转化为
SAM2 可消费的提示，由可训练的 SAM2 MaskDecoder 输出最终实例掩码。

> 本仓库由 `portable_sam2_explicit_coarse` 演化而来（旧项目迁移记录见
> `MIGRATION_MANIFEST.md`）；项目名 **GeoEvi-OVIS** 以仓库所在父文件夹名称为准。

## 目录索引

| 目录 | 说明 | 详细介绍 |
|---|---|---|
| `geoevi_ovis/models/` | 模型层：编码器/neck/MaskHead/coarse/P2-BRR/契约 | [README](geoevi_ovis/models/README.md) |
| `geoevi_ovis/train/` | DDP 训练入口与子模块 | [README](geoevi_ovis/train/README.md) |
| `geoevi_ovis/inference/` | checkpoint 推理与可视化入口 | [README](geoevi_ovis/inference/README.md) |
| `geoevi_ovis/configs/` | MMEngine 配置继承链与机器环境 | [README](geoevi_ovis/configs/README.md) |
| `geoevi_ovis/data/` | WHU COCO 数据集与 loader | [README](geoevi_ovis/data/README.md) |
| `geoevi_ovis/scripts/` | 可执行入口（train / inference） | [README](geoevi_ovis/scripts/README.md) |
| `geoevi_ovis/tests/` | 组件 smoke / 单测 / DDP 控制面 | [README](geoevi_ovis/tests/README.md) |
| `geoevi_ovis/tools/` | 可视化工具与历史分析探针 | [README](geoevi_ovis/tools/README.md) |
| `geoevi_ovis/contrib/` | WHU 之外的数据集实验线 | [README](geoevi_ovis/contrib/README.md) |
| `geoevi_ovis/utils/` | 共享 COCO 评估工具 | [README](geoevi_ovis/utils/README.md) |
| `geoevi_ovis/docs/` | 模块设计文档 | [README](geoevi_ovis/docs/README.md) |
| `sam3_ovis/` | SAM3 开放词汇遥感实例分割主线（design-v2，P-1 进行中；vendored 官方 repo 在 `sam3/`） | [README](sam3_ovis/README.md) |

项目级文档：

- [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md) —— 架构、逐文件说明、
  环境变量表与文档同步规则（**事实来源，改代码必须同步**）；
- [`C5_V2_METHOD.md`](C5_V2_METHOD.md) —— 方法设计、张量数据流、损失与
  论文制图建议；
- [`MIGRATION_MANIFEST.md`](MIGRATION_MANIFEST.md) —— 旧项目迁移与排除项记录。

## 快速开始

```bash
# 1) 本机 conda 环境（Windows CPU；服务器按文件头注释换 CUDA torch）
conda env create -p <envs 路径>/sam2 -f environment.yml

# 2) 组件 smoke（本机 CPU 可跑）
cd geoevi_ovis
SAM2_REPO=<仓库>/sam2 SAM2_CKPT=<sam2_hiera_base_plus.pt>   python tests/smoke_test_components.py

# 3) 训练（消融点筛默认 20% train / 100% validation，两卡 DDP）
bash scripts/train/run_whu1024_explicit_coarse.sh

# 4) 推理
bash scripts/inference/infer_whu_checkpoint.sh /path/to/best_model.pth
```

## 核心工程契约（详见 PROJECT_ARCHITECTURE.md）

1. **自描述 checkpoint（schema v2）**：`model_config` + 训练超参 + 数据协议 +
   架构指纹；推理只依赖 checkpoint，不按文件名猜结构；
2. **影响模型行为的参数必须解析进 `cfg.model`**，禁止只活在环境变量里；
3. **单一环境配置**：机器相关路径集中在 `configs/environment*.sh`；
4. **文档同步**：代码变更与 `PROJECT_ARCHITECTURE.md` 同一提交。

## 消融矩阵（当前存活）

r0（MLP 基线）、r1_c3、r1_c4、r1_c4_rd（±roi_local）、r1_c4_g、c5v2、
paper_promptminer_rd_p2_whu_full(±fast)，全部固定 stride-16 / 64×64
image embedding。矩阵定义见
[`geoevi_ovis/scripts/train/ablations/README.md`](geoevi_ovis/scripts/train/ablations/README.md)。
