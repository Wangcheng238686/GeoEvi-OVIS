# P0 Gate 判据预登记（S2.1 / M1）

> 登记日期：2026-09-02。**本文件在 S2.2 任何零样本推理运行之前提交；登记后
> 阈值与判据不回改**（implementation-v2 §2 S2.1 纪律）。若 gate 触发设计迭代，
> 按迭代规则开新版本，不在本文件上修改。

## 1. SCORE 论文主表基线（抄录自 arXiv 2507.12857 Table 1，ICCV 2025）

mAP（COCO 实例 segm 口径，类平均，×100）。训练 setting 各两行组。

**iSAID 训练 → 测其余四集**（列：SIOR / NWPU / SOTA / FAST / Average）：

| 方法 | SIOR | NWPU | SOTA | FAST | Avg |
|---|---|---|---|---|---|
| ODISE (CVPR22) | 36.40 | 13.91 | 4.65 | 13.68 | 17.16 |
| FC-CLIP (NeurIPS23) | 60.67 | 33.62 | 11.88 | 26.79 | 33.24 |
| MAFT+ (ECCV24) | 35.32 | 6.63 | 5.52 | 9.84 | 14.33 |
| ZoRI (AAAI25) | 62.06 | 30.02 | 12.65 | 26.27 | 32.75 |
| **SCORE (Ours)** | 67.59 | 42.57 | 13.67 | 30.90 | 38.68 |

**SIOR 训练 → 测其余四集**（列：NWPU / SOTA / FAST / iSAID / Average）：

| 方法 | NWPU | SOTA | FAST | iSAID | Avg |
|---|---|---|---|---|---|
| ODISE | 41.77 | 12.30 | 5.88 | 12.68 | 18.16 |
| FC-CLIP | 60.69 | 19.84 | 8.67 | 22.24 | 27.86 |
| MAFT+ | 39.99 | 7.77 | 5.92 | 9.63 | 15.83 |
| ZoRI | 59.77 | 20.26 | 9.58 | 23.46 | 28.27 |
| **SCORE (Ours)** | 69.17 | 23.68 | 10.33 | 27.15 | 32.59 |

自洽性：两块 Average 列均可由四集数字复算（±0.01 舍入）。
**已发表最弱基线（MAFT+，G1 阈值参照）**：NWPU 6.63 / SOTA 5.52 /
FAST 9.84（两 setting 取小）。

## 2. Gate 指标与阈值（登记后不回改）

### G1 类平均实例召回 proxy @IoU0.5 —— 主判据

操作定义：

- **候选池**：SAM3 text-only 在每 (图, 类) 上输出的全部候选中，取
  `score ≥ 0.05` 者（score = sigmoid(logit)×sigmoid(presence)，即 processor
  输出分；0.05 为排序表低尾阈值，与 COCO maxDets=100/类/图上限一并固定）。
- **覆盖判定**：GT 实例与任一同类候选 IoU ≥ 0.5（一对多允许：一个 GT 被多个
  候选覆盖仍计 1；一个候选可覆盖多个 GT）。
- **汇总口径**：三测试集 = NWPU / SOTA / FAST。对三集类表并集中每个类，合并
  该类在三集的全部 GT 实例计算覆盖率；对类取平均（类数 ≈ 65）。micro 变体
  （全 GT 合并算总比例）一并报告但**不作判据**。
- **阈值**（登记）：**G1 ≥ 0.40 → GO**；**0.20 ≤ G1 < 0.40 → 条件 GO**（插入
  P0.5 域适配，只恢复召回，不追域内 SOTA）；**G1 < 0.20 → NO-GO**（回设计层
  评估基座，触发 design-v3 讨论）。

### G2 SCORE 口径 mAP —— 记录用，无硬阈

每测试集 COCO segm AP@[.50:.05:.95]（口径见 sam3_ovis/eval/README.md 预登
记）+ 跨集 Average；与 §1 基线同表汇报（SAM3 零样本是"无 RS 训练"的对照行，
不与闭集训练方法直接比排名）。

### G3 推理可行性 —— 实测登记

单卡显存峰值与 1024² / 2048² 单图时延（整图 / 分块 / 滑窗三模式，S2.3）。
设计文档参照 48G 单卡；**本机为 RTX 3090 24G，如实按 24G 实测**。可行性判据：
基线模式在 24G 内可完成且单图-单类时延支持三测试集全量在一周内跑完（算力
预算核对），超限则登记分块方案与影响。

## 3. 运行约定（影响 G1/G2 数字的可复现性，一并登记）

- 类别短语 = 数据注册表类别名**逐字使用**（含 `Small_Vehicle` 下划线、
  `expressway-service-area` 连字符，token 实测见 sam3-api-notes (c)）。
- 模型：`facebook/sam3` `sam3.pt`（sha256 见 scripts/README.md）；
  `Sam3Processor(resolution=1008, confidence_threshold=0.05)`；全局 bf16
  autocast（官方用法）。
- 图像预处理：processor 内部 resize 到 1008²（整图 resize 模式 = P0 基线；
  分块/滑窗属 S2.3 对照，不进主基线表）。
- GPU：RTX 3090（`CUDA_VISIBLE_DEVICES=0`），bf16。
- 输出：COCO results json（RLE）+ 逐类/逐图候选统计，按数据集落
  `$SAM3_OVIS_LOG_ROOT/zeroshot/<DATASET>/`。
