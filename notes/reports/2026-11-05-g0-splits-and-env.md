# G0 交付：源图级划分、环境依赖与接口契约回归（2026-11-05）

> 对应 implementation-preparation §6 G0（数据/环境/契约）。批准范围：#1 commit
> 资产、#2 split+novel 隔离、#3 依赖+契约。novel 类隔离核查另行出报告
> （`2026-11-05-g0-novel-class-isolation.md`，子任务产出）。未做：任何 GPU
> 作业、模型代码修改、训练。

## 1. 资产固化（#1）

10-28 全部准备资产入库：commit `d166e25`（decisions-2026-10 + 7 份就绪报告
+ README/explainer 同步），工作树恢复干净。

## 2. 源图级划分（#2）

工具 `sam3_ovis/data/g0_splits.py`（协议在 docstring 先于运行冻结），产物
`sam3_ovis/logs/g0_splits/manifest.json`（seed 20261105，幂等校验通过——
重跑逐组一致，不一致即拒绝写盘）：

| split | 源图组 | 切片图 |
|---|---|---|
| train | 1,024 | 10,778 |
| dev | 60 | 606 |
| confirm | 60 | 574 |

- 暴露登记：267 个历史已暴露组（T2 dev 15 + T1 探针 242 + v3 冻结 10）
  全部隔离，不进 dev/confirm；v4-S0 C0 的 DOTA val 100 图名与 train 池
  零碰撞（逐名核对，记录在 manifest）。
- 纪律：confirm 组只用于最终冻结确认；本 manifest 落盘后不得改历史划分。

## 3. 环境与契约（#3）

- **依赖**：hydra-core 1.3.8 / omegaconf 2.3.1 / submitit 1.5.4（+pytest）
  已装入 sam3 env（aliyun 镜像）；官方训练入口 `sam3/train/train.py`
  import 检查通过（SlurmEvent/compose/instantiate 等可见）。
- **接口契约回归**：`sam3_ovis/tests/test_sam3_memory_interface.py`
  （移植自 10-28 临时探针，seed 固定）——reader 形状/三梯度、非方形 RoPE
  与 padding mask 双拒绝、writer 形状/三梯度，4/4 通过；全测试套件
  36/36 无回归。移植中修正一处我的错误假设并固化为契约：**writer 的
  pos enc 返回 BCHW 网格 `[B,64,h,w]`，展平为 reader 的 `[HW,B,64]` 是
  tracker 包装层职责**——两种布局不得混用（interface 报告 §3.1 的坑）。
- **checkpoint 键审计**：`sam3.pt` 共 1,465 键，顶层 `detector.*`(1,156)
  /`tracker.*`(309)。writer 权重在 `tracker.maskmem_backbone`（40 键）、
  reader 在 `tracker.transformer`（106 键）；**无独立 memory_attention /
  memory_encoder 键**，与接口报告"reader=transformer.encoder"结论一致。
  detector 侧 vision/language backbone + geometry encoder 键齐全。

## 4. G0 剩余与 G1 入口

- G0 剩余：novel 类隔离报告（子任务运行中）；训练入口的 dry-run 级
  config 解析（hydra compose）留待需要时做。
- G1（下一步，需另行确认后执行）：完整预训练模型上的 Q-bridge 类别路径
  探针——首遍候选建库 → 原生 reader → mask 池化增量投影 detector query →
  重算概念分数；验证分数变化非零且非纯尺度效应、错误上下文可分辨、
  N/K 显存延迟实测。GPU 现状（每卡 ~17G 空余）满足单图探针。
