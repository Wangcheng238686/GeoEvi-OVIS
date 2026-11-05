# G1 Q-bridge 类别路径探针：协议登记（结果产生前冻结）

> 日期：2026-11-05。对应 implementation-preparation §4（第一技术关口）与 §6 G1。
> 脚本：`sam3_ovis/graph/g1_qbridge_probe.py`。目标：在完整预训练 SAM3 上回答
> **"原生记忆读取的增量注入 detector query 后，概念分数是否发生非零、
> 非纯尺度、且对上下文内容敏感的变化"**。这是类别路径的连通性/敏感性诊断，
> 不声称修复漏检，不选任何最终架构参数。

## 1. 信息路径（全部原生模块，零 vendored 代码修改）

1. 首遍（推理态）：Sam3Processor 15 类 text-only（pool floor 0.05）→ 预测
   实例（box/mask/score/类）——真实预测建库流程，GT 不进入。
2. 画布与记忆（可微，fp32，直接调用裸方法绕过 inference_mode）：
   - query 画布：`model.backbone.forward_image` 的 `backbone_fpn` 中
     stride-16 级 `[HW,1,256]`（1008 输入 → 63×63=3,969 token）；
   - 记录 writer：`inst_interactive_predictor.model.maskmem_backbone`
     （即 tracker 权重）：输入画布特征 BCHW + 实例硬 mask 的 ±10 logits
     （writer 内部 sigmoid×20−10 分布契约），下采样到 63×63 网格；
   - reader：`inst_interactive_predictor.model.transformer.encoder`，
     src=画布、prompt=K 条记录栅格展平 `[K·HW,1,64]`、feat_sizes=[(63,63)]、
     num_obj_ptr_tokens=0（与契约回归测试同调用形）；
   - **Δ画布 = reader(真记忆) − reader(零记忆基线)**。零基线 = 同形状全零
     prompt（reader 拒绝非空 padding mask，无法真"空库"；登记此选择：
     "空库"以零向量记忆近似，非注意力缺席）。
3. 注入（Q-bridge，诊断用固定 mask）：
   - 对每个目标实例 j：`δ_j = maskpool(Δ画布, mask_j@63×63)` → bridge
     `Linear(256→256)`（**非零初始化**：正交初始化，避免零梯度假象）；
   - `q'_j = queries_j + bridge(δ_j)`；用原生 `model.dot_prod_scoring`
     （hs=单层堆叠 queries）重算分数——登记：外部重算绕开
     `supervise_joint_box_scores` 的 presence 耦合，Δ在裸点积分上度量；
     presence 耦合对 Δ的调制另作敏感性记录。
   - 空库等价性：`δ=0` 时分数必须逐位复原（fp32 bit-exact 校验）。

## 2. 登记的度量与判读标准

| # | 度量 | 计算 | "路径有效"的判读 |
|---|---|---|---|
| M1 | 分数变化幅度 | 同一 query 15 文本的 Δs 分布（fp32） | 中位 |Δs| > 1e-3（远超 fp32 噪声 ~1e-6） |
| M2 | 非纯尺度 | 同一 query 跨文本 Δs 的离散度 + 15 类排名翻转率 | 跨文本 Δs 变异系数 > 0.1 且存在排名翻转 |
| M3 | 上下文敏感性 | 三种 bank：本图实例(A) / 异图实例(B) / 零(C)；Δs 模式可分性 | A 与 B 的 Δs 分布可分（AUC>0.7），A 与 C 幅度显著不同 |
| M4 | 梯度连通 | sum(scores) 反传：bridge/reader/writer/主干的梯度范数 | 四处非零（writer 假输入为 detach 画布时允许只到 bridge/reader，分开设登记） |
| M5 | N/K 成本 | K∈{1,2,4,8}：reader 峰值显存与时延 | 曲线登记（无门槛，供 G2 预算） |
| M6 | 空库等价 | δ=0 复原校验 | bit-exact |

M1/M2 为**门槛判读**（不通过=类别路径断路，Q-bridge 方案需回查）；
M3 为方向性证据（弱通过可接受但须登记）；M4/M6 为工程验收。

## 3. 样本与确定性

- 图像：`g0_splits/manifest.json` dev 组前 12 张**有目标**切片（名字排序扫描
  前 60 张候选，seed 20261105）。
  **修订（2026-11-05，结果产生前）**：原登记目标门槛 score≥0.5 经烟雾测试
  证实不切实际——iSAID 文本提示下首遍 max 分数普遍 0.1–0.5（P0001 组实测
  0.14/0.47/0.82），0.5 门槛导致 0 目标。改为 **score≥0.3**（目标与 bank 同
  门槛），跨文本 IoU≥0.5 匹配仍要求。此修订先于任何 M 统计产出。
- 上下文 bank = 同图其余预测实例按分数降序取 K=2（主统计）；异图 bank =
  上一处理图的同构 bank（首图无 M3 异图对照）。
- 精度 fp32（统计段）；M5 成本段允许 bf16 对照并标注。全部张量运算
  固定 seed；脚本落盘逐实例记录 JSON。

## 4. 已知边界（如实带入结论）

- writer/reader 预训练于 tracker neck 分布，本探针喂 detector FPN 特征
  （neck 不匹配，interface 报告已登记）——M1-M3 是"当前可用的原生路径"
  的敏感性，不等于训练后可达幅度；
- 零记忆基线 ≠ 注意力缺席；Δ 的解释限定在"与零向量记忆的差异"；
- 池化 mask 为首遍预测（非 GT），含预测误差——与真实建库流程一致。
