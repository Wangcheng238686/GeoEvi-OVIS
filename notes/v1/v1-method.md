# 方案 v1：双层生命周期图结构记忆用于大幅遥感影像开放词汇实例分割

> 状态：v1 草案（2026-09-01），待用户审阅迭代。基座未定（SAM3 基线实验进行中），
> 本文按"基座无关"撰写；方法名待定。

## 0. 一句话

继 SCORE（ICCV 2025）之后的 OV-RSIS：将 SAM2 式记忆机制从时序改造为**双层生命周期
异构图**——滑窗切片产生的实例/上下文为临时层，跨图持久的类原型/文本嵌入为持久层，
写入/固化/淘汰/检索四算子在图上统一完成，MLLM 精炼持久层语义，不确定性门控驱动
全部生命周期决策。

## 1. 任务设定

- 输入：大幅遥感影像 I（256–4000px+，滑窗切片 W={w_1..w_T}，重叠步进）+ 开放类别名集合
- 输出：全图所有实例 {(mask_i, class_i)}
- 协议：SCORE benchmark（闭集训练 iSAID/SIOR → 开集跨数据集测试）+ 三层扩展
  （多目标 AP 化 / 不确定性校准 / 大图滑窗效率）

## 2. 记忆图形式化

G = (V_temp ∪ V_persist, E, A)

**临时层 V_temp**（单图内，处理完即弃）：
- tile 节点 v_t：窗口特征、全局坐标、重叠拓扑
- 实例节点 v_i：mask embedding、几何（中心/面积/外接框）、类别后验、生命周期状态
  （活跃/冻结/消亡，LOMM occupancy 式）

**持久层 V_persist**（跨图长存）：
- 类原型节点 v_c：CLIP 文本嵌入**初始化**（免训练起步）+ 视觉原型 EMA 动量更新；
  附 MLLM 精炼的语义描述属性（贡献 2 挂点）
- 不确定性属性 A：节点置信度 s = presence × IoU × 多假设分歧（贡献 3 的信号源）

**边 E（类型化）**：
- 空间边：tile↔tile（邻接/重叠权重）、实例↔tile（包含）、实例↔实例（邻近；
  同类高相似时升级为**对抗边**，DAM4SAM 式 distractor 关系）
- 语义边：实例↔原型（匹配度）、原型↔原型（本体层级，可挂 SA-Co/Wikidata）

## 3. 四算子（图上统一，跨两层）

| 算子 | 定义 | 信号/模板来源 |
|---|---|---|
| 写入 write | 实例节点入图 iff s_i > τ_w | presence×多假设分歧门控（SAMURAI/DAM 内省式写入的图化） |
| 固化 consolidate | 高置信实例记忆 EMA 升华为类原型（临时→持久的桥） | XMem potentiation 的空间版+语义版 |
| 淘汰 evict | 容量约束下低置信/被合并节点衰减；冻结节点可复活 | RMem UCB + LOMM 状态机 |
| 检索 retrieve | 当前查询子图路由（原型锚定 k-hop）替代扁平 attention | SAMIX 学习式筛选的拓扑化 |

**红线自查（GMN 教训）**：节点=实例/原型级非帧级；边类型化非全连接；四算子显式可消融；
须给出"图增益 vs 记忆条数"曲线。

## 4. 读出（memory-as-prompt）

检索子图 → 聚合为 prompt tokens（类原型 token + 邻域实例 token + tile 上下文 token）→
注入解码头（C5-v2 prompt 接口 / SAM2 memory attention 适配层，随基座定）。

## 5. 贡献 2：MLLM 语义记忆（持久层）

- 免训练版：CLIP 文本嵌入直接做原型（基线配置）
- 完整版：7B Qwen-VL LoRA（遥感域微调）做 ① 原型语义描述精炼→匹配阈值/边权
  ② 低置信合并仲裁（两原型是否同类）③ SAM3 Agent 式工具循环备选
- 消融：MLLM on/off

## 6. 贡献 3：不确定性门控

- 信号：presence、IoU、多假设分歧（基座现成输出）、原型匹配熵
- 用途：写入阈值、边权、检索范围、固化资格
- 评测：ECE / risk-coverage 曲线（协议扩展，无人做过）

## 7. 训练策略（基座定后细化）

- 冻结 backbone + 文本塔；训练：图算子模块 + 读出适配层（+ MLLM LoRA）
- 数据：OVRSIS95K（主）+ iSAID/SIOR（SCORE 协议对齐）；WHU 线衔接建筑类先验
- 算力核对：冻结主干+<30M 可训模块（SAMIX 4×4090 先例）+ 7B LoRA，4×48G 内

## 8. 实验设计

1. **主表**：SCORE 协议，vs SCORE / FC-CLIP / ZoRI / ODISE / MAFT+
2. **核心消融**：图 vs OmniSAM 式链式 FIFO；**持久层 on/off（预期 novel 类掉点=灵魂消融）**；
   固化 on/off；子图检索 vs 扁平 attention；MLLM on/off；不确定性门控 on/off
3. **图增益 vs 记忆条数**曲线（防 GMN 式"装饰品图"质疑）
4. **大图**：OVRSISBenchV2 分辨率分组 + 滑窗效率协议
5. **多目标**：NWPU-Refer category 子集 + LaSeRS instance 子集 AP 化
6. **校准**：ECE / risk-coverage
7. SAM3 零样本对照行（基座决定依据，无论结果进不进对比表都有信息量）

## 9. Related work 划界（一句话版）

- vs OmniSAM：链式 FIFO→异构图+生命周期；其 regime 为基线
- vs SAMIX：参考池+RL 链→双层图+拓扑检索；其免训练筛选结论为论据
- vs SAM2Long：假设分支树→记忆关系图（DAM4SAM 已证树不解决 distractor）
- vs GMN：帧级全连接小图→异构双层+显式生命周期算子
- vs XMem：时序三层单向流→空间双层+语义持久层+统一算子
- vs SCORE：即席上下文向量→持久图记忆+MLLM+不确定性；其协议为本工作 benchmark

## 10. 遗留开放点

1. 基座（等 SAM3 基线）
2. 持久层容量与跨数据集部署形态（单数据集原型库 vs 全局库）
3. MLLM 三个用途是否全做（工作量 vs 完整性）
4. 方法名
5. 对抗边的判定阈值设计（DAM 面积比法的单帧适配）
