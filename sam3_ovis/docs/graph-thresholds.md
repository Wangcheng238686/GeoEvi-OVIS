# 证据图 v0 阈值与权重登记（S3.2）

> 登记日期：2026-09-03（P1-S3.2 实现开始时）。**本文件登记后不回改**
>（implementation-v2 S3.2 纪律）；若 P1 消融证明某阈值失当，走 S3.5 消融
> 记录与设计迭代流程，不在本文件上修改历史值。
> 代码侧唯一来源：`sam3_ovis/graph/config.py`（值与本文件一一对应）。
> 所有权重域 ∈ [0,1]；上下文/场景只作软分数（红线 5：禁硬规则）。

## 1. 分数归一

| 记号 | 定义 |
|---|---|
| `norm_text` | SAM3 候选分 score ∈ [0.05,1] 线性归一到 [0,1]：`(score-0.05)/0.95` |
| `cos01(c)` | 余弦 ∈ [-1,1] 线性重标 `(c+1)/2` |

## 2. 文本—实例边（TEXT_INSTANCE）

`weight = W_TEXT_SAM3 · norm_text + W_TEXT_CLIP · cos01(clip_cos(crop, 类名))`

| 参数 | 登记值 |
|---|---|
| `W_TEXT_SAM3` | **0.5** |
| `W_TEXT_CLIP` | **0.5** |

## 3. 同类支持边（SAME_CLASS_SUPPORT）

边存在条件：外观臂或尺度臂任一触发；`weight = max(两臂分数)`。

| 参数 | 登记值 | 定义 |
|---|---|---|
| `THETA_S` | **0.70** | 外观臂触发：cos(crop_i, crop_j) ≥ θ_s；臂分 = cos01(cos) |
| `SCALE_K` | **3.0** | 尺度臂：面积比 ratio < K 触发；臂分 = `1 - ratio/K` |

## 4. 相似类竞争边（COMPETING_CLASS）

| 参数 | 登记值 | 定义 |
|---|---|---|
| `COMPETE_TOPK` | **5** | 竞争类 = 类名 CLIP 文本相似 top-K（除自身） |
| `THETA_C` | **0.50** | 且文本相似 cos ≥ θ_c 才算竞争类 |
| `THETA_O` | **0.30** | 跨类候选对 IoU ≥ θ_o 记竞争边 |

`weight = IoU`（截断 [0,1]）。

## 5. 实例—局部上下文边（INSTANCE_CONTEXT）

环带 = 候选 mask 膨胀 r 像素后扣除自身；环带内取 norm_text ≥ 0.5 的高置信
候选作为上下文证据。

| 参数 | 登记值 | 定义 |
|---|---|---|
| `BAND_RATIO` | **0.25** | 膨胀半径 r = `BAND_RATIO · sqrt(area_px)` |
| `PRIOR_ALPHA` | **1.0** | 共现先验拉普拉斯平滑 |
| `PRIOR_FLOOR` | **0.01** | 先验权重下限（避免 0 权边） |
| 先验表 | iSAID-train + SIOR-train 统计 | `P(j|i)` = 同图共现频率，按数据集词表分表；类名无匹配时权重 = `PRIOR_FLOOR` 并标记 `prior=unmatched` |

`weight = max(环带内各证据类的平滑先验 P(j|i))`。先验只作分数缩放。

## 6. 实例—全局场景边（INSTANCE_SCENE）与场景节点

| 参数 | 登记值 |
|---|---|
| 场景短语表（12） | airport, harbor, parking lot, residential area, industrial area, farmland, forest, river, city, port, desert, overpass |
| 场景分 | `cos01(cos(图像CLIP, 场景短语))`，存场景节点 payload（top-3） |
| 边权 | `cos01(cos(crop CLIP, 图像CLIP))`（实例对全局场景的代表性） |

场景只作分数，不产生任何实例（红线 5 / §12.4 留白收敛：场景语义来源 =
RemoteCLIP 文本校验，与 design-v2 §4 边界一致）。

## 7. unknown 冲突边（UNKNOWN_CONFLICT）

触发条件（同时满足）：

| 参数 | 登记值 |
|---|---|
| `THETA_U_TEXT` | **0.20**：`norm_text < 0.20`（文本臂低） |
| 竞争落败 | 存在跨类竞争候选（§4 定义）其 `norm_text` 更高 |

`weight = 1 - norm_text`。unknown 节点只聚合冲突证据，不生成实例（红线）。

## 8. 区域覆盖（INSTANCE_SCENE 的 attrs）

沿 S3.1 builder：候选并集 coverage RLE 存图属性，`area_share` 存边属性；
权重臂已在 §6 定义，不重复计边。

## 9. 复现入口

- 配置：`sam3_ovis/graph/config.py`（dataclass，逐项注释指向本文条目）
- 先验表构建：`sam3_ovis/graph/priors.py`（iSAID/SIOR train → JSON）
- 边实现：`sam3_ovis/graph/edges.py`；集成：`graph/builder.py::build_full_graph`
- 单测：`sam3_ovis/tests/test_graph_edges.py`（mock embedding，逐边验证
  方向与权重域——S3.2 完成判据）

## 10. 完整性指标登记（S3.3，2026-09-03；§12.4 留白收敛）

作用域：**登记候选池 = 原始 score ≥ 0.05**（P0 注册池；在
`compute_completeness` 内显式过滤，不依赖上游；rival 节点剔除）。
高置信地板 `τ_high = 0.5` 在 **norm_text 尺度**（复用 S3.2 已登记
`context_conf_floor`，不新增常数）；重复/匹配 IoU 阈值 = 0.5（沿用评测
惯例）。两套尺度并存：池阈值在原始分尺度、置信地板在归一分尺度。
实现：`graph/completeness.py`；配置：`config.py::COMPLETENESS`。

| 指标 | 公式 | 值域 |
|---|---|---|
| 已解释区域覆盖率 ECR | `|∪ mask(高置信 c : norm_text(c) ≥ τ_high)| / |∪ mask(池内全部候选)|`；池空时定义为 1.0（盲区见下） | [0,1] |
| 未解释响应密度 URD | 计数基：`|{池内低置信候选 : 与所有高置信候选 IoU < 0.5}| / |池|`；池空时 0.0 | [0,1] |
| 冲突候选率 CCR | `|{池内候选 : 参与 COMPETING_CLASS 边 ∨ 存在同类更高分候选 IoU≥0.5}| / |池|` | [0,1] |
| **不完整风险分 R** | `⅓·(1−ECR) + ⅓·URD + ⅓·CCR`（等权手定，P1 不调参） | [0,1] |

用途与红线：R 只**调制最终置信度与告警输出**（S3.4），不生成/删除实例；
ECR 面积基与 URD 计数基互补（面积上被高置信覆盖、计数上仍有成片低分回声
是两类不同症状）。已知盲区：查询类"整图零响应"时 ECR=1、R=0——完整性
不感知完全漏检，这正是它不能替代 G1 召回的原因，论文如实声明。

## 11. 消息传递与合并算子登记（S3.4，2026-09-03）

两轮消息传递（手定权重，P1 不学习）。候选先验 `rel_0 = norm_text(score)`；
每轮聚合四路消息后更新，clip 到 [0,1]：

```text
support_msg(c)   = max over SAME_CLASS_SUPPORT 邻 (w·rel(邻))
compete_msg(c)   = max over COMPETING_CLASS 邻   (w·rel(邻))
context_msg(c)   = INSTANCE_CONTEXT 证据边 w（每候选取 max）
scene_msg(c)     = INSTANCE_SCENE 边 w
rel' = clip(rel + α_s·support·(1−rel) − α_c·compete·rel
            + α_x·context·(1−rel) + α_p·scene·(1−rel))
```

| 参数 | 登记值 |
|---|---|
| 轮数 `ROUNDS` | **2** |
| 支持增益 `α_s` | **0.15** |
| 竞争惩罚 `α_c` | **0.30** |
| 上下文增益 `α_x` | **0.10** |
| 场景增益 `α_p` | **0.05** |
| 风险调制 `LAMBDA_RISK` | **0.5**：`final(c) = rel(c) · (1 − λ·R)`——R 高的图整体降权（跨图排序受影响，图内排序不变） |
| 告警阈值 `THETA_ALARM` | **0.5**：R ≥ 阈值在输出中标记"预测可能不完整" |
| 合并簇 | SAME_CLASS_SUPPORT 边的连通分量，**且要求成对 mask IoU ≥ 0.5**（纯尺度相似不合并） |
| 簇合并分数 | `merged = 1 − Π(1 − rel_i)`（概率或：重复候选互证增强，区别于标准 NMS 的丢弃） |
| 簇代表 mask | 簇内 rel 最高的成员（不做像素平均，避免非同一目标的模糊） |

## 12. 消融矩阵协议登记（S3.5，2026-09-03）

数据：三测试集（NWPU/SOTA/FAST）的 P0 登记候选池（原始 score ≥ 0.05，
predictions.jsonl），**不重跑 SAM3**；四变体共享同一池。

| 变体 | 定义 |
|---|---|
| A0 裸 SAM3 | 池内 score ≥ **0.5**（官方默认置信阈值）+ 标准贪心 mask NMS（类内，score 降序，IoU≥0.5 抑制） |
| A1 无边对照 | 全池（≥0.05）+ `norm_text` 独立打分（无任何边/消息传递/图合并）+ 同款标准 NMS |
| A2 完整图 | 全池 + §2–§7 全部边 + §11 消息传递/合并/风险调制 |
| A3 组件递减 | A2 分别去掉：−竞争（α_c=0 且不建 COMPETING/unknown 边）、−上下文（α_x=0 且不建上下文证据）、−场景（α_p=0）、−风险调制（λ=0） |

工程登记：跨类 IoU 以 bbox 预筛；竞争/上下文的 rival 候选按类取 score
top-10；候选 mask 每图只解码一次、跨类共享；CLIP crop 特征每图一批、
跨类与 A2/A3 共享。指标：各变体 mAP / AP50 / G1 类均召回；**显著性**：
A2−A0 与 A2−A1 的差配对 bootstrap（1,000 次重采样，95% CI，seed=20260903；
配对统计量 = **每图 mAP@[.50:.95]**，图内跨类 micro、全点 PR，**有 GT 无预测
的图计 0**——修正记录 2026-09-04：初版登记为 IoU0.5 每图 AP 且漏计空预测图，
在首个消融结果被判定无效（CLIP 特征跨图串用等实现缺陷）后、任何有效结果
报告之前更正。COCO 全量 mAP 差值作为点估计一并报告）。出口判据（预登记不改）：
**A2−A0 ≥ +2 mAP 且 A2−A1 ≥ +1 mAP → 进 P2**；A2 ≈ A1 → R2 兑现，停线回
设计层。
