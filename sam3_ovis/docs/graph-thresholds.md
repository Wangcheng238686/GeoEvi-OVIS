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
