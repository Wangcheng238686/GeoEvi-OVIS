# graph/ — 同图实例—上下文证据图

design-v2.md §5 的实例化模块。**S3.2 已完成**（2026-09-03）：六类边的可计算
代理全部实现并通过逐边单测；阈值与权重登记见
[`docs/graph-thresholds.md`](../docs/graph-thresholds.md)（登记后不回改）。

## 模块

| 文件 | 内容 | 状态 |
|---|---|---|
| `schema.py` | 节点/边类型（§5.1/§5.2 一一对应）、JSON 序列化、权重域守卫 | ✅ S3.1 |
| `builder.py` | 结构层（类别/场景/unknown/候选/环带节点 + 覆盖属性）+ `build_full_graph`（六类边集成，接受 provider/prior/竞争类共享数据） | ✅ S3.1+S3.2 |
| `config.py` | 登记阈值唯一代码来源（`GraphThresholds` frozen dataclass） | ✅ S3.2 |
| `provider.py` | `RemoteCLIPProvider`（open_clip ViT-L-14，缓存批处理）+ `MockProvider`（测试） | ✅ S3.2 |
| `edges.py` | 文本—实例 / 同类支持（外观+尺度臂取 max）/ 相似类竞争 / 环带上下文 / 场景（覆盖转属性）/ unknown 冲突 | ✅ S3.2 |
| `priors.py` | iSAID+SIOR train 类别共现先验表（拉普拉斯平滑）→ `$SAM3_OVIS_DATA_ROOT/priors/` | ✅ S3.2 |
| `completeness.py` | 完整性三指标 + 图级不完整风险分（公式见下） | ✅ S3.3 |

## 完整性指标（S3.3，登记于 docs/graph-thresholds.md §10）

作用域：**登记候选池 = 原始 score ≥ 0.05**（P0 注册池，`compute_completeness`
内显式过滤；rival 节点剔除）。`τ_high = 0.5` 在 norm_text 尺度（复用 S3.2
`context_conf_floor`）；重复/匹配 IoU = 0.5。

```text
ECR 已解释区域覆盖率 = |∪ mask(高置信 c : norm_text(c) ≥ τ_high)|
                     / |∪ mask(池内全部候选)|        （面积基；空池定义为 1.0）
URD 未解释响应密度   = |{池内低置信候选 : 与所有高置信候选 IoU < 0.5}|
                     / |池内全部候选|                （计数基）
CCR 冲突候选率       = |{池内候选 : 触 COMPETING_CLASS 边 ∨ 同类更高分 IoU ≥ 0.5}|
                     / |池内全部候选|
R   不完整风险分     = ⅓·(1−ECR) + ⅓·URD + ⅓·CCR      （等权手定，P1 不调参）
```

用途与盲区：R 只调制最终置信度与告警（S3.4 消费），不生成/删除实例；
ECR 面积基与 URD 计数基互补；**已知盲区**：整图零响应时 ECR=1、R=0
（完整性不感知完全漏检——这正是它不能替代 G1 召回的原因）。

## 单测

```bash
$PYTHON sam3_ovis/tests/test_graph_schema.py      # 结构 + 序列化
$PYTHON sam3_ovis/tests/test_graph_edges.py       # 六类边方向与权重域 + full-graph 集成（S3.2 判据）
$PYTHON sam3_ovis/tests/test_completeness.py      # ECR/URD/CCR/风险分 + 池过滤（S3.3 判据）
```

## 下一步（S3.4/S3.5）

- S3.4 两轮消息传递打分 → 候选可靠性分 → 分数加权合并（NMS 升级）+
  端到端单图 demo（消费 S3.2 边与 S3.3 风险分）；
- S3.5 消融矩阵 A0–A3（P1 出口判据：A2−A0 ≥ +2 mAP 且 A2−A1 ≥ +1 mAP）。

红线提示：上下文/场景只作软分数（config 内全部权重域 [0,1]，无硬规则）；
unknown 只聚合冲突证据，不生成实例。
