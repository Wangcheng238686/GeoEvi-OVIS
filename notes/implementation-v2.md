# 实施文档 v2.0（对应 design-v2.md）

> 状态：与 `design-v2.md` §1–§11 对齐的分阶段实施计划；风险编号 R*/E* 引用
> `design-v2.md` §12。
> 面向独立执行者（实施代理），自包含，不依赖对话上下文；与设计文档冲突时以
> 设计文档为准，并回报触发设计迭代（按 `notes/README.md` 规则新增版本文件）。
> 版本对应规则：每版 `design-vX.Y.md` 对应 `implementation-vX.Y.md`；设计迭代后
> 实施文档按新版本重建，不跨版本续用。v1 为历史设计，无实施文档。

## 0. 阶段总览与全局约定

### 0.1 阶段表

| 阶段 | 名称 | 估计 | 前置 | 出口 | 关联风险 |
|---|---|---|---|---|---|
| P-1 | 工程与数据准备 | ~1 周 | 无 | M0：环境、数据、评测管线、仓库骨架就绪 | E1–E6 |
| P0 | SAM3 text-only 零样本基线 | 1–2 周 | M0 | M1：Go/No-Go 判定 | R1、E2、E3 |
| P0.5 | 域适配（条件触发） | 2–4 周 | M1=条件 GO | 召回恢复后复测 G1 | R1 |
| P1 | 证据图 v0（training-free） | 3–6 周 | M1=GO | M2：消融 A0–A3 结论 | R2、R3、§12.4 留白 |
| P2 | SCORE 完整协议 + 可学习化 | 4–8 周 | M2 达标 | M3：协议主表 | R4、R6、E4、E5 |
| P3 | exemplar 扩展 | 3–4 周 | M2 达标（可与 P2 并行排布） | M4：增益拆分 + 贡献链消融齐 | R5、红线 4 |

各阶段允许并行（如 P-1 数据下载与 SAM3 接入并行；P3 依赖 P1 的图但不必等 P2）。
P0.5 只在 M1=条件 GO 时插入，范围限定为恢复召回，不追求域内 SOTA。

### 0.2 工程全局约定

1. **新主线位置**：仓库根新增 `sam3_ovis/` 主线包（与 `geoevi_ovis/` C5-v2 线
   平级），内设 `configs/ data/ graph/ eval/ inference/ tests/ scripts/ docs/`；
   vendored `sam3/`（官方 repo）与 `sam2/` 平级。理由：与冻结的 C5-v2 线物理
   隔离，避免 import 面与历史包袱（design-v2.md §12.1）。
2. **C5-v2 冻结规则**：`geoevi_ovis/` 只读。需要其工具（如 COCO 评测）时以
   新增文件的方式在 `sam3_ovis/` 内扩展或薄封装，不修改原文件。
3. **契约移植**（从 README 四条核心工程契约整体继承）：
   - 自描述 checkpoint（schema v2：`model_config` + 超参 + 数据协议 + 架构指纹）；
   - 影响模型行为的参数一律进 `cfg.model`，禁止只活环境变量；
   - 机器路径集中 `sam3_ovis/configs/environment*.sh`，新增
     `SAM3_REPO` / `SAM3_CKPT` 变量，机制照抄 `geoevi_ovis/configs/`；
   - 文档同步：`sam3_ovis/` 内部 README 逐目录维护；仓库根
     `PROJECT_ARCHITECTURE.md` 只追加指针段落，不重写。
4. **环境**：SAM3 依赖可能要求独立 conda env（torch/hydra 版本待实测）；
   `sam3_ovis/scripts/` 全部入口带 env 检查，不假设与 C5-v2 环境共存。
5. **报告制度**：每阶段结束产出 `notes/reports/pX-report.md`，固定四节：
   数字表 / 判据核对（对照预登记阈值）/ 失败案例分类 / 决定与下一步。
   gate 触发设计变化时，停下并按迭代规则开新设计版本，不边跑边改设计。

## 1. P-1 工程与数据准备

**S1.1 SAM3 环境与权重**
安装 vendored `sam3/`（独立 env），下载官方 checkpoint；复核 SAM License 条款
并登记结论（研究使用边界、再分发限制）到 `sam3_ovis/docs/license-note.md`。
输入：官方 repo + 权重链接；产出：可 import 的 env + 权重路径进
`environment*.sh`；完成判据：官方 demo（单图 text prompt）在本机 GPU 跑通。

**S1.2 SAM3 API 面验证**
用脚本验证四件事并落档 `sam3_ovis/docs/sam3-api-notes.md`：
(a) 概念分割输出契约（mask、实例分数、ID 字段与 dtype）；
(b) visual exemplar 输入契约（box/掩码、正负标签、ROI 接口）；
(c) 文本 32-token 上限对遥感类名的实测：SCORE 训练类（iSAID 15 / SIOR 20）+
OVRSISBenchV2 128 类名逐个过文本编码器，记录截断名单（E3）；
(d) 多短语单图行为（为竞争类联合推理做准备）。
完成判据：(a)–(d) 各有可复跑脚本与结论段落。

**S1.3 数据获取与登记**
下载并对齐五个数据集 + 一个补充集，统一转 COCO-instance 格式落到
`environment*.sh` 登记的数据根：
- 训练：iSAID（15 类）、SIOR-DIOR（20 类）；
- 测试：NWPU-VHR-10、SOTA、FAST（split 以 SCORE HF `HuangShiqi008/SCORE`
  为准，协议引用注明来源，E4）；
- 补充：NWPU-Refer category 子集（"all X" 表达式 → 实例 mask；连通域分解的
  粘连风险在报告标注）。
逐测试集登记 GT 来源（人工 / SAM 伪标签，E5），写入
`sam3_ovis/docs/gt-provenance.md`。
完成判据：每数据集 loader 冒烟通过（图数、实例数、类别表与官方一致）。

**S1.4 SCORE 评测管线**
基于 `geoevi_ovis/utils/` 的 COCO 实例评测薄封装，实现 SCORE 跨数据集口径：
逐数据集逐类 AP → 汇总 mAP，与 SCORE 论文口径对齐；另实现本方案自用诊断量：
类平均实例召回 proxy（GT 实例被任一候选 IoU≥0.5 覆盖比例）、查准、
mask-IoU 条件质量。完成判据：用伪造预测自测通过，口径写进
`sam3_ovis/eval/README.md`。

**S1.5 仓库骨架与契约移植**
建 `sam3_ovis/` 目录树、环境脚本、tests 冒烟框架；根 README 目录索引加一行
指针。完成判据：`sam3_ovis/tests/smoke_pipeline.py` 空跑通过（M0）。

## 2. P0 SAM3 text-only 零样本基线（Go/No-Go gate）

**S2.1 判据预登记**
跑任何推理之前，把 SCORE 论文主表基线数字抄录进
`notes/reports/p0-gate-criteria.md`，并登记三个 gate 指标与阈值（此后不回改）：
- G1 类平均实例召回 proxy @IoU0.5（三测试集汇总）——主判据；
- G2 SCORE 口径 mAP——记录用，无硬阈；
- G3 推理可行性——单卡（48G）显存峰值与 1024²/2048² 单图时延。
初始阈值：G1 ≥ 0.40 → GO；0.20 ≤ G1 < 0.40 → 条件 GO（插入 P0.5）；
G1 < 0.20 → NO-GO（回设计层评估基座，触发 design-v3 讨论）。
完成判据：文件存在且含全部阈值与来源。

**S2.2 零样本推理 harness**
实现 `sam3_ovis/inference/zeroshot_text.py`：每图 × 每类短语 → SAM3 text-only
概念分割 → 候选（mask + 分数）→ S1.4 管线评测。同时记录逐类、逐图候选统计。
完成判据：三测试集 + NWPU-Refer category 子集全量跑完，数字进报告。

**S2.3 大幅图推理开销实测（E2）**
同一子集（每测试集抽 50 图）对比三种模式：整图 resize / 固定网格分块 /
重叠滑窗；记录显存、时延、指标差。完成判据：开销表 + 选定默认模式及理由。

**S2.4 失败分类**
对每测试集抽 100 失败案例人工归类：漏检（无候选）/ 误检（错类）/ mask 质量
（对但形状差）/ 重复（一实例多候选）。此分类直接决定 P1 图的发力点。
完成判据：分类计数表 + 每类 ≥10 个可视化样例存 `notes/reports/figures/`。

**S2.5 gate 判定（M1）**
对照 S2.1 预登记阈值出结论，写 `notes/reports/p0-report.md`；NO-GO 或条件 GO
均须同步给出证据链。后续所有阶段的对照基线数字以本报告为准。

## 3. P1 证据图 v0（training-free）

目标：验证图的独立作用（红线 2、R2）。本阶段全部模块为手定权重启发式，
不训练任何参数；每个设计留白在此收敛并落档。

**S3.1 图 schema 与构建器**
实现 `sam3_ovis/graph/builder.py`，把 design-v2.md §5 实例化为数据类：
- 节点：类别节点（查询短语）；候选节点（SAM3 候选 + crop 特征）；局部上下文
  节点（实例环带区域特征）；全局场景节点；unknown 节点。exemplar 节点结构
  预留、P3 才启用。
- 单图单查询类建图；多类查询时底层数据共享、图按类构建。
完成判据：单测覆盖节点构造与序列化。

**S3.2 边的可计算代理（v0 定义）**
逐条实现 §5.2 关系，初始阈值登记后不回改：
- 文本—实例：SAM3 concept 分 + CLIP（crop vs 类名）余弦，加权归一；
- 同类支持：crop 外观相似（CLIP/DINOv2 余弦 ≥ θ_s）或尺度比 < k；
- 相似类竞争：竞争类 = 类名 CLIP 相似 top-K；跨类候选 IoU ≥ θ_o 记竞争边；
- 实例—局部上下文：环带内高置信候选类别构成的软共现分（训练集统计先验）；
- 实例—全局场景：场景节点 = CLIP 图像级 top scenes；scene-class 软先验表
  （训练集统计；只作分数，禁硬规则，红线 5）；
- 区域覆盖：候选并集 coverage mask；
- unknown 冲突：低文本分 + 竞争未胜 + 上下文矛盾的候选。
完成判据：每类边有单测（构造样例图验证方向与权重域）。

**S3.3 完整性指标实例化（§12.4 留白收敛）**
定义并实现三个统计量：已解释区域覆盖率、未解释响应密度（低阈值候选/热区
未被高置信覆盖部分）、冲突候选率；输出图级"预测不完整风险分"。该分只调节
最终置信度与告警输出，不生成新实例（红线：上下文不得硬造目标）。
完成判据：指标公式落 `sam3_ovis/graph/README.md` + 实现过单测。

**S3.4 推理算子**
两轮消息传递式打分（手定权重）→ 候选可靠性分；支持边连通分量作重复簇，
分数加权合并（NMS 升级）；完整性分调制最终排序。完成判据：端到端单图 demo。

**S3.5 消融矩阵 A0–A3（本阶段核心产出）**
- A0：裸 SAM3（官方阈值 + 标准 NMS）；
- A1：**无边对照**——同特征、同分数体系，仅去掉消息传递（逐实例独立打分）；
- A2：完整图；
- A3：组件递减（−竞争 / −局部上下文 / −场景 / −完整性调制）。
指标：三测试集 mAP + 召回 + 完整性指标；显著性用 bootstrap。
出口判据（预登记初始值）：A2−A0 ≥ +2 mAP 且 A2−A1 ≥ +1 mAP → 进入 P2
可学习化；A2 ≈ A1 → R2 兑现，停线回设计层（图的联合判断需重新设计不可
替代环节，如竞争消解或完整性的显式用途）。
完成判据：`notes/reports/p1-report.md` 含完整矩阵与判据核对。

## 4. P2 SCORE 完整协议与可学习化（条件触发）

**S4.1 基线对照决策**
登记决定：SCORE 对照采用论文数字引用或自行复训（复训条件：审稿要求受控
对比；论文配置开源、单卡 L40S 级，算力内）。完成判据：决定与理由入报告。

**S4.2 训练信号自举**
iSAID + SIOR 上跑 SAM3 候选 → 与 GT 匹配（IoU≥0.5）生成节点可靠性与边标签
（以 S3.2 启发式代理的目标值为弱标签）。完成判据：标签统计与抽样可视化。

**S4.3 可学习图模块**
冻结 SAM3/CLIP，训练轻量边打分器 / ≤2 层 GAT（可训参数 < 30M，4×48G 内）；
超参搜索范围预登记。完成判据：训练收敛 + 验证集增益。

**S4.4 跨集泛化报告（R6）**
训练分布（iSAID/SIOR）vs 三测试集分别报告；若可学习版跨集差于 training-free
版，主结果回退 training-free 并如实报告。完成判据：对照表进报告。

**S4.5 SCORE 协议主表（M3）**
text-only 主结果 + 全消融，口径与 S2.1 登记一致。完成判据：主表定稿进
`notes/reports/p2-report.md`。

## 5. P3 exemplar 扩展

**S5.1 GT exemplar 上限测量**
每图每类随机取 1 个 GT 实例作 visual exemplar，SAM3 组合 prompt 重分割；
结果**单独表格**报告，不与 text-only 混排（红线 4）。
完成判据：上限表 + 与 text-only 的逐类差。

**S5.2 self-exemplar 闭环**
选择器 = 图可靠性 top-1 候选（带最低分阈值与多样性约束）→ 作 exemplar 重
分割 → 图再融合；轮数上限 2。完成判据：闭环 demo + 逐轮数字。

**S5.3 增益拆分消融（R5）**
把 self-exemplar 增益拆为：实例发现增益（新增正确实例的 AP 贡献）与 mask
质量增益（已有实例 IoU 提升），分别报告。完成判据：拆分表进报告。

**S5.4 贡献链闭环消融（M4）**
graph on/off × exemplar on/off 的 2×2，加完整性组件开关，对应论文三贡献
（关系组织 → 完整性判断 → 视觉证据增强）。完成判据：矩阵齐、可绘图。

## 6. 论文实验矩阵（贡献 ↔ 实验映射）

| 贡献（design-v2.md §9） | 支撑实验 | 来源阶段 |
|---|---|---|
| 同图实例—上下文证据图 | A0/A1/A2/A3 + 组件递减 | P1 |
| 完整性证据记忆 | 完整性指标 vs 漏检相关性 + 调制消融 | P1/P3 |
| 图驱动 exemplar 增强 | 2×2 闭环 + 增益拆分 + GT 上限分表 | P3 |
| 主结果 | SCORE 三测试集 text-only mAP 主表 | P0/P2 |
| 基线与协议 | SCORE 数字/复训对照、SAM3 零样本基线 | P0/P2 |

## 7. 文档与同步规则

1. 本文档只管"怎么做"；科学边界一律以 `design-v2.md` 为准。
2. 阶段报告是唯一的进度事实来源；gate 判定只对照预登记阈值。
3. 代码合入与 `sam3_ovis/` 内 README 同一提交；根 `PROJECT_ARCHITECTURE.md`
   追加指针段落，不展开新主线细节。
4. 任何与设计红线冲突的实现便利一律禁止；确需突破时先触发设计迭代。
