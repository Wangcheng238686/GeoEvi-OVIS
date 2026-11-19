# G3b 待查重目标的先验工作查重（2026-11-21）

日期：2026-11-20/21。检索窗口：2025-06 之后为重点（含窗口前的通路先例）。
基准材料：`2026-11-21-g3a-results.md`（信息差证据）、
`2026-10-28-score-learnable-dilation.md`（SCORE 源码级分析）、
`2026-10-25-sam3-relational-memory-prior-art.md`（上轮关系记忆查重）。

**待查重机制（G3b）**：SAM3 基座上，把候选实例与图筛选的高置信上下文实例的
crop 经 RemoteCLIP 文本锚定编码（crop 与 15 类文本嵌入的余弦逻辑）作为记忆
记录写入原生 memory（writer/reader），经可训 bridge 注入 detector query，
改善开放词汇类别排序。
差异轴：①外部编码器信息差；②图＝上下文证据源筛选（质量税 2.2pt 已量化）；
③自身与上下文统一编码（self 不单列）。

本文只做检索与判定，不冻结设计、不启动训练。

## 1. 检索方法与覆盖

- SCORE（arXiv [2507.12857](https://arxiv.org/abs/2507.12857)，ICCV 2025）引用者：
  Semantic Scholar citations API 全量拉取（15 篇，2026-11-19 索引；Google Scholar
  显示约 17 篇，差额未能逐一核对——S2 索引延迟，见 §6 局限）。
- 方向检索：`SAM 3 × memory / remote sensing`、`RemoteCLIP × memory`、
  `open-vocabulary × memory × instance`（arXiv API 全字段短语检索，按提交日期
  倒序）、OpenReview API、NeurIPS 2026 / CVPR 2026 会场页、项目 GitHub README
  直查。Web 搜索引擎于 2026-11-20 触发周限额，后续以 API 直查补足。
- 上轮已覆盖工作（CA-SAM2、SAM3-DMS、VL-SAM-v3、OmniSAM、MPG-SAM2、
  MemorySAM、SAM2Object、ReG-SAM、GPRN）不再重复判定，仅在汇总时引用。

## 2. SCORE 引用者逐条判定（任务 1）

**结论先行：15 篇引用者中没有任何一篇做"实例/区域级 CLIP 证据写入记忆或
注入 query"。** 全部是语义分割/训练自由管线/域适配/评测/综述，无记忆写入
机制。逐条：

| 引用者 | 一句话方法 | 判定 |
|---|---|---|
| SegEarth-OV3（[2512.08730](https://arxiv.org/abs/2512.08730)，2025-12） | 免训练把 SAM3 用于遥感 OVSS：语义头+实例头 mask 融合，presence 分数过滤不存在类别 | 仅相关（重要基线；无外部编码器 crop、无记忆、无图） |
| ConInfer（[2603.29271](https://arxiv.org/abs/2603.29271)，CVPR-F 2026） | 训练自由遥感 OVSS 的上下文感知推理：跨空间单元联合预测并显式建模单元间语义依赖，注入全局上下文线索 | 部分重叠（轴②弱形态：上下文依赖建模已存在；但 patch 级一致性、无实例 crop 编码、无记忆写入、无 query 注入） |
| DR-Seg / Decouple and Rectify（[2604.02010](https://arxiv.org/abs/2604.02010)，ACM MM 2026） | 把 CLIP 特征解耦为语义/结构子空间，RS-DINO 结构增强，先验驱动图修正模块 + 不确定性自适应融合 | 部分重叠（轴①：辅助编码器融合已有——DINO 结构证据补 CLIP；轴②有图模块但用于结构先验修正而非上下文实例筛选；无记忆、无实例级文本锚定、语义分割） |
| SeeCo / Seeking Consensus（[2604.26221](https://arxiv.org/abs/2604.26221)） | 测试时几何一致性（多视图）+ 语义一致性（文本自适应校准）在线再校准 OVSS | 仅相关 |
| Perturbation-Regularized OVRSIS（[2604.15652](https://arxiv.org/abs/2604.15652)） | 扰动正则化像素-文本对应，统一多域评测 | 仅相关 |
| ProSM（CVPR-F 2026，DOI 10.1109/CVPRF71909.2026.00612） | 软掩蔽微调保留上下文信息，细粒度遥感分割 | 仅相关 |
| ProtoFlow（[2604.03212](https://arxiv.org/abs/2604.03212)） | 类增量遥感分割的低曲率原型流（持续学习记忆） | 仅相关（"memory"属持续学习语义，不同物） |
| 几何变换流形恢复（[2610.04300](https://arxiv.org/abs/2610.04300)） | 利用 D4 群几何不变性稳定 SAM3-OVSS 的响应 | 仅相关 |
| GeoRel-Prompt（ACM ICIIoCT 2026，DOI 10.1145/3821820.3821832） | 灾害场景指代定位的地理关系伪标签精炼 | 仅相关（关系推理用于伪标签，非记忆/证据写入） |
| TerraScope（[2603.19039](https://arxiv.org/abs/2603.19039)，CVPR 2026） | 像素接地的对地观测 VLM 推理 | 仅相关 |
| RSGround-R1（[2601.21634](https://arxiv.org/abs/2601.21634)） | 遥感视觉定位的空间推理由衷学习 | 仅相关 |
| BFTS-Net（Neurocomputing 2026，DOI 10.1016/j.neucom.2026.135562） | 师生框架+边界感知精炼的闭集遥感实例分割 | 仅相关 |
| RS-ReXSeg（JSTARS 2026，DOI 10.1109/JSTARS.2026.3740289） | 拒绝感知的解释性语言引导分割（开放查询） | 仅相关 |
| Dual Context-Conditioned OVSS（IEEE Access 2026，DOI 10.1109/ACCESS.2026.3674757） | 双路上下文条件的 OVSS | 仅相关 |
| 遥感基础模型综述（Sensors 2026，DOI 10.3390/s26154671） | 四类范式的遥感分割综述 | 仅相关 |

## 3. 方向检索命中（任务 2、3）

### 3.1 部分重叠（必须逐一对比）

| 工作 | 一句话方法 | 与三差异轴的重叠 |
|---|---|---|
| **EviSAM3**（NeurIPS 2026 poster，[会场页](https://neurips.cc/virtual/2026/poster/150827)，[代码](https://github.com/YaofireJinglian/EviSAM3)） | 指代遥感分割：BEiT3 跨模态先验经 rectified adapter 在 SAM3 image encoder 各层推断"证据"，动量更新累积进**证据记忆库**，证据驱动 MoE 路由并精炼 mask decoder 的 dense/sparse prompt | **最接近的一篇**："证据+记忆+SAM3+遥感"四词组合已出现。但证据源是 BEiT3 跨模态 adapter 特征（非 RemoteCLIP crop-文本锚定逻辑）、记忆为自建动量 bank（非原生 memory writer/reader）、注入点是 encoder adapter + prompt 精炼（非 detector query）、任务为指代分割（非开放词汇实例分割）、无图筛选。判定：**部分重叠（轴③弱重叠 + 轴①②不重叠）** |
| **ViTA-SAM**（[2610.25641](https://arxiv.org/abs/2610.25641)，2026-10） | 少样本分割：SAM3 上建视觉记忆模块（support 特征+mask 写入），SAM3 文本编码特征作为互补先验，跨模态对齐模块把文本条件化在记忆增强的视觉表示上，伪标签回写更新记忆 | SAM3 上"记忆写入/读取 + 文本条件化"先例已出现。但记忆内容是 support 样本特征（跨 episode、非同图上下文实例）、任务为 FSS（非 OV 实例分割）、无外部 CLIP crop 证据、无图、无 detector query 注入。判定：**部分重叠（轴③弱重叠）** |
| **SegRAG**（[2605.17630](https://arxiv.org/abs/2605.17630)，IP&M 2026） | 训练自由：外部标注视觉知识库按类建 bank，跨图像检索命中转为 SAM3 点提示（空间提示而非语义注入） | 轴①弱重叠（外部证据源已与 SAM3 结合；与上轮 VL-SAM-v3 同族）。无 crop-文本逻辑、无原生记忆写入、无 query 注入、无图。判定：**部分重叠（轴①弱）** |
| **CorrCLIPv2**（[2610.24757](https://arxiv.org/abs/2610.24757)，IJCV 2026） | 训练自由 OVSS：跨图像对象检索把对象语义从上下文中剥离，缓解 CLIP 上下文偏差 | 轴②的**反方向**先例（他们视上下文为偏差源并剥离；我们视高质量上下文为证据源并筛选）。无记忆、无实例分割、非遥感。判定：**部分重叠（轴②反向旁证，可用作"上下文角色需治理"的引证）** |
| **FARM**（[2606.15476](https://arxiv.org/abs/2606.15476)，2026-06） | 机器人实时构建对象级关系空间记忆（几何+视觉语言描述子+VLM 证据评分+空间谓词检索） | "关系结构筛选对象级记忆 + VLM 评分证据"的组合已在机器人检索任务出现。非分割基座、非原生 memory、非 OV 类别排序。判定：**部分重叠（轴②旁系先例）** |

### 3.2 仅相关

- **ProC-SAM3**（[2606.21863](https://arxiv.org/abs/2606.21863)）：SAM3 prompt 接口离线校准（MLLM 生成 prompt 池 + presence 门控融合），文本侧改进，无证据写入。
- **ENEAS**（[2609.03756](https://arxiv.org/abs/2609.03756)）：跟踪+发现的语义验证层（视觉嵌入匹配 + 条件 VLM 精炼）——"验证层"概念已出现，但无记忆写入与图。
- **UAV-OVVIS / AeroTrack**（[2607.08075](https://arxiv.org/abs/2607.08075)）：UAV 开放词汇视频实例分割新任务+免训练基础模型协同管线——任务设定邻近（OV 实例级），机制无关。
- **Cross-View OVD**（[2510.03858](https://arxiv.org/abs/2510.03858)）：地面-航空域对齐的 OV 检测——轴①的域适配旁证。
- **Promptable Concept Segmentation from Above**（[2607.09583](https://arxiv.org/abs/2607.09583)）：SAM3 遥感零样本/单样本系统评测——证明 SAM3 遥感域差距（我们的动机引证），非机制竞争。
- **Bonn RA-L 2025**（[Paetzold & Nogga](https://ais.uni-bonn.de/papers/RA-L_2025_Paetzold_Nogga.pdf)）：OVD 检测框 crop 的 VLM 验证（含 invalid 判定）——"crop 验证"用于自身检测框，非上下文实例证据入记忆。
- **SAMWISE**（[2411.17646](https://arxiv.org/abs/2411.17646)，2024-11，窗口前）：文本驱动 SAM2，语义注入 SAM2 memory attention——"外部语义进 SAM 记忆通路"的最早先例，必须引用，但为视频文本条件化、非实例证据记录。
- **kNN-CLIP**（[2404.09447](https://arxiv.org/abs/2404.09447)，窗口前）：检索式 CLIP 分类记忆——检索范式旁证。
- 其余（MemMTL [2608.28078](https://arxiv.org/abs/2608.28078) 原型记忆多任务、SAM3RL [2610.19931](https://arxiv.org/abs/2610.19931) RL 记忆控制、OmniOVCD [2601.13895](https://arxiv.org/abs/2601.13895)、SpotDiff [2611.17172](https://arxiv.org/abs/2611.17172)、ReAttnCLIP CVPR 2026、VIRST [2603.27060](https://arxiv.org/abs/2603.27060) CLIP 引导锚帧选择等）：任务或层级不同，仅相关。

## 4. 三差异轴组合判定（任务 4 上半）

1. **轴①（外部编码器信息差）：单独不成立。** RemoteCLIP 证据入遥感 OV 系统
   就是 SCORE 本身；DR-Seg（DINO）、SegRAG（检索库）、Cross-View（地面 CLIP）
   均为"外部证据补基座盲区"的实例。G3a 的价值是把信息差**量化**（自身 crop
   文本逻辑 +5.49pt、p=1e-4），这是证据强度，不是新颖性。
2. **轴②（图＝上下文证据源筛选）：作为独立新颖性弱，作为耦合组件可守。**
   上下文建模本身高度活跃（ConInfer 单元依赖、DR-Seg 图修正、FARM 关系记忆、
   CorrCLIPv2 反向剥离），但"图筛选**同图高置信上下文实例**的文本锚定证据并
   决定其作为记忆记录写入"这一具体角色未检索到。2.2pt 质量税（R1g−R1）是
   轴②的量化支撑：筛选质量决定信息价值，这是图区别于几何膨胀/无筛选池化的
   可测差异。
3. **轴③（统一编码 + 原生 memory 写读 + 可训 bridge 注入 detector query）：
   完整通路未出现。** 最接近的两个（EviSAM3、ViTA-SAM）都不写"实例 crop 与
   类文本嵌入的余弦逻辑记录"、不经可训 bridge 注入 detector query：EviSAM3
   的记忆是自建动量 bank、注入在 encoder adapter/prompt 层；ViTA-SAM 是 FSS
   support 记忆。写入**原生** memory（writer/reader 接口）并回接 detector
   query 改善 OV 类别排序的通路，检索范围内无先例（含上轮已查的 SAM3-DMS、
   VL-SAM-v3、CA-SAM2，其记忆均为身份/检索用途，不直接改类别打分——与
   2026-10-25 报告 §5 的判断一致）。
4. **组合结论：在本次检索范围内三轴组合仍成立，无撞车。** 但成立的形态是
   "组合级"而非"组件级"：写作时应把新颖性主张落在轴②③的耦合（图筛选的
   上下文证据 → 原生记忆记录 → bridge 注入 query 的可训通路）上，并显式承认
   轴①是已确立的范式。

## 5. SCORE 威胁评估（任务 4 下半）

SCORE 是最近邻，也是最强的对照与引用义务（同任务、同编码器、同动机）：

| 维度 | SCORE（[2507.12857](https://arxiv.org/abs/2507.12857)，源码分析见 2026-10-28 报告） | G3b（本机制） |
|---|---|---|
| 上下文获取 | 自身 mask 可学习膨胀（**含对象**），共享 kernel 混合 | 图筛选**其他**高置信实例（质量税 2.2pt 量化） |
| 证据形态 | 膨胀区域 RemoteCLIP patch 特征（连续向量，池化） | crop 与 15 类文本嵌入余弦逻辑（文本锚定记录） |
| 证据治理 | 无筛选（几何决定一切） | 高置信门控（R1g 过线、R1 不显著） |
| 架构位置 | decoder 内直注实例类别表示，一次前向 | 原生 memory writer/reader + 可训 bridge → detector query |
| 自身处理 | 膨胀区含自身 | 自身与上下文统一编码（轴③） |

- **威胁等级：中高，但非撞车。** SCORE 占据"遥感 OV 实例分割 + 区域级
  RemoteCLIP 上下文注入类别表示"的概念高地；我们与其共享动机与编码器，故
  论文必须在相关工作直接对比，且实验须以 SCORE（或其膨胀直注机制）为强
  对照，而非只比闭集基线。
- **真正的风险不在查重而在实证**：G3a 报告已警示（v3 教训）——B1（CLIP 融合）
  数据集级 mAP 仅 +0.65 不显著，而目标级排序 +5.49pt 不等于 mAP 增量。若
  G3b 在同预算 mAP 口径下不能显著超过"SCORE 式膨胀直注"消融，则图筛选+
  记忆写入+可训注入会被审稿人评为对已验证机制的工程化包装。建议把
  "SCORE 式自身膨胀直注"列入 G3b 第一关的强制消融臂。

## 6. 检索局限（如实）

- Semantic Scholar 对 SCORE 索引 15 篇引用者，Google Scholar 显示约 17 篇，
  差额 2 篇未能核对（S2 对新引用有收录延迟）。
- CA-SAM2 引用者追踪未完成：S2 API 在本次会话后半持续 429 限流。CA-SAM2
  为 2025-10 发表，窗口内后续应有限；且本会话的 SAM3×memory 全量清扫与
  OpenReview 检索未发现其方向上做 CLIP 证据写入的新作。必要时补查。
- Web 搜索引擎周限额耗尽（2026-12-01 恢复），后续以 arXiv/OpenReview API、
  会议页面、GitHub README 直查补足；未索引的 workshop/在审稿件不在覆盖内。
- 本报告是检索结果的收敛陈述，不构成"组合创新成立"的证明——成立与否最终
  由 §5 的对照实验决定。

## 7. 原始来源

- SCORE：[arXiv:2507.12857](https://arxiv.org/abs/2507.12857)，
  [代码](https://github.com/HuangShiqi128/SCORE)。
- EviSAM3：[NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/150827)，
  [OpenReview](https://openreview.net/forum?id=f1JHxI12WT)，
  [代码](https://github.com/YaofireJinglian/EviSAM3)。
- ViTA-SAM：[arXiv:2610.25641](https://arxiv.org/abs/2610.25641)。
- SegRAG：[arXiv:2605.17630](https://arxiv.org/abs/2605.17630)，
  [IP&M 版](https://www.sciencedirect.com/science/article/pii/S0306457326005807)。
- CorrCLIPv2：[arXiv:2610.24757](https://arxiv.org/abs/2610.24757)，
  [代码](https://github.com/zdk258/CorrCLIP)。
- FARM：[arXiv:2606.15476](https://arxiv.org/abs/2606.15476)。
- SegEarth-OV3：[arXiv:2512.08730](https://arxiv.org/abs/2512.08730)。
- ConInfer：[arXiv:2603.29271](https://arxiv.org/abs/2603.29271)，
  [代码](https://github.com/Dog-Yang/ConInfe)。
- DR-Seg：[arXiv:2604.02010](https://arxiv.org/abs/2604.02010)。
- ProC-SAM3：[arXiv:2606.21863](https://arxiv.org/abs/2606.21863)。
- UAV-OVVIS：[arXiv:2607.08075](https://arxiv.org/abs/2607.08075)。
- Cross-View OVD：[arXiv:2510.03858](https://arxiv.org/abs/2510.03858)。
- SAM3 遥感评测：[arXiv:2607.09583](https://arxiv.org/abs/2607.09583)。
- ENEAS：[arXiv:2609.03756](https://arxiv.org/abs/2609.03756)。
- SAMWISE：[arXiv:2411.17646](https://arxiv.org/abs/2411.17646)。
- SAM3：[arXiv:2511.16719](https://arxiv.org/abs/2511.16719)，
  [代码](https://github.com/facebookresearch/sam3)。
