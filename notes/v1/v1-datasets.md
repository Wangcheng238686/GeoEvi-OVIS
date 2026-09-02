# 遥感数据集调研（8 篇数据集论文精读）

> 调研日期：2026-09-01，两个并行代理精读，全文提取件在 `papers/_extract/`
> 用户方向：滑窗遥感 spatial memory（图结构组织）+ 单帧开放词汇**实例**分割，文本类名优先
> 未决问题：开放词汇走 SAM3 还是 SAM2+外挂文本通道（本文件提供决策证据）

## ⚡ 三个核心发现（先读这个）

1. **"OVRSIS" 名字有误导**：OVRSISBench 系的 IS = **Image** Segmentation，不是 Instance。
   OVRS / LandDiscover50K / OVRSISBench V1&V2 全部是开放词汇**语义**分割（mIoU 协议），
   全部走 CLIP cost-aggregation 路线，**没有一篇用 SAM 系当分割基座**，全部没有 base/novel 划分。
2. **"开放词汇遥感实例分割"已被 SCORE 占位**（ICCV 2025，"Scene Context Matters in OV RS
   Instance Segmentation"，arXiv 2507.12857；同团队还有 MARIS 海洋版；SAM 系有
   AerOSeg/AerOSeg++ CVPR 2025）→ **必须查重**。但其配套大规模实例级数据集仍稀缺。
3. **数据空白 = 机会**：没有任何大规模 OV 遥感数据集发布实例级标注——OVRSIS95K 的半自动
   标注管线**内部生成了实例级 mask 但最终合并成语义标注发布**（其管线 = captioner→segmentor
   出实例 mask→合并）。"图结构 spatial memory + OV 实例分割 + 实例级评测协议"组合没有
   现成数据集与协议，协议本身可主张为贡献点。

## A. Referring 组（文本表达式→mask）

### NWPU-Refer（arXiv 2506.03583，西工大）— ★与方向距离最近
- **任务**：referring，但**原生支持 category-based referring**（"All vehicles"/"All airplanes"）
  = "类名→全实例"的社区先例；**唯一支持 single/multi/non-object 三模式**——non-object
  （查询无目标→空 mask）是独有的"实例不存在"监督信号，直接对应记忆删除/存在性门控实验。
- **规模**：15,003 图 / 49,745 标注（3.3 条/图）/ 32 类 / 1024–2048px / 0.12–0.5m / 30+ 国。
- **语言**：中英双语纯人工；6 维度（尺寸/空间关系/颜色/类别关系/运动状态/联想）。
- **协议**：7:1:2 按地理区域划分；P@0.7-0.9/oIoU/mIoU。MRSNet test mIoU 43.60（任务更难，
  比 RRSIS-D 低 20+ 点）；**无 SAM 系方法**。
- **适配**：category 子集即用为"类名→全实例"训练/评测集（连通域分解可近似还原实例级监督，
  同类相邻实例有粘连风险）；**1024–2048px 是本组唯一真正需要滑窗的尺寸档**，与 C5-v2 管线
  吻合；GitHub 公开（CVer-Yang/NWPU-Refer）。

### LaSeRS + SegEarth-R2（CVPR 2026，西交大）— 类别最广 + 粒度最细
- **任务**：MLLM 对话分割，一条指令→N 个 [SEG]→**N 张独立 mask**（最接近"记忆图一次写入
  N 个实例节点"的监督形式）；四维度：**层级粒度（semantic/instance/part 三级）**、目标多重
  性、推理需求（explicit/implicit，含知识推理）、语言变异（短语到 80+ 词）。
- **规模**：40,396 mask / 122 类 / 30,830 QA；mask 来源 = **SAMRS（继承 DIOR/DOTA/iSAID/
  FAIR1M 类别体系）+ SAM 全图网格点 prompt**（part 级靠局部稠密 prompt）；输入 1024²。
- **协议**：test 1,900 mask 按四维度切分（gIoU/cIoU）+ 大量跨库迁移。SegEarth-R2-3B avg
  gIoU 57.2；**multi-target 是所有模型最弱维度**（38.3-49.2）→ 我方机会；part 级领先 20 点。
- **⚠ 对基座决定的硬证据（Table 6 消融）**：换 SAM decoder(ViT-H) 62.7 / SAM2 decoder 64.3，
  均显著低于 Mask2Former+Swin-B 的 72.3 → **SAM 系 backbone/decoder 在 RS 比层级架构差
  8–10 gIoU，小目标+极端尺度是系统性弱点，与 prompt 形态无关**。
- **适配**：explicit-short + instance 级子集 = 现成"类名→单/多实例"评测集（需剥掉 QA 回答
  文本）；GitHub 公开（earth-insights/SegEarth-R2）。

### RRSIS-D + RMSIN（CVPR 2024，厦大）
- **任务**：单目标 referring（无 multi/non-object），方位词推理最重（right/left/upper/lower
  最高频）；17,402 三元组 / 800×800 / 0.5–30m / **20 类 = DIOR 体系**（源 RSVGD bbox）。
- **标注**：SAM bbox-prompt 生成 + 人工修补（作者自述 SAM 航空域差）——SAM 系 RS 弱点在
  数据集构建时已显形。
- **适配**：**不适合**"类名→全实例"（表达式绑定单个目标+方位语境）；只当域内预训练/泛化
  测试用。GitHub 公开（Lsan2401/RMSIN，RefCOCO 格式）。

### RefSegRS（TGRS 2024，TUM）
- **任务**：单表达式→单张二值 mask（类名表达式=该类全实例**并集** mask）；4,420 三元组 /
  285 景 / 14 类 / 512×512（源 SkyScapes 13cm 航空，1200 窗/600 步滑窗裁切降采样——**构建
  方式与滑窗管线同构**）。
- **价值主要是方法学**：其 identity/inclusion 概念映射证明"语义标注零成本合成类名驱动数据"
  可行——**该配方可直接套到 WHU/iSAID 自建训练信号**。LAVT+LGCE mIoU 0.60。

## B. 开放词汇组（类别名→语义 mask）

### OVRSIS95K + OVRSISBenchV2 + Pi-Seg（arXiv 2604.15652，USTC/TeleAI）— ★主数据集推荐
- **OVRSIS95K**：94,620 训练图 / **35 类** / 5 场景域；半自动管线（captioner→解析名词→
  segmentor 出**实例级** mask→合并成语义标注）+ 人工审计（类别 97.25% 直收，mask 91.66%
  直收）；HF 公开 **58.5GB apache-2.0**（kkk2026/OVRSIS95K）。
- **OVRSISBenchV2**：170,036 图 / **128 类**（train 35 / test 93）；10 个下游测试集 + 3 个
  下游任务协议（**建筑提取含 WHU_Aerial/WHU_Sat.II/Inria/xBD——与已有 WHU 训练线直接对口**；
  道路；洪水）；**分辨率分组协议**（<800² vs >800²）+ **滑窗效率协议**（首次把滑窗推理开销
  量化进 benchmark：滑窗版 3,014 GFLOPs/24.2 FPS vs 非滑窗 608/66.9）→ 滑窗记忆叙事的
  现成评测框架。Pi-Seg ViT-L m-mIoU 44.40 SOTA；**无 SAM/SAM2/SAM3 基线参赛**。
- **适配**：主训练集 = OVRSIS95K；主 benchmark = V2；base/novel 需自建（V2 Fig 5 的
  test-only 类别清单 + V1 类重叠统计可构造零重叠 split）。

### OVRSISBench V1 + RSKT-Seg（AAAI 2026）
- 8 公开数据集重划分（DLRSD/iSAID/Potsdam/Vaihingen/UAVid/UDD5/LoveDA/VDD），双配置
  55,628 图/72 类；**跨数据集 OV transfer 协议**（iSAID→UAVid/UDD5/VDD 强域移）= 现成
  泛化测试床；Fig 2(b) 类重叠统计可直接用来挑"零重叠数据对"构造图记忆新类实验。
  RSKT-Seg ViT-L m-mIoU 46.21。图宽 256–4000。代码链接占位、无独立数据发布。

### LandDiscover50K + GSNet（AAAI 2025，大连理工）
- 51,846 图 / 40 类（**物体性最强**：iSAID 物体类 ∪ DIOR 风格类 ∪ 土地覆盖），源
  OEM/LoveDA/DeepGlobe/SIOR/SOTA（间接引入 DOTA/DIOR/xBD/Inria 等）；400–1200px。
  明示**不做 novel 划分**；跨数据集测 FLAIR/FAST/Potsdam/FloodNet，GSNet avg mIoU 31.25。
  GSNet 结构 = **冻结 specialist(DINO) + generalist(CLIP) 双流**——"SAM2 记忆+外挂文本
  通道"的直接模板。HF 有数据、**无 license 文件**。

### OVRS benchmark（arXiv 2409.07683，TGRS 2025）
- 4 数据集重打包（DLRSD 7,002/17 类 + iSAID 24,439/15 + Potsdam/Vaihingen），跨数据集
  transfer 协议的最早版本；OVRS m-mIoU 55.47/48.52。规模老、训练已 resize 384²，只当
  sanity-check。MIT license（github.com/caoql98/OVRS）。

## C. 实验角色分配（推荐组合）

| 角色 | 数据集 | 理由 |
|---|---|---|
| **主训练集（OV 语义）** | OVRSIS95K | 94.6K 图/35 类，类平衡，apache-2.0，含建筑/道路与 WHU 线互补 |
| **主 benchmark** | OVRSISBenchV2 | 170K/128 类协议最全；下游建筑/道路对口 WHU；分辨率分组+滑窗效率协议天然适配滑窗记忆叙事 |
| **类名→全实例训练/评测** | NWPU-Refer category 子集 | 原生 all-X 表达式 + multi/non-object 三模式 + 唯一真需滑窗的尺寸档 |
| **实例级多目标评测** | LaSeRS instance 子集 | 122 类最广 + 一指令 N 独立 mask + multi-target 是全场最弱维度（我方机会） |
| **跨域泛化** | OVRSISBench V1 协议（iSAID→UAVid/UDD5/VDD） | 现成强域移 + 类重叠统计可构造零重叠新类 split |
| **域内预训练/泛化** | RRSIS-D | 单目标 referring，DIOR 类别体系 |
| **自建数据配方参考** | RefSegRS 的 identity/inclusion 映射 | 语义标注→类名驱动并集 mask 零成本合成，可套 WHU/iSAID |
| **sanity-check** | OVRS | 便宜快速 |

## D. 关键缺口（= 可主张的贡献点）

1. **实例级评测协议缺口（部分被 SCORE 补上）**：8 篇数据集论文全无 instance-level AP 协议
   （全是并集 mask 的 oIoU/mIoU/P@X/gIoU）；但 SCORE（ICCV 2025）在方法论文侧建立了
   **跨数据集实例级 mAP 协议**（iSAID/SIOR 训练 → NWPU-VHR-10/SOTA/FAST 测试，FreeSeg 式，
   HF 发布 split）→ **直接复用 SCORE 协议当基准，不必自建**；自建部分只剩 NWPU-Refer
   category 子集与 LaSeRS 多[SEG]子集的 AP 化（SCORE 未覆盖多目标表达式设定）。
2. **实例级标注缺口**：无大规模 OV RS 数据集发布实例级标注（OVRSIS95K 内部有但合并销毁）
   → 自建"OVRSIS95K 语义→实例化"或 NWPU-Refer/LaSeRS 子集实例化是数据贡献。
3. **SAM 系基线缺口**：全部 OV RS benchmark 无 SAM/SAM2/SAM3 基线 → 跑 SAM3 zero-shot
   对照本身就是有信息量的实验。

## E. 对未决问题（SAM2 vs SAM3 基座）的证据汇总

- **支持 SAM3**：任务形式已收敛（NWPU-Refer/LaSeRS 证明"类名→全实例"可评测）；SAM3 原生
  text→concept→instances 语义覆盖该任务；可跳过外挂文本通道；OVRSISBenchV2 的文本查询+
  分辨率分组+滑窗效率协议对两条路线都是现成公平评测框架。
- **反对（更硬）**：SAM/SAM2 decoder 在 RS 比层级 Swin+Mask2Former 差 8–10 gIoU
  （SegEarth-R2 Table 6）；RRSIS-D 构建时记录 SAM 航空域差；本领域全部 SOTA 是 CLIP 外挂
  文本通道 + 层级解码头，SAM 从未当过 OV RS 分割基座。
- **行动建议**：先跑 SAM3 在 NWPU-Refer category 子集 + LaSeRS instance 子集 + V2 的
  zero-shot 基线，对照 Pi-Seg/CAT-Seg；不达标则走 SAM2(C5-v2 层级解码头)+CLIP/MLLM 外挂
  文本通道（GSNet 双流为模板）。
