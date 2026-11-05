# SAM3 关系上下文记忆：近邻查重与动机复核

日期：2026-10-25。范围：公开可检索论文、会议原文、作者仓库；属于定向查重，不能
据此保证穷尽所有未公开或未索引工作。本文记录证据与候选问题，不冻结新方法或启动训练。

## 1. 已确认的研究约束

负责人要求图与 SAM 记忆本身构成研究核心，尽量保留原生 memory encoder / attention，
接受小幅适配，并允许对主干进行遥感微调。优先研究其他实例和场景上下文如何帮助当前
目标识别与分割，而非仅关联同一实例的跨窗重复观察。二维滑窗只是可选的观测协议。

## 2. 检索口径与结论

检索组合包括 SAM3 / SAM 3 / SAM2 × graph / relational / scene graph / context ×
memory / memory bank / memory selection × remote sensing / segmentation；进一步追查
CA-SAM2、SAM3-DMS、VL-SAM-v3、GPRN、ReG-SAM、SCORE、OmniSAM 和 MPG-SAM2。
技术判断优先依据会议全文、arXiv 原文及作者仓库；搜索聚合页仅用于发现线索。

结论分开表述：

- 已有 SAM 结合图、SAM 结合上下文记忆、SAM3 结合记忆选择、遥感结合上下文的工作。
  因此不能宣称这些组件或两两结合首次出现。
- 本次未找到同时明确满足“显式实例—上下文关系图控制 SAM3 原生记忆读写，并用于
  遥感开放词汇实例识别与分割”的直接同类论文。
- 后一项是检索结果，不是不存在的证明，更不是组合创新已经成立的证明。

## 3. 最接近的证据

| 工作 | 已有机制 | 与当前意图的交集和边界 |
|---|---|---|
| CA-SAM2，MICCAI 2025 | 邻近 patch 的环境和前景纹理分别进入双记忆并用于 SAM2 attention | 最接近“周围环境通过原生记忆帮助当前实例”；无显式实例关系图、无开放词汇类别识别 |
| SCORE，ICCV 2025 | 区域上下文调整类别表示，全局上下文适配文本 | 直接覆盖遥感开放词汇实例类别消歧动机；未采用 SAM3 原生图记忆 |
| GPRN，AAAI 2025 | 在 SAM visual prompts 间建图并进行关系聚合 | “SAM + 图 + 语义一致性”已有；属于跨域 few-shot prompt 推理 |
| SAM3-DMS，arXiv 2026 | 依据每个对象可靠性独立控制 memory 更新 | “对象级 SAM3 记忆选择”已有；视频身份稳定，并非邻居关系语义记忆 |
| VL-SAM-v3，arXiv 2026 | 外部视觉原型检索，形成稀疏/稠密先验，适配 SAM3 等检测器 | “SAM3 + 外部视觉记忆 + 开放词汇识别”已有；不能等同 native tracker memory |
| OmniSAM，ICCV 2025 | 滑窗 patch 序列复用 SAM2 memory | “静态图空间切片调用原生记忆”已有 |
| MPG-SAM2，ICCV 2025 | pixel/object 两级聚合全局与历史信息 | “SAM2 记忆补全局上下文”已有；任务为 RVOS |
| MemorySAM，arXiv 2025 | 多模态序列使用 SAM2 memory，并以训练期语义原型辅助语义学习 | “原生记忆适配非视频语义任务”已有 |
| SAM2Object，CVPR 2025 | SAM2 多视图一致分割后构造 superpoint 图聚类 | “SAM2 记忆与图出现在同一系统”已有；图处于后续 3D 聚类阶段 |
| ReG-SAM，arXiv 2026-09 | 参考血管图与原型生成结构/模态表示 | “SAM + 参考图数据库”已有；不能仅凭数据库称原生记忆控制 |

### CA-SAM2 的决定性细节

已读取官方 PDF 第 3–6 页。它将二维病理图像从中心向外逆时针扫描成 256×256
patch 序列。EMB 保存同图已处理 patch 的 image embedding，以二维坐标欧氏距离选近邻，
经独立训练的 SAM2 memory-attention 层注入环境；TMB 保存 memory encoder 产生的
前景记忆，以相似度和预测质量维护多样性，再选择与 query 最相似的记忆供 SAM2
memory attention 读取。这里的 SDF 指 Shortest Distance First，不是 signed distance field。

所以“用空间近邻与相似度替换 FIFO”“加环境/前景双 bank”“二维图像切片使用 SAM
记忆”均已被覆盖。作者已公开训练与测试仓库，但本次未执行其代码验证复现性。
这也补正此前医学论文定位：Slide-SAM 只是一个候选；CA-SAM2 更符合“二维平面切窗
结合 SAM 记忆”的技术描述，仍不能断定它就是负责人当时看到的论文。

## 4. 可研究的具体问题（推论，尚未冻结）

候选动机：遥感小目标的对象内证据不足，而普通上下文池化和按相似度检索不区分
“同一对象”“附近不同对象”“背景环境”三类证据。将不同对象的 mask memory 当作
当前对象的身份记忆直接混合，会有串实例和背景偏置风险。图可以编码这些证据的
角色与关系，决定它们以身份证据还是上下文证据参与读取。

需要证明的不是“图存了更多东西”，而是：同预算下，关系约束能选择普通近邻和
相似度检索无法选择的有效上下文，帮助类别判别，同时避免串 mask。若空间近邻或
普通 attention 已达到相同效果，图的独立动机不成立。

可供后续并行探针比较的三类假设：

1. 关系选择：图筛选当前对象可读的邻近实例/环境 memory，与 CA-SAM2 式双 bank、
   距离与相似度 top-K 比较。
2. 关系角色编码：在保留原生 attention 主体下，对身份和上下文 memory 加小型角色/
   关系适配，验证能否兼顾类别识别与实例隔离。
3. 冲突控制：实例—环境关系检测不可靠写入或冲突读取，与单对象置信度门控比较，
   验证改善的是语义错类而非仅分数阈值。

这三项是调研后的候选，不代表已批准实现。均需相同主干遥感微调、相同 token 容量、
相同附加参数/训练预算的无图对照，并用删边/打乱边及错误上下文干预检验关系贡献。

## 5. SAM3 架构中的关键约束

SAM3 官方架构分为概念 detector 与 memory-based tracker。把图记忆只接在 tracker
上可能改善 mask，却不会自动改变 detector 对 ship/car 的类别判断。后续设计必须
明确 memory-conditioned feature 如何影响开放词汇打分或概念预测；这可能需要小型
回接适配并联合微调，尚不能宣称是零改动的现成功能。

还需定义同图记忆在“第一次观察—保存—读取—更新”中的实际作用，并以同图上下文
attention 作对照。若一次建图一次融合就达到相同收益，应如实称上下文融合，不能
仅因使用了 memory encoder 就声称记忆管理成立。

## 6. 原始来源

- CA-SAM2：[官方全文](https://papers.miccai.org/miccai-2025/paper/1881_paper.pdf)，
  [会议页面](https://papers.miccai.org/miccai-2025/0131-Paper1881.html)，
  [作者代码](https://github.com/HanbinHuang123/CA-SAM2)。
- SCORE：[论文](https://arxiv.org/abs/2507.12857)。
- GPRN：[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32695)，
  [代码](https://github.com/CVL-hub/GPRN)。
- SAM3-DMS：[全文](https://arxiv.org/html/2601.09699v1)，
  [代码](https://github.com/FudanCVL/SAM3-DMS)。
- VL-SAM-v3：[全文](https://arxiv.org/html/2605.03456v3)。
- OmniSAM：[论文](https://arxiv.org/abs/2503.07098)。
- MPG-SAM2：[ICCV](https://openaccess.thecvf.com/content/ICCV2025/papers/Rong_MPG-SAM_2_Adapting_SAM_2_with_Mask_Priors_and_Global_ICCV_2025_paper.pdf)。
- MemorySAM：[论文](https://arxiv.org/abs/2503.06700)，
  [代码](https://github.com/Chenfei-Liao/MemorySAM)。
- SAM2Object：[CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_SAM2Object_Consolidating_View_Consistency_via_SAM2_for_Zero-Shot_3D_Instance_CVPR_2025_paper.pdf)。
- ReG-SAM：[论文](https://arxiv.org/abs/2609.31160)。
- SAM3：[论文](https://arxiv.org/abs/2511.16719)，
  [官方代码](https://github.com/facebookresearch/sam3)。
