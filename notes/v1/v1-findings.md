# 论文精读知识库（19 篇）

> 每篇格式：定位 → 机制要点 → 关键数字 → 与本项目接口 → 风险
> 详细全文在 `papers/_extract/`；关键页码/表号已标注，可回查原 PDF
> 研究方向：图结构记忆组织 + 单帧遥感开放词汇实例分割 + MLLM 语义记忆 + 不确定性

---

## A. 基础模型

### SAM2（arXiv 2408.00714v2，42p，Meta）— 改造对象
- **定位**：统一图像/视频可提示分割基础模型；流式记忆让轻量 decoder 逐帧传播 masklet；class-agnostic、无文本。
- **机制**：memory bank = 近 7 帧 FIFO 空间记忆（num_maskmem=7，消融 N=6）+ 提示帧队列（首帧常驻）+ object pointer 列表（256 维=4×64，16 个）；空间记忆 64 通道；memory attention L=4（self→cross→MLP，2d-RoPE 仅近期帧）；occlusion token 判目标可见性，遮挡帧记忆加可学习 embedding。
- **关键数字**：MOSE 77.9 / SA-V test 78.4 / LVOS 78.0 (J&F, Hiera-L)；训练 256×A100×108h（p.32）。
- **消融**：GRU 记忆无增益；记忆条数 4→8 不敏感；64 通道无损 → **记忆模块是低风险可替换件**（Table 9/11, p.13-15）。
- **接口**：节点=帧记忆+pointer 天然可图化；occlusion embedding→节点置信度；pointer 可与 MLLM 文本嵌入对齐；IoU/occlusion/多 mask=三种现成不确定性信号。
- **局限**：镜头切换/拥挤/长遮挡失效；多目标独立零通信（§C 原话指向 shared context——图记忆切入口）；**单帧时记忆闲置**。
- **风险**：视频侧记忆改进是红海。

### SAM3（arXiv 2511.16719v2，78p，Meta）— ★推荐基座
- **定位**：概念级分割基础模型 ~850M；短名词短语(NP)或图像 exemplar（可组合）→ 图/视频中**所有**匹配实例 mask+ID；架构=DETR 式 detector（无记忆、身份无关）+ SAM2 式 tracker（记忆、管身份）共享 PE 骨干。
- **记忆机制（tracker）**：置信度选择性写入（只存"目标确信在场"帧）+ MDS 评分 + 确认延迟 T=15 + 重复 masklet 删除 + **周期重提示 N=16**（IoU/置信度双≥0.8 用检测 mask 重置记忆）+ 漂移引导重提示（§C.3）；**Multiplex**：16 槽桶共享空间记忆+槽级身份嵌入，128 目标提速 5.2×且指标略升（§H）；**§B 官方明说"跨目标共享全局记忆"是未来工作**。
- **Prompt**：NP 训练分布=字幕解析+alt-text+本体节点+Llama4 硬负例，文本编码器 32 token 上限；exemplar=框+正负标签+ROI 特征（现成记忆写入接口）；组合最优 pmF1 74.6 > 纯文本 66.4 > 纯图 69.5。
- **关键数字**：LVIS 零样本 mask AP 48.5；SA-Co/Gold cgF1 54.1（人类 72.8）；MOSEv2 60.3（SAM2.1-L 47.9）；训练 172k A100h + 86k H200h（p.68，不可复现）。
- **遥感弱项**：训练池仅 Visdrone 一个航拍集；RF-100VL Aerial 零样本 AP 24.0（微调 56.9）、ODinW13 航拍 20.9（p.48-49）→ **俯视+文本类名是 SAM3 最弱象限 = 切入空间**；域适配配方见 §A.3（合成数据+AI verifier）。
- **MLLM 接口**：官方 Agent=MLLM 编排 4 工具（segment_phrase/examine/select/report），7B 可行（Qwen2.5-VL-7B ReasonSeg 62.2 gIoU）；H200 单图 30ms。
- **语义骨架**：SA-Co 本体=22.4M 节点 Wikidata 图（17 大类/72 子类），已用于负例采样——现成的概念图层。
- **消融**：硬负例短语是识别能力最大来源（0→30 负例，IL_MCC 0.44→0.68）；presence head +1.5 cgF1；PE 对齐编码器远胜 Hiera。
- **结论**：首选基座（文本原生+detector 单帧可用+tracker 可插拔 ~100M 参数可模块级训练）；SAM2 作消融对照。

---

## B. 记忆管理改进（怎么选/存/淘汰）

### SAMURAI（arXiv 2411.11922v2，UW）— training-free
- **机制**：运动感知记忆准入——回溯窗口内一帧入库须同时过 mask 分、object 分、Kalman 运动分（KF 状态=[x,y,w,h,ẋ,ẏ,ẇ,ḣ]，s_kf=IoU(预测框,mask框)）三阈值，选满 N_mem=7；选 mask 用 argmax(0.15·s_kf+0.85·s_mask)。只替换打分函数，不动 memory attention。
- **数字**：LaSOT 74.23 (+5.69 vs SAM2.1-L)；消融：记忆选择 +4.4 > 运动建模 +2.5（LaSOT，Table 3）→ **"挑对记忆"比"选对mask"值钱**。
- **接口**：三分数门控是"节点准入打分"，可推广为图上评分；单帧下 motion 死，但"任务相关第三方分数参与准入"可换成本文-掩码对齐分。
- **风险**：低（标量阈值，无关系建模）。

### DAM4SAM（CVPR 2025，Ljubljana）— training-free，离图最近
- **机制**：记忆功能分区 DAM=RAM（3 槽近期外观 FIFO，带时间编码）+ DRM（init 帧固定槽 + 3 槽"锚帧"FIFO——含临界 distractor 且被正确分对背景的帧，**无时间编码**，加了反而掉 3.6%）；内省式写入：输出 mask 与备选 mask 并集框面积比 <0.7 判定发现 distractor，且仅当 IoU>0.8 + 面积稳定时写入；RAM 每 5 帧更新、目标缺席时冻结。
- **数字**：DiDi Q 0.694 (+7%)；VOT2022 EAO 0.753 (+9%)；DiDi 上超 SAM2Long 7%、超 SAMURAI 2%；**树搜索不解决 distractor**。
- **消融**：目标在场才更新 +2.5%；DRM 双条件缺一不可；DRM 去时间编码 +3.6%。
- **接口**：分区=最朴素的图（硬编码二分）；DRM 无时先验=类原型雏形；多假设内省写入=负例节点触发器；**结论原话呼吁 learnable management policies=我们在做的事**。
- **风险**：中（只做"记忆分类"会被视为增量，必须做出边+图上聚合）。

### SAM2Long（ICCV 2025，CUHK）— training-free，必划界对象
- **机制**：P=3 条假设路径并行，每路径独立记忆库；每帧 3 候选 mask 分叉成 3P，按累计 log-IoU 剪枝留 top-P；不确定时（|occlusion|<δ_conf=2）强制各路径选 IoU 取整后互不相同的 mask 维持多样性（成对差异 51.4→84.5）；路径内记忆=质量过滤（IoU>0.3 且 occlusion>0），memory key 按 occlusion 排序加权 [0.95,1.05]。注意：正式版打分是 log-IoU 累加，**非 EWMA**（EWMA 说法来自代码库/更新版，引用需核实）。
- **数字**：平均 +3.7 J&F；SA-V test +5.3；**增益与视频时长正相关**（LVOS +3.2 vs DAVIS +0.1）；开销仅 -14% FPS（encoder 每帧仍跑一次）。
- **消融**：P=4 无增益；时间衰减/空间相似度加进记忆选择**反而掉点**（Table 7）→ 纯质量过滤胜过任何加权——图边权设计的警示。
- **接口**：树=假设空间图而非记忆关系图（兄弟分支互不知晓）；可取内核=多假设延迟决策+不确定时强制多样性（可用于原型库去冗余）；败因自述"缺乏语义理解"= MLLM 侧直接论据。
- **风险**：中（"图 vs 树"必被问，见 collision-map 标准答案）。

### RMem（CVPR 2024，UIUC）— 论据最强
- **机制**：memory deciphering 受控实验（额外解码器从记忆解码第 0 帧，记忆刻意不含第 0 帧）：记忆 4→32 帧解码质量反而 0.86→0.83，注意力 0.247→0.056 → **冗余稀释注意力**；解法：K=8 定长 + UCB 式淘汰 O_k=R_k+α√(2logT/t_k)（注意力相关性+新鲜度）+ 可学习时序 PE（须搭配受限记忆才有效）。
- **数字**：Long Videos 91.5 J&F vs AOT 84.3（基线是 AOT/DeAOT 非 SAM2）；记忆长度呈**倒 U**（3→8 升，8→20 降回）。
- **接口**：图记忆的核心 motivation——它证明瓶颈在读取端注意力被冗余稀释，但只会"删帧丢信息"；图上"合并+继承"不丢信息地压缩是严格升级；**deciphering 协议可移植为遥感单帧动机实验（无人做过）**。
- **风险**：低；非 training-free（要训 PE）。

---

## C. 记忆读取与生命周期

### Cutie（CVPR 2024）— 物体级读取
- **机制**：像素记忆（XMem 式）+ 物体记忆 S（N=16 向量，mask-pooling 生成，前景/背景各半，streaming average 更新，pooling 权重全 0 时冻结防漂移）；object transformer L=3：masked cross-attention（前半 query 只看前景/后半只看背景）→query 自注意→pixel←query 回写。所有全局通信经 16 个 query 中转（N×HW 规模，故实时）。
- **数字**：MOSE 68.3 (+8.7 over XMem)；36-45 FPS。
- **消融**：**动态物体记忆承担几乎全部物体级信息**（去 S 掉 0.4，去静态 query 掉 0.1）→ 图节点应存"动态聚合内容"而非可学习模板；masked attention 掉 3.5 且训练不稳。
- **接口**：S 的向量=物体摘要节点雏形；N×HW 轻量读取头可嫁接到 SAM2 记忆库上；单帧下 S 天然重定义为类原型/exemplar 记忆；X0=X+S 入口可用 MLLM 文本嵌入调制。
- **训练**：30h×4×A100（small）——4×48G 可复现。

### XMem（ECCV 2022）— 分层存储骨架
- **机制**：Atkinson-Shiffrin 三级：感觉记忆（GRU）→工作记忆（高分辨率 KV，5≤T<10）→长期记忆（原型库 ≤10K 元素/1.4GB，压缩比 6328% 精度不降）；固化=usage 选 top-P 原型 + **key 空间非局部加权聚合（potentiation）**+ LFU 淘汰；相似度=各向异性 L2（shrinkage=元素置信度 + selection=通道重要性）。
- **数字**：Long Video 90.0 全场唯一不衰减；22.6 FPS。
- **消融**：**k-means 聚类原型崩溃（82.4±10.3）vs usage-based 90.0**——重要警示；potentiation +2.1；shrinkage/selection 有协同。
- **接口**：potentiation 本质=**免训练 graph coarsening**——图生命周期的直接骨架；usage/shrinkage=节点权重两个现成定义；免训练配方可先做图记忆 v0 再谈可学习版；长期原型=语义槽位（文本嵌入可直接检索）。
- **风险**：中（只做"三级记忆搬单帧"= XMem 移植，必须让图产生分层之外的新能力）。

### LOMM（ICCV 2025）— 节点生命周期状态机
- **机制**：LOM=每物体槽位一向量，M_t=(1-p_t)M_{t-1}+p_t·Q̇_t，p_t=前景概率（不可见即冻结防背景污染）；occupancy 布尔向量单调累积存在性；DOA 两阶段：已有物体先用记忆做 anchor 跟踪→occupancy 引导匈牙利分配（occupied 槽先匹配，新物体只进空槽）；自适应 anchor=sim 凸组合。
- **数字**：YTVIS-22 41.1 AP（+7.2 over DVIS，R50 在线）；tracker 仅 9.88M 参数 +10ms/帧；**记忆类型对比 LOM 34.0 > momentum 30.9 > similarity 28.5**。
- **接口**：p_t=节点活跃度；occupancy=节点状态机（活跃/冻结/占用）；OHM=图的增量构建规则（先挂已有节点、失败开新节点）；L_sim=现成的边监督信号；**文本类名初始化记忆槽=最自然的 MLLM 注入点**。
- **风险**：低-中（若只包装"诞生-冻结-复活"无边的建模会同质化）。

### LiVOS（CVPR 2024）— 聚合算子的数学参照+反面警示
- **机制**：softmax 匹配核化重排为递归 S_t=G_t⊙S_{t-1}+φ(K)ᵀV（S∈R^{64×256} 常数状态，时空无关）；门 G=α·1ᵀ 低秩参数化（通道级遗忘，数据依赖）。
- **数字**：DAVIS 84.4（低于 Cutie 3.5-6.6，但 Cutie 限单帧记忆掉到 49-52）；**2048p 仅 7.7GB、4096p 30.4GB 独家可跑**；门控 +1.4。
- **接口**：S 按节点分块→分块门控=图上节点/边级遗忘算子（从通道级升维到拓扑级）；48G 显存预算基准；警示=无结构大状态掉精度，**图拓扑本身要作结构先验**。
- **风险**：低（效率赛道）；若"图压缩"最终等价于换个状态空间模型会被 Mamba 系质疑。

---

## D. 领域改造与开放词汇（单帧先例组）

### OmniSAM（ICCV 2025）— 单帧先例 No.1：空间序列化
- **机制**：全景滑窗切 9 帧重叠 patch 序列当伪视频喂 SAM2（bank n=9），记忆机制原样保留建模跨 patch 依赖；Hiera 只加 LoRA(q/v) <3MB；换语义 decoder 出固定 K 类；FPA 原型对齐（逐帧类别原型 vs 冻结源原型 MSE）；动态伪标签双向投票。
- **数字**：SPin8→SPan8 79.06 mIoU（+10.22）；消融：记忆注意力开比关好（59.34→62.46），bank 9>6>3>0。
- **接口**：链式滑窗记忆可升级为 patch 邻接图（重叠区=天然边权）；uncertain 像素判定=不确定性仲裁接口；<26MB 可训参数在 4×48G 内。
- **风险**：低；但**"单帧→伪视频用记忆"叙事已被占位，不能再当贡献点，必须引用**。

### Spatial-SAM（CVPR 2025）— 单帧先例 No.2：外部记忆替换
- **机制**：轻量 3D U-Net 预计算 SDF（符号距离场）替换 SAM2 在线 logit 记忆（SDF 符号内正外负对齐 logit→接口无感替换）；检索双向邻域 t±6 且**排除当前帧**（防自耦合）；双轨半监督（1/64 标注起步，伪标签↔SDF 迭代 3 轮）。
- **数字**：OOMLM 96.51 Dice（超半监督 CPS +8.07，与全监督 SAM4EM 差 0.09）；消融：原记忆 92.62→双向 SDF 94.11→+排除自身 94.45。
- **接口**：证明记忆库只是"插槽"，内容可整体替换为离线预计算、双向可检索的结构化表示；邻域检索=K-NN 图可推广为显式图；"排除自身"=图记忆筛选现成准则。
- **风险**：低；注意它占的是"换记忆内容类型"，我们占"图组织"，不冲突。

### MPG-SAM2（ICCV 2025，武大）— 文本注入完整模板
- **机制**：BEiT-3-L 联合编码视频+文本→V/T/CLS；MPG 用 CLS 对 V cross-attn 生成逐帧伪 mask 作 dense prompt + CLS 作 sparse prompt；HGA 在 memory attention 之后、decoder 之前并联全局支路（patch 级+目标级各 1 层）；mask-text 相似度损失。
- **数字**：Ref-YT-VOS 73.9 J&F（+6.8）；新增仅 21M 参数/1.3G 显存；8×A800（MeViS）/8×4090。
- **接口**：文本→dense+sparse 双 prompt=单帧遥感文本分割的完整可搬运模板；星型全局注入可升级为图结构（文本节点↔mask 节点↔原型节点）；BEiT-3 可换 MLLM hidden states（文列 LISA/GLaMM/EVF-SAM 三路线）。
- **风险**：中⚠——**同组作者姊妹篇 RS2-SAM2 已占"遥感+文本+SAM2"（见 collision-map）**。

### SAM-I2V（CVPR 2025）— 低成本改造标杆
- **机制**：不用 SAM2 权重，SAM+时序分支（TFI）+ 记忆筛选（过去帧分 local/global 组按相似度 softmax 采样）+ 记忆当 prompt（learnable tokens 对记忆 masked cross-attn）。
- **数字**：达到 SAM2.1 的 90-91% 性能，训练成本 0.2%（4.6k vs 2.2M G·Hour，8×24G 一天）。
- **接口**：训练经济学引用依据；MSA 分组采样=学习式记忆筛选轻量先例（可推广为图邻居采样）；"记忆聚合成少量 token"=图 readout 对照基线。
- **风险**：低。

### SAMIX（CVPR 2025，HUST）— 单帧先例 No.3：跨图像参考记忆 + RL 选参考
- **机制**：冻结 SAM2 + AdaptFormer 适配器（10M，插 Hiera 各 block MLP 旁）把特征调成语义一致；记忆重定义=跨图像参考池（每图存 {E 空间记忆, e=AvgPool+MaskPool 紧凑语义嵌入}），query 经 memory attention 检索参考作 dense contextual prompt；SPNet 策略网络自回归选 T=4 参考对，GRPO 强化学习（奖励=quality+consistency+supervision 三项，stochastic beam search 采样）；HGP 层次引导（候选只含比 query 粒度更细的）。
- **数字**：VOC 75.9 mIoU（+4.1 over WISH）；**4×RTX 4090 训练（与本项目算力完全匹配）**；选择策略消融：random 70.4 < DINOv2 72.0 < 记忆嵌入相似度 72.7 < SPNet 75.9 → **学习式筛选≫启发式检索**。
- **接口**：exemplar 记忆完整实现（visual exemplar 备选路线的直接模板）；{E,e} 对=现成节点表示与动量更新算子；三奖励可改造成图更新的不确定性驱动奖励；**选参考是序列决策（链），图检索是其拓扑化升级**。
- **风险**：中（RL 选参考思想被占，related work 须对比）。

### EfficientTAM（ICCV 2025，Meta）— 效率与粒度参照
- **机制**：SAMI 预训练 vanilla ViT-T/S 替换 Hiera；记忆空间 token 局部性强→K/V 2×2 池化近似交叉注意力（30K token→7.6K）；**实测线性注意力掉 >10 J&F（关键负结果）**。
- **数字**：SA-V test 74.5（SAM2 74.7）；iPhone 15 Pro Max ~28FPS(Mobile)；训练 256×A100（不可复刻，只用权重）。
- **接口**：记忆 token 高度冗余→图节点可用区域级粗粒度（与 tile/超像素对齐）；线性注意力失败→图压缩须保留结构信息（拓扑即先验）。

---

## E. 查重专题（详见 collision-map.md）

### SCORE（ICCV 2025 Highlight，arXiv 2507.12857）— 重叠度：中（任务层高/方法层低）
首个占位 OV-RSIS 任务名。任务设定与本项目完全同构（单帧、类名→全实例、跨数据集、实例级
mAP 协议 + HF split）。方法 = FC-CLIP 范式（冻结 RemoteCLIP ViT-L + ConvNeXt-CLIP +
Mask2Former 300 query）+ 双粒度场景上下文（RAI 掩码膨胀池化注入 + GCA [CLS] 域适配分类器）；
**逐项确认：无记忆机制、无图结构、无 MLLM、无不确定性**——四条创新轴全部正交。
iSAID 训练平均 mAP 38.68；FAST 仅 ~13 mAP（细粒度类硬伤=我方机会）；单张 L40S 可训；
代码 Apache-2.0 开源（无 checkpoint）。→ 任务名引用纪律："继 SCORE 之后"；
其 benchmark/协议直接复用为对照基线。

### RS2-SAM2（AAAI 2026，arXiv 2503.07266）— 重叠度：低
Referring RRSIS（一文本→单目标单 mask）；**原文弃用记忆机制（p.5）**；创新在 BEiT-3 联合编码+BHFM adapter+MPG 伪 mask prompt；8×4090 训练，代码开源；RefSegRS test mIoU 73.90 / RRSIS-D test 66.72。→ 动机证据（连它都绕开记忆）+ 必引文献（对照，不进实验表）。

### GMN（ECCV 2020，GraphMemVOS）— 重叠度：中低
帧级 3-5 节点全连接图（借 AGNN 边构造）；读=KV attention、写=GRU 邻居聚合；图只在单帧 K=4 步内存活，跨帧重采样重建；无生命周期/语义/不确定性；DAVIS17 82.8；5 节点饱和。→ 双重引用：图记忆先驱+反面教材；设计红线见 collision-map。

---

## F. 横向结论

**四条确认的空白（16 篇记忆工作共同天花板 = 本项目创新轴）**：
1. 记忆条目间没有显式边（关系全隐式存在于注意力）
2. 压缩=删除，无人做"合并+继承"（RMem 只会删帧）
3. 读取全是扁平 attention，无子图路由检索
4. 单帧记忆重定义（类原型/tile 上下文/exemplar/文本节点异质图）完全空白

**两条叠加空白**：MLLM 语义节点（19 篇全零）；不确定性止步标量阈值（occlusion/IoU/presence），无图上传播。

**算力结论**：冻结主干+<30M 可训模块路线（SAMIX 4×4090 / OmniSAM 26MB）完全落在 4×48G；SAM3 全量不可复现（25.8 万 GPU 时），模块级训练可行。

**数据/工程备忘**：SAM2 用 WSL（Windows 环境）；torch≥2.5.1；SAM 2.1 checkpoints；SAM3 checkpoint 经 AI Playground；文本编码器 32 token 上限待验证遥感类名适配性。
