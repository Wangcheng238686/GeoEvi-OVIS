# 外部调研：微调/适配 SAM3 到遥感域的真实增益与配方

日期：2026-11-22。为 R1（源域微调）提供外部锚点。本项目冻结基座现状：
iSAID dev 46 图，NMS 池 mAP 0.223、any-class 召回 55%、same-class 40.4%、104 FP/图。

检索方式：WebSearch/联网阅读器当月配额已尽（2026-12-01 重置），本报告改经
arXiv 检索页/API 与 curl 直接抓取原文（arXiv HTML 全文 9 篇、摘要 4 篇），
全部数字均从原文表格/正文逐字核对，未经二手转述。

## 1. 汇总表

| 工作 | 任务/数据 | 微调方式 | 增益数字 | 关键配方 | 来源 |
|---|---|---|---|---|---|
| SAM 3 官方（Meta, 2511.16719v2） | RF-100VL **Aerial** 子集检测 | 少样本 FT（10-shot）与全量 FT | **0-shot AP 24.0 → 10-shot 35.4（+11.4）→ 全量 56.9（+32.9）**；RF-100VL 全体 15.2→36.5→61.6；ODinW13 61.0→71.8→75.8 | §F.3：40 epochs、lr=标准值 1/10、batch 2、无 mask 标注时**关掉全部 mask 分量与损失**、不做 prompt tuning、用数据集原始标签做 prompt。§A.3 域适配：SAM3 伪标签+AI verifier 合成数据两步训练（先预训练再用 1:1 混合预训练数据 FT）。输入固定 1008×1008 | [arXiv:2511.16719](https://arxiv.org/abs/2511.16719) Tab.2/35/36, §A.3/§F.3 |
| GOOSE 2D 自蒸馏（2606.20130，ICRA 2026 挑战第 4） | GOOSE 2D+GOOSE-Ex 语义分割（11,234 训练图） | 部分解冻：**encoder 最后 12 个 block+特征融合**（{4,8,12,16} 搜出来，16 无增益）+轻量 FPN 解码器，非 LoRA | 最终 composite mIoU 69.73；消融：CE 基线 67.71 → 自蒸馏 +0.67 → flip TTA +0.15 → **光度增广 +0.86（单项最大）** → 多尺度 TTA +0.34 | 1008×1008 固定输入；滑窗 stride 840（~17% 重叠）Hann 融合；AdamW lr 3e-5、**encoder 再除 4**、wd 0.01、poly 衰减、batch 2、AMP、≤25 epochs、单卡 4090 8.5h；光度增广每项独立 p=0.5 开关（亮度 ±40/255、对比度/饱和度 [0.7,1.3]、色相 ±0.1）；**GT-box 提示的冻结 SAM3 当 teacher 自蒸馏**（仅 22/56 类、teacher mask 预计算、λ=0.3）；文本提示 teacher 失败（"water" 打满所有纹理区）；多尺度 {0.75,1.0,1.25} 以缩放图像而非输入实现 | [arXiv:2606.20130](https://arxiv.org/abs/2606.20130) |
| Dabaja & Celik 零样本评测（2607.09583） | iSAID 实例分割 / DIOR 检测，零微调 | 无（零样本，5 种 prompt 配置） | **iSAID：Text-Only mAP 11.85 / AP50 22.73；Box-Only 32.77；Text+Box 35.54/61.65；+Oracle 36.30/63.09**。DIOR：Text-Only 11.67 vs Box-Only 66.55；Airport 类文本 0.00 vs Text+Box 92.25。GZSI：base 59.14 / novel 30.49 AP50，超 ZoRI(47.05/9.30) | 结论：SAM3 视觉骨干强、文本对齐弱；"域微调 strictly required，既要校准文本投影层，也要为高密度小目标优化视觉编码器"。presence head 可当零样本场景分类器/过滤器 | [arXiv:2607.09583](https://arxiv.org/abs/2607.09583) Tab.IV/VI/VIII |
| 文本提示+轻量微调（2512.15564，光伏/小目标） | 4 类遥感目标（250–975 tile，目标 3.7–70.9 m²） | 全解冻 text encoder+mask decoder（视觉/几何编码器冻结），非 LoRA | **最大跳变发生在 ZS→FT100**（仅 100 个标注对象）：DT F1 0→58.1、SH 16.3→57.1；FT200→FT300 趋平或负；hybrid text+box 最优（RSt IoU 70.1→91.9）；ResUNet 全配置下被碾压 | lr 1e-5、25 epochs、batch 2；每类 **10 个候选名词短语、≤4 词（SAM3 官方建议）**，用 50 对验证样本按 F1+IoU 排序选优；"俯视外观描述性短语"（如 tilted solar panels）胜过抽象同义词；训练/推理 prompt 不一致会退化；Precision-IoU 持久鸿沟=欠分割 | [arXiv:2512.15564](https://arxiv.org/abs/2512.15564) Tab.II/III |
| VPRef（2609.16486） | 跨域 referring 分割（Vaihingen→Potsdam，46,972 三元组） | **SAM3+LoRA**：r=8、α=16、dropout 0.5，插在视觉 qkv+文本 MHA+融合层，仅动 **1.08%** 参数 | 未调 SAM3 跨域全挂；SAM3-ft 源域 mIoU 79.66、跨域 71.53（-8.11）；多粒度文本混排使目标测试 65.78→**69.76（+4.0）** | 明确动机："全量反传在无监督域适配循环下有灾难性遗忘与发散风险"；AdamW 5e-5、wd 0.01、batch 16、480×480；伪标签自训练+多粒度文本混排；文本混排防语言捷径、自训练管宏观空间对齐 | [arXiv:2609.16486](https://arxiv.org/abs/2609.16486) Tab.III/V |
| ProC-SAM3（2606.21863） | 8 个遥感 OVSS 基准，training-free | 无参数训练（prompt 界面校准） | 平均 mIoU 56.1%，**较此前最佳 training-free +3.9pp** | 离线 MLLM 生成 prompt 池+类别匹配器分组+类别先验约束；文本 embedding 缓存复用；presence 置信门控融合+保峰值聚合（保小目标细粒度激活）；动机即"单个类名 prompt 语义覆盖不足" | [arXiv:2606.21863](https://arxiv.org/abs/2606.21863) |
| ConceptBank（2602.06333） | 自然场景+遥感 OVSS，data/concept drift | 免参数校准 | 恢复 drift 下的对齐（新鲁棒性基线） | 从目标域统计构建数据集专属概念库+类级视觉原型，挖掘支撑样本抑制离群，融合候选概念纠 concept drift | [arXiv:2602.06333](https://arxiv.org/abs/2602.06333) |
| DCR-SAM（2610.27960） | 多概念同时分割 | 适配框架（解耦+密集竞争头） | 超 SOTA 2.3~10.7%，**推理比原生 SAM3 快 33×** | 解决逐概念独立推理的延迟与语义重叠误分类；可学习 background token 吸收非目标——对 15 类逐类推理的我们直接相关 | [arXiv:2610.27960](https://arxiv.org/abs/2610.27960) |
| SAHI（经典参照，2202.06934） | VisDrone/xView 航拍小目标检测 | 切片推理（免训练）/切片辅助微调 | 切片推理 +6.8/+5.1/+5.3 AP（FCOS/VFNet/TOOD）；切片+微调累计 **+12.7/+13.4/+14.5 AP** | 通用滑窗切片可叠加任何检测器，无需微调即得 ~5-7 AP | [arXiv:2202.06934](https://arxiv.org/abs/2202.06934) |

同域邻接（未展开）：ASCoR（2611.03382，SAM3 源-free 域泛化）、HyperSAM（2609.37340，高光谱侧编码器）、光伏小目标提示研究（2608.10801，hybrid 提示最优）、GeoNLI/DiffuSAM（2609.28741/2604.18201，SAM3+系综/扩散定位，无需训练）。

## 2. 通用结论：LoRA/部分微调 vs 全量

- 没有任何一篇对 SAM3 做全量 FT 的遥感工作发表；三个微调工作分别选了：部分解冻最后 12 block（GOOSE）、LoRA r=8 动 1.08%（VPRef）、只解冻 text encoder+decoder（2512.15564）。共同理由：冻结视觉主干保预训练特征、防灾难性遗忘（VPRef 明说全量反传在自训练循环下会发散）。
- GOOSE 的消融给出部分解冻的边界：12 block 够用，16 block 无增益——**可训练容量不是瓶颈，数据/增广/prompt 才是**。
- 学习率三家高度一致地小：3e-5~5e-5（GOOSE 对 encoder 再除 4）、1e-5（2512.15564）、SAM3 官方少样本用标准值 1/10；epochs 25~40；batch 2~16。
- SAM3 官方 §A.3 的另一条路线：**零人工标注域适配**——用冻结 SAM3 伪标签+AI verifier 清洗的合成数据，两步训练并按 1:1 混入预训练数据；合成数据随规模最终追平人工标注。R1 若标注预算受限可降级为此路线。

## 3. 小目标召回：分辨率/多尺度/切片的量化

- SAM3 输入固定 1008×1008（论文 §C.4.2，训练用 crop/resize 变化长宽比并"帮助小目标"，评测时不保长宽比直接 resize）。iSAID 800×800~4000×13000 原图整图压到 1008 会杀死小目标——GOOSE 的做法是**滑窗 1008/stride 840（17% 重叠）+ Hann 加权融合**。
- 多尺度 TTA 量化：GOOSE {0.75,1.0,1.25} 图像级缩放 +0.34 mIoU（放在全部训练技巧之后仍有净收益）；SAHI 在航拍检测上切片推理免训练 +5~7 AP。
- Dabaja & Celik 把 SAM3 的失效归因于"有限分辨率阈值+跨模态失配"两个正交因素；2512.15564 显示最小目标（DT 3.7 m²）零样本 text-only IoU 为 0，FT100 后到 41——**小目标主要靠微调+切片救，不靠 prompt 救**。

## 4. 类名 prompt 处理的已知影响

- 单类名 prompt 语义覆盖不足是遥感 OVSS 的公认首要问题（ProC-SAM3 立论；+3.9pp 的收益全部来自 prompt 侧）。
- SAM3 官方建议**短名词短语（≤4 词）**；2512.15564 实测：描述俯视外观的短语（"tilted solar panels"、"dark circular void in soil"）显著优于抽象类名/地面视角词；训练与推理 prompt 不一致会退化（FT300 时 text 召回反降，疑过拟合窄 prompt 分布）。
- RF-100VL 官方协议直接用数据集原始标签、不做 prompt tuning——Aerial 0-shot 24.0 即"原始类名直出"的水平；iSAID 类名带下划线（storage_tank）与 DIOR 的 Airport 文本 0.00 vs 框提示 92.25 说明**词面-域失配是零样本文本 grounding 崩坏的主因之一**，微调官方数字中相当一部分增益（+11.4@10-shot）可理解为修这个失配。
- 免训练替代：MLLM 生成每类 prompt 池+小验证集排序（2512.15564：50 对即可）+presence 置信门控（ProC-SAM3）——prompt 池/embedding 缓存一次离线完成，推理零额外文本编码开销。

## 5. 本项目 R1 的合理预期区间

参照系换算：Dabaja & Celik 在 iSAID **全 val**（458 图）标准协议下 text-only mAP 11.85；本项目 46 图 NMS 池协议 mAP 0.223、any-class 召回 55%，属召回导向的宽松池评估，数值天然偏高，不能直接对比绝对值，但三条外部曲线给出增益倍率：

- 官方 Aerial：10-shot（每类 10 图）+11.4 AP（×1.48），全量（百图级/类）+32.9 AP（×2.37）。
- 2512.15564：百级标注对象即拿到 ZS→FT 的 80% 以上增益，300 对象后趋平。
- GOOSE：11k 图部分微调即达 69.7 mIoU，其后所有推理期技巧合计只再 +1.5。

对 R1（iSAID train 1411 图、15 类、有 mask 标注，LoRA 或部分解冻）：

- **mAP 0.223 → 0.35~0.50**（×1.6~2.2）：微调主收益来自修文本对齐+小目标；若低于 0.32（×1.45）应视为配方失败而非数据不足。
- **any-class 召回 55% → 70%~85%**：召回上限由切片/分辨率决定（滑窗 1008/840 后小目标才进有效感受野）。
- **same-class 40.4% → 55%~70%**：类间混淆（如 vehicle 三兄弟）主要靠文本投影校准消解。
- **104 FP/图 → 30~60/图**：presence 置信门控（ProC-SAM3 思路）+FT 后阈值重校准；FP 不会归零，GZSI 数字显示 base/novel 差距会长期存在。

## 6. 三个最值得抄的配方要素

1. **部分微调 + 小学习率 + 独立开关光度增广**：解冻 encoder 末 12 block（lr 再除 4）+decoder 或 LoRA r=8（1.08% 参数），AdamW 3e-5~5e-5、25~40 epochs、batch 2~16；GOOSE 证明每项独立 p=0.5 的光度增广是单项最大增益（+0.86 mIoU，个别类 +21 IoU），成本为零。iSAID 有 mask 标注，不必像官方少样本那样关 mask loss。
2. **滑窗多尺度推理与微调解耦**：1008 窗/840 stride/Hann 融合 + {0.75,1.0,1.25} 图像级多尺度 TTA（GOOSE +0.34；SAHI 免训练 +5~7 AP）。iSAID 大图必须切片，这是召回 55%→80% 档的主引擎，且可与任何微调状态叠加、随时上线。
3. **prompt 池 + presence 门控（先于/伴随 R1）**：每类 ≤4 词、俯视外观描述的名词短语池（MLLM 生成、50 对验证集排序、embedding 离线缓存），类名弃用下划线原样直出；推理用 presence 置信门控砍 FP（ProC-SAM3 +3.9pp、免训练）。进阶可加 GOOSE 式 GT-box teacher 自蒸馏（冻结 SAM3 自教，+0.67）与官方 §A.3 的 1:1 混源数据防遗忘。

## 附：来源清单

- SAM 3 官方论文 <https://arxiv.org/abs/2511.16719>（v2, 2026-03-28；Tab.2/35/36、§A.3、§C.4、§F.3）
- GOOSE 2D 自蒸馏 <https://arxiv.org/abs/2606.20130>
- SAM3 遥感零样本评测 <https://arxiv.org/abs/2607.09583>
- 文本提示+轻量微调 <https://arxiv.org/abs/2512.15564>
- VPRef 跨域基准 <https://arxiv.org/abs/2609.16486>
- ProC-SAM3 <https://arxiv.org/abs/2606.21863>（代码 <https://github.com/YanghuiSong/ProC-SAM3>）
- ConceptBank <https://arxiv.org/abs/2602.06333>
- DCR-SAM <https://arxiv.org/abs/2610.27960>
- SAHI <https://arxiv.org/abs/2202.06934>
