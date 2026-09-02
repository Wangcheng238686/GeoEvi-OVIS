# 设计指导 v2.0

> 状态：已收敛的概念设计稿。本文记录研究设计边界，不规定具体代码实现。

## 1. 研究目标

在 SCORE 定义的开放词汇遥感实例分割任务上，研究如何利用同一图像中的实例、区域和场景证据，提升 text-only 条件下的实例识别、实例完整性和相似类别区分能力。

## 2. 主任务

主任务保持 SCORE 的 text-only 设定：

```text
遥感图像 + 类别短语 → 该类别的全部独立实例 mask
```

主结果使用实例级评测。GT exemplar 和 self-exemplar 不改变主任务，只作为增强模式。

## 3. 总体架构

```text
图像 + 类别短语
        ↓
SAM3 text-only 概念分割
        ↓
初始实例候选
        ↓
同图实例—上下文证据图
        ↓
支持、竞争、场景一致性和未解释区域推理
        ↓
实例可靠性与预测完整性判断
        ↓
可选 GT/self exemplar 增强
        ↓
SAM3 visual exemplar 分割
        ↓
最终实例 mask、类别和置信度
```

## 4. 基座边界

SAM3 是主分割基座，负责概念级实例发现与视觉 prompt 分割。CLIP 不作为必需的主 proposal generator，可作为语义相似度校验、候选表示或对照分支。

MLLM 暂不进入第一版主模型，避免与 SAM3 Agent、ProC-SAM3 等 prompt expansion/agent 路线重叠。后续如加入，应作为独立扩展重新评估创新性。

## 5. 证据图

### 5.1 节点

- 类别节点：输入的开放词汇类别；
- 候选实例节点：SAM3 产生的实例候选；
- 局部上下文节点：实例周围的区域和背景；
- 全局场景节点：机场、港口、城市等场景证据；
- unknown/background 节点：无法可靠归类或存在冲突的候选；
- exemplar 节点：只在 GT/self exemplar 扩展模式中出现。

### 5.2 关系

- 文本—实例匹配；
- 同类实例支持；
- 相似类别竞争；
- 实例—局部上下文兼容；
- 实例—全局场景兼容；
- 实例—区域覆盖；
- 实例—unknown 冲突；
- exemplar—候选相似性。

### 5.3 图的科学角色

图不是新的分割器，也不是普通的 self-attention 或 NMS。图负责组织和联合判断：

- 哪些候选互相支持；
- 哪些候选属于相似类别竞争；
- 哪些候选与上下文不一致；
- 哪些区域已经被解释；
- 哪些区域可能存在漏检；
- 哪个候选适合作为视觉 exemplar。

## 6. 记忆定义

第一版的 memory 定义为“同图动态实例证据记忆”，包括：

- 已确认实例证据；
- 上下文支持证据；
- 类别冲突证据；
- 已解释与未解释区域；
- unknown 候选；
- exemplar 候选。

它不是 SAM2/SAM3 的视频 temporal memory，也不是跨图长期 prototype bank。

## 7. Exemplar 扩展

### GT exemplar

用于测量可靠视觉示例带来的理论上限。该结果必须与 text-only 主结果分开报告。

### Self-exemplar

由 text-only 候选经证据图选择得到，用于检验模型能否自动发现可靠视觉证据。

SAM3 原生已经支持 visual exemplar，因此创新点不在“使用 exemplar”，而在“图驱动的 exemplar 发现与验证”。

## 8. 完整性建模

除实例 mask 质量外，方案还关注类别实例是否可能漏检：

- 哪些区域已有可靠实例解释；
- 哪些区域仍有未解释响应；
- 哪些候选重复或互相冲突；
- 当前类别的预测是否可能不完整。

完整性是方法的主要差异化方向之一，但不能用场景共现规则强行生成目标。

## 9. 论文贡献框架

1. 同图实例—上下文证据图：显式建模文本、实例、场景、支持、竞争和 unknown 关系。
2. 面向实例完整性的证据记忆：联合判断已解释区域、未解释区域和候选冲突，而非只依赖单实例置信度。
3. 图驱动的视觉 exemplar 增强：从 text-only 结果中自动发现可靠视觉证据，辅助 SAM3 改善困难实例分割。

三个贡献是一条链：

```text
关系组织 → 证据完整性判断 → 视觉证据增强
```

## 10. 相关工作边界

- SCORE：任务、跨数据集协议和实例级 mAP来源；其区域/全局上下文不是显式实例证据图。
- SAM3：原生 text/exemplar 概念分割基座。
- FSS-SAM3：support/query few-shot exemplar 强基线。
- ELFSS-AR：遥感、SAM3、prototype/entity/semantic refinement 强近邻，但主要是 few-shot semantic segmentation。
- ProC-SAM3：遥感 SAM3、prompt pool、MLLM prompt expansion 和 confidence fusion 强近邻。

不能声称首次使用 SAM3、exemplar、prototype、MLLM 或 segmentation graph。

## 11. 后续收敛时必须保持的红线

- text-only 必须是主任务；
- 图在没有 exemplar 时也必须有独立作用；
- semantic segmentation 与 instance segmentation 不混淆；
- GT exemplar 不得冒充公平的 text-only 结果；
- 上下文只能作为软证据，不能成为硬类别规则；
- MLLM、跨图持久记忆和视频记忆不进入第一版主线；
- 新增模块必须说明它解决的具体科学问题以及与已有工作的差异。

## 12. 风险与隐患登记

> 2026-09-02 可行性分析结论记录。本节是风险登记附录，不修改 §1–§11 的设计
> 边界，也不构成设计版本迭代，故不新建版本文件；实验证据出现后按
> `notes/README.md` 迭代规则处理。SAM3 域弱点数字引自
> `notes/v1/v1-findings.md`（SAM3 论文 p.48-49）。

### 12.1 前置结论：换轨而非增量改进

SAM3 text-only 主线与当前 C5-v2 代码库（SAM2 Hiera + PAFPN 检测器 + ECPG +
P2-BRR，WHU-1024 封闭类）在模型层零交集。v2 按新主线实施：

- 复用工程契约（自描述 checkpoint schema、`cfg.model` 契约、单一环境配置、
  文档同步流程）、COCO 实例评测工具、loader 骨架（`whu_instance_dataset.py`
  的类别映射机制）与 tests/scripts 组织模式；
- 不复用 `geoevi_ovis/models/` 任何模块；红线未要求 SAM2 机制对照，无回归路径；
- C5-v2 WHU 线原地冻结，作为工程契约验证场与后备论文线，不做迁移改造。

### 12.2 科研风险

| 编号 | 风险 | 等级 | 关联 | 检验/缓解 |
|---|---|---|---|---|
| R1 | SAM3 text-only 遥感候选召回未知，是全部图推理的输入上限；SAM3 航拍零样本已知偏弱（RF-100VL Aerial AP 24.0） | 高 | §3 | Phase 0 零样本基线（SCORE 测试集 + NWPU-Refer category 子集），测查准/查回/mask AP；召回不足先做域适配，不直接上图 |
| R2 | 证据图独立增益不显著，启发式图退化为高级 NMS/后处理，贡献 1 失效 | 高 | §9 贡献 1；§11 红线 2 | 图消融：无边（逐实例独立打分）vs 有边（联合推理）；training-free v0 先行，正信号后再学习化 |
| R3 | 完整性（漏检/未解释区域）评测协议需自建，SCORE 仅提供实例 mAP；自定指标的审稿接受度不确定 | 中 | §8 | 指标只作 SCORE 主协议之外的补充报告，不改动主结果口径 |
| R4 | 撞车：ProC-SAM3（prompt expansion + confidence fusion）与证据图置信度推理最近；FSS-SAM3 已占 exemplar 路线 | 中 | §10 | 持续查重；方法表述维持“text-only 实例级 + 同图证据图”差异轴 |
| R5 | self-exemplar 循环性：候选与 exemplar 同源 SAM3，增益可能集中于 mask 质量而非实例发现 | 中 | §7 | 消融拆分“实例发现增益 vs mask 质量增益”；exemplar 结果与 text-only 分开报告（红线 4） |
| R6 | 可学习图模块过拟合闭集训练分布（iSAID/SIOR），SCORE 跨数据集测试失效 | 中 | §9 | 先以 training-free 版本定增益上限；学习化后必须报告跨数据集差值 |

### 12.3 工程与数据风险

| 编号 | 风险 | 等级 | 缓解 |
|---|---|---|---|
| E1 | SAM3 为自定义 SAM License（非 Apache/MIT），条款可能影响成果发布形式 | 低 | 发表前复核 license 条款；只依赖官方权重与公开 API 面 |
| E2 | 大幅遥感图（1024–4096px）上 SAM3 推理显存/延迟未实测 | 中 | Phase 0 顺带实测整图/分块/滑窗三种推理开销 |
| E3 | SAM3 文本编码器 32 token 上限对遥感类名的影响未验证 | 低 | Phase 0 一并验证（v1 待办 #5）；RS 类名多为短名词短语，预计影响有限 |
| E4 | SCORE 无 checkpoint，受控对照需自行复训 | 中 | 对照或引论文数字或自行复训（L40S 级，现有算力内）；协议引用注明来源 |
| E5 | SCORE 部分测试集 GT 为 SAM 伪标签 | 中 | 论文按测试集逐个标注 GT 来源（v1 既定约定） |
| E6 | 五个数据集（iSAID/SIOR/NWPU-VHR-10/SOTA/FAST）获取与类别对齐工作量 | 低 | 全部公开；loader 从现有类别映射机制扩展 |

### 12.4 设计留白待收敛

- 全局场景节点的场景语义来源（§5.1）未定。候选方案为 CLIP 场景校验，与 §4
  “CLIP 可作语义相似度校验”的边界吻合；实现不得升级为硬类别规则（红线 5）。
- 完整性判断的量化指标（§8）未定。候选为已解释区域覆盖率、未解释响应密度、
  重复/冲突候选率三类统计，需给出实例化定义与阈值来源。

### 12.5 Go/No-Go 门槛

Phase 0（1–2 周）：SAM3 text-only 零样本在 SCORE 测试集上的查准/查回/mask AP
作为整条主线的 gate。判据须在实验开始前登记，避免事后调整：召回阈值以 SCORE
已发表最弱基线为参照设定；若召回不可用，先解决域适配（SAM3 论文 §A.3 配方）
再进入 Phase 1 证据图开发。

