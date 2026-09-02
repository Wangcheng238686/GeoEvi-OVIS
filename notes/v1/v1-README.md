# 项目知识库与代理系统说明

> 研究项目：图结构记忆组织 × SAM2/SAM3 × 单帧遥感开放词汇实例分割
> 建立日期：2026-09-01

## 系统架构

```
主代理（协调者，唯一全局视图）
 ├─ notes/ 知识库（文件落盘 = 无限上下文，跨会话存活）← 本目录
 │   ├─ README.md          本文件：系统规范
 │   ├─ findings.md        19 篇论文精读结论（每篇一张压缩卡片）
 │   ├─ collision-map.md   撞车地图：威胁排序 + 划界表 + 动机证据链
 │   └─ decisions.md       用户拍板的方向决定（只记用户确认过的）
 ├─ papers/                原始 PDF + _extract/ 全文提取件（精读原始材料）
 └─ 可续接专家代理（agentId 续接，上下文累积）见下方名册
```

## 代理名册（2026-09-01 会话创建）

| 角色 | agentId | 已读内容 | 备注 |
|---|---|---|---|
| 基座专家（SAM2/SAM3） | `agent_d65729b8-1317-4822-8415-bca84d7a7214` | SAM2 42p + SAM3 78p | 已验证可续接，细节记忆完整 |
| 记忆管理专家 | `agent_acb9f6b8-5d5c-4b4c-9dcd-b388ebc10af3` | SAMURAI/DAM4SAM/SAM2Long/RMem | |
| 记忆读取专家 | `agent_29272d8f-0b1a-4a95-b774-92eb7eb77ba3` | Cutie/XMem/LOMM/LiVOS | |
| 领域改造专家 | `agent_519c31b2-9175-402b-b02d-93738d9c633a` | OmniSAM/Spatial-SAM/MPG-SAM2/SAM-I2V/SAMIX/EfficientTAM | |
| RS2-SAM2 查重员 | `agent_552f8874-326e-4e25-9fa8-d2ce04b164f0` | RS2-SAM2 | |
| GMN 查重员 | `agent_f5bbc234-fa68-40da-be7a-5ea21e1c806a` | GMN ECCV2020 | |

**重要限制**：agentId 与会话存储绑定（见 `~/.zcode/cli/agents/sess_*/`），跨会话可能失效。
因此**文件层是唯一跨会话保证**，代理层是会话内的效率优化。新会话恢复项目 =
读 `notes/` 三件套，而不是依赖旧代理。

## 文件维护规范（所有代理必须遵守）

1. **强制落盘**：任何代理产出新结论（读完新论文 / 用户拍板新决定 / 撞车评估变化），
   必须写入对应文件后才算任务完成。findings 每篇一个 `###` 节，不动别人的节。
2. **decisions.md 只记用户明确拍板的事**，代理的建议不进此文件。
3. **来源可溯**：卡片中关键数字/引句标注论文页码或章节。
4. 原始 PDF 不移动、不重命名；新论文放 `papers/`，提取文本放 `papers/_extract/`。

## 项目环境备注（2026-09-01 确认）

- 项目根目录为**用户已有工程**：git 仓库（PromptMiner / WHU fast-150 训练线 /
  论文 main5 制图记录）、`C5_V2_METHOD.md`（C5-v2 方法：ECPG 粗掩码转提示生成器 +
  P2-BRR 边界残差修正器，输出 SAM2 可消费的点/框/dense mask prompt，待消融）、
  `MIGRATION_MANIFEST.md`（自 RSPrompter 系代码迁移：PAFPN/RoIHead/SAM2 MaskHead 接线）。
- **用户拍板**（详见 decisions.md）：C5-v2 是后续实现基座；方案收敛前**任何代理不得
  读改项目代码**；实施阶段任务外包给其他 agent。
- 主代理当前职责 = 调研 + 方案设计 + 收敛讨论。findings 中"RSPrompter 系文献另行调研"
  一条已被自家代码部分覆盖。

## 新代理冷启动协议

任何新派发的代理，prompt 末尾附上：
1. 先读 `notes/README.md`（本文件）→ `notes/findings.md` 相关章节 → `notes/collision-map.md`
2. 研究方向一句话：图结构记忆组织 + MLLM 语义记忆 + 不确定性方法，
   应用于单帧遥感开放词汇实例分割（文本类名 prompt 优先，exemplar 备选），算力 4×48G，目标发论文
3. 产出必须回写知识库
