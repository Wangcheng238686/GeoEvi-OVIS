# 设计指导文档

本目录用于跟踪开放词汇遥感实例分割方案的设计收敛与后续迭代。

## 当前版本

- 当前方向决定：[`decisions-2026-10.md`](./decisions-2026-10.md)（图与 SAM3 原生记忆、同图上下文、允许遥感微调）
- 当前查重：[`关系上下文记忆调研`](./reports/2026-10-25-sam3-relational-memory-prior-art.md)
- 候选探针：[`三条图记忆路线草案`](./reports/2026-10-25-graph-memory-candidate-routes.md)（关系选择 / 角色区分 / 冲突门控，尚未实现）
- 9 月大图方案设计：[`design-v4.0.md`](./design-v4.0.md)（TG-CWP 历史收敛方案，供参考）
- 9 月大图方案实施：[`implementation-v4.0.md`](./implementation-v4.0.md)（S0 准入 2026-10-26 判不足已止损，见 [`reports/v4-s0-c0-verdict.md`](./reports/v4-s0-c0-verdict.md)）
- v3 收尾设计：[`design-v3.0.md`](./design-v3.0.md)（已关账，2026-10-26：负结果，见 [`reports/v3-frozen-eval-results.md`](./reports/v3-frozen-eval-results.md)）
- v3 收尾实施：[`implementation-v3.0.md`](./implementation-v3.0.md)（不再扩展滑窗或 memory）
- 面向项目负责人的现状说明：[`project-explainer.md`](./project-explainer.md)（每次项目解释后同步维护）
- 状态：2026-10-25 重新访谈与调研；核心约束已确认，具体新设计与并行探针尚待形成
- 主任务：遥感开放词汇实例分割；大图切窗作为可选观测场景
- 主基座：SAM3
- 核心构想：图组织同图实例与环境上下文，控制原生记忆的写入/读取，帮助当前实例识别与分割
- 适配边界：尽量保留原生 memory encoder / attention；接受小幅适配与适当主干微调
- 记忆边界：第一版仅同图内，不保留跨测试图的样本记忆；参考 CA-SAM2 组织方式
- 风险登记：[`design-v2.md` §12](./design-v2.md)（2026-09-02 可行性分析）

## 历史版本

- v2：[`design-v2.md`](./design-v2.md) / [`implementation-v2.md`](./implementation-v2.md)，保留原始阈值与结果；后续推进改按 v3.0。

- v1 旧设计与调研材料：[`v1/`](./v1/)
- 该目录由原 `notes/` 整体迁移而来；其中与当前 v2 不一致的内容均视为历史方案，不作为现行设计口径。
- v1 为历史设计，无对应实施文档。

## 设计—实施对应规则

- 每版 `design-vX.Y.md` 对应一份 `implementation-vX.Y.md`；设计迭代后实施文档按新版本重建，不跨版本续用。
- 设计文档只定科学边界（做什么、为什么）；实施文档只定执行步骤（怎么做、判据、产出）。
- 冲突时以设计文档为准；实施中确需突破边界，先按迭代规则开新设计版本。

## 迭代规则

后续遇到实验问题时，新增版本文件，例如：

```text
design-v2.1.md
design-v3.0.md
```

每次迭代记录：

1. 触发问题或新证据；
2. 原设计的失效位置；
3. 修改内容；
4. 对创新性、任务定义和实验可比性的影响；
5. 是否需要重新核对相关工作。

不要直接覆盖旧版本。重大方向变化使用主版本号，局部调整使用次版本号。
