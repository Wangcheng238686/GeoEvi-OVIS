# SAM License 复核结论（S1.1 / E1）

> 复核日期：2026-09-02。对象：vendored `sam3/`（官方 repo，commit 浅克隆
> 2026-09-02）随附 `LICENSE`（"SAM License", Last Updated: November 19, 2025）
> 与 HF `facebook/sam3` 权重仓库条款。本文只记录结论与出处，不构成法律意见。

## 1. 许可证识别

- 官方 repo `LICENSE` 文件为 **SAM License**（自定义协议，非 OSI 标准许可证），
  覆盖 "SAM Materials" 全部元素：文档、模型、软件、算法、模型权重、推理/训练
  使能代码。**代码与权重同证**，没有分开的 code/weights 双许可。
- 注意：`sam3/pyproject.toml` 的 classifier 写 "License :: OSI Approved :: MIT
  License"，与 LICENSE 文件冲突，属上游陈旧元数据；以 LICENSE 文件为准
  （README §License 亦指向 SAM License）。引用本仓库许可信息时不要抄
  pyproject classifier。

## 2. 研究使用边界（对本项目的直接约束）

| 条款 | 内容 | 对本项目 |
|---|---|---|
| §1a Grant | 非独占、全球、不可转让、免版税；允许使用/复制/再分发/修改/创作衍生品 | 训练、修改、封装 sam3 均允许；**无 research-only 限制**，也无 Llama 式商业审批条款 |
| §1b.i 再分发 | 分发 SAM Materials 或衍生品必须同按 SAM License 条款，并随附协议副本 | 若将来开源含 vendored `sam3/` 的代码或微调权重，须附 SAM License 原文 |
| §1b.ii 发表致谢 | 以研究成果发表时**必须在出版物中致谢使用了 SAM Materials** | 论文须写明基于 SAM 3（Meta）；投稿前检查致谢段 |
| §1b.iii 合规 | 遵守适用法律、贸易管制与数据保护法 | 常规科研使用，满足 |
| §1b.iv 禁止逆向 | 不得逆向工程/反编译/探测底层组件 | 只依赖官方权重与公开 API 面（design-v2 §12.3 E1 缓解措施的另一半理由） |
| §1b.v 管制用途 | 不得用于军事/核/间谍等 ITAR 受限终途 | 不适用 |
| §5a 衍生品归属 | 我们的衍生品（微调权重、修改代码）归我们所有 | 训练产物归属清晰 |
| §6 终止 | Meta 可因违约终止；终止后须删除并停用 | 常规违约风险，低 |

## 3. 结论（登记）

1. **可用于本研究**：text-only / exemplar 分割研究用途在授权范围内；发表需
   致谢 SAM Materials（§1b.ii）。
2. **再分发限制**：仅当对外分发含 SAM Materials 的衍生品时生效（同条款 +
   附协议）；本仓库现阶段为私有科研仓库，不受影响。
3. **风险评级维持 E1 = 低**。

## 4. 权重获取渠道备注（如实登记）

- HF `facebook/sam3` 为 gated（manual），本机账号申请被拒，且 HF 被拒后无
  重新申请入口（huggingface_hub#2960；官方 repo issue #420 记录过表单非英文
  被自动拒的先例）。
- 当前权重取自 ModelScope 官方命名空间镜像 `facebook/sam3`
  （https://modelscope.cn/models/facebook/sam3）：`sam3.pt` 字节数与 HF 官方
  完全一致（3,450,062,241 bytes）；本地 sha256 于下载完成后记录在
  `sam3_ovis/scripts/README.md`。
- 待办：以英文表单重新申请 HF 官方访问（新账号或 GitHub issue 人工复核）；
  获批后用官方源校验和复核本地权重，本文随之更新。
