# G2b 诊断协议：记忆路线失败的根因分离（2026-11-20）

> 上承 G2a 否定结论（`2026-11-17-g2a-results.md`）。负责人指令：实施 A，
> 排查更深层原因——"路线为什么失败"。本阶段**诊断优先**：先用廉价实验
> 分离四个候选假设，再决定修复方式。资源：四卡已空闲（24GB/卡），诊断
> 全程 no_grad 前向 + CPU 探针，单卡 ~4GB。

## 1. 候选假设与可检验预言

| 假设 | 内容 | 可检验预言 |
|---|---|---|
| **H1 neck 失配** | writer/reader 预训练于 tracker(SAM2) neck 分布；G2a 喂 detector FPN 特征 → 记录为分布外噪声 | D1：det 与 trk 特征逐位余弦/尺度差异大；D2：原生路线上 Δ 幅度与结构显著强于现路线 |
| **H5 差分病理** | M 臂注入 `out_real − out_null`，减掉了 N 臂获益的通用画布信号 → M = N + 噪声残差，M−N 差距被设计混淆 | D3：cos(pool(real), pool(null)) ≈ 1 且 Δ 幅度远小于分量 |
| **H3 信息缺失** | 记忆记录本身不含线性可解码的类别判别信息（机制前提弱） | D4：线性探针 probe(real) ≯ probe(null)/probe(canvas) |
| **H2 logit 分布** | ±10 硬 mask logits 偏离原生契约 | 次要：D2 附带检查记录统计是否异常 |

判定逻辑：H1/H5/H3 不互斥；按证据强度决定 G2b 修复组合（neck 适配 /
改注入设计 / 止损）。**诊断不构成任何机制主张的正向证据**，只定位失败
原因。

## 2. 诊断设计（D1–D4，全部未训练的预训练权重上）

- 样本：G2a 缓存的 30 图（train 18 / val 122 目标 + bank 候选），同一批
  目标/掩码/文本匹配复用，保证与 G2a 可比。
- 特征对：同一图同一次 `forward_image`（fp32，no_grad）产出
  ① detector FPN 最近 63 的层（G2a 现路线，实测 72×72）与
  ② `sam2_backbone_out` 的方形网格层（tracker 原生 neck）。

**D1 分布差距**：两组特征的通道均值/方差、逐位归一化余弦、RMS 比。
**D2 writer/reader 响应**：同 bank 掩码分别在 det/trk 特征上建记录
（记录 RMS/通道 std）；reader 在四种组合（src∈{det,trk} × rec∈{det,trk,
null}）下的 Δ 幅度与 Δ 的通道相关性——原生组合 vs 现组合的响应比。
**D3 差分病理**：对 val 目标，pool(out_real)、pool(out_null)、pool(Δ) 的
范数与两两余弦（det 与 trk 两条路线各算一遍）。
**D4 线性探针（信息含量）**：每目标取 pool(out_real) / pool(out_null) /
pool(canvas) 三种 256 维特征，logistic regression（train 171 目标训练、
val 122 评估）分别报 top-1。probe(real)−probe(null) 为记忆内容信息量的
线性下界。

## 3. 判读（预登记）

- H1 成立条件：D1 逐位余弦 < 0.5 且 D2 原生路线 Δ 幅度 ≥ 3× 现路线；
- H5 成立条件：D3 cos(real, null) > 0.95 且 |Δ|/|real| < 0.3（两路线一致）；
- H3 成立条件：D4 probe(real) − probe(null) < +2pt（线性下界无信息）；
- 修复映射：H1→G2b 加 neck 适配（或直接用 trk 特征）重测 M' vs N；
  H5→M' 改为注入 out_real（不差分）；H3 单独成立→机制前提不足，
  止损论证成立，不再投入。

## 4. 边界

- 诊断用未训练 reader/writer（预训练原样），反映"信息存在性/分布"，
  不直接预测训练后行为；
- trk 特征网格若非方形或与 72 不一致，掩码/记录按其网格重采样（登记）；
- 全部数字落 `logs/g2b_diag/diag_summary.json` + 报告
  `notes/reports/2026-11-20-g2b-diagnosis.md`。
