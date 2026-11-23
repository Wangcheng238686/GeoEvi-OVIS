# EXP-0001 结论：R1 v2 源域 LoRA 微调——PARTIAL（2026-11-23）

> 协议 `2026-11-22-r1-protocol-v2.md`（§1 判据先冻结；§5 偏离先登记）。
> 启动 commit 2541e3f / tag exp-EXP-0001；审计 FAIL→修复→复审 PASS 全档
> `docs/reviews/2026-11-22-r1v2-implementation-audit.md`。
> 训练：DDP-4，5 epoch（36 epoch 等效），10.0h，每 rank 9.0GB（11GB 上限
> 内；一次 23.5GB 卡顶报警，无 OOM）。评估：last-checkpoint（09:21 落盘
> 即所评），117 val 组图，6,721 GT。产物：`logs/g5a/{ablation.json,
> r1v2_verdict.json}`、ckpt `/data/wangcheng/geoevi-r1/run1/checkpoints/`。

## 1. 主结果（vs COMBO 基线，预登记口径）

| 指标 | COMBO | R1v2 | 门 | 判 |
|---|---|---|---|---|
| segm mAP（主，池化） | 0.2700 | **0.3341** | GO≥0.39 / PARTIAL≥0.30 | PARTIAL 区间 ✓ |
| R1v2−COMBO 逐图配对 | — | **+3.24pt CI[2.14, 4.57]** | CI 不含 0 | ✓ |
| same-class 召回 | 51.51%* | 45.5% | ≥45% | ✓（见*口径注） |
| FP/图 | 289.8 | **440.3** | ≤350 | **✗ 越护栏** |
| center 召回 | 95.25% | 96.13% | 不回归 | ✓ |

**判定：PARTIAL**（mAP∈[0.30,0.39) 且 CI>0；FP 护栏失守同时成立——
预登记门为合取，PARTIAL 条款独立成立）。

***基线口径注（如实登记）**：协议 §1 引用的 same-class 基线 31.8% 来自
交接文档；但奇偶校验所锚定的存档 `ablation.json`（与 mAP 0.2700 同源
同批）给出 **51.51%**。按权威存档口径，微调的 same-class 实为**回退
6.0pt**，而非"从 31.8% 升至 45.5%"。45.5%≥45% 的门在字面与预登记数字
下仍判通过，但其解释必须以 51.51% 为基线。

## 2. 解读

1. **项目首个显著正向机制结果**：mAP +6.41pt（池化）/ +3.24pt（配对，
   CI 不含 0）——此前全部机制线（v1/v3/v4/G2a/G3b）为零或负。梯度
   访问主导损失通道（G4a 审计的结论）首次兑现为显著收益。
2. **代价结构**：raw 候选 528→740/图、FP +150/图、same-class −6pt——
   微调推高了整体发射率与分数质量（排序改善），但 IoU0.5 口径的
   类别正确率回退。mAP 升与 same-class 降并存说明收益集中在**分数
   排序/校准**，而非分类边界。
3. 距 GO（0.39）差 5.6pt；FP 护栏是最明确的可修复缺口（440→350 需
   压 20% FP，同时保住 mAP）。

## 3. 后续（PARTIAL 条款：配方调参一次，不另设新门；二选一待负责人裁定）

- **T1（训练侧，协议菜单内：lr/解冻/增广）**：lr 5e-5→2e-5 + 增广
  p 0.5→0.7（压过拟合发射率），同门重跑一轮（~10h×4 卡）。
- **T2（评估侧，协议 NO-GO 分支的校准线前移使用）**：冻结本 ckpt，
  presence/温度/NMS 重标定攻 440 FP/图——不训不烧卡，若校准后
  过门可并入 GO 证据（需登记口径：GO 门指标在校准后分数上重算）。
- 过任一门后补 2 seed 稳定性验证（协议 §1）。

## 4. 归档

- 结论 MD：本文件；判定原文见 `r1v2_verdict.json`（VERDICT: PARTIAL）；
- ckpt：`/data/wangcheng/geoevi-r1/run1/checkpoints/checkpoint.pt`（3.4GB，
  数据盘，不进 git）；
- 训练日志：`/data/wangcheng/geoevi-r1/run1.out` + tensorboard 同目录；
- EXP-0001 状态更新为 PARTIAL-完成（registry 同步）。
