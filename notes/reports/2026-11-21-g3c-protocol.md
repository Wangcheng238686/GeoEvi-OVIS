# G3c 协议：冻结基座可训适配器的确认实验（2026-11-21）

> 对象：G2a N 臂副产品——冻结 SAM3 + 可训小路径（bridge+reader+writer
> 7.4M 参数，实际有效：reader 空库变换 + bridge），CE 训练后 val top-1
> +2.5~+4.1pt（3 种子同向，p=0.06–0.25，n=122，30 图）。
> 任务：在 **100 图缓存**（773/401 目标，train 54/val 46 图）上确认两个
> 口径：目标级（统计功效升级）与**数据集级 segm mAP**（G3b 教训前置）。

## 1. 设计

- 训练：与 G2a N 臂完全同机制（Δ=reader(canvas, 零bank)，pool→bridge→
  注入匹配 query→DPS 15 文本分数→CE vs GT 类）；AdamW 1e-4、批 4、
  **3000 步**（数据 ×4.5，步数 ×2，登记：≈15.5 epoch）、3 种子。
- 推断扩展到全候选（mAP 口径）：候选 j 在文本 t 下的适配分数
  s' = scale·(hs_proj(q_j+dq_j)·P_t) 再 clamp——hs_proj 线性故可批量：
  s' = clamp(scale·(hs_proj(q)·P_t + hs_proj(dq_j)·P_t))。每候选 dq_j
  = bridge(pool(Δcanvas, mask_j))；候选与 query 按框 IoU≥0.5 匹配
  （G3b cands 缓存的 mask + g2a 缓存的 per-text boxes/queries）；
  未匹配候选保留基线分数（登记）。

## 2. 度量与判定（预登记）

val（组级隔离，46 图）：
- **M-target**：401 目标 top-1 提升 + 逐目标配对符号检验（3 种子）。
- **M-mAP**：全候选 COCO segm mAP（A_adapt vs A_base）+ 逐图 AP
  源图组配对 bootstrap。

| 结果 | 判定 |
|---|---|
| M-mAP ≥ **+1.0pt** 且 CI 不含 0，且 M-target ≥ +2pt 且 p<0.05（3 种子同向） | **适配器线确认成立**，值得发展为正式贡献候选 |
| M-target 成立但 M-mAP ≈ 0 | 与 G3b 同命（目标级不传导 mAP），B 线诚实关闭 |
| M-target 在大样本下消退（<+2pt 或方向不稳） | 30 图结果为过拟合，B 线关闭 |
| M-mAP < 0 且 CI 不含 0 | 适配器损害 mAP，B 线关闭 |

## 3. 边界

- reader 空库变换 + bridge 无记忆内容（G2b 已证零信息差），本实验
  测的是**纯可训容量效应**——结论措辞不得含记忆/图主张；
- GT 仅作训练标签与评测，不进推理；mAP 分母含全部候选（含 FP）；
- 候选-query IoU 匹配失配率登记（预期 <5%）。

产物：`sam3_ovis/graph/g3c_adapter_confirm.py`；结果
`logs/g3c/confirm.json`；报告 `notes/reports/2026-11-21-g3c-adapter-confirm.md`。
