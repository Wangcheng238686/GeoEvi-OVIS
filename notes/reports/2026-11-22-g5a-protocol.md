# G5a 免费杠杆消融协议（2026-11-22，结果产生前冻结）

> 依据：三路独立审计（`2026-11-22-{independent-conclusion-audit,
> pipeline-protocol-audit, external-benchmark-survey}.md`）。目标：在不动
> 权重的前提下量化三个免费杠杆对现行管线的单独与联合贡献，重建基线，
> 并为 R1 重设判据。全程单卡（GPU0，共享卡 ~8G 余量，autocast bf16）。

## 1. 臂设计（同一 46 val 图，同 GT，同 NMS 口径）

| 臂 | prompt | pool floor | 分辨率 | 检验什么 |
|---|---|---|---|---|
| B0 | 下划线原样 | 0.05 | 1008 | 基线复现（应≈g4a：NMS mAP 0.223，center 召回 77.3） |
| L1 | **下划线→空格** | 0.05 | 1008 | prompt 格式（审计①：worst 图 +10.1pt 召回） |
| L2 | 下划线原样 | **0.01** | 1008 | 低分带候选（审计②：最难图 90.4% 漏检有覆盖） |
| L3 | 下划线原样 | 0.05 | **1344** | 有效分辨率（审计②：+16.5pt 召回） |
| COMBO | 空格 | 0.01 | 1344 | 联合 |

全部臂：逐类提示（合并提示已被证产出 0 候选）、类内 NMS（bbox-IoU
0.5 近似，g4a 同款）、presence 耦合分数原样。

## 2. 度量（每臂）

- NMS 池 COCO segm mAP（主指标）+ 相对 B0 的逐图配对 bootstrap
  （2000 次，95% CI）；
- 召回两口径：bbox-IoU≥0.5 / 中心点命中（any-class 与 same-class）；
- FP/图；候选数/图。

## 3. 判读（预登记）

- **方向性结论标准**：COMBO − B0 的 mAP ≥ **+3pt 且 CI 不含 0** →
  免费杠杆构成实质新基线，R1 在新基线上重立项；mAP 增益 <1pt 或 CI
  含 0 → 免费杠杆只改善召回不改善排序（与"FP 排序支配"一致），
  R1 判据仍按原基线、但动机表述更新；
- 单杠杆归因表如实报告（预期 L1/L3 主召回、mAP 增益主要看 COMBO）；
- L2 预期候选暴涨 → NMS 后 FP/图 与 mAP 的权衡单独解读。

产物：`sam3_ovis/graph/g5a_free_lever.py`；
候选缓存 `/data/wangcheng/geoevi-r1/freeler/{arm}/`；
结果 `sam3_ovis/logs/g5a/ablation.json`；
报告 `notes/reports/2026-11-22-g5a-free-lever-ablation.md`。
