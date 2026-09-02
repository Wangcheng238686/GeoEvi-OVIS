# eval/ — SCORE 评测管线

SCORE 跨数据集口径（implementation-v2.md S1.4）：基于 `geoevi_ovis/utils/` 的
COCO 实例评测薄封装（只读复用，不改原文件），实现于 `score_eval.py`。

## 口径登记（2026-09-02，S2.1 gate 前对照论文表格复核）

- **主指标**：每测试集 COCO 实例 segm AP@[.50:.05:.95]（pycocotools 默认：
  maxDets=[1,10,100]，areaRng=all），类平均；跨数据集 "Average" = 各数据集
  mAP 无权平均。
- **依据**：SCORE（ICCV 2025, arXiv 2507.12857）§4.1 "We follow previous work
  [39] to evaluate … with mean Average Precision (mAP) for open vocabulary
  instance segmentation"，即 FreeSeg/FC-CLIP 一脉的 COCO 式口径。论文未细化
  maxDets 等，采用 pycocotools 默认值——本登记即为预登记口径，改动须留痕。
- **逐类 AP**：从 COCOeval precision 张量 [T,R,K,A,M] 对 T/R/A(全)/M(首) 维
  求均值，-1 哨兵剔除；无 GT 类返回 nan。

## 自用诊断量（R3；只作补充报告，不进主口径）

| 量 | 定义（S1.4 登记文字的实例化） |
|---|---|
| recall proxy | 逐类：GT 实例被任一同类候选以 IoU≥0.5 覆盖的比例 → 类平均（micro 变体一并输出） |
| precision proxy | 逐类：候选按分数贪心一对一匹配同 GT（IoU≥0.5），TP/候选数 → 类平均 |
| conditional IoU | 已匹配 (≥0.5) 候选对的 mask IoU 均值（定位成功后的形状质量） |

诊断量预测统一 COCO RLE 格式（SAM3 mask 在上游编码）；多边形预测不被诊断
路径接受。G1 gate（S2.1）使用 recall proxy 的类平均值。

## 完成判据与自测

`tests/test_score_eval.py` 用合成 GT + 伪造预测验证：完美预测 mAP/recall/
precision/condIoU 全 1；带漏检/重复/偏移/纯 FP 的降级预测各自落到解析可预期
的数值（含 cat 平均与 micro 两口径）。2026-09-02 于 base 与 sam3 env 双通过。

## 入口

```python
from sam3_ovis.eval.score_eval import run_score_eval
result = run_score_eval(gt_json, predictions=[...])   # summary / per_class_ap / diagnostics
```
