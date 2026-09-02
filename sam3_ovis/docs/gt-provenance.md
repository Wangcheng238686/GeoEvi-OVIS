# GT 来源登记（S1.3 / E5）

> v1 既定约定：论文按测试集逐个标注 GT 来源。split 与标注均以 SCORE HF
> `HuangShiqi008/SCORE` 发布为准（MIT）；论文协议表述引自 SCORE
> （ICCV 2025, arXiv 2507.12857）§4.1。登记日期 2026-09-02。

| 数据集 | 角色 | 图数 / 实例数 / 类数 | GT 来源 | 依据 |
|---|---|---|---|---|
| NWPU（NWPU-VHR-10） | 测试 | 731 / 2,661 / 10 | **人工标注** | SCORE §4.1："instance masks further annotated by [44]"；原检测框出自 Cheng et al.（SCORE ref [7]）；ref [44] 全名待 P0 报告前补引 |
| SOTA | 测试 | 874 / 29,247 / 18 | **SAM 伪标签** | SCORE §4.1：SAMRS 提供，源自 DOTA-V2.0 检测框 |
| FAST | 测试 | 3,207 / 43,258 / 37 | **SAM 伪标签** | SCORE §4.1：SAMRS 提供，源自 FAIR1M 2.0 检测框（细粒度航模类） |
| SIOR | 训练 | 11,725 / 68,062 / 20 | SAM 伪标签 | SAMRS，源自 DIOR 检测框（SCORE §4.1：11,725 图 / 20 类，与 json 一致） |
| SIOR-val | 测试（协议） | 11,738 / 124,373 / 20 | SAM 伪标签 | SCORE Table 1 在 iSAID 训练时以 SIOR 为测试集 |
| iSAID | 训练 | 28,029 / 704,684 / 15 | 人工标注 | iSAID 原生人工实例标注；⚠️ SCORE 论文 §4.1 称 18,732 训练图，其 HF json 实为 28,029——split 以 HF 发布为准（E4），差异不裁决、如实双登记 |
| iSAID-val | 测试（协议） | 9,512 / 235,531 / 15 | 人工标注 | SCORE Table 1 在 SIOR 训练时以 iSAID 为测试集 |
| NWPU-Refer category 子集 | 补充 | 未开始 | referring 人工表达式 + 实例 mask | arXiv 2506.03583（西工大），GitHub CVer-Yang/NWPU-Refer |

含义与影响（对应 design-v2 §12.3 E5）：

1. **测试侧两档 GT 质量**：NWPU 是纯人工；SOTA/FAST 的 mask 是 SAM 伪标签
   （框为人工/竞赛标注，mask 由 SAM 生成）。SOTA/FAST 上报数字时必须在论文
   里注明伪标签性质；SAM3 生成的候选与 SAM 伪标签 GT 存在同源偏差的可能
   （对召回 proxy 略有利、对边界质量评估偏松），P0 报告单列讨论。
2. FAST 37 类细粒度（A220/Boeing737 级文本可分性存疑），v1 已把 FAST 低 mAP
   列为机会点；text-only 短语表设计时注意细粒度类名。
3. 训练侧 iSAID（人工）/SIOR（伪标签）混合；P2 训练信号自举（S4.2）时分别
   记录标签噪声水平。

冒烟核对：`tests/smoke_datasets.py`（2026-09-02 三测试集 PASS，数字与 SCORE
论文 §4.1 声明逐项一致）。
