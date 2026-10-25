# v4-S0 数据摸底：本地大图数据的跨窗作用空间（iSAID + DOTA 原图）

> 日期：2026-10-25。对应 `implementation-v4.0.md` S0 第 1–3 条中不依赖 GPU 的部分。
> 脚本：`sam3_ovis/eval/v4_s0_recon_isaid.py`、`sam3_ovis/eval/v4_s0_recon_dota.py`
> （协议在 docstring 登记，数字产生前冻结）。结果 JSON：
> `sam3_ovis/logs/v4_s0_recon_isaid.json`、`sam3_ovis/logs/v4_s0_recon_dota.json`。

## 1. 数据事实

| | iSAID train / val | DOTA-v1.5 train / val |
|---|---|---|
| 原图（未预切） | 1,411 / 458 | 1,411 / 458（**与 iSAID 同一批原图**，尺寸分布逐分位一致） |
| 实例数（评估口径） | 354,789 / 116,649 | 115,346 / 34,008（剔除 difficult 95,285 / 35,557） |
| 图宽 min/med/max | 387 / 1824 / 12029 px | 同左 |
| 图高超 tile(1024) 的图 | 1,210 / 400 | 同批图 |
| tiles/图（中位） | 6 | 6 |
| 标注格式 | COCO JSON，全局坐标多边形，实例 ID 完整 | 逐图 txt，8 坐标旋转框，全局坐标，实例身份 =（图, 行） |
| 实例面积中位 / p95 | 124 / 1,993 px² | — |

关键发现：**iSAID 与 DOTA 共享同一批原始大图**（iSAID 在 DOTA 图像上重标注）。
本地没有第二家"保留原始大图"的数据源（SAMRS/SOTA/FAST/SIOR 均为预切小图，
NWPU/DIOR 图幅 ≤1024，不满足 design-v4.0 §5 的原图硬要求）。

## 2. tile=1024 模拟切窗统计（overlap=200 主案；overlap=128 副案结论同向）

实例 bbox 严格包含口径（mask 可能略超 bbox，解释时保留不确定性）；
"跨窗" = cross-x + cross-y + corner（无任何 tile 完整包含且跨内网格线），
"超 tile" = 实例比 tile 还大；贴图像边缘不算截断。

| 数据集 | 完整落单 tile | 跨窗 | 超 tile |
|---|---|---|---|
| iSAID train | 99.88% | **0.10%** | 0.02% |
| iSAID val | 99.86% | **0.12%** | 0.02% |
| DOTA-v1.5 train | 99.41% | **0.43%** | 0.15% |
| DOTA-v1.5 val | 99.27% | **0.73%** | 0.15% |

DOTA 逐类 inside%（val，不足 98% 的类）：
soccer-ball-field 82.0、ground-track-field 87.2、harbor 95.8、plane 96.3、
bridge 97.7。跨窗集中在大型/带状目标类，但这些类的绝对实例数很少
（val：soccer 89、ground-track 133、bridge 427）。

多 tile 可见性（DOTA val，bbox 与 ≥2 个 tile 相交 = 落在 overlap 区）：
1 tile 35.6%、2 tiles 43.8%、3 tiles 1.0%、4 tiles 19.5%。
注意 4-tile 比例偏高源于边缘锚定 tile 与前一 tile 形成大重叠
（图尺寸略超 tile 时锚到边缘的末 tile 与前 tile 重叠可达数百 px），
正式 manifest 冻结时需处理（例如边缘半 tile 对齐）。

## 3. S0 判读

1. **几何截断作用空间不足**：跨窗实例仅 0.1%（iSAID）/ 0.4–0.7%（DOTA）。
   若 TG-CWP 的收益只能来自"恢复被边界截断的实例"，其 mAP 上限远低于
   可检测的效应量——按 design-v4.0 §5 的准入逻辑，仅凭该空间应判
   **作用空间不足，负报告停止**。
2. **剩余的两个可能作用空间**（S0 尚未关闭的部分）：
   - **邻窗漏检差异**：同一实例（未被截断）在 tile A 被检出、tile B 漏检。
     其规模 = 多 tile 可见质量（44–64%）× 单 tile 漏检率 × 跨 tile 独立性，
     只能由 C0（独立 tiles + 全图融合）实跑量化——这是 S0 剩余的
     决定性统计（"C0 漏检中可由邻窗可靠观察覆盖的比例"与 oracle 正框恢复上限）。
   - **overlap 区去重质量**：44–64% 实例被多 tile 重复观测，融合器的重复率
     是 TG-CWP 的次要收益轴（design-v4.0 已登记该指标）。
3. **数据结论**：本地唯一可用大图源 = DOTA-v1.5 原图（train 1,411 / val 458，
   全局坐标 + 实例身份可恢复，v1.5 十六类）。iSAID 标注可作补充词表/对照，
   但两者不是独立图像来源，论文中须如实登记"单一图像来源"。

## 4. 决策建议（供项目负责人裁定）

- **A（建议）**：先跑 C0 一次性统计（DOTA val 子集 ~100 图，GPU 单卡数小时），
  量化"邻窗可见漏检"与 oracle 上限，再决定 TG-CWP 去留。成本小，
  且这是 design-v4.0 §5 明确要求的 S0 第 3 步，不算超纲。
- **B（严格止损）**：以几何截断空间 0.4–0.7% 直接判 S0 不通过，出负报告
  停止 v4，避免 GPU 投入。符合"不以人为切得更碎制造问题"的红线。
- A/B 的选择本质是对"邻窗漏检差异是否可能是主要收益源"的先验判断：
  SAM3 text-only 单 tile 漏检率在 20–45% 量级（P0/T1 实测），只要跨 tile
  漏检有中等独立性，该空间可能达实例级的 5–15%，远大于截断空间。

## 5. 口径备忘

- bbox 口径高估 inside、低估跨窗（mask 边缘略超 bbox）；对 0.1–0.7% 的
  结论量级无影响。
- difficult 实例（DOTA）未计入比例；计入会进一步压低跨窗比例（小实例占比更高）。
- 本摸底不含任何模型推理，不构成对 C0/C1/C2/C3 的任何选参。
