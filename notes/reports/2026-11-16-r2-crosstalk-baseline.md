# R2 零训练串扰基线：结果与判定（2026-11-16）

> 临时编号 R2-XTALK-20261116。问题（routes 文档 §4：R2 立项前必须验证
> "把其他实例放入 reader 是否发生身份混淆"）：**把已知类别 c_n 的强邻居
> 实例记忆读入目标 j 的 query 后，j 的 15 文本分数是否被拉向邻居类方向？**
> 脚本 `sam3_ovis/graph/r2_crosstalk_probe.py`（设计/判读门登记于其
> docstring，冻结先于任何统计产出）；数据
> `logs/r2_crosstalk/{records.jsonl, summary.json}`。本报告为该探针唯一口径。

## 1. 协议（要点；全文见脚本 docstring）

- 样本：g0_splits dev 组扫描前 60 候选，前 12 张有 score≥0.3 目标图
  （复用 G1 选择逻辑与 seed 20261105）；12 图 / 125 目标（Top-16/图）。
- 强邻居 n：first_pass 中与 j 异类且（mask IoU≥0.5 或 bbox IoU≥0.2）的
  最高分预测。实际选中 57 个：Bridge↔Harbor 49、Small↔Large_Vehicle 8，
  mask IoU 中位 0.947、bbox IoU 中位 0.975、分数差中位 −0.043（邻居分数
  略高于目标）——即真正的近重复混淆对，正是串扰最该发生的场景。
- 三臂 bank（K=1 单条记忆，同目标同批匹配 query 配对）：neighbor（n 的
  记忆，处理臂）／self（j 自身记忆，内容对照）／foreign（上一处理图 top
  目标的记忆，sham 对照；首图无）。
- 路径与 G1 同构（import 复用 g1_qbridge_probe 的 grounding_once/
  pick_level/build_records/reader_delta/maskpool/score_queries，未改该
  文件）：writer 记录→reader Δ画布→maskpool@j.mask→正交 bridge
  （seed 20261105）→q'_j 重打分。
- 度量（预注册）：P1 Δs_{c_n}^{neighbor} 在匹配文本 Δs 中的分位（0.5=无
  拉动）；P2 pull 强度 Δs_{c_n}−mean_t(Δs_t) 的 neighbor−self／
  neighbor−foreign 配对差；P3 方向性 Pearson(Δs_t, ⟨unit(p_t),
  unit(p_{c_n}−p_{c_j})⟩)（p_t 为 dot_prod_scoring 原生文本方向）；P4
  argmax 移向邻居类比例。显著性＝以图为聚类的配对簇 bootstrap（10,000
  次，seed 20261116，95% CI）。
- 判读门（先于结果登记）：显著＝P1≥0.60 且两配对差 CI>0 且方向性>0；
> 零＝P1∈[0.45,0.55] 或任一配对差 CI 含 0；中间如实报告。

## 2. 精度口径修订（先于统计产出；烟雾阶段发现）

事实：`build_sam3_image_model(enable_inst_interactivity=True)` 构建时经
`sam3_tracking_predictor.py:51` 进入**全局 bf16 autocast 且永不退出**。
G1 的运行即处于该漏出状态——其 records 的 Δ 全部落在 bf16 网格（如
0.078125=10×2⁻⁷；192 目标 15 文本 Δ 中位仅 7 个不同值），G1 报告的
"fp32（统计段）"与事实不符（该报告口径需以此为准更正）。仓库既有登记：
demo_text.py:34、validate_api.py:87（官方用法＝全局 bf16 autocast，
fp32-only 会被 perflib.fused.addmm_act 的无条件 bf16 输出打断）。

本探针修订为混合精度（固定，无选项）：特征路径（首遍/画布/writer/
reader/grounding）运行在与 G1 完全相同的全局 bf16 autocast 下（若构建
未漏出则显式进入同款）；**打分/bridge/maskpool/文本方向在
autocast(enabled=False) 的 fp32 岛内**——实测 57 目标全部 Δ 无量化并列
（每目标不同 Δ 值≈匹配文本数），而 bf16 口径的中位 7/15 并列会直接稀释
分位类度量。精度对结论无方向性影响，只提高分辨力。

## 3. 结果

| 度量 | neighbor 臂 | self 臂 | foreign 臂 |
|---|---|---|---|
| P1：Δs_{c_n} 分位（中位，n=57） | **0.409** | 0.833 | 0.321 |
| P2：pull 强度（中位） | −0.0034 | **+0.0099** | −0.0049 |
| P3：方向性 Pearson（中位） | −0.076 | +0.096 | −0.007 |
| P4：argmax 移向邻居类 | **0/57** | 0/57 | — |
| argmax 任意翻转 | 0/57 | 0/57 | 0/57 |

配对差（簇 bootstrap，按图聚类，7 张图贡献有效行）：

| 配对 | 中位 | 95% CI | 两尾 p |
|---|---|---|---|
| pull：neighbor−self | **−0.0105** | [−0.0157, −0.0070] | <1e-4 |
| pull：neighbor−foreign | +0.0028 | [−0.0045, +0.0061] | 0.40 |
| 分位：neighbor−self | −0.267 | [−0.467, 0] | 0.055 |
| 分位：neighbor−foreign | +0.067 | [0, +0.2] | 0.85 |
| 方向：neighbor−self / −foreign | 0 / −0.070 | 均含 0 | 1.0 / 0.31 |

伴随观察：|Δs| 中位 0.046、p90 0.134（n=2,247，路径活跃，接 G1 M1
0.078 的 bf16 量化口径）；self 臂在 c_j 槽位分位中位 **0.864**（89% 目标
>0.6）；空池 bit-exact 57/57；峰值显存 4.29 GiB（≤6GB 预算）；总时长
1.1 分钟；无强邻居目标 68/125；4/57 目标的 self 与 neighbor 记忆在
72×72 网格量化后逐位相同（近重复 mask 下采样碰撞，已计入并在统计中
呈现，非聚合主因——其余 53 目标上两臂分位中位 0.833 vs 0.367）。

### 判读

1. **零训练串扰不存在**：邻居记忆读入 j 后，邻居类文本分数既不升到
   机会带之上（P1=0.409），pull 与异图 sham 无差（CI 含 0，p=0.40），
   方向性不显（≈0/负），57 个真混淆对上 argmax 移向邻居 0 次。四项
   预注册信号一致指向 CROSSTALK_NULL。
2. **反向发现——身份保持而非混淆**：self 臂把"描述本位置的类簇"
   （c_j 与近重复 c_n 同时）整体拉高（0.864/0.833），而换成邻居的记忆
   （写入位置 95% 重叠）该效应即消失（0.409≈foreign 0.321），配对差
   −0.0105（p<1e-4）。零训练 reader+随机 bridge 的读取效应是内容/
   写读位置一致的，不发生跨实例的身份转移。
3. 边界（如实）：仅 7/12 图贡献有效行（其余图无异类近重复目标或 c_n
   文本未匹配），簇 bootstrap CI 偏宽；混淆对集中于 Bridge↔Harbor，
   结论对该混淆族最有说服力；首遍预测存在 GPU 非确定性常规抖动
   （12 图集合与 G1 的个别图不同，门槛 0.3 边缘检测来回横跳，登记）。

## 4. R2 判定

**零训练串扰基线为 NULL：R2 的"身份混淆"前提在零训练原生路径上不成立。**

- routes 文档 §4 要求的 R2 前置验证（"必须先验证把其他实例放入 reader
  是否发生身份混淆"）得到否定答案：零训练状态下，邻居实例记忆不会把
  目标分数拉向邻居类。R2 以"防串扰"为存在空间的论证在零训练层面**无
  依据**。
- 但 G1 已证该路径可训（梯度全链路连通、跨文本非均匀 Δ）；串扰是否在
  **训练后**出现（训练把 readerΔ 与类别方向耦合后可能产生零训练所没有
  的转移）是开放问题。R2 若继续，立项依据应改写为"训练后角色机制能否
  超过无角色同预算对照"（与 §5 的 B/U/A 对照直接比较），并与 G2 训练
  regime 合并测量，而非引用零训练串扰作为动机。
- 对 R2 有利的次要证据：self 臂的位置-类簇对齐效应（0.86/0.83）说明
  记忆的"写读一致性"在零训练下已有可测信号，角色机制若要在训练中
  放大或利用这种一致性，此效应可作为其作用面之一。

## 5. 复现命令

```bash
cd /home/wangcheng2021/project/GeoEvi-OVIS
nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1   # 需空余 ≥6GB
source sam3_ovis/scripts/load_environment.sh
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" sam3_ovis/graph/r2_crosstalk_probe.py --mode smoke
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" sam3_ovis/graph/r2_crosstalk_probe.py --mode full --n-images 12
# 产物：sam3_ovis/logs/r2_crosstalk/{records.jsonl, summary.json}
#       （smoke 写 *_smoke.*；全量自动截断 records.jsonl）
```

## 6. 执行 trail（治理记录）

- 烟雾阶段拦截的自身缺陷（未污染正式数据）：① 原设计以
  autocast(enabled=False) 包裹特征路径，触发 addmm_act bf16/fp32 断裂，
  由此发现全局 autocast 漏出事实并修订精度口径（§2）；② 异图单条记录
  错误切片 [1,64,h,w]→[64,h,w] 导致第二图 reader_delta 形状断言失败，
  修复为保持 4-D；③ smoke 改为取首个存在强邻居目标的图（无邻居图无法
  验证全路径）。
- 环境扰动如实登记：GPU1 由他人任务共享，两次启动间一度满载（空余
  0 GiB），按 ≥6GB 门等待后重跑；全量启动时空余 6.07 GiB，峰值
  4.29 GiB，未挤占。
- 未修改任何现有文件；未 git commit。G1 报告的"fp32"口径更正建议见 §2，
  归档于本报告而不回改 G1 文档。
