# Neck 失配敏感度探针：协议登记（结果产生前冻结）

> 日期：2026-11-16（协议段）。脚本：`sam3_ovis/graph/neck_mismatch_probe.py`
> （复用 `g1_qbridge_probe.py` 的机械，原文件禁改、仅 import）。上游背景：
> G1 协议 §4 与结果 §2.2 登记的 writer/reader 预训练于 tracker neck 分布、
> 探针却喂 detector FPN 特征的 neck 失配，及 own-vs-foreign AUC 0.609 弱敏感。
> 本探针回答：**换记忆特征来源（detector FPN → sam2_backbone）后，分布差多大、
> 敏感性 AUC 变多少（ΔAUC），neck 失配是否足以解释 0.609**。

## 0. 源码确认（写脚本前完成，先于任何测量）

- `sam2_backbone_out` 来源链（已逐行读源码确认）：
  `build_sam3_image_model(..., enable_inst_interactivity=True)` →
  `_create_vit_neck(add_sam2_neck=True)`（`sam3/model/necks.py`：
  `sam2_convs = deepcopy(self.convs)`，与 detector 颈同构不同权）→
  `SAM3VLBackbone._forward_image_no_act_ckpt`（`vl_combiner.py`，scalp=1 丢弃
  最低分辨率级）→ `backbone.forward_image` 返回键 `sam2_backbone_out`。
- 输出键与分辨率（1008² 输入，ViT patch 14 → trunk 72×72×1024）：
  `sam2_backbone_out["backbone_fpn"]` = 3 级
  [288×288（stride 3.5，256ch；`set_image` 后被 `conv_s0` 投到 64ch，本探针不用）,
   144×144（stride 7，256ch；`set_image` 后被 `conv_s1` 投到 32ch，不用）,
   **72×72（stride 14，256ch，`backbone_fpn[-1]`，不被 set_image 改写）**]，
  另有 `vision_pos_enc` 同构列表。
- tracker 原生记忆像素特征（`sam3_tracker_base._encode_new_memory` +
  `_prepare_backbone_features`，`num_feature_levels=3`）即
  `sam2_backbone_out["backbone_fpn"][-1]` → 72×72×256。
- **更正 G1 协议笔误**：G1 协议 §1 称画布为"stride-16 级 63×63"，实际
  `pick_level` 选中 72×72（1008/14，stride 14；G1 records.jsonl grid=[72,72]）。
  因此 detector 画布与 sam2_backbone 记忆级**同网格同通道**，主对比不需要代理；
  任务提示的"若不可得则做代理对比"不触发。补充口径仍登记 144×144 高分辨率级
  （avg-pool 到 72²）的分布统计。

## 1. 两口径（arm）定义——唯一变量是造记忆的特征来源

| arm | 造记忆特征（writer 输入 pix_feat） | 说明 |
|---|---|---|
| det | `backbone_out["backbone_fpn"]` 中 `pick_level` 选中级（72×72×256，detector 颈） | G1 现状，机械逐条复刻 |
| trk | `backbone_out["sam2_backbone_out"]["backbone_fpn"][-1]`（72×72×256，tracker 颈） | tracker 原生记忆分布 |

两 arm 共享（成对控制）：同一 writer（`maskmem_backbone`）与同一调用形
（mask ±10 logits @ 网格、`skip_mask_sigmoid=False`）、同一 reader 调用形
（src=detector 画布 + detector pos，Q-bridge 语义不变，**不**换 reader 的 src）、
同一首遍预测/目标/匹配、同一 bridge（seed 20261105 正交，同 G1）与第二个
bridge（seed 20261116，稳健性）。foreign bank 按同 arm 来源分别传递
（det 用 det 特征造的上一图 bank，trk 用 trk 的）。

## 2. 登记测量

- **D1 分布差**（同图同区域，区域=首遍目标 mask ∪ bank mask，全部插值到 72²）：
  逐通道区域内均值/方差（|Δμ|/σ_det、方差比的中位数与分位）、全图逐通道
  均值/方差、通道均值向量互余弦、逐位置模长（均值/分位）、区域池化向量模长比、
  互余弦矩阵（det 区域 × trk 区域：对角=同区域跨源、非对角=跨区域，衡量几何
  是否共享/区域是否仍可分辨）、正交 Procrustes（A=det 区域池化 [R,256]，
  B=trk 同区域 [R,256]；逐图与全图拼接两种；报告对齐前后相对 Frobenius 残差、
  对齐后成对余弦、AᵀB 奇异值谱 top-k 能量占比）。补充：det 72² vs trk
  144²（pool 到 72²）同表。
- **S2 记忆敏感性**（规则与 G1 相同：dev 前 12 张有目标图、目标 score≥0.3
  上限 16、bank K=2 且与全部目标 IoU<0.5、跨文本 IoU≥0.5 匹配、foreign=上一图
  bank、空库=同形状零 prompt）：每 arm 重算 M1（|Δ| 中位/p90/超 1e-3 占比）、
  M2（跨文本 CV、argmax 翻转率）、M3（own vs foreign AUC + 双中位）。
  **主统计量 ΔAUC = AUC_trk − AUC_det**，附逐图聚类 bootstrap（重采样图，
  n=1000）95% CI 与逐 (目标,文本) 成对 |Δ| 比值中位、sign 率；双 bridge 并列。
- **V3 判定门槛（预登记）**：AUC_trk ≥ 0.7 ⇒ neck 来源是 0.609 的主要解释
  （"换记忆来源"有效）；ΔAUC ∈ [0.02, 0.7−AUC_det) ⇒ 部分解释（建议适配投影
  或换记忆来源，视 D1 的 Procrustes 残差二选一）；|ΔAUC| < 0.02（CI 含 0）
  ⇒ neck 来源不足以解释，0.609 的弱敏感主要来自 reader/bridge 零训练状态，
  建议训练吸收（G2/G3 路线）而非接口修补。D1 侧：Procrustes 对齐后残差
  < 0.3 且对齐后成对余弦 > 0.7 ⇒ 线性适配投影大概率够用；否则倾向换记忆来源。

## 3. 工程与精度（B 段 GPU 窗口约束下的预登记）

- 显存窗口：轮询 nvidia-smi（5 分钟一次，≤4 小时），任一卡空余 ≥7GB 即用，
  多卡合格取计算利用率最低者；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 精度：`--precision auto`——启动时空余 ≥10.5GB 用 fp32（G1 全对齐）；空余
  ≥7GB 用混合 bf16（**标注**：`backbone.forward_image/forward_text/
  forward_grounding` 与首遍在 bf16 autocast 下跑，权重仍 fp32；writer/reader/
  打分/全部统计 fp32；Δ 的逐元素相减共享同一 queries，q 的 bf16 误差在
  s_own−s_base 中相消，剩余特征噪声 ~0.4% 量级，低于 M1 的 1e-3 门槛一个量级
  以内；bf16 轮次的目标集在 0.3 门槛边缘可能与 fp32 轮差个别目标——ΔAUC 为
  轮内成对比较，不受跨轮漂移影响，如实登记）。
- 相比 G1 删减（与 neck 问题无关且省显存）：不做 M4 梯度段、不做 M5 K 曲线；
  保留一次 δ=0 bit-exact 校验。每图 backbone 只前向一次：复用
  `processor.set_image` 的 `state["backbone_out"]`（detector 与 sam2 两颈输出
  都在其中；`set_image` 对 sam2 级 0/1 的 conv_s0/s1 改写不触及本探针用的
  级 −1；数值与 G1 二次前向同权重同输入等价）。
- 输出：`sam3_ovis/logs/neck_mismatch/{records.jsonl, dist_stats.json,
  summary.json}`；脚本含 `--self-test` CPU 自测（统计函数/AUC/Procrustes/
  bootstrap 的合成数据校验，不加载模型不碰 GPU）。
- 确定性：seed 20261105 全局；逐图逐实例落盘 JSONL。

## 4. 已知边界（如实带入结论）

- trk arm 仍是"writer 的原生输入分布"，但 reader 的 src 仍是 detector 画布
  （Q-bridge 语义决定的成对控制）——这是 neck 来源的单变量隔离，不是完整
  tracker 管线的复刻；"src 也换"的完整原生口径超出本探针登记范围。
- 空库=零向量记忆近似（同 G1）；首遍预测含预测误差（同 G1）。
- 12 图样本量下 AUC 的 bootstrap CI 较宽，ΔAUC 判读以 CI 与门槛双看。

---

## 5. 执行 trail 与结果（B 段后补记，2026-11-16）

### 5.1 A 段完成项

- 协议（本文 §0–§4）先于任何测量冻结；§0 的源码结论（sam2_backbone_out
  可得、键/分辨率、writer 契约、G1 "63×63" 笔误更正）均在写脚本前逐行核读：
  `sam3/sam3/model/necks.py`（sam2_convs=deepcopy）、`vl_combiner.py`
  （scalp=1 → 双颈各 3 级 [288²,144²,72²]）、`sam3_image_processor.set_image`
  （仅改写 sam2 级 0/1，级 −1 即 72²×256 不受影响）、
  `sam3_tracker_base._encode_new_memory/_prepare_backbone_features`
  （原生记忆 pix_feat = sam2 `backbone_fpn[-1]`，72×72×256）、
  `memory.py SimpleMaskEncoder`（mask 先插值 1152² 再 ×16 下采样 → 输出
  `vision_features` 64ch @ 72²）。**结论：主对比无需代理，同网格同通道。**
- 脚本 `sam3_ovis/graph/neck_mismatch_probe.py`：import 复用
  `g1_qbridge_probe`（原文件零改动），CPU 自测 14/14 通过（恒等/旋转+噪声/
  无关特征三档对 dist 统计与 Procrustes、AUC 边界、bootstrap CI 含 0、
  预登记 verdict 分支），`py_compile` 通过。
- 双 bridge（20261105/20261116）、逐图聚类 bootstrap、精度策略（≥10.5GB
  fp32 / ≥7GB bf16 混合并标注）均按 §3 实现。

### 5.2 B 段：4 小时无显存窗口，GPU 测量未运行（如实登记）

- 轮询 trail（`sam3_ovis/logs/neck_mismatch/poller.log` + 会话内 10 秒粒度
  手动监测）：13:53–18:02（>4 小时）。规则执行：free≥7168MiB、多卡合格取
  最低利用率、先 smoke 后 full（`/tmp/sam3_ovis/neck_poller.sh`，可复用）。
- 全程无任何一卡空余 ≥7GB：4×RTX3090 被 8 个同用户训练进程
  （`ue_iterative_refined_*`，每卡 2 个各 ~8–9GB，pgrep/etime 核实）持续
  占用，自由显存稳定在 5.8–6.9GB；期间计算利用率 0–100% 波动（进程未退出）。
  自动 poller 进程曾被会话终止一次（14:02 前，无窗口错过），其后由手动
  监测补足全程，覆盖率近连续。
- 因此 **D1 分布差量级、S2 的 ΔAUC、V3 判定均未测量**；`logs/neck_mismatch/`
  仅有 poller trail，无 records/dist_stats/summary JSON。本报告不给出任何
  未测数值。

### 5.3 复跑指引（窗口出现后一条命令）

```bash
source sam3_ovis/scripts/load_environment.sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=<gpu> \
  "$PYTHON" sam3_ovis/graph/neck_mismatch_probe.py            # full 12 图
# 先验：--self-test（CPU）；--mode smoke --n-images 2 冒烟；--precision auto
# 产物：logs/neck_mismatch/{records.jsonl, dist_stats.json, summary.json}
# 读数：summary_b1.delta_auc_trk_minus_det + bootstrap.ci95 + verdict 字段，
#       D1_global.procrustes_det_trk / pooled_cos_global 判适配投影是否够用。
```

### 5.4 当前可给的结论（仅基于源码核读，不含未测数值）

- "neck 失配"在接口层面成立且可消除：writer 的原生输入（sam2 fpn[-1]）与
  G1 实喂的 detector 画布**同网格同通道**（72×72×256），切换记忆来源或加
  线性适配投影都是零结构改动；但二者分布差与 ΔAUC 未测，"是否足以解释
  0.609" 保持 open，待 §5.3 复跑后按 §2 门槛判定。
- 给 G2 的顺序建议不变：先跑本探针（分钟级、零训练）再决定"适配投影 vs
  换记忆来源 vs 训练吸收"；若 ΔAUC<0.02 且 CI 含 0，neck 来源不是主因，
  修补应转向训练吸收（bridge/adapter 联合微调）。

## 6. B 段结果（2026-11-17 补测，node01，按 §5.3 一条命令执行）

> 环境：新打通的 node01（10.171.103.175，空闲 3090；本项目栈已按相同路径
> 镜像：env/ckpt/代码/dev 数据）。执行 bf16 混合精度（§3 预登记允许；fp32
> 模式撞上模型构建路径泄漏的全局 bf16 autocast，dtype 混杂报错——与 R2
> 发现同源）。12 图 / 192 目标 / 2,085 对，0.6 分钟。过程修复 3 处脚本缺陷
> （procrustes/statistics 段 bf16 落 fp32 + 关 autocast；trkhr 补充臂
> 64 通道的逐行余弦跳过；跨通道 thin-SVD），均为执行层修复，不动协议。

### 6.1 主结果（b1 桥，两臂同桥同规则）

| 臂 | M1 中位 Δ | M3 AUC(own vs foreign) |
|---|---|---|
| det（detector FPN 造记忆，G1 现状） | 0.078 | **0.607**（复现 G1 的 0.609） |
| trk（tracker 主干造记忆，"匹配"来源） | 0.055 | **0.557** |
| **ΔAUC (trk − det)** | — | **−0.049**；配对 frac(trk 更大)=0.342 |

### 6.2 分布差（D1，真实存在）

- 区域池化互余弦对角 0.344（同区域跨 neck 特征仅 ~1/3 对齐），离对角
  0.089；Procrustes 对齐后残差 1.29（res_before 1.93），cos_after 0.914，
  主能量集中（top-4 奇异方向 82%）。
- 记忆记录级：同 mask 双源 record 余弦 0.702、模长比 0.95。
- trkhr 补充臂（fpn[-2]，64 通道）：Procrustes res_after 9.46——与
  det/trk 主臂不同构，仅作登记不作判定。

### 6.3 判定（按 §2 预登记门槛）

**"neck 失配解释 0.609"被否定，且方向相反**：换成"匹配"的 tracker 主干
来源后 AUC 反而降 0.049（|ΔAUC|>0.02 门槛触发；方向与失配假设相反）。
分布差真实（§6.2）但不转化为敏感性差异——**敏感性瓶颈不在记忆来源，
应转向训练吸收**（bridge/adapter 联合微调），换记忆来源或零成本适配投影
都不再是候选修补。

### 6.4 桥稳健性（b2 第二随机桥，登记项）

b2 桥下两臂 AUC 均回落到 ~0.51（机会水平），M1 变大但 CV 降至 0.087：
**零训练 AUC 由桥的随机抽取主导**——再次印证"敏感性主张必须训练后评估"，
也解释了 G1/b1 的 0.61 只应解读为"路径有信号"而非稳定属性。

### 6.5 附加价值

- G1 核心数字在异机（3090/node01）+ 修正精度协议下复现（0.607 vs 0.609），
  M6 空库 bit-exact 12/12。
- node01 现已具备本项目完整执行环境（含 editable sam3 重锚定），可作为
  G2 训练的算力补充。
