# R1 v2 全量启动前独立实施审计（2026-11-22）

审计对象：R1 v2（源域 LoRA 微调 SAM3）训练 + 评估全链路。
依据：`notes/reports/2026-11-22-r1-protocol-v2.md`（下称"协议"）、d3aa4ce 提交、
烟雾日志 `/data/wangcheng/geoevi-r1/smoke_run.out`、烟雾 ckpt
`/data/wangcheng/geoevi-r1/smoke/checkpoints/checkpoint.pt`。
方法：只读取证 + CPU 级复算（`CUDA_VISIBLE_DEVICES=""`，未占用 GPU、未改任何
项目文件、未启动训练）。除本报告外无文件改动。

复算环境说明：在 CPU 上重建模型结构时对 `torch.zeros/arange` 等做了
device="cuda"→"cpu" 的进程内 shim（仅存在于审计进程，未写入仓库），不影响
模块结构枚举与 state_dict 键比对结论。

---

## A. 判定门实现 vs 协议 §1 —— 相符（但其结果当前不可达，见 G）

证据：`sam3_ovis/graph/r1_v2_eval.py` L156–169

- `ok = (mAP >= 0.39 and lo > 0 and same_class >= 45 and fp <= 350)` —— 与协议
  GO 门（0.39 / CI 下界>0 / 45% / 350）逐项一致；
- `partial = (0.30 <= mAP < 0.39 and lo > 0)`，否则 NO-GO —— 与协议
  PARTIAL 区间及"NO-GO 其余"一致；
- 反例检验：mAP≥0.39 但 same-class<45 → `ok=False`、`partial=False`（mAP 越
  上界）→ NO-GO，与协议"GO 需四条同时成立，其余 NO-GO"一致，无漏判路径；
- 显著性：R1v2−COMBO 逐图配对 bootstrap，`BOOT_N=2000`、percentile [2.5,97.5]
  （L142–147），与协议"2000 次 CI95 逐图配对"一致；verdict 用的 `lo` 即 CI 下界；
- 主指标 `r1["mAP"]` 来自 eval_arms 的 117 图 pooled segm mAP（协议主指标口径）。

次要备注（非反例）：协议 PARTIAL 写"CI 不含 0"，字面含 hi<0（显著变差）情形；
实现取 `lo > 0`，hi<0 落 NO-GO——比字面更保守且符合"止损"语义，不构成偏离。

**结论 A：数字与逻辑相符。但该段代码位于奇偶校验断言之后，当前根本执行不到
（见 G），判"实现正确、链路断链"。**

## B. prompt 一致性 —— 通过

- 训练 JSON `/data/wangcheng/geoevi-r1/isaid_train_g0_space.json`：15 个类别名
  全部空格式（'Small Vehicle'、'storage tank'、'Ground Track Field'…），
  `any('_' in name)` = False；配置 `ann_file` 即此文件（r1_isaid.yaml L197）。
  训练期诊断用 `isaid_tiny_val_space.json` 类别名同为空格式。
- 评估 R1v2 臂：`COMBO_CFG = {"prompt": "space", "floor": 0.01, "res": 1344}`
  （r1_v2_eval.py L39），`g5a.gen_arm("R1v2", COMBO_CFG, ...)` 生效。
- `g5a_free_lever.py` L53–54：`prompt_of(cname, style) = cname.replace("_"," ")
  if style=="space" else cname`；L105–106 每个 cname 过 `prompt_of` 后才进
  `set_text_prompt`。
- GT 匹配不受名字影响：候选侧 `cls` 与 GT 侧 `id_to_name` 出自同一 gt 词表
  （`instances_train.json` 类别名，带下划线），映射走 `name_to_id[cls]` /
  `a["category_id"]`（eval_arms L206–209, L225–234）。
- 下划线泄漏扫描：训练侧唯一 prompt 来源是 space JSON；评估侧唯一 prompt 来源
  经 `prompt_of(...,"space")`。未发现任何带下划线类别名进入训练/评估 prompt。

**结论 B：通过。**

## C. LoRA 正确性 —— 数量与冻结/加载正确；但"50 个=视觉主干"的登记与事实不符（18 个被静默旁路）

代码与烟雾证据（`sam3/r1_lora.py`、smoke_run.out L431）：

- 冻结全部基座：`apply_lora` 先 `for p in model.parameters(): p.requires_grad_(False)`
  （L61–62）。CPU 重建实测：trainable = 1,122,304 参数 / 100 个张量，全部为
  lora_A/lora_B；烟雾日志 L1352–1353 "Trainable parameters 1.1 M /
  Non-Trainable 838 M"。
- B 零初始化：`lora_B = nn.Parameter(torch.zeros(...))`（L29–30）。单元实测：
  新建 `LoRALinear` 前向与冻结基座逐位相同（allclose=True，scale=alpha/r=2.0）
  —— 起步等价冻结成立。
- 50 替换 / 36 MHA 排除：烟雾日志 L431 `[lora] replaced 50 Linears (r=8,
  alpha=16; skipped 36 MHA modules)`；本审计 CPU 重建独立复现同样输出。
- ckpt 键等价（"烟雾 ckpt 已验证 strict 加载路径的等价形式"复核）：CPU 上以
  训练同构结构（enable_segmentation=False + apply_lora）对烟雾 ckpt 做
  state_dict 集合比对：missing=0、unexpected=0（即 strict 可过）；以评估结构
  （enable_segmentation=True + apply_lora(r=8, alpha=16)）`load_state_dict(
  ck["model"], strict=False)`：unexpected=0，missing=28 个键且全部以
  `segmentation_head.` 开头——与 r1_v2_eval.py L67–69 的断言完全吻合。评估侧
  apply_lora 与训练同构（同一 `r1_lora.apply_lora`，r/alpha 一致）。
- ckpt 事实：`epoch=1, steps={'train':24,'val':0}`；无 segmentation_head 键。

**与登记不符的事实（CPU 结构枚举 + 前向代码取证）**：

50 个被替换 Linear 的真实构成为 **32 + 18**：

1. 32 个 = `backbone.vision_backbone.trunk.blocks.{0..31}.attn.qkv`
   （vitdet `Attention`）。`vitdet.py` Attention.forward 以 `self.qkv(x)`
   模块调用 → **LoRA 生效**。
2. 18 个 = `transformer.encoder.layers.{0..5}.{self_attn,cross_attn_image}.
   out_proj`（12）+ `transformer.decoder.layers.{0..5}.cross_attn.out_proj`
   （6）。这些 out_proj 属于 `sam3/model/model_misc.py` 的**自定义**
   `MultiheadAttention`（别名 MultiheadAttentionWrapper，非 torch
   nn.MultiheadAttention，故未被 L65–67 的 MHA 跳过逻辑排除）。其 forward
   将 `self.out_proj.weight, self.out_proj.bias` 直接传入
   `multi_head_attention_forward(...)`（model_misc.py forward，两处分支均如此）
   ——只读权重张量、不调用模块 → **LoRA 增益在训练与评估中均被静默旁路**。
   `LoRALinear` 的 `.weight/.bias` 属性委托（L35–41）保证了不崩溃、基座结果
   不变，这正是"看似生效实际没做"的形态。单元实测：B≠0 时模块调用路径与
   weight-only 路径输出不同，旁路路径永远拿不到 LoRA 增益。
3. 后果量化：有效 LoRA = 32 个 trunk qkv = 1,048,576 参数；其余 73,728 参数
   （占"trainable 1.12M"的 6.6%）为死参数（无前向依赖 → 无梯度 → AdamW 不
   更新；DDP `find_unused_parameters=True` 掩盖了本应报错的事实）。
4. 协议 §5.1 登记"LoRA 覆盖 = 视觉主干 50 个注意力投影（1.12M 参数）；
   概念 transformer …… 排除"——**计数与参数量对，构成为错**：实际是
   "32 个视觉主干 qkv（生效）+ 18 个概念 transformer out_proj（替换但不生效）"。
   就"概念 transformer 实际未被训练"这一结论而言，有效行为与登记的偏离结论
   一致（训练科学性未被污染：等价于只对 trunk qkv 做 LoRA），但登记文本必须
   修正后再启动，且"1.12M 可训练参数"口径需改为"1.05M 有效"。

**结论 C：冻结/零初始化/评估同构/ckpt 键全部核实通过；50/36 计数一致但
登记的覆盖构成与事实不符（18 个静默旁路），需在启动前重新登记。**

## D. DDP-4 变更 —— 语义安全，但属未登记、未提交的启动配置改动

- `gpus_per_node: 4`：存在于工作区 `sam3/sam3/train/configs/r1_isaid.yaml`
  L317；但 `git diff HEAD` 显示这是 d3aa4ce 之后的**未提交改动**（同批还有
  `num_train_workers: 8→4`；文件 mtime 2026-11-22 22:59 > 提交时间
  2026-11-21 20:31）。协议 §5 四条偏离均未涉及 world size；notes/ 内亦未
  检索到 DDP-4 登记。因 `/data/wangcheng/geoevi-r1/run1` 尚不存在（无任何
  结果），该改动属**结果产生前**的启动配置决策、非事后改参——但按本项目
  "全量先 commit / 偏离登记"规程，启动前必须提交并登记。
- 显存：烟雾单卡实测 `Mem (GB): 9.00`（smoke_run.out L1401–1402，训练期峰值
  9.00GB）。DDP 每 rank 模型/批量与烟雾相同（batch/rank=1），新增仅 LoRA
  梯度桶（1.12M 参数 < 5MB）与 NCCL 通信缓冲（几十 MB 量级）→ 每 rank
  ≈ 9.0GB < 11GB 上限（机器为 4× RTX 3090 24GB）。备注：协议头部"共享卡
  ~8G 余量/卡"的表述低于 9.0GB 实测，启动排卡时需按 ≥10GB 余量预留。
- lr 无 world-size 缩放：grep `sam3/sam3/train/`（optim/trainer/train_utils）
  无任何按 world_size 缩放 lr 的代码；`lr_vision_backbone=5.0e-5` 落在协议
  3e-5~5e-5 区间。有效 batch = 4×1=4（DDP 梯度取均值）。
- 无其他语义漂移：max_epochs=5 不随 DDP 变化（epoch=全数据集一遍的口径不
  变）；§5.2 的"53.9k 迭代 / ~18.7h"按单卡口径登记，DDP-4 下每 rank 优化器
  步数 ≈ 13.5k（总样本数不变），墙钟约 1/4——属登记口径需补注，非正确性问题。
- `find_unused_parameters: True` 在提交配置中即有——恰是 18 个死 LoRA 参数
  所必需（否则 DDP 会因未收到梯度报错）；这也反向印证 C 项的旁路事实。

**结论 D：DDP 语义与资源安全（9.0GB<11GB、lr 不缩放、batch 4×1），但
gpus_per_node 1→4 + workers 8→4 为未提交、未登记的启动配置改动，须先
commit + 登记。**

## E. 防泄漏 —— 通过

- val 集合：`/data/wangcheng/geoevi-r1/freeler/{B0,COMBO,L1,L2,L3}` 各臂
  `cands_*.json` 的 img_id 并集 = **117 图**（每臂各 117），组前缀 18 组
  （P2052/P2054/P2067/P2100/P2103/P2122/P2131/P2145/P2177/P2217/P2225/
  P2359/P2373/P2411/P2436/P2475/P2572/P2590），全部 ⊆ manifest dev 组。
- 训练集合：`isaid_train_g0_space.json` 10,778 图 / 1,024 组，与
  `sam3_ovis/logs/g0_splits/manifest.json` train 组**逐组全等**（双向差集
  均为空）；与 dev/confirm 组交集为 0。
- 交集检验：val 117 图与训练 JSON 在 file_name 层与组层交集**均为 0**。
  抽查 20 个 img_id（如 25157→P2572_600_1400_273_1073.png、
  22333→P2067_375_1175_0_800.png、24192→P2475_600_1400_600_1400.png 等，
  全清单见审计过程）均为 val 组图。
- 候选生成无 GT 介入：`g5a.gen_arm` 中 gt 仅用于取图列表与类别名
  （L82–86）；候选只由模型前向 + floor 阈值产生（L100–125）；GT 框/类别
  仅在 `eval_arms` 的指标计算侧出现。
- 训练期 tiny val（5 图，P0000，属 train 组）不触碰 val 组，且只作诊断
  （见 F：无 best 挑选）。

**结论 E：通过。**

## F. last-checkpoint 口径 —— 通过

- `r1_isaid.yaml` `checkpoint.save_freq: 0`，`save_list` 未设（默认空）：
  `trainer.py` L351–357，`save_freq > 0` 为假 → `checkpoint_names` 恒为
  `["checkpoint"]` → 只落 `checkpoint.pt`，无 `checkpoint_<epoch>.pt`。
- best 挑选路径关闭：`_log_meters_and_save_best_ckpts`（L991–1019）仅当
  `checkpoint_conf.save_best_meters is not None` 才另存 best ckpt；配置未设
  （CheckpointConf 默认 None）→ 无 best ckpt。`skip_saving_ckpts: false`。
- 保存时机：训练结束 `self.save_checkpoint(self.epoch + 1)`（L609）。
- 评估默认读该文件：`r1_v2_eval.py --ckpt` 默认
  `/data/wangcheng/geoevi-r1/run1/checkpoints/checkpoint.pt`（L44–45）。
- 烟雾侧实证：`smoke/checkpoints/` 下仅 `checkpoint.pt`（3.38GB，
  epoch=1/steps=24）。

**结论 F：通过（无 best 挑选路径，last-ckpt 口径成立）。**

## G. 复算一致性（R1v2−COMBO 配对 + 奇偶校验）—— **不通过（阻断性缺陷）**

配对部分正确：`dets_of` 复用 `g5a.nms_image`（类内 NMS IoU<0.5 保留），
R1v2 与 COMBO 在同一 117 图、同一 name_to_id、同一 sub-GT 上逐图求
single-image segm mAP，差值做 2000 次自助（固定种子 20261122）——与协议
显著性口径一致。

奇偶校验错误（`r1_v2_eval.py` L135–141）：

```python
m_combo_re = float(np.mean([p_combo[i] for i in val_ids]))   # 逐图 mAP 的均值
assert abs(m_combo_re - ablation["COMBO"]["mAP"]) < 1e-6, ...
```

两处独立致命问题：

1. **数学上就不是同一个量**：`m_combo_re` 是 117 个"单图 mAP"的算术平均
   （每图各自建 COCO 索引、类别按该图出现情况计），而 `ablation["COMBO"]
   ["mAP"]` 是 eval_arms 的 pooled COCOeval mAP（跨图按分数全局排序、按全
   数据集类别汇总）。二者一般不相等。
2. **容差异想**：即便量相同，eval_arms 落盘前 `round(mAP, 4)`（g5a L246），
   1e-6 容差小于 4 位小数舍入误差（≤5e-5）。

在真实 COMBO 数据上的 CPU 复算（本审计，117 图全量）：

- pooled 复算 mAP = 0.27000709（与 `ablation.json` 存档 0.27 一致——eval_arms
  自身口径无漂移）；
- 逐图均值 = 0.16584233；
- |Δ| = 1.04e-01 ≫ 1e-6 → **该断言在真实数据上确定性触发 AssertionError**，
  位置在 verdict 计算之前 → `r1v2_verdict.json` 永远不会产出。即：全量训练
  + 117 图候选生成完成后，判定脚本必然中断在自检断言上，GO/PARTIAL/NO-GO
  无法落判（A 项代码因此"正确但不可达"）。

修复方向（供实施方参考，本审计不改动代码）：将奇偶校验改为同口径比较——
例如以 r1_v2_eval 侧对 COMBO dets 重算一次 pooled COCOeval(sub) 与
`ablation["COMBO"]["mAP"]` 比对（容差 ≥ 1e-4 以吸收 4 位舍入），或让
eval_arms 额外落盘逐图 mAP 供逐图口径互验。

**结论 G：不通过。此为阻断项。**

## H. 红线扫描 —— 通过（含一项治理缺口，见 D）

- val 组不用于选参：训练数据 = manifest train 组（E 项逐组全等已证）；
  训练期 val 用 train 组内 5 图（P0000）；无 best-ckpt/早停路径（F 项）；
  选择口径 = last checkpoint。
- 无事后判据修改迹象：协议 §1 四要素与判定门在 c6ab5b9 随 R1 v2 重登记写定；
  d3aa4ce 对协议文件仅 +16 行（§5 偏离登记，git show --stat 证实），四条偏离
  （LoRA 覆盖、target_epoch_size 死旋钮→max_epochs=5、评估侧 seg head
  strict=False、烟雾四项验证）均登记于全量启动前；R1v2 无任何结果产物
  （`/data/wangcheng/geoevi-r1/run1` 不存在，freeler 下无 R1v2 臂目录），
  不存在"结果后改门"的时间窗口。
- 工作区唯一未提交改动 = r1_isaid.yaml（gpus_per_node/num_train_workers，
  见 D）——属启动配置而非判据，但未走 commit+登记流程，构成治理缺口。

**结论 H：判据与数据红线未破；DDP-4 改动未登记为流程缺口。**

---

## 总结

| 项 | 结论 |
|---|---|
| A 判定门数字/逻辑 | 相符（但被 G 阻断不可达） |
| B prompt 空格式一致性 | 通过 |
| C LoRA 冻结/零初始化/同构加载 | 通过；覆盖构成与登记不符（18/50 静默旁路），需重登记 |
| D DDP-4 | 语义与显存安全；未提交、未登记 |
| E 防泄漏 | 通过（组/图双零交集，抽查 20 id） |
| F last-checkpoint | 通过 |
| G 奇偶校验 | **不通过（确定性 AssertionError，verdict 无法产出）** |
| H 红线 | 通过（D 为治理缺口） |

阻断原因（按严重度）：
1. `sam3_ovis/graph/r1_v2_eval.py` L140 奇偶校验把"逐图 mAP 均值"当
   "pooled mAP"比对（真实数据差 0.104，且 1e-6 容差小于 4 位舍入），全量
   训练完成后判定必然中断，四要素 verdict 无法落盘；
2. 协议 §5.1 对 LoRA 覆盖的登记与事实不符：50 个替换 = 32 个生效的视觉主干
   qkv + 18 个被 `model_misc.MultiheadAttention` weight-直传路径静默旁路的
   transformer out_proj（有效可训练 1.05M 而非 1.12M），须先修正登记；
3. DDP-4（gpus_per_node 1→4，及 workers 8→4）为未提交、未登记的启动配置
   改动，违反"全量先 commit + 偏离登记"规程。

VERDICT: FAIL: r1_v2_eval.py L140 奇偶校验量纲错误（逐图均值≠pooled mAP，真实数据 |Δ|=0.104，判定链必然断链）+ §5.1 LoRA 覆盖登记与事实不符（18/50 替换被 out_proj weight 直传静默旁路）+ DDP-4 改动未提交未登记；三项修复/重登记后方可全量启动

---

# 复审（2026-11-22，commit abe39ab，定向复核三项修复）

复审范围：仅 abe39ab 改动分支（`git show abe39ab`：r1_lora.py / r1_v2_eval.py /
r1_isaid.yaml / 协议 §5；工作区相对 abe39ab 无其他改动，仅本报告未跟踪）。
方法同首轮：CPU 结构复算 + 真实 COMBO 数据复算，未用 GPU、未改项目文件。

## C 修复复核 —— 通过

- `sam3/r1_lora.py` apply_lora 现排除两类：`torch.nn.MultiheadAttention` 与
  `sam3.model.model_misc.MultiheadAttention`（自定义类，forward 直传
  out_proj.weight/bias）；vitdet `Attention` 不受影响。
- CPU 独立复算（与提交者 GPU 实测对照）：
  - 训练构型（seg=False）：replaced=32，全部为
    `backbone.vision_backbone.trunk.blocks.{0..31}.attn.qkv`（位置集合仅
    {trunk_qkv}，无 LEAK）；trainable=64 张量 / 1,048,576 参数（=1.049M，
    与登记一致）；skipped=60（torch MHA 36 + 自定义 24）。
  - 评估构型（seg=True）：replaced=32 且与训练侧同一集合（ckpt 键对称），
  skipped=61（seg head 内多 1 个自定义 MHA，且 seg head 无任何 Linear 被
  误替换 → eval strict=False 的 missing 仍仅 segmentation_head.* 族）。
  - 全新模型 lora_B 逐位为零（起步等价冻结保持）。
- 协议 §5.1 已如实改写为"32 个 qkv / ~1.05M"，并保留首轮误登 50/1.12M 的
  更正记录。

## G 修复复核 —— 通过（容差正当）

- `r1_v2_eval.py` 奇偶校验改为：对 `dets_of("COMBO")` 在同一 `sub`（117 图）
  上重跑 COCOeval(segm) 得池化 mAP，与 ablation 存档值比对，容差 6e-5。
- 量纲：两侧均为池化 mAP，且评估流程第 3 步 `g5a.eval_arms` 先在本进程内
  重写 ablation.json，故该校验实测的是两条独立代码路径（g5a.eval_arms 内联
  dets 构建 vs r1_v2_eval.dets_of）的一致性——正是奇偶校验的目的。
- 容差正当性：存档值经 `round(·,4)` 量化（步长 1e-4，最大舍入偏差恰为
  5e-5）；复算路径确定性的唯一差异来源即该舍入 → 6e-5 = 5e-5 + 1e-5 浮点
  余量，不多不少。真实漂移不会被掩盖：候选/NMS/映射/score 任一实质变化在
  117 图池化 mAP 上的影响 ≥1e-4 量级（G5a 最小臂间差 2.65pt；即便单 det
  扰动也 ≥1e-4），且显著性判定另有 2000 次配对 bootstrap 承担，本校验仅
  为口径防漂移哨兵。真实数据实证：复算 0.27000709 vs 存档 0.27，
  |Δ|=7.1e-6 < 6e-5，断言通过；首轮的量纲错配（0.166 vs 0.270）已消除，
  verdict 段恢复可达。
- 外观小疵（不阻断）：pairing["parity_check"] 文案仍写 "(<1e-6)"，实际
  容差 6e-5——仅日志字符串，建议顺手更正。

## D 修复复核 —— 通过

- `r1_isaid.yaml`：gpus_per_node=4、num_train_workers=4 已随 abe39ab 提交
  （diff 与首轮指出的两行完全一致，无夹带）。
- 协议 §5 新增第 5 条（DDP-4：lr 不缩放 5e-5、有效 batch 4×1、11GB 上限
  来源=负责人设定本任务专用）与第 6 条（奇偶校验修正），均为全量启动前
  登记；`run1/` 仍不存在，无事后改参窗口。

## 复审结论

三项 FAIL 均已按审计意见修复并提交、登记；首轮通过项（A/B/E/F/H）未受
abe39ab 影响（改动不触及判定门数字、prompt 链路、数据划分、ckpt 口径）。
C 项有效覆盖现为 32 个 trunk qkv（1.049M），与重登记一致；G 项判定链可
达且容差有据；D 项配置与登记闭环。

VERDICT: PASS（复审 abe39ab：C=32 trunk-qkv/1.049M 无泄漏（CPU 复算 60/61 skipped 与 GPU 实测一致）；G=池化对池化 7.1e-6<6e-5（舍入半宽 5e-5+余量，真实漂移≥1e-4 不可掩）；D=DDP-4 已提交并入 §5.5/§5.6 登记；外观小疵 parity_check 文案 "<1e-6" 未更新，不阻断）
