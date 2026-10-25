# v4-S0 C0 统计运行计划：DOTA val 子集实跑协议（结果产生前冻结）

> 日期：2026-10-25。对应 `implementation-v4.0.md` S0 第 3 条、`design-v4.0.md` §5
> 准入统计；路线 A 已获批准（见 `v4-s0-recon-large-images.md` §4）。
> 脚本：`sam3_ovis/eval/v4_s0_c0_run.py`（GPU 推理）、
> `sam3_ovis/eval/v4_s0_c0_analyze.py`（CPU 分析）。
> 本文档在**任何结果产生之前**登记抽样、匹配口径与指标定义；运行后不得回头修改。

## 1. 研究问题（本统计要回答什么）

摸底报告确认几何截断空间不足（DOTA val 跨窗仅 0.73%），剩余的候选作用空间是
**邻窗漏检差异**与 **overlap 区去重质量**。本运行量化：

- c：C0 融合后漏检的 GT 中，有多少在**融合前**的某个 tile 观察里其实可见
  （存在正框可回注的可靠观察）；
- d：上述比例构成 TG-CWP oracle 正框恢复上限（S0 准入的决定性数字）；
- e：融合器重复实例率（TG-CWP 次要收益轴的基线）；
- a/b：C0 全图 segm mAP/AP50 与逐实例命中率（后续 C1/C2/C3 的对照锚点）。

## 2. 冻结协议

### 2.1 数据与抽样

- 数据：`/data/wangcheng/dataset/DOTA` val 原图 458 张
  （`val/images/part*/images/*.png`）+ `val/labelTxt-v1.5/DOTA-v1.5_val/*.txt`，
  按文件名 stem 配对；DOTA-v1.5 十六类（含 container-crane）。
- 抽样：seed=20261025，`--n-images 100`（默认）。优先抽含 ≥1 个非 difficult
  实例的图（摸底实测该池为 456/458 张），不足再从其余有标签图补；两池均按
  名字排序后用 `numpy.random.default_rng(seed).choice(replace=False)` 抽取，
  确定性可复现。**完整名单（含 W/H、实例数、路径）在推理前写入 manifest.json。**
- 标签解析：跳过 `imagesource`/`gsd` 头行（按内容识别）；每行 8 坐标 OBB +
  类名 + difficult；与 `v4_s0_recon_dota.py` 同规则。

### 2.2 tiling 与推理

- tile=1024，overlap=200（stride=824）；锚点序列**逐字复用**
  `v4_s0_recon_dota.py:tile_anchors`（含"末 tile 吸附边缘"）。图小于 tile 的
  维度单锚点 0，crop 即整图/整带。
- 模型：`build_sam3_image_model(checkpoint_path=$SAM3_CKPT, load_from_HF=False)`
  + `Sam3Processor(model, confidence_threshold=POOL_FLOOR=0.05)`，bf16 autocast，
  推理骨架照抄 `graph/t2_candidate_cache.py`（`set_image(tile)` → 16 类逐类
  `set_text_prompt` → 读 `state["scores"]`/`state["masks"][j,0]`）。
- prompt = DOTA 原始类名（连字符形式，如 `small-vehicle`），不做任何改写。
- 每 tile 串行处理，同一时刻只保留一份 image embedding；逐图落盘 flush + fsync
  + `torch.cuda.empty_cache()`；逐图 done ledger 支持断点续跑（崩溃半图重跑后，
  分析端按 done 记录的候选数截尾丢弃旧半块）。
- **显存：预计峰值 ~6 GB**（SAM3 image model @1008, bf16）；磁盘预算
  candidates.jsonl 约几十 MB。

### 2.3 候选记录（candidates.jsonl，全部 global 坐标）

`{img, W, H, tile_origin, tile_size, cat_name, score, mask_rle(global, [H,W]),
bbox_global(xyxy), bbox_local(xyxy), bbox_full_in_tile(bool)}`

- tight bbox 由 mask 像素重算（先贴全局画布再算，local = global − origin）。
- `bbox_full_in_tile` 定义：tile-local tight bbox **不接触任何内部分割 tile 边**
  （tile 左/右/上/下边若与图像边缘重合则豁免——贴图像边缘不算截断，与摸底
  报告口径一致）。语义 = 该观察未被 tile 边界截断，是 TG-CWP 可靠 anchor 的
  候选条件。注意：末 tile 吸附边（= 图像边缘）已按豁免处理。
- GPU 阶段不做融合、不做 NMS——所有候选原样落盘，融合离线做。

### 2.4 分析口径（analyze 脚本，CPU）

- GT：OBB polygon → pycocotools RLE（全局 [H,W]）；外接 AABB = 8 坐标 min/max。
  **difficult 实例整体剔除、不进 GT；落在 difficult 实例上的预测按普通 FP 计**
  （DOTA 官方为 ignore 口径，本处为简化口径，见 §4 注意项）。
- C0 融合：global 坐标下**严格逐类**（按 cat_name 分组）mask NMS，IoU≥0.5
  抑制，score 降序贪心。零预测图保留在 mAP 分母。
- 指标：
  - **a**：全图 segm mAP/AP50/AP75（COCOeval, iouType=segm, RLE）+ 逐类 AP50；
  - **b**：逐 GT 命中 = 存在同类别融合预测 mask IoU≥0.5（非互斥匹配）；
  - **c**：b 未命中的 GT 中，存在**融合前**候选同类别且 IoU≥0.5 的比例，
    按 `bbox_full_in_tile`（有 full 观察 / 仅截断观察）分账；
  - **d**：oracle 正框恢复上限 = c 的分子，双分母登记：/漏检 GT（可恢复率）
    与 /全部 GT（对实例空间的绝对余量，mAP headroom 口径）；
  - **e**：同一 GT 被 ≥2 个融合预测匹配的比例（分母 = 至少被 1 个预测匹配的
    GT）；边界截断子集（classify 的 cross-x/cross-y/corner，与摸底同逻辑）单列，
    同时报告各边界子集命中率。
- 边界分类与锚点逐字复用 `v4_s0_recon_dota.py`，保证与已批准摸底同口径。

## 3. 运行命令与预估成本

```bash
source sam3_ovis/scripts/load_environment.sh
# GPU 阶段（单卡；确认显存余量 ≥ ~7G 后启动）
CUDA_VISIBLE_DEVICES=0 nohup "$PYTHON" sam3_ovis/eval/v4_s0_c0_run.py \
    --n-images 100 --seed 20261025 --tile 1024 --overlap 200 --pool-floor 0.05 \
    > "$SAM3_OVIS_LOG_ROOT/v4_s0_c0_run.log" 2>&1 &
# 断点续跑：同参数重跑即可（done ledger 跳过已完成图）
# CPU 分析阶段（不占 GPU）
"$PYTHON" sam3_ovis/eval/v4_s0_c0_analyze.py
# 输出：stdout 摘要 + $SAM3_OVIS_LOG_ROOT/v4_s0_c0_stats.json
```

- 成本预估：tiles/图中位 6（摸底），大图至 ~40 tile；每 tile 16 次 text-only
  forward。参照 t2 cache 与摸底报告"~100 图 GPU 单卡数小时"的量级，预计
  **2–5 小时**（RTX 3090 单卡），峰值显存 **~6 GB**（bf16, resolution 1008）。
  GPU 阶段不含 NMS/融合，无额外显存峰值。
- 自检（已通过，2026-10-25）：两脚本 `py_compile` 通过；run 脚本 10 组、
  analyze 脚本 7 组 CPU 断言通过（tile_anchors 原点/吸附/步长、global↔local
  往返、full_in_tile 四边、RLE round-trip、合成 IoU=0.5、全局 mask RLE 与 OBB
  polygon RLE 互通 IoU=1、逐类 NMS 跨类不抑制、截断两半对同一 GT 双匹配、
  边界分类、COCOeval 完美匹配 AP50=1）；另用合成 run 目录端到端验证分析
  管线（手算 IoU 0.6/2/3/0.46 场景下 a–e 全部数字吻合、difficult 剔除生效、
  崩溃半块截尾生效）。

## 4. 需主控裁定/知悉的口径注意项

1. **difficult 口径**：按任务书"difficult 剔除"执行（不进 GT、不 ignore），
   对 difficult 密集类（DOTA-v1.5 小目标类 difficult 占比高）会**低估 AP**。
   若要与外部 DOTA 榜单对比，需另跑 ignore 口径；作为 C0/C1/C2/C3 内部对照
   不受影响。
2. **oracle 双分母**（d）：`/漏检 GT` 是"可恢复率"，`/全部 GT` 才是 mAP 量级的
   headroom；两者都会写入 stats JSON，解读时不要混用。
3. **bbox_full_in_tile 用 bbox 而非 mask 判截断**：bbox 不触边但 mask 像素触边
   的极端情况会漏判为 full（摸底报告 §5 已登记同类不确定性；对比例结论量级
   无影响，候选记录保留了完整 mask 可事后复核）。
4. **prompt 未改写**：连字符类名直接作 text prompt（如 `soccer-ball-field`）。
   若 SAM3 文本编码器对此敏感，c/d 会系统性偏低——这本身是可报告的观察，
   不在本统计里做 prompt 工程（避免选参嫌疑）。
5. **NMS/匹配阈值 0.5**：与 design §5"所有变体使用相同候选阈值与融合器"一致，
   不在本统计中搜索。
6. 抽样 100 图中边界截断实例绝对数很小（摸底：跨窗+corner ≈ 0.73%，100 图约
   5,000–7,000 实例 → 预计 <60 个）；e 的边界子集单列仅作方向参考，显著性
   不足以单独作为准入判据；准入判据以 c/d（邻窗漏检空间）为主。
