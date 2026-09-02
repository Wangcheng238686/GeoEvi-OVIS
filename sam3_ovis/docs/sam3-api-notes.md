# SAM3 API 面验证笔记（S1.2）

> 复跑入口：`sam3_ovis/inference/validate_api.py`（tokens / contract / exemplar
> / multi 四个子命令）；权重与路径经 `scripts/load_environment.sh`。
> 验证日期 2026-09-02；env = `/data/wangcheng/envs/sam3`（torch 2.10.0+cu128）；
> checkpoint = `$SAM3_CKPT`（facebook/sam3 `sam3.pt`，sha256 见 scripts/README.md）。

## 0. 使用前置（两个官方坑，先记）

1. **必须全局 bf16 autocast**：官方 notebook 显式
   `torch.autocast("cuda", dtype=torch.bfloat16).__enter__()`。ViTDet FFN 走
   `sam3/perflib/fused.py::addmm_act`，无条件把 FFN 输出转 bf16；纯 fp32 推理
   会在 `F.linear` 报 "BFloat16 and Float" dtype 混用。我们的两个入口脚本均已
   照官方方式包 autocast（不改 vendored 代码）。
2. **`Sam3Processor.set_image` 传 PIL 对象，不要传 HWC numpy**：numpy 分支取
   `shape[-2:]` 当 (H,W)，对 (H,W,C) 数组会读成 (W,C)，导致后续 mask
   interpolate 到转置尺寸。官方 notebook 一律传 PIL。

## (a) 概念分割输出契约 —— PASS

`processor.set_text_prompt(state, prompt)` 返回 state，含：

| 字段 | 形状 / 类型 | 说明 |
|---|---|---|
| `masks` | `[N, 1, H, W]` **bool**，原图分辨率 | 二值实例 mask（logits>0.5） |
| `masks_logits` | `[N, 1, H, W]` float32 | ⚠️ 名不副实：已是 **sigmoid 后的概率**（interp 后统一 sigmoid），不是原始 logit |
| `boxes` | `[N, 4]` float32，**xyxy 绝对坐标** | 由 cxcywh→xyxy 后乘原图 (W,H) |
| `scores` | `[N]` **bfloat16** | `sigmoid(pred_logit) × sigmoid(presence_logit)`（presence 门控已乘入） |

- 候选过滤在 processor 内完成：`out_probs > confidence_threshold`（默认 0.5，
  可 `set_confidence_threshold` 调）。
- 实测（test_image.jpg + "shoe"，thr=0.5）：12 实例，score∈[0.72, 0.89]；
  thr=0.05 时 43 候选 → 阈值敏感，P0 harness 需把阈值作为登记超参。
- `presence_logit_dec` 的门控意味着"该概念是否在场"参与打分——图 v0 可直接
  复用该信号做 unknown 判断的输入之一。

## (b) visual exemplar 输入契约 —— PASS

- 接口：`processor.add_geometric_prompt(state, box, label)`，box =
  **归一化 cxcywh**（[cx,cy,w,h]∈[0,1]），label True=正 / False=负。
- 多 exemplar **累加**（逐个 add），`reset_all_prompts(state)` 清空。
- 实测：正框（画面中心 0.3×0.3）+ 负框（角落）→ 1 实例，score 0.914。
- 与 text 组合：`set_text_prompt` 后再 add 几何 prompt 可行（geometric 作为
  额外过滤/定位证据）；纯几何（无 text）也允许，模型走 "visual" 通道。
- ⚠️ processor 面上只暴露 **box** exemplar；mask exemplar / 独立 ROI exemplar
  图（SAM3 论文的 image exemplar）在 image processor 内无直接接口，P3 如需
  要，走 `build_sam3_video_model`/veval 路径或查 `SAM3InteractiveImagePredictor`
  （`enable_inst_interactivity=True`，SAM1 式点/框）。已在 P3 排期中标注。

## (c) 文本 32-token 上限 × 遥感类名 —— PASS（E3 解除）

- 运行时确认：`SimpleTokenizer(bpe_path=…, context_length=32)`，**默认即 32**
  （tokenizer_ve.py 的 DEFAULT_CONTEXT_LENGTH=77 只是基类默认，SAM3 builder
  显式传 32）。
- 实测 100 个类名（SCORE 五个 split 全量）：SIOR 20 / NWPU 10 / SOTA 18 /
  FAST 37 / iSAID 15，token 数 2–6，**零截断**。最长：`expressway-service-area`、
  `expressway-toll-station`、`soccer-ball-field`、`ground-track-field`、
  `ground_track_field`、`Ground_Track_Field`、`Soccer_ball_field`（6）。
- FAST 细粒度航型号（A220 / Boeing737 / C919 等）为 5 token，完全无压力。
- OVRSISBenchV2（2026-09-02 补齐）：从官方 [Pi-Seg 仓库](https://github.com/LiBingyu01/Pi-Seg)
  的 17 个 register_*.py 抽取全部类别名，去重后 **103 个唯一类名**（34 训练 +
  83 下游并集；论文的"128 类"为跨数据集重复计数），token 数 2–6，**零截断**。
  清单入库 `data/vocab/ovrsisbench_v2_class_union.txt`，复跑：
  `tokens --classes-file data/vocab/ovrsisbench_v2_class_union.txt`。
  **S1.2(c) 至此全部完成：203 个遥感类名实测，无一超 32-token。**
- 结论：遥感类名是短名词短语，32-token 上限对本任务无实际影响；风险 E3 从
  "低" 降级为 "已解除"（登记后不回改）。

## (d) 多短语单图行为 —— 结论明确

- `set_text_prompt` **替换**式：每次调用擦除上一个 text prompt（docstring 明示
  "will erase the previous text prompt"）。
- 实测：同一图，"shoe"→12 实例；"person"→0；**联合串 "shoe; person"→0**——
  整串被编码为单一概念查询，分号不产生多概念分解。
- **对本方案的含义**：竞争类联合推理（design §5.2 相似类竞争）在 API 面上
  必须实现为"每图 × 每类各跑一次、底层图像特征共享"——`set_image` 的
  backbone_out 可复用（state 缓存），仅文本与 grounding 重算；这与 S2.2
  harness 的"每图 × 每类短语"设计完全吻合，且 `set_image_batch` 支持批量图
  （P0 大幅图分块模式可利用）。
- 另注意：文本 prompt 替换后 `geometric_prompt` 仍在 state 里，跨类复用 state
  时如需干净竞争比较，应每类 `reset_all_prompts` 或新建 state。

## 复现记录

```
$PYTHON sam3_ovis/inference/validate_api.py tokens --dataset SIOR|NWPU|SOTA|FAST
$PYTHON sam3_ovis/inference/validate_api.py contract --image $SAM3_REPO/assets/images/test_image.jpg --prompt shoe
$PYTHON sam3_ovis/inference/validate_api.py exemplar --image .../test_image.jpg
$PYTHON sam3_ovis/inference/validate_api.py multi --image .../test_image.jpg --phrases shoe person
$PYTHON sam3_ovis/inference/demo_text.py --image .../test_image.jpg --prompt shoe   # S1.1 出口
```
