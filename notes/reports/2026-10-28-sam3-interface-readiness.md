# SAM3 原生记忆与概念检测回接：实施准备核验

日期：2026-10-28。范围：只读源码检查与 CPU 小张量诊断；没有修改 vendored SAM3、安装依赖、占用 GPU 或训练模型。约束以 `../decisions-2026-10.md` 为准。

## 1. 判断与版本

**原生 writer/reader 可以保留；同图关系记忆尚不存在现成的端到端接线。** 最先需要验证的不是图模型，而是「原生 memory → detector query → 文字条件分数」的信息路径。当前 image detector 不会自动读取 tracker memory。

本次项目 HEAD 为 `88c5216b174286b5a730550dec139b1663f6bc28`。`git -C sam3 rev-parse --show-toplevel` 返回项目根目录，说明它不是独立 SAM3 git checkout；不能把此 SHA 与 10/25 的项目 SHA 差异解读为上游 SAM3 升级。tracker 文件最后提交为 `a569b910611fc261e8960ee3806b297bb835fbc7`（2026-09-02）；API 判断以本次文件内容为准。import 后确认 memory、decoder 模块均来自本工作树 `sam3/sam3/model/`。

已有结论不能忽略：`v3-frozen-eval-results.md` 已关闭空间相对惩罚主张；`v4-s0-c0-verdict.md` 已停止跨窗正框传播。它们不等价于证明可训练原生上下文记忆失败，也不能继续使用旧的“v3 等待验证 / v4 主线进行中”状态。本报告不借边界实例恢复为新方案的核心动机。

## 2. 可复用模块与张量契约

以下路径均相对仓库根目录，行号对应本次工作树。

| 接口 | 源码 | 契约 / 限制 |
|---|---|---|
| tracker writer 构造 | `sam3/sam3/model_builder.py:344` | `SimpleMaskEncoder`，视觉输入 256 通道，输出 64 通道；mask 先插值至 1152×1152 再 /16 为 72×72 |
| writer 包装 | `sam3/sam3/model/sam3_tracker_base.py:796` | 当前特征 `[HW,B,256]`、mask logits `[B,1,h,w]`、object score；返回 `[B,64,H,W]` 与同形 position 列表 |
| writer 内部融合 | `sam3/sam3/model/memory.py:187` | 像素投影 + mask 下采样 + ConvNeXt fuser + 256→64 投影；mask 下采样图和视觉图必须同尺寸 |
| tracker reader 构造 | `sam3/sam3/model_builder.py:380` | 4 层，query=256、key/value=64，RoPE，自注意力及跨注意力；实际模块名 `transformer.encoder` |
| reader 直接调用 | `sam3/sam3/model/decoder.py:693` | `src=[HW,B,256]`、`src_pos` 同形，`prompt=[KHW,B,64]`、`prompt_pos` 同形；列表形式要求 `src_key_padding_mask=[None]`；输出 `memory=[HW,B,256]` |
| tracker 取库包装 | `sam3/sam3/model/sam3_tracker_base.py:559` | 默认取历史条件帧/非条件帧、加时间位置、附 object pointers；不是通用同图图检索函数 |

writer 包装会对 mask logits 做 sigmoid，再乘 20、减 10（tracker `:115`, `:819`）；直接调用 `maskmem_backbone` 时必须自行遵循该分布契约，不能把二值 mask 当 logits 再 sigmoid。低 object score 会加入 no-object spatial embedding（`:843`），不能把图节点质量分未经校准塞进 object score 来充当图门控。`is_mask_from_pts=True` 的 eval 路径有硬阈值，训练模式则不走此分支。

memory 返回值本身没有 detach；`_maybe_clone` 仅在编译时 clone（tracker `:1172`），不截断梯度。是否对第一遍预测 / 特征 stop-gradient 是我们需要明确登记的训练选择，不是原生模块的必然行为。

## 3. 不能机械复用的细节

### 3.1 原生 reader 要完整网格，不能直接塞任意图节点向量

`sam3/sam3/sam/transformer.py:309` 用 query token 数的平方根重建方形 RoPE；`:318` 排除末尾 object pointer 后，对 memory 栅格重复 query 的 RoPE。`sam3/sam3/sam/rope.py:61` 按整数倍重复频率。

因此第一版应保持每条 memory 为同尺寸完整栅格，图选择的是 K 条记录，而不是 K 个池化 token。任意 pooling / token pruning / 矩形网格 / 不同分辨率 crop 都不是直接兼容。把所有节点冒充 object pointers 虽可能绕开旋转长度，但改变了预训练语义，不能称为原生栅格 memory 的无修改复用。

完整栅格也有代价：72×72=5,184 token；一条 64 通道 BF16 memory 约 0.633 MiB，若 position 同精度再约 0.633 MiB，不含第一遍特征、attention 激活及反传。4 层 reader 对每个目标重复完整画布，成本会随目标数和 K 增长。不能只报告小型图控制器参数量来代表成本。

### 3.2 时间、空间与身份

tracker 包装的条件记录用同一个时间位置，非条件记录按历史顺序加时间 embedding（tracker `:601`, `:678`）。二维上下文没有真实帧序，不能把图节点编号作为时间距离。建议首个接口实验在同一图像画布建库，固定相同的条件 memory slot 作为无图/图共享设置，并做记录顺序置换检查。后续若引入关系位置编码，要列为显式适配，而不是声称原生 RoPE 已表达全图关系。

原生 object pointer 是同一对象历帧的身份线索（tracker `:695`）。其他实例及环境通常没有当前对象的合法 pointer；第一轮可用 `num_obj_ptr_tokens=0` 直接读 spatial memory。若保留身份 bank，只有该实例自身观察使用身份 pointer，其它角色另编码并单独对照。

同图相同画布的不同 mask memory 中，像素特征仍然覆盖整图：writer 是相加 mask，并未把 mask 外视觉特征清零（memory `:203`）。所以“选择了哪个 mask 记录”不等于“只读到哪个实例的像素”。环境区域仍会带入当前对象视觉信息，需另设非 mask memory / 重复同特征不同 mask 对照，验证结果到底来自写入对象还是共享整图特征。预训练特征的全局感受野也使严格像素级排除不可能凭一个 mask 实现。

### 3.3 空库与设备

tracker `:578` 的 `num_maskmem==0` 直接返回原始视觉特征；`:767` 的 init/no-prev 分支返回视觉特征 + 256 维 `no_mem_embed`，二者并不相同。`:773` 后的 dummy token 代码在 return 后不可达，而且其 256→64 expand 不能作为修复方案直接复用。wrapper 非初始分支还要求至少一个 cond output（`:590`）并含 `.cuda()`（`:657`, `:665`），不适合直接当 CPU 通用 bank 接口。

共同 bridge 应明确规定空库时 residual 为零，输出退化为同一基线；不能让空库仅因加不同 embedding 就改变模型。额外比较 init/no-memory 选择需单独登记。

此外 `PositionEmbeddingSine(precompute_resolution=1008)` 在构造时硬分配 CUDA tensor（`sam3/sam3/model/position_encoding.py:52`）；仅 `device='cpu'` 不保证完整 builder 可在无 GPU 环境构造。小张量 writer 使用 `precompute_resolution=None`，不是已完成原生尺寸 CPU 全模型验证。

## 4. Detector 回接：实际影响类别的路径

默认 `build_sam3_image_model(... enable_inst_interactivity=False)`（builder `:575`）只构造 detector。打开该选项才建立 tracker（`:628`）和 sam2 neck（`:110`）。`_load_checkpoint` 只在 tracker 存在时加载对应权重（`:548`）。必须记录 missing/unexpected keys；“构造了模块”不足以证明已载入预训练 memory 权重。

`Sam3DualViTDetNeck` 有独立 `convs` 与 `sam2_convs`（`sam3/sam3/model/necks.py:98`）；共享 trunk 不等于共享 neck。`SAM3VLBackbone` 返回 `sam2_backbone_out`（`sam3/sam3/model/vl_combiner.py:101`），tracker 原生使用它；detector 从自己的 `backbone_fpn` 取特征。两者都是 256×72×72，但语义分布和权重不同，不能只按尺寸直接互换。

概念路径清楚可追：`sam3_image.py:441` → `_run_encoder`（`:213`）→ `_run_decoder`（`:253`）→ `_update_scores_and_boxes`（`:297`）→ `dot_prod_scoring(hs,prompt,prompt_mask)`（`:321`）。`DotProductScoring` 在 `model_misc.py:763` 投影 query 和池化 prompt 后取点积，输出每个文字条件 query 的单个匹配 logit，而不是一个固定 K 类 softmax。改变 tracker mask 结果不会自动改变这些 logits。

可比较的 bridge 候选：

| 候选 | 接线 | 优点 / 限制 |
|---|---|---|
| F：空间特征残差 | tracker reader 的 256 维栅格输出，经小型投影，将相对无 memory 参考的增量加到 detector 对应 feature，再运行 detector encoder/decoder | 类别、box、mask 都可受影响，较完整；目标特异检索若逐目标执行，会重复整图 detector；共享一次检索又弱化实例条件选择 |
| Q：目标 query 残差 | 对 reader 栅格增量按第一遍预测 mask 池化并投影，加到相应 detector query 的 `hs`，再走原生 scoring；mask 可用同一更新 query 重算 | 首个信息路径诊断优先，目标ID易追踪、成本较可控；仍需整图 reader/目标或共享bank方案；固定候选无法修复首遍零响应漏检，不能充当最终召回方案 |
| P：视觉 prompt 回接 | reader 特征池化成 prompt token，调用现有视觉 prompt 接口 | 可跑，但 prompt 会进入编码和最终 prompt pooling，可能直接改变类别模板语义/提示数量，归因不清；暂不优先 |

建议先用 Q 验证“换上下文能改变同一实例对不同文本的兼容分数，梯度可达原生 writer/reader”，并以 F 作为完整检测/分割可行性检查。不能在未测吞吐前冻结最终 bridge；也不能把 Q 的候选重排包装成完整检出能力改善。

共同接口草案：`write_record(canvas_features, predicted_region_logits, source_score, metadata)` 返回 `{image_uid, record_uid, source_candidate_uid, role, memory:[B,64,H,W], pos:[B,64,H,W]}`；`read_records(target_uid, canvas_features:[HW,B,256], canvas_pos, selected_record_ids)` 只接收同图、同canvas的完整网格记录，排除当前目标自己的候选记录，拼接至 `[KHW,B,64]`，返回 `[HW,B,256]`。K=0返回零增量。相同物理目标的跨文本重复候选也须按预测共指组排除，不能只排一个类别编号下的uid；共指不确定性另外记录，不能用GT认定。

Q诊断先对少数固定目标逐目标读完整canvas，再在该目标预测mask上池化相对于共同no-memory路径的增量，得到 `[B,256]`，小投影加到对应query；不能把池化后的单query直接作为原生reader query，否则空间RoPE语义改变。F候选可在目标区域对reader增量加权后叠加为一张共享detector残差画布，重跑整图detector；但多个目标的残差冲突、全局self-attention影响非目标、query匹配变化均待测。逐目标重跑整个detector是昂贵上界对照，不建议首个大规模方案。ROI裁剪reader涉及新的坐标/尺度适配，先后置。此草案尚未实现或测过detector桥接。

Q 要在 `_update_scores_and_boxes` 之前更新 `hs`，或明确只重算 scores 而保留 boxes/masks 作诊断。训练含 DAC/O2M，`hs` 是 `[layers,B,N,256]`，源码 `:307` 拆 O2O/O2M；只修改 `out['queries']` 不会回写已经生成的 logits。给一类文本的 query 注入记忆，并不等于已验证船/汽车跨文本竞争，需要显式匹配同一物理实例的候选或固定实例后逐文本评分。

## 5. R1/R2/R3 最小改动与梯度

- **R1 选择**：图在外部选择 K 条完整栅格记录，直接调原生 reader；不改 attention 主体。离散 top-K 对边评分不可导，若图边需学习，须登记连续训练代理或先规则构图，不能宣称天然端到端学习。无图距离/相似度同 K、同 reader、同桥接。
- **R2 角色**：给完整栅格的 `prompt_pos` 加 64 维角色 embedding，或在写入/读取前加角色投影；保留 RoPE 网格。保留相同 K 和集合，做角色打乱及等参数无角色对照。若直接把其他实例当自己的 identity memory，身份串扰需先测。
- **R3 门控**：硬记录准入/选择可复用 reader；真正的连续边权 attention bias 需要小改 reader。`TransformerDecoderLayerv2.forward_pre`（decoder `:982`）assert `memory_mask`、padding mask 和 `attn_bias` 全部为 None，现有接口不支持直接传图 mask。仅把 feature 乘权重并不等价于控制 attention 概率；零 value 仍占 softmax 分母，K 投影 bias 和 position 仍会产生 key。不能当成严格“拒绝读取”。

R3二值record准入与R1硬top-K一样，不会自动向门控打分器传梯度。第一接口版本可用确定性关系路由，训练reader/bridge/主干；若要学习图边或准入分数，必须另列辅助监督、连续松弛或其他可验证学习路径。第一轮R3应先采用二值完整record准入，真正attention bias后置，避免把三条路线共同接口扩大成attention重写。

不要通过 `Sam3Processor` 训练：`set_image`、`set_text_prompt` 等均有 `@torch.inference_mode()`（`sam3_image_processor.py:41`, `:113` 等）。应走原生 `BatchedDatapoint`/`model.forward`（`sam3_image.py:557`）或可微 wrapper。model.eval 不等于 requires_grad=False，但 inference tensor 不能作为保留主干训练梯度的缓存。

官方 ODinW 训练 config 给出 model `eval_mode:false`（`train/configs/odinw13/odinw_text_only_train.yaml:349`）、视觉/文本不同 LR 分组（`:388` 起）、主干 layer decay 和 AMP；它可作为主干微调训练壳参考，尚不是图记忆训练 recipe。scratch 的 freeze 字段是否被当前 builder 使用应按实例化链核验，不能只看到 `NoFreeze` 就认定主干实际训练。

主干需要通过当前路径保留梯度；第一遍建库可选择停止梯度以控制成本，但应明确：若 bank features.detach，writer 和建库主干不会从第二遍 loss 得到梯度。若只 detach 第一遍预测 mask、保留视觉特征则二者不同。初期硬 mask/argmax、object appearing threshold、proposal threshold和离散图选择会截断相应梯度，第二遍检测损失仍可训练已连接的视觉/reader/bridge参数。真正参数分组和有无梯度必须运行验证。

## 6. 本次实际执行的 CPU 证据

环境 `/data/wangcheng/envs/sam3/bin/python`，PyTorch `2.10.0+cu128`；命令统一 `CUDA_VISIBLE_DEVICES=''`。import 成功并打印模块实际路径。没有加载 checkpoint。

1. 直接使用 builder `_create_tracker_transformer().encoder.eval()`，query `[4,1,256]`、memory `[8,1,64]`，pos同形，`feat_sizes=[(2,2)]`。输出 `[4,1,256]`，平方均值反传成功：query grad norm `2.30e-7`，memory `1.23e-7`，首层 cross-attention q_proj `4.69e-7`。该损失接近 LayerNorm 常数，仅用于证明连接；非梯度大小的有效训练论据。
2. 同一 reader 换成 `[5,1,64]` memory，实际报 `RuntimeError: size ... (5) ... (4)`，验证任意节点 token 长度不兼容默认 RoPE。
3. writer 保留原生通道、ConvNeXt/fuser/输出结构，仅将 mask interpolation 改为 `[32,32]`，position 不预计算，以匹配 `[1,256,2,2]` CPU 特征。输出 memory/pos 均 `[1,64,2,2]`；feature、mask、out_proj gradient norm 分别约 `0.02075 / 0.007605 / 0.6024`。这是缩小网格组件诊断，不是 1008 输入、1152 mask、预训练权重或整模型功能验证。

为便于独立复跑，另保留固定seed脚本 `/tmp/geoevi_sam3_memory_interface_probe_20261028.py`，执行命令：

```bash
CUDA_VISIBLE_DEVICES='' /data/wangcheng/envs/sam3/bin/python /tmp/geoevi_sam3_memory_interface_probe_20261028.py
```

脚本seed=`20261028`、CPU threads=2，reader改用随机方向线性loss；本次输出 reader/query/memory/q_proj grad norm=`30.38779 / 11.35502 / 47.54651`，writer feature/mask/out_proj grad norm=`0.019150 / 0.014685 / 0.533887`，并实际确认ragged keys及非None padding mask分别报RuntimeError与AssertionError。此脚本不加载checkpoint，也不构建或测试detector bridge。

## 7. 下一步接口探针与准入顺序

1. **P0 权重与契约**：空闲 GPU 后加载本地 SAM3 checkpoint 的 detector+tracker，记录完整 key 覆盖；1008输入实际取出两套neck、mask memory、position 的 shape/dtype/device，确认 72×72。测试同图多实例 batch、记录序列、K=0/1/多条及reset；此阶段不训练。空库应与共同无memory基线数值一致。
2. **P1 可微回接**：先 Q 后 F，保留固定第一遍候选，使用非饱和原生概念 logits 做标量loss。检查主干选定层、两套neck、writer、reader、bridge 的梯度/优化器覆盖和单步参数变化；不期待第一步就有指标提升。零残差初始化会使上游第一步梯度为0，需先验证随机非零小残差或第二步，不能误判断路。
3. **P2 上下文干预**：固定图像/文本/目标/预算，替换正确邻居、远处邻居、同图随机、同值不同mask、打乱边、空bank；检查 logits/mask 变化及跨对象串扰。换memory导致数值变化只证明路径连通；有方向的类别改善需独立源图dev和负例统计。
4. **P3 规模与候选限制**：记录单图N实例、K、完整栅格token数和真实显存/耗时；分别评估 Q 固定候选分类作用空间与 F 可变候选能力。若只改变已有候选，不能用它承诺解决 v4 中的零响应漏检。
5. **P4 最小并行**：相同可微bridge/主干微调/预算下再实现 R1/R2/R3，各自对 CA-SAM2式距离/相似度检索与一次性上下文attention。R3软边权需先完成最小attention bias适配及默认等价验证。

本轮做到组件可复用和局部反传证据；尚未证明预训练端到端接线、类别收益、novel迁移或图必要性。上述P0–P4属于实施准备建议，不表示用户已授权启动训练。
