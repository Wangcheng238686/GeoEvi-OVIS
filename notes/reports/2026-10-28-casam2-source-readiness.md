# CA-SAM2 源码实施准备复核

日期：2026-10-28。范围：官方论文与源码静态深读，为后续 SAM3 同图关系记忆提供接口参考；没有安装其环境、加载权重、运行第三方 Python、执行训练或验证性能。

## 1. 证据与结论

- 官方仓库：[HanbinHuang123/CA-SAM2](https://github.com/HanbinHuang123/CA-SAM2)。本次 clone 得到 SHA `7d85ff493c463774fc8b83515d384756aeb61a67`，与 10 月 25 日草案一致。
- 官方论文：[MICCAI 2025 PDF](https://papers.miccai.org/miccai-2025/paper/1881_paper.pdf)，已下载至本次只读检查目录 `/tmp/geoevi-casam2-0SrQEpiv/paper.pdf`；仓库位于同目录 `source/`。临时目录不是项目依赖，不能假设长期存在。
- 下文 `file:line` 均相对此固定 SHA，持久链接可用 `https://github.com/HanbinHuang123/CA-SAM2/blob/7d85ff493c463774fc8b83515d384756aeb61a67/<file>#L<line>`。
- 已读 `notes/decisions-2026-10.md` 与 `2026-10-25-graph-memory-candidate-routes.md`。用户要求同图建库、逐原图清空、SAM3 遥感微调、尽量保留原生 memory 主体、测试仅图像与文本。

**判断：可参考双 bank 的职责分离、距离/相似度检索、memory 读写接入点；不能整体移植该训练/验证循环。** 其 texture bank 生命周期与我们的约束不同，且存在列表别名、硬编码布局、论文/实现不一致等问题。它也没有图，没有文本开放词汇类别预测，更没有 SAM3 tracker→detector 的回接。

## 2. 实际 pipeline 和 bank 内容

论文 §2.1–2.2：256×256 patch、stride 164（overlap 92），中心向外螺旋扫描；每个 patch 先进行环境读取，再纹理读取，再解码实例 mask。环境 bank 保存同图既往 patch 的 image embedding；纹理 bank 保存图像与预测 mask 编出的 memory，使用多样性和预测 IoU 管理。

源码主流程：

```text
point_encoder(image) → SAM2.forward_image(image, domain_features)
    → vision_feats[-1], vision_pos[-1]
    → 最近空间 patch 的 context attention
    → 相似 patch 的 texture attention
    → 点 prompt encoder → 多实例 mask decoder
    → 合成一张实例 ID 图 → _encode_new_memory
    → 保存 texture memory；另存 context feature
```

| 项目 | 静态证据 | 实际含义 |
|---|---|---|
| 编码与读取顺序 | `run/run_on_epoch.py:66–127`、`:620–677` | context 后 texture，两次独立 reader 调用 |
| context 记录 | `:251–260`、`:685–693` | `[image_feature, local_position, x_origin, y_origin]`，每条是 patch，不是实例 |
| texture 记录 | `:212–217`、`:485–491` | `[mask_memory, position, mean_predicted_iou, image_key]`，每条仍是 patch，多实例已合并 |
| context 检索 | `:972–1006` | 用 patch 左上角坐标欧氏距离排序，依次读取最近 K 条；默认 K=1 |
| texture 检索 | `:97–127`、`:649–677` | key 与当前 context-conditioned feature 的展平向量 cosine；`topk(..., B)`，检索条数绑定 batch size，验证 B=1 |
| texture 更新 | `:219–250`、`:493–518` | 根据 memory 相似性与平均预测 IoU 替换；不是基于人工 GT IoU |
| memory 编码 | `sam2_train/modeling/sam2_base.py:665–703` | 低分辨率特征与 mask 融合，经原生 memory encoder 输出 memory 与位置 |
| reader 修改 | `sam2_train/modeling/memory_attention.py:113–117,155–183` | 原四层 texture reader 外，新增一层独立 context reader 与一个 stride-2 conv；不是完全原封不动复用原生模块 |

图像坐标只参与 bank 排序；没有将全图 x/y 编入原生 reader 的 positional encoding。读取仍使用 patch 内位置与 RoPE，不能据此声称已解决遥感空间坐标适配。

## 3. 生命周期：必须改写

`main.py:125` 每个 epoch 初始化 texture bank，然后整个 epoch 跨训练图更新。训练循环在每张图仅清空 context（`run/run_on_epoch.py:29`）。验证接收训练 texture bank 的浅拷贝（`main.py:152`）；独立 eval 从 checkpoint 载入 texture（`:74–78`），并在验证内持续更新（`run/run_on_epoch.py:485–518`），新验证图只清空 context（`:311`）。论文 §3.1 也明确训练 TMB 存入 checkpoint 供测试使用；测试图之间继续写入的具体行为来自源码。

这不是本项目允许的协议。我们需要 `begin_image(image_id)` 令所有样本 bank 同时为空，`end_image()` 丢弃全部记录；图内 tiles 共享原图 ID。checkpoint 只保存模型、optimizer 和训练状态，不保存测试样本 bank。任何跨图 texture 原型若未来要研究必须另设明确协议，不能沿用此实现暗中引入。

其 context 仅历史 patch 可见，因此有扫描顺序依赖；训练按 batch 读后写，同一 batch patch 不能互读，`cfgs.b` 会同时改变检索量与可见上下文。我们的两遍预测快照方案应明确全部第一遍记录建成后只读，避免 batch/scan order 改变机制。

## 4. 张量契约与不应复制的细节

这里列的是 **CA-SAM2 的 256 输入配置**，不是 SAM3 已验证形状：

| 数据 | 布局与维度 | 证据 |
|---|---|---|
| 三尺度特征 | spatial sizes `(64,64),(32,32),(16,16)`；低尺度 `[256,B,256]`（L,B,C） | `run/run_on_epoch.py:24,129–137` |
| texture memory | `[B,64,16,16]` → `[256,B,64]` | `:193–200,93–94`；YAML `kv_in_dim:64` |
| context memory | `[256,1,256]`，依 query batch 组装为 `[256,B,256]` | `:257–258,995–996` |
| texture key | `[256*16*16]` | `:217,653–658`；`reshape(B,-1,64,64)` 后又 flatten，空间重排本身不参与最终 cosine |
| predicted mask | 多实例 logits 先 threshold/RLE/NMS，合成实例 ID 图 `[B,1,256,256]` | `:182–197,806–875` |

需特别审查：

1. **context 特征并非可靠的原始快照。** `memfeatures = vision_feats` / `memfeatures_pos = vision_pos_embeds`（训练 `:72–73`、推理 `:629–630`）只是 Python 列表别名；后续 `vision_feats[-1] = ...` 覆盖的是同一个列表。写入 context 的实际是经历读取后的 feature，与“Original image embedding”注释不符。应在读前保存 tensor 引用或显式 immutable snapshot，并以测试证明未被覆盖。
2. **context reader 的布局很特殊。** `memory_attention.py:155–157` 在 batch-first 后将 `[B,256,256]` 直接 `view(B,256,16,16)`、stride-2 conv、`flatten(2)`，得到 `[B,256,64]`；没有先按常规 LBC→BCHW permute，也没有将 `[B,C,H*W]` 转为 `[B,H*W,C]`。结果把 256 轴作为 token、64 轴作为 kv channel，能匹配 `kv_in_dim=64`，但语义上不能直接当作常规空间 token 下采样。我们需自行定义、断言正确布局；是否维持该特殊变换是单独消融，不能静默照搬。
3. **实例 ID 被当 mask logits。** `combine_mask` 返回 0/1/2/... ID 图，而 `_encode_new_memory` 默认对输入 sigmoid 再 scale 20、bias −10（`sam2_base.py:686–696`；`sam2_hiera_l.yaml:91–92`）。0 对应 0.5，不是背景低概率；不同 ID 也有不同强度。若其意图是前景纹理，移植时应用二值前景或保留单实例概率/logit 契约，不能将 ID 图直接输入。该判断来自算子静态推导，尚无权重实测影响。
4. **context 容量满后停止写入。** `:255` 仅当 `len+B<=capacity` 才追加，不是 FIFO；不能称其保留最近 patch。
5. **纹理检索 K 与 batch size 耦合。** `topk(similarity_scores,B)` 在 bank 长度小于 B 时可能失败；首个 batch 写入也可能越过容量。必须独立 `K=min(requested_K,len(bank))` 并验证 B=1、B>1 的输出布局。
6. **验证辅助列表会积累。** `to_cat_memory/*pos/*image_embed` 在每张图开始初始化（`:308–310`），`inference` 每次 append 全 bank（`:642–647`），没有逐次清空。它们跨 patch/TTA 调用重复收集旧记录；实际检索候选可能包含多份旧 bank，不等价于仅当前 bank。新实现应每次从 snapshot 局部构造 tokens。
7. **单候选 patch 被跳过。** 验证 `:407–408` 对 0 或 1 个点都 continue；不能拿来适配稀疏遥感实例。多实例 nuclei 假设也体现在 `inference :712` 用 `len(gt_points)` 参数，尽管实际 prompt 来自预测；我们的测试接口不应含 GT 参数。
8. **非正方形切窗风险。** `run/dataset/cpm.py:239–284` 把按 Y 长度定义的 `down` 用作 `X_points[down]`、按 X 长度定义的 `right` 用作 `Y_points[right]`。正方形网格可能掩盖错误；长条遥感图必须重写扫描器并测覆盖与唯一性。`stride=256-overlap`（`:206`）也无视可变 crop size。
9. 空 bank 临时创建 `nn.Parameter(zeros)`（`:975–976` 等）没有注册到模块/optimizer，本质只增加零，不是可训练的 no-memory embedding。设备也部分硬编码 `.cuda()`。应使用原生 no-memory 路径或注册参数，跟随 tensor device/dtype。
10. checkpoint 用 `strict=False` 且屏蔽 missing/unexpected 报错（`sam2_train/build_sam.py:79–90`）；新增 context 参数可能随机初始化，不能因加载成功断言其有预训练权重。

## 5. 论文—实现差别与梯度边界

论文 §2.2 的 TMB 满库策略：先把新记录加入，计算全部 pairwise similarity，移除相似度总和最大的冗余项，再按 IoU 容差检查。代码 `run/run_on_epoch.py:222–245` 实际先找与新记录最不相似的旧项 a，再找旧库中与 a 最相似的 b，满足条件后替换 b。两者不是同一算法，报告应注明“源码式无图对照”还是“论文字面式对照”。

论文 memory 记号容易让人误以为按 memory embedding 检索；源码读 key 是 image embedding cosine，替换时才用 mask-memory cosine。应分别记录 `retrieval_key` 和 `encoded_value`，不可混用。

训练存储 texture feature、position、image key 与 context feature 全部 `.detach()`（`:214–217,247–258`）。`combine_mask` 也通过 threshold、RLE/NumPy 丢失 mask 梯度（`:806–875`）。当前 loss 来自 decoder 输出（`:263`），新增 `_encode_new_memory` 输出仅被 detach 存储，因此按本训练路径 **memory encoder 没有通往 loss 的可微路径**；原生 reader 和当前特征路径可微。此为静态计算图判断，未做 backward 实测；不能声称该仓库实际联合训练了 memory encoder。

主干冻结策略为 `main.py:42–44`：image_encoder 除 prompt_generator 外冻结，不符合用户本轮希望适度微调主干的约束。可学习域分支和 MFRB 属于参考工作本身，不能无代价作为我们的简单 adapter；具体 SAM3 解冻范围应另登记。

## 6. 我们需要自己实现的最小接口

以下是后续接口草案，不是已落地代码：

```python
MemoryRecord(
    image_id, record_id, role, source_prediction_id,
    value_LBC, position_LBC, retrieval_key_C,
    global_bbox_xyxy, local_grid_hw, source_transform,
    predicted_quality, text_embedding, predicted_mask,
)

begin_image(image_id)  # 清空全部样本记录
observations = first_pass(image, texts)  # 训练测试均预测建库
snapshot = encode_snapshot(observations, sam3_native_writer)
for target in observations:
    ids = selector(target, snapshot, budget)  # 无图距离/相似度或 R1/R2/R3
    memory, position, metadata = assemble(ids, target)
    updated = native_reader(target.feature, memory, position)
    prediction = detector_reentry(updated, target, texts)
end_image()  # 第二遍不回写；无跨图样本缓存
```

必须自己写并核验的部分：原图 ID 与生命周期管理、实例/环境节点而非整 patch 混合节点、预测 quality、三条图干预、token budget、同图坐标/role 适配、SAM3 原生 writer/reader 的实际接口、reader→概念 detector 的回接与微调方案。CA-SAM2 没有这些模块；它改善 nuclei mask 不证明记忆能改善开放词汇类别判断。

写端梯度有两种需要预登记的合法选择：缓存 detach、只训练 reader/当前主干；或同图两遍保留必要图，使第二遍 loss 训练 writer。两者训练成本和可学习组件不同，不能无声切换；若设计声称微调 writer，应对 writer 梯度做正向验证。

无图 U 对照应同图两 bank、同样第一遍预测、同样原生模块与回接、相同 memory/token/训练预算，只用距离/相似度检索。由于逐图 reset 与不可变快照改变了 CA-SAM2 原协议，应称“CA-SAM2 启发的同图无图基线”，不能称论文复现。

## 7. 实施前最小测试清单

下列尚未执行；是进入实现后的验收要求：

1. **reset / 无标签接口：** 图 A→B 时 bank=0；打乱评测图序不改变每图结果；测试 memory record 无 GT mask/类别字段、checkpoint 无样本 bank。
2. **布局：** 不同 H×W、不同 B、空库、1 条记录、K>len(bank)；对 token/channel 做带坐标的合成 tensor 检查，不能仅看 shape 不报错。
3. **快照：** reader 前后原始记录 tensor checksum 与对象引用保持约定；TTA/第二遍不会重复 append 当前记录。
4. **mask 契约：** 空 mask、概率 mask、二值前景、单实例与重编号实例 ID；ID 重编号不应改变同一前景内容的 memory。
5. **梯度：** 对主干选中层、writer、reader、graph/controller、detector adapter 分别检查有限非零梯度，并验证 detach 开关与设计一致。
6. **信息路径：** 正确/打乱/空 memory 在相同 query 下是否改变类别 score 与 mask；同时测无缓存一次性 context attention，防止把任意残差扰动当记忆作用。
7. **扫描与预算：** 500×2000、2000×500、边长小于 tile、非整倍数；tile 覆盖完整且无非法索引；K 与 B 解耦；固定 graph/无图总 token 和 reader 调用数。
8. **公平干预：** 同图无图 U、随机边、正确边、错误环境；基于预测建库的主结果与 GT oracle 分开。

## 8. 许可与复现状态

在此 SHA 的完整 Git tree 中没有 `LICENSE`、`COPYING` 或 `NOTICE` 文件；README 只有引用与致谢。多个 SAM2 文件头（例如 `memory_attention.py:1–5`）指向根目录 LICENSE，但该文件缺失。公开可读不等于已获明确授权复制其新增代码。当前可以继续阅读、引用和按接口独立实现；计划直接复制其实现并发布前需向作者确认适用许可，并分别核验 SAM2、MedSAM2、PromptNucSeg 及本项目 SAM3 的上游条款。这里未推定其属于任何特定开源许可证。

本次证明的是“源码与论文有可用的结构参考，并定位了移植边界”；未证明它能在现有环境运行、checkpoint 完整、论文数字可复现、SAM3 可接入、图有效或遥感迁移有效。
