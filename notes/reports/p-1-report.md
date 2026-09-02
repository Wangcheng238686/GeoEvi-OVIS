# P-1 报告：工程与数据准备（M0）

> 阶段：P-1（implementation-v2.md §1）；报告日期 2026-09-02；
> 主机 lthpc（RTX 3090 ×4 24G，driver 570.133.20 / CUDA 12.8）。
> 状态：**S1.1 / S1.2 / S1.4 / S1.5 完成；S1.3 五个 SCORE split 全部落地冒烟
> PASS，仅 NWPU-Refer 补充集待用户百度网盘下载。M0 达成。**

## 1. 数字表

### 1.1 环境（S1.1）

| 项 | 值 |
|---|---|
| conda env | `/data/wangcheng/envs/sam3`（python 3.12.14，独立于 C5-v2 全部环境） |
| torch / torchvision | 2.10.0+cu128 / 0.25.0+cu128（3090 sm_86 上 CUDA 可用，matmul 通过） |
| sam3 | vendored 官方 repo（仓库根 `sam3/`），`pip install -e` |
| checkpoint | `$SAM3_CKPT` = `/data/wangcheng/pretrained-models/sam3/sam3.pt`，3,450,062,241 bytes，sha256 `9999e234…c8c9e`（与 ModelScope 元数据一致、与 HF 官方字节数一致） |
| 官方 demo 出口 | test_image.jpg + "shoe" → **12 实例**，score∈[0.72,0.89]，mask [12,1,720,1280] bool，overlay 落盘 |
| 网络注意 | 本机代理 7890 对 HF 时通时断；国内源（ModelScope/阿里云）必须绕代理直连 |

### 1.2 数据（S1.3，冒烟 = `tests/smoke_datasets.py`）

| 数据集 | 角色 | 图 / 实例 / 类 | GT 来源 | 冒烟 |
|---|---|---|---|---|
| NWPU | 测试 | 731 / 2,661 / 10 | 人工 | PASS |
| SOTA | 测试 | 874 / 29,247 / 18 | SAM 伪标签 | PASS |
| FAST | 测试 | 3,207 / 43,258 / 37 | SAM 伪标签 | PASS |
| SIOR（train） | 训练 | 11,725 / 68,062 / 20 | SAM 伪标签 | PASS |
| SIOR-val | 测试（协议） | 11,738 / 124,373 / 20 | SAM 伪标签 | PASS |
| iSAID（train） | 训练 | 28,029 / 704,684 / 15 | 人工 | PASS（zip 全档 CRC 校验通过） |
| iSAID-val | 测试（协议） | 9,512 / 235,531 / 15 | 人工 | PASS |
| NWPU-Refer 子集 | 补充 | – | referring 人工 | **待用户百度网盘**（官方仅发网盘，HF 无镜像） |

注：iSAID 训练图数 SCORE 论文称 18,732、其 HF json 实为 28,029；split 以
HF 发布为准（E4），差异在 gt-provenance.md 双登记、不裁决。

### 1.3 API 面（S1.2，详见 `sam3_ovis/docs/sam3-api-notes.md`）

| 检查 | 结论 |
|---|---|
| (a) 输出契约 | masks [N,1,H,W] bool；boxes [N,4] fp32 xyxy 绝对坐标；scores [N] bf16 = sigmoid(logit)×sigmoid(presence)；`masks_logits` 实为 sigmoid 后概率（命名陷阱） |
| (b) exemplar | add_geometric_prompt（归一化 cxcywh，正/负 label）可用、可叠加；processor 无 mask/独立图 exemplar 接口（P3 需走 veval/interactive 路径） |
| (c) 32-token | 运行时确认 context_length=32；SCORE 五 split 共 **100 类名 token 2–6，零截断**（iSAID 数据落地后补跑完成）；OVRSISBenchV2 128 类名清单待取（脚本支持 `--classes-file`） |
| (d) 多短语 | set_text_prompt 为替换语义；联合串 "shoe; person" 整体成单概念查询返回 0 → 每图×每类独立调用，图像特征可共享（与 S2.2 harness 设计吻合） |

### 1.4 评测管线（S1.4）

- `eval/score_eval.py`：逐数据集 COCO segm AP@[.50:.05:.95]（pycocotools 默认
  maxDets）→ 类平均 mAP；跨数据集 Average = 无权平均；逐类 AP 从 precision
  张量聚合。口径预登记于 `eval/README.md`，S2.1 gate 前对照 SCORE 主表复核。
- 诊断量（自用，R3）：recall proxy（G1 同形）/ precision proxy /
  conditional IoU，类平均 + micro 双口径。
- 自测：合成 GT + 伪造预测（漏检/重复/偏移/纯 FP）双环境 PASS（base + sam3）。

## 2. 判据核对（对照 implementation-v2 §1 完成判据）

| 子任务 | 判据 | 状态 |
|---|---|---|
| S1.1 | 可 import env + 权重路径进 environment*.sh；官方 demo 单图 text prompt 跑通 | ✅（含 license-note.md 登记结论；E1 维持"低"） |
| S1.2 | (a)–(d) 各有可复跑脚本与结论段落 | ✅（validate_api.py 四子命令 + sam3-api-notes.md；(c) iSAID/OVRSISBenchV2 补跑项已列） |
| S1.3 | 每数据集 loader 冒烟通过（图数/实例数/类别表与官方一致）；GT 来源落档 | ✅ 五个 SCORE split 全 PASS + zip CRC 校验；gt-provenance.md 已登记；NWPU-Refer 待网盘（不阻塞 P0 三测试集主协议） |
| S1.4 | 伪造预测自测通过；口径写进 eval/README.md | ✅ |
| S1.5 | `tests/smoke_pipeline.py` 空跑通过（M0） | ✅ 2026-09-02 M0 四项全绿（iSAID 不在 M0 必需集内） |

## 3. 失败案例分类（工程向）

| 现象 | 根因 | 处置 |
|---|---|---|
| pip 装torch 长时间无进展 | 阿里云 wheel 单连接约 900MB 后断/停；-q 吞掉可见性 | 本地 wheel + 并行分段 curl（峰值 ~48MB/s）+ `--no-index --no-deps` 安装；记录于 scripts/README.md |
| HF 下载间歇 Connection refused | 本机 clash 代理不稳定 | 数据集走 hf-mirror.com（免代理）；代理仅作兜底 |
| bf16/fp32 dtype 崩溃 | 官方设计依赖全局 bf16 autocast（vitdet FFN fused op 强制 bf16） | 照官方 notebook 包 autocast，不改 vendored 代码 |
| mask 尺寸转置 | set_image 的 numpy 分支按 shape[-2:] 取 (H,W)，HWC 数组误读 | 一律传 PIL 对象（官方 notebook 同款用法） |
| `sam3` import 被 repo 根目录 shadow | 仓库根 `sam3/`（clone 根）在 cwd 于 sys.path 时构成命名空间包 | 入口脚本以自身路径导入（sys.path[0]=脚本目录），不受影响；勿在 repo 根直接 `python -c "import sam3"` |

## 4. 决定与下一步

1. **M0 判定：达成**（2026-09-02 冒烟四项全绿）。iSAID 已落地并通过 CRC 校验
   与冒烟；唯一缺项 NWPU-Refer 补充集在网盘侧，只影响 P0 的补充实验，不影响
   SCORE 三测试集主协议。
2. 权重渠道：当前 ModelScope 镜像（sha256 验证）；**用户待办**：以英文表单
   重新申请 HF `facebook/sam3` 访问（被拒后无重新申请入口，需新账号或 GitHub
   issue 人工复核），获批后用官方 LFS oid 复核本地权重并更新 license-note。
3. 下一步进入 **P0**（顺序按 implementation-v2 §2）：S2.1 判据预登记
   （p0-gate-criteria.md，抄录 SCORE 主表基线）→ S2.2 零样本 harness →
   S2.3 推理开销实测 → S2.4 失败分类 → S2.5 gate 判定。
4. GPU 说明：设计文档 G3 以 48G 单卡为参照；本机为 24G×4，P0 实测时按本机
   如实记录显存/时延（1024²/2048² 单图），若超 24G 需分块推理（S2.3 恰好
   覆盖该实验）。
