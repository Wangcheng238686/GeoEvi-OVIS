# 交接文档：node01（单卡 RTX 3090）接力（2026-11-24）

> 本文档替代 `/tmp/geoevi-handoff/HANDOFF.md`（后者按 skill 规范写在原机
> 临时目录，不随仓库走）。原机 = lthpc（4×3090），新机 = node01
>（1×3090，`ssh node01`，免密已配）。

## 0. node01 上已就绪的内容（本节即启动清单）

| 项 | node01 路径 | 状态 |
|---|---|---|
| 仓库 | `/home/wangcheng/project/GeoEvi-OVIS` | HEAD `805a8ad`，989 文件与本机一致，工作区干净；origin 已改为 SSH |
| conda 环境 | `/data/wangcheng/envs/sam3`（**9 月 2 日已有**，非本次传输） | torch 2.10+cu128 CUDA 可用；本次补装 fvcore/tensorboard/torchmetrics/sklearn/scipy（numpy 1.26.4 锁定） |
| sam3 editable | `…/__editable___sam3_0_1_0_finder.py` | MAPPING 指向 `/home/wangcheng/project/GeoEvi-OVIS/sam3/sam3`（正确，无需改） |
| SAM3/RemoteCLIP 权重 | `/data/wangcheng/pretrained-models/{sam3,RemoteCLIP}` | 9 月已有，rsync 校验一致 |
| R1v2 checkpoint | `/data/wangcheng/geoevi-r1/run1/checkpoints/checkpoint.pt` | 3,375,421,957 字节，与原机逐位一致 |
| val 候选 6 臂 | `/data/wangcheng/geoevi-r1/freeler/` | 6 臂 × 118 文件，与原机一致 |
| 训练标注（空格类名） | `/data/wangcheng/geoevi-r1/isaid_{train_g0,tiny_val}_space.json` | 已传 |
| SCORE 数据集 | `/data/wangcheng/dataset/SCORE` | 11/16 旧副本 + 本次后台 rsync 增量（见 §4） |
| environment.local.sh | `sam3_ovis/configs/environment.local.sh` | **已生成**（gitignored），PYTHON/SAM3_CKPT/DATA_ROOT 全部解析 |

**环境加载**：`source sam3_ovis/scripts/load_environment.sh`。

**远端已验证**（2026-11-24）：pytest 50/50；数据集 train json = 28,029 图
/ 704,684 标注；GPU 复算 R1v2−COMBO = **+3.24pt CI[2.14, 4.57]**、COMBO
池化 0.2700、R1v2 0.3341——与原机存档逐位一致。

## 1. 与原机的路径差异（唯一一处）

`/home/wangcheng2021/project/GeoEvi-OVIS` → `/home/wangcheng/project/GeoEvi-OVIS`。
`r1_v2_eval.py` 已改为从 `__file__` 推导（commit 805a8ad）。其余硬编码
路径均在 `/data/wangcheng/` 下，两机镜像、零改动。历史脚本
（`pilot_ablation_eval.py`、`gpu_window_wait.sh` 等）仍写死原机仓库路径
——它们属于已归档阶段，如需使用要先改。

## 2. 当前结论状态（详见 notes/project-explainer.md 与各报告）

- 五条机制止损维持；G3a 正向已勘误为缓存污染。
- 根因（G4a/G5b）：prompt 下划线伪影（修 +8.16pt）、排序通道上限
  +11.4pt > 分类通道 +3.4pt。
- **EXP-0001（R1v2 微调）= PARTIAL，项目首个显著正向结果**：
  mAP 0.2700 → 0.3341（+3.24pt CI[2.14,4.57]）；代价：FP/图 290→440、
  分类纯度 44.74%→40.33%。
- **same-class GO 门不可判**（S1 勘误，协议 §6）：原指标非确定；两个
  确定性口径对 ≥45% 给出相反答案。重定门限须负责人裁定。

## 3. S2（下一步，协议已登记、代码+测试就绪、零数字产生）

协议 `2026-11-24-s2-protocol.md`（温度维度已在登记时撤销，网格 = 5 个
逐类阈值配置 q∈{0,50,80,90,95}）；判据 FP≤350 且 mAP 配对 CI 下界
> −1.0pt。

**node01 的优势：790 图 train 池候选生成（`--phase gen`）在单卡上无
竞争，~2–3 小时可完成**。运行：

```bash
cd /home/wangcheng/project/GeoEvi-OVIS && source sam3_ovis/scripts/load_environment.sh
CUDA_VISIBLE_DEVICES=0 "$PYTHON" sam3_ovis/graph/s2_rescore.py --phase gen
CUDA_VISIBLE_DEVICES="" "$PYTHON" sam3_ovis/graph/s2_rescore.py --phase select
CUDA_VISIBLE_DEVICES="" "$PYTHON" sam3_ovis/graph/s2_rescore.py --phase eval
```

gen 从未运行过（原机无显存窗口），先小规模试跑再全量。红线：val 18 组
不得用于选参；网格封闭；5 配置全不过 = 如实记录"分数重标不足"。

## 4. 数据集 rsync 注意事项

后台任务把整个 SCORE（含 67GB 的 `zips/`）做增量同步。若要省时，可
`--exclude zips`；iSAID/SIOR 等解压目录才是 DATASETS 注册表引用的。

## 5. 待负责人裁定（勿自行决定）

1. same-class 门口径与阈值；
2. PARTIAL 的"配方调参一次"用 T1（训练侧）还是 S2 结果（评估侧）；
3. 后续是否扩 LoRA 到概念 transformer（分类退化的指向，需新配方设计）。

## 6. 已知环境坑（远端同样适用）

见 `notes/reports/2026-11-22-r1-protocol-v2.md` §5 与
`2026-11-23-exp0001-conclusion.md` 勘误节：vitdet 融合核（训练必须用
`sam3/r1_train_wrapper.py`）、MHA 内 Linear 不可被 LoRA 替换、
`target_epoch_size` 死旋钮、位置编码硬编码 cuda、numpy<2 锁定。
