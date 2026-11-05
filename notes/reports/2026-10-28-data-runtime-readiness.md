# 图记忆实施准备：数据、评测与运行条件

日期：2026-10-28。只读核查；未安装依赖、未加载大模型、未运行 GPU 作业。

## 1. 当前历史状态的使用边界

`v3-frozen-eval-results.md` 已将 v3 相对惩罚与空间竞争图关闭；
`v4-s0-c0-verdict.md` 已将 TG-CWP 跨窗正框传播在 S0 止损。新方案不再接续这些
机制；复用的是数据注册、候选缓存格式和实例评测经验。

旧结论的固定候选上限、特定跨窗匹配比例不能外推为“主干微调、原生 memory 和
detector 特征干预也无效”。旧报告中把观测候选差异率换算成严格 mAP 数学上限的
说法尚无充分推导，本准备报告不采用。新方案须重新检验上下文语义作用空间。
跨图数据集性能下降、逐图 AP 统计与全局 COCO AP 不是同一个量，未来配对统计
须按源图组重采样并重算全局 COCO AP。

## 2. 源域与确认数据建议

优先考虑 iSAID 的人工实例标签作为适配源域。注册表登记 train 28,029 图、704,684
实例、15 类；原始 json 存在于 `/data/wangcheng/dataset/SCORE/iSAID/train/instances_train.json`
（647,753,143 字节）。本次未重新逐条解析整份标注，不把注册数量称为新复算结果。

开发/验证必须按原图 ID 分组，不能随机按发布切片划分；既往已暴露记录包括：

- `sam3_ovis/logs/t2_cache/manifest.json`：237 切片、15 源图组。
- `sam3_ovis/logs/v3_frozen_eval/manifest.json`：146 切片、10 源图组。
- T1 242 源图组清单见上份 frozen manifest 的 `excluded_probe_groups`。

旧确认图已经被观察，后续可以明确标为开发/旁证，不能重复宣称未暴露确认数据。
训练监督可以使用规定的源域数据，但最终确认组必须在拟定的新训练 split 中排除。
跨数据集图像重合还需另检，不能仅依靠源图字符串不同。

## 3. 开放词汇的类隔离不能省略

对 `sam3_ovis/data/score_datasets.py` 注册类别执行了大小写/连字符/下划线归一，并
显式将 plane→airplane、baseball field→baseball diamond。此检查只针对注册名字，
不是最终人工语义审计，也未读取测试像素或评估新结果。

- NWPU 与 iSAID 大量类别相同；唯一非精确同名的 vehicle 是 small/large vehicle 的
  上位概念，不能据此称整套 NWPU 为 novel 类测试。
- SOTA 与 iSAID 的 15 类经别名转换全部重合；airport、container crane、helipad
  是初步非同名类，仍需源图、父子类别与数据来源审计。
- SIOR-val 非精确同名类别有 airport、chimney、dam、expressway service area、
  expressway toll station、golffield、overpass、stadium、train station、vehicle、
  windmill。其中 vehicle、overpass/bridge、stadium/体育场相关类需单列层级关系，
  不能直接计作严格 novel。
- FAST 的飞机/船/车子类可能是源域大类的下位类；应报告 seen、hierarchical transfer
  与严格 novel，不能以不同字符串充当未见类证据。

SIOR、SOTA、FAST 的实例 mask 来自 SAM 伪标签，iSAID 是人工实例标签；出处登记见
`sam3_ovis/docs/gt-provenance.md`。模型适配、novel 泛化和边界质量结果须分别说明
标签质量，不把目标测试 GT 用于图构建、memory 写入、选参或人工 exemplar。

## 4. 当前机器实况

只读命令：`nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader`
及 `df -h /home/wangcheng2021/project/GeoEvi-OVIS /data /tmp`。

核查瞬间实际为 4× RTX 3090，每卡 24,576 MiB，已用约 22,700 MiB，剩余约
1,300–1,450 MiB。它们是即时值，不代表后续预约资源。未检查或终止其他任务。
9 月的 4×48GB 记录不能用于当前容量承诺；本轮不具备并行大模型训练的空闲显存。

项目所在 `/home` 分区 98% 已用、约 9.5GB 可用；`/data` 约 499GB 可用，`/tmp`
所在分区约 247GB 可用。后续大缓存和 checkpoint 应显式配置数据分区或已有输出根，
不能将多实验大文件默认写入项目目录。

## 5. 运行环境核查

`/data/wangcheng/envs/sam3/bin/python` 存在，版本为 3.12.14。metadata 核查：

| 包 | 结果 |
|---|---|
| torch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| pycocotools | 2.0.11 |
| sam3 | metadata 为 0.1.0；单独 find_spec 显示 namespace，不能据此证明导入正确 |
| iopath | 0.1.10 |
| hydra-core、omegaconf、submitit | 未安装/不可发现 |

`sam3/sam3/train/train.py` 顶层导入 submitit/hydra/omegaconf，训练 extras 在
`sam3/pyproject.toml` 已列出。因此现有推理环境不等于官方训练入口可运行。
后续需先核验实际 `sam3.__file__` 与子模块定位，按正式环境流程补齐训练依赖；
本轮未变更任何环境。单卡前向/反向与显存冒烟尚未执行。

权重 `/data/wangcheng/pretrained-models/sam3/sam3.pt` 存在，大小 3,450,062,241 字节；
仅检查文件存在与大小，本轮未重算权重 hash 或加载。

## 6. 版本取证纠正

`git -C sam3 rev-parse --show-toplevel` 返回项目根，而非单独 SAM3 仓库。当前 root
HEAD `88c5216b174286b5a730550dec139b1663f6bc28` 是项目 SHA；10 月 25 日记录的
`5a5df...` 也不能直接解释为 SAM3 上游提交。具体源文件 hash 是更可靠的取证：

| 文件（项目根相对路径） | SHA256 |
|---|---|
| sam3/sam3/model/sam3_tracker_base.py | b2b52409c002e1590262375aa794f8ab67e7476f42f8fee41a76de0c14aa62e2 |
| sam3/sam3/model/sam3_image.py | 2c20de90fb0b76b01dd437385eb7226f8c3cd4757a4c80cc8dec943dcf46a4a0 |
| sam3/sam3/model_builder.py | d71d6d3e485ec3eae48bbc2ba676f401b5853d65c4195a91d077b04da38121c2 |
| sam3/sam3/model/sam3_image_processor.py | d8738a0efb6138b01c0dc5deceffd29de9e675860a9a1ed3822b08766373333b |

## 7. 实施前的就绪要求

科学需求已经足够明确，暂不需要负责人再回答图算子或接口细节。技术准入还需完成：
训练环境导入、原生 writer/reader 的小张量前反向、mask/grid/位置编码契约、类别
detector 回接、逐图 reset 与读取排除、源图/类别隔离、单卡峰值显存测量。
这些先于 R1/R2/R3 并行短训练；当前“调研准备已可交付”不等于“训练已就绪”。
