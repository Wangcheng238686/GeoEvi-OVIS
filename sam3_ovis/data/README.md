# data/ — 数据集登记与 loader

数据根：`$SAM3_OVIS_DATA_ROOT`（本机 `/data/wangcheng/dataset/SCORE`）。
来源协议：**split 以 SCORE HF `HuangShiqi008/SCORE` 为准**（协议引用注明来源，
E4）；原始 zip 落 `<数据根>/zips/`，各数据集解压为 `<数据根>/<DATASET>/` 下的
COCO-instance 子树 `val/instances_val.json (+ images/)`（训练集为
`train/instances_train.json`）。

## 数据集注册表（S1.3，2026-09-02 核实）

| 数据集 | 角色 | 图数 / 实例数 / 类数 | GT 来源 | 状态 |
|---|---|---|---|---|
| NWPU | 测试 | 731 / 2,661 / 10 | 人工 | ✅ 冒烟 PASS（与 SCORE 论文 §4.1 一致） |
| SOTA | 测试 | 874 / 29,247 / 18 | SAM 伪标签 | ✅ 冒烟 PASS |
| FAST | 测试 | 3,207 / 43,258 / 37 | SAM 伪标签 | ✅ 冒烟 PASS |
| SIOR（train） | 训练 | 11,725 / 68,062 / 20 | SAM 伪标签 | ✅ 冒烟 PASS |
| SIOR-val | 测试（协议） | 11,738 / 124,373 / 20 | SAM 伪标签 | ✅ 冒烟 PASS |
| iSAID（train） | 训练 | 28,029 / 704,684 / 15 | 人工 | ✅ 冒烟 PASS（论文称 18,732 图，以 HF json 为准，差异双登记） |
| iSAID-val | 测试（协议） | 9,512 / 235,531 / 15 | 人工 | ✅ 冒烟 PASS |
| NWPU-Refer category 子集 | 补充 | – | referring 人工 | **待用户百度网盘下载**（官方仅发网盘：`https://pan.baidu.com/s/1HzLVXSqk_m4zWWJ-EJ5SkA` 提取码 `NWPU`；HF 无镜像） |

GT 来源逐条依据见 `docs/gt-provenance.md`（E5）。

## 模块

- `score_datasets.py`：注册表（`DatasetSpec` + 官方期望值）+ stdlib loader；
  新数据集在 `DATASETS` 登记后自动纳入冒烟。
- 冒烟入口：`tests/smoke_datasets.py`（图数/实例数/类别表/图片文件抽查）。

## Loader 约定

从 `geoevi_ovis/data/whu_instance_dataset.py` 的类别映射机制扩展（只读复用，
不改原文件）；P0 推理 harness 直接消费注册表的 COCO json 与图片路径。
