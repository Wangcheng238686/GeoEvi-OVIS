# configs/ — sam3_ovis 机器环境与运行配置

| 文件 | 用途 |
|---|---|
| `environment.sh` | 共享默认环境（PYTHON / SAM3_REPO / SAM3_CKPT / 数据根 / 输出根）。 |
| `environment2.sh` | lthpc 主机（RTX 3090 x4）已核实路径版本；经 `environment.local.sh` 软链激活。 |
| `environment.local.sh` | 本机软链（gitignored，不随仓库分发）；缺失时 loader 回退 `environment.sh`。 |

继承机制与 `geoevi_ovis/configs/` 一致：入口统一 `scripts/load_environment.sh`，
外层 shell 已导出的变量优先于文件内默认值。

## 变量契约

影响模型行为的参数一律进 `cfg.model`；本目录只登记机器路径与执行默认：

| 变量 | 含义 |
|---|---|
| `PYTHON` | sam3 独立 conda env 的解释器（torch 2.10.0+cu128, python 3.12）。 |
| `SAM3_REPO` | vendored 官方 repo（仓库根 `sam3/`，与 `sam2/` 平级）。 |
| `SAM3_HUB_DIR` | `facebook/sam3` HF 快照目录（config + tokenizer + 权重）。 |
| `SAM3_CKPT` | 主 checkpoint（`$SAM3_HUB_DIR/sam3.pt`）。 |
| `SAM3_OVIS_DATA_ROOT` | 登记数据根；五个数据集 + 补充集以其下子树注册（见 `data/README.md`）。 |
| `SAM3_OVIS_LOG_ROOT` / `SAM3_OVIS_TMP_ROOT` | 可写日志 / 临时目录。 |
| `CUDA_VISIBLE_DEVICES` | 本机执行默认（P-1 期间其他任务占用 GPU 1–2，默认 0）。 |
