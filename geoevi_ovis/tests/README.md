# tests/ — 轻量测试与 smoke（本机 CPU 可跑）

| 文件 | 用途 |
|---|---|
| `smoke_test_components.py`（`.sh` 为入口） | 组件契约：PAFPN 双分辨率、coarse/P2/2P2N、no-mask 契约、坐标后处理回归等。 |
| `test_dense_prompt_utils.py` | dense prompt 工具单测（raw detach、Gaussian 映射、空 ROI 回退）。 |
| `smoke_ddp_control_plane.py` | 双进程 CPU/Gloo 延迟 rank-0 控制面 smoke（经 torchrun 启动）。 |
