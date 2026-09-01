# scripts/train/ — 训练入口

| 路径 | 用途 |
|---|---|
| `run_whu1024_explicit_coarse.sh` | 主线训练分发器：按 EXPLICIT_PROMPT_MODE/P2 组合路由到 ablations wrapper。 |
| `ablations/` | 消融矩阵 runner、wrapper、契约校验、test-only、总 smoke（见 `ablations/README.md`）。 |
