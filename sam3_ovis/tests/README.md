# tests/ — sam3_ovis 冒烟与单测

| 文件 | 用途 |
|---|---|
| `smoke_pipeline.py` | M0 出口判据：加载环境 → 数据注册表 → 评测管线空跑（P-1）。 |

运行方式：

```bash
source sam3_ovis/scripts/load_environment.sh
"$PYTHON" sam3_ovis/tests/smoke_pipeline.py
```

后续：S1.4 评测口径自测（伪造预测）、S3.x 图构建/边/指标单测在此扩展。
