# inference/ — 推理入口

| 文件 | 用途 |
|---|---|
| `infer_from_checkpoint.py` | 从自描述 checkpoint 重建模型，在 WHU validation/test/自定义 COCO split 上输出 predictions/metrics/manifest。 |
| `visualize_p2_refiner.py` | P2-BRR 修正前后的掩码可视化（配 `scripts/inference/visualize_p2_checkpoint.sh` 使用）。 |
