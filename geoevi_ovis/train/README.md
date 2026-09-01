# train/ — 训练层

| 文件 | 用途 |
|---|---|
| `train.py` | DDP 训练主入口：CLI、配置构建、训练/验证循环、checkpoint 调度、`--test-only`。 |
| `eval_metrics.py` | COCO GT/DT 构造、bbox/segm COCOeval、RLE 编码、指标表达式解析。 |
| `distributed.py` | DDP/gloo 初始化、rank-0 延迟控制广播、DDP timeout、git 状态采集。 |
| `optim.py` | 分组优化器、norm eval、`ExponentialMovingAverage`。 |
| `checkpointing.py` | verified best/常规 checkpoint 保存、RESUME 严格加载、INIT_FROM 迁移。 |
| `monitors.py` | prompt/coarse 诊断日志、dense residual 与 P2-BRR epoch 监控。 |
| `losses.py` | detector loss 阶段权重、coarse/final loss 缩放、P2-BRR loss 物化。 |
