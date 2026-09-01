# data/ — 数据层

| 文件 | 用途 |
|---|---|
| `loader.py` | `create_train_loader`（WHU COCO 训练/验证加载 + 确定性子集抽样）与 `create_test_loader`。 |
| `whu_instance_dataset.py` | `WHUCocoInstanceDataset`：WHU COCO 实例分割数据集（支持类别映射，供 iSAID 复用）。 |
