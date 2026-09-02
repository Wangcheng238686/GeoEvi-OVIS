# inference/ — SAM3 推理 harness

规划（implementation-v2.md S2.x，P0 启用）：

- `zeroshot_text.py`：每图 × 每类短语 → SAM3 text-only 概念分割 → 候选
  （mask + 分数）→ S1.4 管线评测；记录逐类、逐图候选统计；
- 大幅图三种推理模式对比（整图 resize / 固定网格分块 / 重叠滑窗，E2）；
- exemplar 组合 prompt（P3 启用）。

P-1 阶段本目录只放官方 demo 复跑脚本（S1.1 完成判据：单图 text prompt 在本机
GPU 跑通）。
