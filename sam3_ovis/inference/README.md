# inference/ — SAM3 推理 harness

约定：全局 bf16 autocast（官方用法，见 docs/sam3-api-notes.md §0）；
`set_image` 传 PIL 对象；阈值/口径以 `notes/reports/p0-gate-criteria.md`
预登记为准。

| 脚本 | 用途 | 阶段 |
|---|---|---|
| `demo_text.py` | 官方 demo 复跑（S1.1 出口判据） | P-1 ✅ |
| `validate_api.py` | S1.2 四项 API 面验证（tokens/contract/exemplar/multi） | P-1 ✅ |
| `zeroshot_text.py` | 每图×每类 text-only 候选 → JSONL RLE；断点续跑 + 逐图 OOM 韧性；产出逐类/逐图候选统计 | P0 ✅ |
| `inference_modes.py` | 整图/网格分块/重叠滑窗三模式对比（E2：显存、时延、指标差） | P0 ✅ |
| `demo_graph.py` | S3.4 端到端单图 demo：P0 预测 → 全图构建 → 完整性 → 两轮消息传递 → 合并 → 风险调制排序（`--min-gt` 区分正样本/纯FP 展示） | P1 ✅ |
| `failure_mining.py` | 四类失败（漏检/误检/mask质量/重复）挖掘 + 可视化导出 | P0 ✅ |

## P0 运行口径（已预登记，不回改）

- 候选池 `score ≥ 0.05`（processor confidence_threshold），类名逐字取注册表；
- 输出 `$SAM3_OVIS_LOG_ROOT/zeroshot/<DATASET>/predictions.jsonl` +
  `run_meta.json`（逐类/逐图候选数、耗时分解、OOM 记录）；
- gate 汇总：`eval/run_gate_eval.py --datasets NWPU SOTA FAST`。
