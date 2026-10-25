# 实施文档 v4.0：TG-CWP 最小验证

日期：2026-09-29。对应 `design-v4.0.md`。本阶段不修改模型权重和原生 memory。

## S0：数据与作用空间准入

1. 选择保留原始大图、全局实例标注与 tile 坐标的数据；登记 license、split 和暴露状态。
2. 固定二维 tile size/stride/padding，在任何结果运行前写 manifest 与可视化抽检。
3. 统计跨 tile GT、边界截断、C0 漏检但邻窗可见实例及 oracle 正框恢复上限。
4. 若跨窗可恢复空间不足，输出负报告并停止，不以人为切得更碎制造问题。

交付：`notes/reports/v4-s0-action-space.md`、split/tile manifest、10 张坐标回贴图。

## S1：C0 独立二维 tile 基线

复用现有 SAM3 text-only harness；增加二维 tiler、tile-local/global 坐标转换、embedding
缓存和全图实例融合。候选与零预测图均原子落盘。验证同一大图以不同 batch/遍历顺序
运行得到一致输出；计算全图 segm mAP、跨窗子集和重复率。

## S2：正框传播接口探针

只用现有 `add_geometric_prompt(..., label=True)`。构造 20–50 个已知跨窗实例：从邻窗
独立预测生成 overlap box，映射至目标 tile；检查 cxcywh 归一化、裁剪、边距、空框与
reset 纪律。比较目标 tile text-only 与 text+positive-box 的召回、mask IoU、假阳和
重复，不选参数。若正框不能形成有利取舍，停止。

## S3：C1 固定邻接传播

建立四邻接 tile 图。Pass A 全部完成后统一生成传播计划；Pass B 严格只执行计划，
不把第二遍结果再次传播。实现固定几何匹配、单 anchor/target/class 预算和统一融合。
保存每次调用的 source/target tile、global/local box、类别、源质量及输出候选映射。

## S4：C2/C3 图选择对照

实现确定性 anchor 排序：完整度、边界距离、SAM3 分数、跨窗一致性；在开发集冻结
词典序和阈值。C2 取首个可靠 anchor；C3 在相同可用集合和调用预算内固定种子随机取。
不得加入 GNN、学习权重、类别上下文或 native memory。

## S5：冻结验证与决策

在未参与规则选择的大图上一次性运行 C0/C1/C2/C3。以原图为 bootstrap 单位，报告
全图 segm AP、跨窗/边界子集、重复与错误传播、调用数和耗时。严格按 design-v4.0
止损：C1≤C0 则终止；C2≤C1 或 C2≤C3 则取消图主张；只有 C2 稳定领先才制定后续主
实验，不在本阶段追加模块。

## 测试要求

- x/y 两轴切窗覆盖、右/下边缘和 padding 回贴无缝；global↔local box 往返误差受控。
- 邻接对称、无跨图边；同一全图实例不会因 tile 顺序改变 tracklet。
- 所有 geometric prompt 为归一化 cxcywh 且位于 [0,1]；每类重跑前清空旧 prompt。
- Pass B 非递归；C1/C2/C3 第二遍调用数预算一致；GT 不进入传播计划。
- 融合后 mask、bbox、area 与 COCO RLE 一致；零预测大图保留在评测分母。
