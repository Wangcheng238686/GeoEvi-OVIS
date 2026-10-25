# 设计指导 v4.0：二维拓扑引导的跨窗正框提示传播

日期：2026-09-29。状态：路线已收敛，尚未通过数据与作用空间准入门。
对应实施：`implementation-v4.0.md`。v3 仅完成冻结验证，不与本路线混跑。

## 1. 唯一研究问题

面向无法整图送入 SAM3 的遥感大图，验证如下可证伪命题：

> 在二维重叠 tile 中，一个实例在某个窗口内完整、可靠，在相邻窗口内被边界截断或
> 漏检时，使用全图拓扑和跨窗实例对应选择该可靠观察，并把其重叠区域映射成目标
> tile 的正 box exemplar，能否提高开放词汇实例分割，同时少引入重复和错误传播？

方法名称暂定 **Topology-Guided Cross-Window Prompting（TG-CWP）**。准确定位是
图控制的 prompt propagation，不宣称修改 SAM3 原生 memory。

## 2. 冻结边界

- 使用 SAM3 已验证的公开接口：`set_text_prompt` 后调用
  `add_geometric_prompt(state, box, label=True)`；box 为目标 tile 内归一化 cxcywh。
- 只使用正 box；不使用负 prompt、mask prompt 或未验证的 tracker memory 接口。
- 不修改 SAM3 权重、memory encoder、attention、RoPE 或 decoder；不训练 LoRA、
  adapter、GNN 或额外分类器。
- 不使用类别共现和全局场景硬先验；不跨测试图保存状态；GT 不进入推理。
- 只做两遍、非递归流程。第二遍结果不再成为新 anchor，防止错误级联并保证顺序不变。

## 3. 最小 pipeline

### Pass A：独立二维侦察

原始大图按固定 tile size、x/y overlap 切成二维网格。每个 tile 保存全图 origin、
有效区、padding 和邻接；对冻结类别词表逐类运行 text-only SAM3，缓存图像 embedding、
候选 mask/box/score/presence。所有坐标同时保存 tile-local 与 global 两套表示。

### 全图观察图

候选观察是节点；节点只保存类别、global mask/box、源 tile、SAM3 分数、与 tile 边界
距离和是否穿过 overlap。仅在源 tile 空间相邻且同类别观察在全图重叠区几何一致时，
建立 `cross-window-correspondence` 边。tile adjacency 是建边约束，不把扫描次序当边。

对每个跨窗实例簇选择一个 anchor：优先选择离 tile 边界更远、可见面积更完整、原始
分数更高且与邻窗观察一致的节点。第一版使用登记的确定性规则，每个实例簇只保留一个
anchor；不得通过 GT 或测试集搜索权重。

### Pass B：拓扑门控正框回注

只处理满足以下任一条件的目标 tile：该实例在邻接 tile 有可靠 anchor，但目标 tile
对应观察缺失；或目标观察接触边界且质量低于 anchor。将 anchor 在两 tile overlap 内的
global mask 外接框映射到目标 tile，扩固定小边距并转成归一化 cxcywh。对同一类别使用
text prompt + 一个正 geometric box 重新推理。第二遍结果与 Pass A 候选用统一类内
mask NMS/跨 tile 实例融合，输出全图 COCO 实例结果。

## 4. 唯一对照链

| 变体 | 定义 | 回答的问题 |
|---|---|---|
| C0 | 独立二维 tiles + 固定全图融合 | 大图基础线 |
| C1 | 固定四邻接，满足几何重叠即传播正框，不做 anchor 质量选择 | prompt propagation 本身是否有效 |
| C2 | TG-CWP：同预算、图选择一个可靠 anchor 后传播 | 拓扑/实例关系选择是否有独立价值 |
| C3 | 同 C2 候选数，但从相邻可用 anchor 中固定种子随机选择 | 收益是否只是多跑一次 SAM3 |

OmniSAM 式蛇形 FIFO memory 只作为相关工作/可选外部强基线，不进入第一轮实现；
Slide-SAM 对应 C1 的方法来源。第一轮不得同时加入 native memory，保证因果解释单一。

## 5. 准入、指标与停止条件

必须使用保留原始大图、tile 坐标与跨 tile GT 身份的数据。已经独立发布、无法恢复
原图位置的小图不能用于证明跨窗收益。先统计：跨窗实例数、边界截断率、C0 漏检中
可由邻窗可靠观察覆盖的比例，以及 oracle 正框可恢复上限；作用空间不足则停止。

主指标为全图实例 segm mAP/AP50；同时报告跨窗实例 AP/Recall、边界截断子集、重复
实例率、错误传播率、每图第二遍调用数、延迟和显存。所有变体使用相同 tiles、候选
阈值与融合器。

- C1 不优于 C0：停止跨窗提示传播，不做图。
- C1 优于 C0、但 C2 不优于 C1/C3：保留简单传播基线，放弃图贡献。
- C2 稳定超过 C0/C1/C3，且错误传播和计算预算可控：才考虑扩大数据和论文主实验。
- 只有 C2 成立后，才另立版本研究 native spatial memory；不得在 v4 内追加。

## 6. 与现有工作的边界

OmniSAM 是一维横向 patch sequence 的 SAM2 FIFO memory；Slide-SAM 是三切片链上的
mask-to-box/point prompt 双向传播。TG-CWP 的差异必须由实验支撑：二维非因果 tile
拓扑、开放词汇实例级共指、可靠 anchor 选择，以及两遍非递归的顺序不变传播。仅将
Slide-SAM 从 z 轴改成 x/y 扫描不构成贡献。
