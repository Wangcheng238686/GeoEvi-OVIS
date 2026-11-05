# 同图实例—上下文图记忆：实施前文献复核

日期：2026-10-28。研究准备报告；不构成模型已实现、已复现或新颖性已证明的声明。
约束依据 `../decisions-2026-10.md`，候选依据
`2026-10-25-graph-memory-candidate-routes.md`。本报告只修改自己的文件。

## 1. 判断与证据强度

可以开始共同接口准备和机制探针。更准确的待研究问题是：同图中其他实例与环境的
关系，能否指导 SAM3 原生记忆的访问，帮助概念判别，同时避免把其他对象的证据当作
当前对象身份。不能把“二维切窗”“双 bank”“图聚合”“质量写入”各自包装成首次。

本次未核实到完整同时满足“显式实例—环境关系控制 SAM3 原生 memory、同图内生命周期、
遥感开放词汇实例判别”的直接同类。该结果仅覆盖下述定向检索及全文核验，不证明不存在。
新颖性应由具体机制相对近邻的差异和实验支撑，而非 SAM3/图/遥感三关键词的组合支撑。

核验深度如下；“阅读全文方法”表示已打开原始全文并核对相应方法段，不表示已逐行证明
全部推导或复现结果。

| 文献 | 本轮核验层次 | 未完成范围 |
|---|---|---|
| CA-SAM2 | MICCAI 正式全文 §2.1–2.2；官方仓库；`run/run_on_epoch.py` 读写与 reset | 未执行训练/评测；未穷举其所有文件许可 |
| SCORE | 原始全文 §3.2–3.5、官方仓库及 LICENSE | 本轮未审计全部训练源码、数据下载及重现指标 |
| GPRN | AAAI 正式 PDF 方法与公式；官方 `model/GPRN.py` | 未运行；未穷举附带代码来源及许可 |
| SAM3-DMS | arXiv v1 方法 §3.1–3.2、官方仓库与 LICENSE | 未对整个 fork 与 SAM3 做逐提交 diff；未运行 |
| MPG-SAM2 | 最新打开版本 arXiv v3 §3.3–3.5、官方 README 与 LICENSE | 未逐行审计 HGA 实现、未运行 |
| Online Reasoning Video Segmentation with Just-in-Time Digital Twins | ICCV 正式 PDF 摘要、框架和 §3.2 | 周边机制查重，未检查全部代码及实验 |

## 2. 最近邻分别覆盖什么

### CA-SAM2：主要组织参考，同时是不能省略的强对照

其 EMB 保存已处理 patch 的图像表示，按二维欧氏距离选近邻，经独立训练的一层
SAM2 memory attention 融合；TMB 保存前景记忆，按相似度及质量维护和读取。它已经
覆盖环境/前景分工与非 FIFO 检索。EMB 每图清空，但代码将 texture list 从函数外传入，
我们的全部样本记忆必须另行强制逐原图清空，不能照搬其生命周期。
[正式全文 §2.1–2.2](https://papers.miccai.org/miccai-2025/paper/1881_paper.pdf)，
[官方源文件](https://github.com/HanbinHuang123/CA-SAM2/blob/main/run/run_on_epoch.py)。

### SCORE：上下文帮助遥感开放词汇判别已有直接证据

RAI 根据预测 mask 扩展区域，从 RemoteCLIP 池化上下文并更新类别表示；GCA 用全局
场景适配文本分类器。故“周围环境帮助分船/车”是动机依据，不能作为我们独有结论。
我们应比较对象周围区域池化/attention 与关系记忆的增量。
[全文 §3.2–3.5](https://arxiv.org/html/2507.12857v1)，
[官方代码](https://github.com/HuangShiqi128/SCORE)。

### GPRN：给 pooled-mask 节点加相似度图，已有明确先例

图节点是 SAM mask 池化的视觉提示；投影后余弦相似度形成边权，多轮残差聚合，再将
节点信息回填区域，服务跨域 few-shot 分割。它不通过 SAM3 原生 memory 读写。但若 R1
仅做余弦图聚合或按边权 top-K，机制与其非常接近，不能仅改数据/基座就主张强创新。
[正式 PDF 方法，图 2、式 3–5](https://ojs.aaai.org/index.php/AAAI/article/view/32695/34850)，
[GraphAttention/enhance_feature 源码](https://github.com/CVL-hub/GPRN/blob/main/model/GPRN.py)。

### SAM3-DMS：单对象质量门控不是新空间

DMS 将 SAM3 组平均质量门槛改为逐对象质量与 presence 联合门槛，以防空白/错误记忆
污染对象身份；不额外增加关系控制器。它是 R3 必须比较的对象独立门控，而非关系方法。
[全文 §3.1–3.2，式 1–4](https://arxiv.org/html/2601.09699v1)，
[作者仓库](https://github.com/FudanCVL/SAM3-DMS)。

### MPG-SAM2：上下文与历史双层融合已有先例

HGA 在像素与对象 token 两级融合全局视频语义与历史信息；object token 先读全局视频
特征、再读历史 mask token。它是 RVOS，不是同图实例—环境关系检索，但足以否定
“给 SAM 同时加全局上下文和历史对象记忆”本身的新颖性。作者特别声明 ICCV 图 2 有误，
应以最新 arXiv 图示核对实现。
[arXiv v3 §3.3–3.5](https://arxiv.org/html/2501.13667v3)，
[官方仓库及图示更正](https://github.com/rongfu-dsb/MPG-SAM2)。

### 图、SAM 和记忆共存的系统也已存在

ICCV 的 Just-in-Time Digital Twins 使用感知模型（含 SAM2）建立对象场景图，并以时间
窗口保留图状态做视频关系推理。其图记忆位于外部推理系统，不能等同控制 SAM 原生 reader，
但不能宣称首次让 SAM、scene graph 和 memory 出现在一个系统。
[正式 PDF §3.2](https://openaccess.thecvf.com/content/ICCV2025/papers/Shen_Online_Reasoning_Video_Segmentation_with_Just-in-Time_Digital_Twins_ICCV_2025_paper.pdf)。

## 3. R1/R2/R3 的独立性与最低可检验内容

| 路线 | 独立变化 | 最容易与已有工作重合的实现 | 值得检验的差异 |
|---|---|---|---|
| R1 关系选择 | 固定 bank、reader、角色/质量规则；只改选取的记录 | 距离/余弦 top-K；mask 节点相似度传播 | 对象—对象与对象—环境关系是否提供单纯近邻/相似度不能提供的 query 相关性 |
| R2 角色读取 | 固定相同记录和选择规则；只改 reader 输入角色 | 环境/前景双 bank；一般 type embedding；global/history 两阶段融合 | 同为其他对象，其身份证据与上下文证据的使用方式是否需要区分，能否降低串实例而保留类别收益 |
| R3 关系门控 | 固定候选 bank 和 reader；只改准入/读取权重 | 单对象置信度阈值；相似度多样性淘汰 | 对象自身质量相同条件下，关系是否还能识别有害记忆；是否会压制罕见但真实的场景 |

三条路线是可分离实验干预，不自动构成三个互不重叠的论文创新。第一轮不能将选择、角色、
质量同时改掉再各取名字。R2 尤其不能把“身份/上下文/环境”三个标签视作已证明的新机制；
需要把角色作用落实到读取路径，并控制额外参数。R3 必须在预测关系上有效，不能只有 GT
环境或 GT 类别 oracle 才有效。

当前设计允许适当遥感微调；“无图版本”也必须相同微调。静态两遍快照初版主要管理有限
memory 的访问，尚无充分理由宣传长期更新/生命周期管理。缓存是否有独立价值需测复用次数、
计算成本，以及不显式建库的一次性同图 attention。SAM3 tracker 的改善也不能自动证明类别
改善，须有通向概念 detector 的实质路径。

## 4. 最少必要对照与失败解释

1. 相同源域微调 SAM3，无新增 memory；并匹配 detector 回接 adapter 的参数对照。
2. CA-SAM2 式环境距离 + 前景相似度检索，匹配 tokens、容量、训练和回接路径。
3. 相同上下文输入的一次性 attention/区域池化对照：检查是否需要 memory 表示和访问机制。
4. R1 对距离 top-K、余弦 top-K、匹配度数随机选择；固定 bank/reader。
5. R2 对无角色、随机角色、同参数普通投影；固定实际读取记录，并分别去掉其他实例与环境。
6. R3 对独立对象质量门控（DMS 式）、无门控、同写入数量的随机门控；固定 reader。
7. 各路线做删边、匹配度数打乱边、错误环境替换与正常环境的配对实验；报告 novel AP、
   错类和实例串扰，不能用 mask AP 的变化单独代替类别证据。

建议优先看 R1 的空间/语义关系能否超过无图检索，再看 R2 是否真有身份污染现象；R3
需要先证明存在可预测的关系冲突。此排序是工程准备判断，不是替用户决定放弃其他路线。
若无图上下文达到同等效果，当前图机制失败；不应以研究主题必须含图推断图必然有效。
旧 V3 固定候选重排及 V4 跨窗恢复的局部负结果既不支持新路线成功，也不能外推否定经过
微调、改变特征与原生记忆的新干预；本轮不复活 prompt-only 路线。

## 5. 代码与许可可用性

| 仓库 | 公开与读取证据 | 许可状态（本次实际看到的文件） | 实施含义 |
|---|---|---|---|
| [CA-SAM2](https://github.com/HanbinHuang123/CA-SAM2) | 正式论文链接仓库；已读取读写代码 | 根目录未列 LICENSE，raw `/main/LICENSE` 返回 404；未确认完整派生代码授权 | 可读作机制参考；勿把公开可见等同已有清晰复制/再分发授权 |
| [SCORE](https://github.com/HuangShiqi128/SCORE) | 作者仓库，训练/评测入口公开 | [根 LICENSE 为 Apache-2.0](https://raw.githubusercontent.com/HuangShiqi128/SCORE/main/LICENSE) | 可参考上下文与对照；数据和依赖许可另查 |
| [GPRN](https://github.com/CVL-hub/GPRN) | 核心 `model/GPRN.py` 可访问 | 根目录未列 LICENSE，raw `/main/LICENSE` 返回 404；未穷举子目录 | 不将其称作已明确宽松授权的可直接拷贝组件 |
| [SAM3-DMS](https://github.com/FudanCVL/SAM3-DMS) | 作者公开 fork；README 仍有旧 clone 名称 | [LICENSE 是 SAM 自定义协议](https://raw.githubusercontent.com/FudanCVL/SAM3-DMS/main/LICENSE)，非 Apache | 按实际协议及 checkpoint 获取条件处理；DMS 贡献授权未另做法律解释 |
| [MPG-SAM2](https://github.com/rongfu-dsb/MPG-SAM2) | 官方提供训练/推理与权重链接 | [根 LICENSE 为 Apache-2.0](https://raw.githubusercontent.com/rongfu-dsb/MPG-SAM2/main/LICENSE) | 可作 global/history 融合对照；BEiT-3、SAM2 权重与数据条件单独核对 |

本轮 GitHub API 请求未取得有效树/commit 数据，因此不凭失败响应声称仓库没有文件，
也未生成新的源码检查点。正式实施应固定下载版本与 SHA，并逐文件记录来源；
CA-SAM2 历史 SHA 仅沿用候选报告记载，未以本次网络读取独立重算。

## 6. 检索范围与剩余不确定性

在线组合查询包括 `SAM3 graph memory remote sensing`、`SAM3 relational memory`、
`SAM3 graph memory segmentation`、`SAM2 graph memory context segmentation`、
`SAM scene graph memory segmentation remote sensing`，以及上述五篇准确标题/作者仓库。
聚合站/综述仅用于找线索，技术比较以作者全文、会议原文、作者源码为依据。

剩余项：SAM3/3.1 是否已有对象共享上下文 memory 的更新版本；近几个月未被索引的论文；
R2 对 typed relational attention 的更广泛先例；R3 对关系一致性记忆管理的非 SAM 先例。
这些不应阻止接口准备，但在论文创新声明定稿前需继续查。首次设计只应声称在所检验任务
和机制上相对所列近邻的差异，不能写“首次提出图记忆”或“目前没有类似工作”。
