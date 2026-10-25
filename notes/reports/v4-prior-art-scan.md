# v4 TG-CWP 撞车检查（2026-10-25）

> 检索范围：3 组 web 检索 + 1 篇重点论文全文摘要核对（Remote SAMsing）。
> 覆盖有限，论文写作前建议按 awesome-list（见文末）再做一轮系统扫描。
> 对照基准：`notes/design-v4.0.md` 的差异轴——二维非因果 tile 拓扑、跨窗实例
> 共指、可靠 anchor 选择、两遍非递归正框回注 SAM3、开放词汇实例级。

## 逐工作判定

| 工作 | 一句话方法 | 与 TG-CWP 的重叠 | 判定 |
|---|---|---|---|
| [samgeo：Tiled Segmentation with SAM 3](https://samgeo.gishub.org/examples/sam3_tiled_segmentation)（gishub，2025） | SAM3 滑窗切 GeoTIFF，逐 tile 独立推理再拼接 | 即我们的 C0（独立 tiles + 融合） | **仅相关**：证明 C0 是标准工程实践，TG-CWP 的贡献必须落在 C1/C2 的差异上 |
| [Remote SAMsing](https://arxiv.org/abs/2605.00256)（arXiv 2605.00256，2026-04） | SAM2 无类别自动分割：逐 tile 多遍绘制+阈值松弛提覆盖；跨 tile 碎片用 contextual padding + 免参数 best-match merge 几何拼接 | 跨 tile 碎片合并（= C0 融合器设计参照）；无文本 prompt、无类别、无 anchor、无重提示 | **不撞车**，但压缩了"仅做跨 tile 融合"的宣称空间 |
| OmniSAM（ICCV 2025，已源码级分析，explainer 2026-09-29） | 一维横向 patch 序列当视频帧，复用 SAM2 FIFO memory，固定类语义 UDA | 跨窗 memory（一维、语义、最近邻序） | **部分重叠**，已登记；TG-CWP 用正框回注而非 memory，二维非因果 |
| Slide-SAM（MIDL 2024，已分析） | 三张相邻 CT/MRI 切片堆窗，中心切片 prompt，边缘 mask→box/point 沿 z 轴传播 | prompt 跨窗传播（z 轴、医学、三片链） | **部分重叠**，已登记；TG-CWP 是 x–y 网格 + 实例级共指 + anchor 选择 |
| [V2-SAM](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_V2-SAM_Marrying_SAM2_with_Multi-Prompt_Experts_for_Cross-View_Object_Correspondence_CVPR_2026_paper.pdf)（CVPR 2026） | 跨视角（多视图）anchor prompt 生成做目标对应 | 仅"anchor prompt"术语相近；问题是跨视角对应，不是同图 tile | **仅相关** |
| [SAM3 self-distillation GOOSE 2D](https://arxiv.org/html/2606.20130v1)（2026-06） | 自蒸馏 + 滑窗多尺度推理（训练改造） | 大图 tile 推理（训练路线） | **仅相关**：v4 红线本就不训练 |
| SAM3.1（Meta，2026-03 [博客](https://ai.meta.com/blog/segment-anything-model-3)） | 视频 masklet 传播加速 | 官方视频传播，非大图 tile | **背景**：若官方后续发布大图方案需重查 |
| [MV-SAM](https://jaesung-choe.github.io/mv_sam) | 多视图 promptable 分割 3D 一致性 | 跨视图，不同问题 | 仅相关 |
| [PanoEarth](https://www.scilit.com/publications/9b1bf41d31e8b7eb0f2edf36f4dbcdc5)（2026） | 大规模 OV 全景分割数据集 | 数据资源（大图+类别词表） | 仅相关，可作未来数据候选 |

## 结论

1. **未发现正面撞车**：没有工作组合"二维 tile 图拓扑 + 跨窗实例共指 + 可靠
   anchor 选择 + 正 box 回注 SAM3 + 开放词汇实例级"。TG-CWP 的差异轴目前成立。
2. **两个挤压效应**：samgeo/Remote SAMsing 表明"滑窗 + 几何融合"是常识
   （C0 不可作为贡献）；OmniSAM/Slide-SAM 表明"跨窗 memory/prompt 传播"在
   一维/医学已有先例。TG-CWP 的论文主张必须严格锚定在
   **C2 与 C1/C3 的对照差**（关系选择 vs 固定邻接 vs 随机 anchor）。
3. 遗留动作：投稿前按
   [Awesome-Remote-Sensing-Open-Vocabulary-Learning](https://github.com/zhangda1018/Awesome-Remote-Sensing-Open-Vocabulary-Learning)
   再做一轮系统扫描；Remote SAMsing 需在相关工作章引用并划界。

Sources: 检索结果见上表链接（samgeo、arXiv 2605.00256、arXiv 2606.20130、
CVPR 2026 V2-SAM、Meta SAM3.1 博客、MV-SAM、PanoEarth、SCORE arXiv 2507.12857）。
