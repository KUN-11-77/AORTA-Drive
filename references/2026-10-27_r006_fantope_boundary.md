# R006 Fantope 理论出处与创新边界

日期：2026-10-27。用途：结果前理论核验；不是 R006V0F 科学证据。

## 已核验原始来源

1. Vincent Q. Vu, Juhee Cho, Jing Lei, Karl Rohe. *Fantope Projection and Selection:
   A near-optimal convex relaxation of sparse PCA*. NeurIPS 2013。
   [官方论文页](https://papers.nips.cc/paper_files/paper/2013/hash/81e5f81db77c596492e6f1a5a792ed53-Abstract.html)。
   官方摘要明确把 Fantope 定义为 rank-\(d\) projection matrices 的凸包，并以其构造 sparse
   principal-subspace estimation 的凸松弛；原文还使用 ADMM 求解并给出统计收敛结果。
2. Yixuan Qiu, Jing Lei, Kathryn Roeder. *Gradient-based Sparse Principal Component Analysis
   with Extensions to Online Learning*. arXiv:1911.08048, 2019。
   [arXiv 页面](https://arxiv.org/abs/1911.08048)。摘要明确说明 FPS 是凸 formulation，但原始
   算法计算代价较高，并研究更高效的梯度算法及优化/统计误差界。

本地可追溯来源保存于：

- references/2026-10-27_r006_fantope/sources/fantope_neurips2013.html*
- references/2026-10-27_r006_fantope/sources/fantope_gradient_1911.08048.html*
- references/2026-10-27_r006_fantope/sources/arxiv_fantope_query_20261027.xml*

## 对 AORTA 的合法借用

标准 Fantope

\[
\mathcal F_r=\{P=P^\top:0\preceq P\preceq I,\operatorname{tr}P=r\}
\]

及其“rank-\(r\) projector 集合的凸包”解释属于已有数学，不能作为 AORTA 创新。AORTA 的候选诊断
把 planner-pullback residual

\[
\sum_i\|J_i(I-P)\delta_i\|_{W_i}^2
\]

放在同一凸域上。由于 \(P\mapsto W_i^{1/2}J_i(I-P)\delta_i\) 是仿射映射，该目标是凸的；这是针对
AORTA 对象的直接数学推导，但仍是标准凸分析应用，不应包装成核心理论贡献。

## 它能回答与不能回答的问题

能回答：

- 当前 Stiefel 结果的 seed instability 是否主要来自非凸求解；
- 最优对象更像硬 projector 还是具有分数特征值的软 task metric；
- top-\(r\) 舍入是否产生不可忽略的 objective gap。

不能回答：

- 外观漂移是否存在跨场景共享的真实低维因果机制；
- soft \(P\) 是否可由 RGB/DINO 在组外预测；
- 有限 activation patch、NavDream 或驾驶行为是否改善；
- AORTA 是否优于 Constant Eye 或容量匹配 adapter。

因此 Fantope 只在 R006V0F 返回 estimator instability 时作为最小失败诊断，不属于当前主方法，也不
自动增加论文创新性。真正可能形成贡献的仍是：任务敏感机制的发现、由机制自然导出的可恢复接口，以及
matched baseline 无法替代的跨外观行为收益。
