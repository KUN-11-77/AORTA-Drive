# Visual-First 文献与创新边界

## 1. Constant Eye 的真实贡献

原文与官方 arXiv 页面支持以下范围：

- navdream 用像素对齐的生成式外观变换隔离 appearance shift；
- 冻结 DINOv3-H+ 作为统一视觉底座；
- 对 LTF、DiffusionDrive、GTRS-Dense 分别训练 adapter 与 paradigm-specific planning head；
- 在 navtest/navhard/NavDream 上报告显著改善。

因此 Constant Eye 是 AORTA 的最近方法和必做基线。不能把“冻结 DINO + adapter”写成 AORTA 创新，也不能把 per-planner 训练描述为原 planner 完全不变。

本地原文：`C:/Users/19706/Desktop/Shannon/essays/AutoDriving/端到端自动驾驶与大模型/Constant_Eye.pdf`。
官方页面：https://arxiv.org/abs/2602.12563

## 2. Constant Eye 尚未闭合的机制问题

1. 分析主要是 PCA、t-SNE 与注意力可视化，没有用内部干预区分相关性与介导；
2. adapter 消融只在 navtest 展开，不能回答哪种结构真正带来 OOD 增益；
3. 不证明 DINO 信息能识别原生 planner 的任务统计量；
4. 不区分 planner null-space drift 与行为敏感 drift；
5. 没有证明修复 feature 或接近 clean trajectory 会提高 official utility；
6. 没有完整时延、显存、FLOPs、置信区间与随机种子不确定性；
7. 仍需训练 planner-specific 接口，未回答是否能保持原生 planner 完全冻结。

## 3. 理论工具的先行工作边界

- Zhao et al., ICML 2019 证明 domain-invariant source representation 并不足以保证目标适配，并给出与 conditional shift 有关的误差界。因此“更 invariant”不能作为 AORTA 结论。
  https://proceedings.mlr.press/v97/zhao19a.html
- Zhao et al., JMLR 2022 系统刻画 accuracy–invariance 的 information-plane tradeoff。因此 AORTA 只能声称把该一般限制具体化到 frozen planning interface，而不能把该权衡当作新理论。
  https://jmlr.org/papers/v23/21-1078.html
- Geiger et al., CLeaR 2024 的 DAS 已经研究 distributed causal variable alignment；分布式子空间搜索本身不是新贡献。
  https://proceedings.mlr.press/v236/geiger24a.html
- Geiger et al., JMLR 2025 已将 activation/path patching、causal mediation、DAS、steering 等统一在 causal abstraction 下；activation patching 本身不是创新。
  https://www.jmlr.org/papers/v26/23-0058.html

## 4. AORTA 可争取的窄创新

不是：DINO、feature invariance、Jacobian、activation patching、低秩 adapter 或 source augmentation。

而是以下联合机制：

> 对冻结端到端 planner 的 appearance failure，先用受控 interchange intervention 定位因果介导接口，再用 planner pullback metric 与 appearance drift 提出任务敏感低秩子空间，最后让冻结 DINO 只预测该子空间中的最小接口残差，并保持 planner/head 全冻结。

这项联合只有在以下三点都成立时才具有方法创新性：

1. 干预发现是 scene-specific 且优于 random/energy controls；
2. DINO 对该目标的 correct-vs-mismatch/removed 增量成立；
3. 相同参数/FLOPs/数据下优于普通 adapter 与 Constant Eye。

任一不成立，AORTA 都不能用理论语言掩盖负结果。
