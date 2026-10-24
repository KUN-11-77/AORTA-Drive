# Preference quotient / graph-ranking 查新边界

检索时间：2026-10-25（Asia/Shanghai）  
状态：`TARGETED_BOUNDARY_REVIEW__NOT_A_SYSTEMATIC_REVIEW`

本轮只回答一个问题：AORTA 新提出的 candidate-graph、Laplacian quotient loss、Hodge/cycle 与谱隙
解释中，哪些是已有数学，哪些仍可能形成任务与算法贡献。原始官方来源、文本及 SHA 元数据保存在
`F:\AORTA-Drive-artifacts\literature\2026-10-25_preference_quotient`。

## 1. 直接相关来源

| 工作 | 状态 | 官方来源 | 已覆盖的内容 | 对 AORTA 的约束 |
|---|---|---|---|---|
| Jiang, Lim, Yao, Ye, *Statistical Ranking and Combinatorial Hodge Theory* | arXiv:0811.1067v2；后续发表于 Mathematical Programming | https://arxiv.org/abs/0811.1067 | pairwise ranking 作为图 edge flow；gradient 与 divergence-free/cyclic residual 的 Hodge 分解；L2 global ranking | gradient/cycle decomposition、Hodge 术语和“全局 score 产生可积偏好”不是 AORTA 原创 |
| Hirani, Kalyanaraman, Watts, *Least Squares Ranking on Graphs* | arXiv:1011.1716v4 | https://arxiv.org/abs/1011.1716 | 用节点值差拟合 pairwise edge data 的 graph least squares | quotient-Laplacian squared loss 的基本线性代数不是新理论 |
| Shah et al., *Estimation from Pairwise Comparisons: Sharp Minimax Bounds with Topology Dependence* | arXiv:1505.01462v1 | https://arxiv.org/abs/1505.01462 | BTL/Thurstone pairwise estimation；误差与 comparison-graph Laplacian spectrum 相关 | 用谱结构讨论估计误差、比较图设计不是新颖性来源 |
| Hajek, Oh, Xu, *Minimax-optimal Inference from Partial Rankings* | arXiv:1406.5638v1 | https://arxiv.org/abs/1406.5638 | partial ranking 的误差界与比较图 Laplacian spectral gap 相关 | “谱隙越大估计越稳”的一般思想已有明确先例 |
| Li et al., *Expand-and-Prune: Test-Time Candidate-Space Adaptation for End-to-End Autonomous Driving* | 2026 preprint；arXiv:2610.23262v1 | https://arxiv.org/html/2610.23262 | 保持 planner 参数与 native scoring procedure 不变，在测试时扩展候选并按中间 score entropy 自适应剪枝；跨四种 planner/四个 benchmark | “冻结 planner + 测试时候选适配 + 不确定性 + 跨架构 + 效率”已被直接占据；AORTA 不能以 plug-and-play candidate assistance 为主新意 |
| Zheng et al., *Gen-Drive* | 2024 preprint；arXiv:2410.05582v1 | https://arxiv.org/html/2410.05582 | 用 VLM 辅助 pairwise preference 训练 scene/plan scalar reward model，并用于候选选择和 RL 微调 | 驾驶中的 pairwise preference data、scalar reward scorer 与 generation-then-evaluation 已有先例 |
| *DriveReward* | 2026 preprint；arXiv:2606.08525v2 | https://arxiv.org/html/2606.08525 | temporally grounded trajectory evaluation dataset、VLM reward model、multi-modal trajectory scoring | 学习驾驶 reward/trajectory score 及跨基线使用不是 AORTA 原创 |

原始文件 SHA：

- `hodge_rank_abs.html`: `49425ca1c1c8dcb75bb2ee7bc84474204d59a88e016f131d023a7fcd5fda49ef`
- `least_squares_ranking_abs.html`: `0ed5314a57917679bb2c4341622d46bbb6928b70258d2682d71a854d24225663`
- `topology_minimax_abs.html`: `a3159ff1ffff3b02c57c6809c80a697510332b2f80b2be8b475bc5976f3469a7`
- `partial_rankings_abs.html`: `44975a4a6d3850e27f78e146dbf8ed37da6848e4110b84c7badca4433577e71c`
- `expand_prune_html.html`: `86cb3f1f5de2fb0e73a3f42c6adfce91d707781b2b6d334ebd49763a5229b54d`
- `gendrive_html.html`: `1db909908353cd5b25c29538a17303ebaf30477edec97c213d43caf3b5d17445`
- `drivereward_html.html`: `34cfb34237aa135127d6f869931b96a6f9e051a9f2bfdcbc884708b9545c12fb`

## 2. 明确不能宣称原创的内容

1. 把 pairwise preference 表示成图上的 edge flow；
2. 用节点 score difference 得到反对称、cycle-consistent preference；
3. 将自由 edge flow 分解为 gradient 与 cyclic residual；
4. 用 graph least squares / Laplacian energy 拟合 pairwise comparison；
5. 观察 comparison topology 或 spectral gap 影响估计误差；
6. 单独使用 Bradley–Terry、Thurstone、ranking loss 或 Hodge projection。
7. 从 pairwise driving preferences 训练 scalar reward/scorer；
8. 给冻结 planner 增加测试时候选处理并跨架构验证；
9. 单独用 score entropy/confidence 调整候选数量或计算预算。

因此 `PREFERENCE_QUOTIENT_THEORY_V0_CN.md` 中的命题 1、Hodge decomposition 及谱工具只能作为
设计地基与分析语言。单独投稿这些数学会构成明显的已有工作重述。

## 3. 仍可能成立的 AORTA 特定贡献

潜在新意不在图论公式本身，而在以下组合是否由驾驶问题必然推出、并具有不可替代的实证价值：

1. **估计对象：**在 appearance-only intervention 下，以 frozen planner native trajectory 为上下文的
   evaluator-legal preference，而不是通用用户/item ranking；
2. **问题反例：**即使视觉道路场正确，常用 scalar road-margin readout 仍因 common-min domination
   删除决策差异；这迫使学习对象从 field correctness 转向 decision-sufficient preference；
3. **视觉条件模型：**共享视觉证据与完整轨迹相互作用，直接估计候选关系，同时通过 correct-image
   vs mismatch/removed controls 证明不是场景先验；
4. **结构必要性：**exact-gradient head 是否在 appearance shift、candidate-policy/planner transfer、
   regret 与 latency 上胜过容量匹配的自由 pair head；
5. **选择性使用：**同时误差 lower bound 是否能把 preference 预测转化为 native-preserving intervention，
   并以 clean harm、false improvement 和 abstention 审计，而不是只提高 ranking accuracy。

相对于 Expand-and-Prune，AORTA 的区别不能写成“我们也不改 planner”。必须正面证明：Expand-and-Prune
解决的是候选 support/capacity，而 AORTA 解决的是 appearance shift 下 **候选优劣证据本身是否正确、
是否依赖正确图像、以及何时有资格覆盖 native**。两者原则上互补，因此最终实验还需要把
Expand-and-Prune 风格 candidate-support control 与 preference-estimation gain 分离，避免把新增候选
headroom 误归因给视觉任务关系。

只有第 1–5 项作为一个闭环被验证，才可能形成 ICCV/CVPR 方法贡献。叙事应写成“把成熟的
graph-ranking structure 用于一个此前定义错误的视觉驾驶 estimand，并证明这一结构在外观 shift 下
具有必要性”，而不是“我们发明了 Hodge preference”。

## 4. 对核心公式的降级与保留

### 作为地基保留

\[
\mathcal L_Q=(s-y)^\top L_w(s-y)
\]

它精确匹配 gauge-invariant pair error，且连通图的零空间只有共同 offset，适合作为 principled core
loss。完全图统一权重下，它可用 centered-node loss 以 \(O(K)\) 实现。

### 不能单独作为 novelty

\[
q=D L^\dagger D^\top q+(I-D L^\dagger D^\top)q
\]

是标准 least-squares/Hodge projection。正文若使用，应直接引用 Jiang et al. 与 graph ranking 文献，
并把贡献放在 native-conditioned visual estimand 和 matched empirical necessity 上。

### 可作为项目自有的任务桥梁，但需谨慎措辞

若单场景 \(\mathcal L_Q\le\eta^2\)，则由 Laplacian Poincaré bound 与 score-selection inequality 得到

\[
\operatorname{Regret}\le 2\eta/\sqrt{\lambda_2(L_w)}.
\]

该推导对 AORTA 的 loss—decision 对齐很有用，但其数学构件属于标准谱图/排序理论。除非后续全文
查新证明同样形式在该 native-conditioned driving estimand 中尚未出现，否则应称“task-specialized
bound”而非新的一般 ranking theorem。

## 5. 后续查新任务

B4 通过后、matched-structure 预注册前，还需补充：

1. 视觉/多模态 preference learning 中通过 scalar potential 强制 transitivity 的神经架构；
2. autonomous driving candidate ranking/scoring 中的 pairwise、listwise 与 native-relative labels；
3. distribution shift 下的 ranking calibration、selective ranking 与 conformal ranking；
4. graph topology / hard-negative weighting 与 decision regret 的现有界；
5. 2025–2026 candidate-conditioned driving scorer 的最新论文与代码合同。
6. Expand-and-Prune 与 AORTA 在 frozen-planner、candidate support、uncertainty 和 runtime 上的
   matched boundary；若公开代码可用，应纳入公平系统基线或明确无法对齐的接口差异。

在这些检索完成前，C16 仍保持 `proposed hypothesis only`。

