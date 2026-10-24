# AORTA ICCV 故事与方法树 v2

状态：`POST_B3_CANDIDATE_NARRATIVE__B4_RUNNING__NOT_A_PAPER_CLAIM`

目的：把问题、数学对象、网络、损失、仲裁与实验约束成一条可证伪主线。本文档不是宣传稿；
任何未通过对应资格门的句子都不能升级为论文贡献。

## 1. 唯一记忆点

> **天气改变图像，却不改变同一场景中候选驾驶行为的真实优劣。外观鲁棒驾驶真正应保持的，
> 不是特征或稠密道路场本身，而是正确图像所支持的候选偏好。**

这一定义同时排除两个错误捷径：

- 表示跨外观不变但不含场景信息，例如常数表示；
- 道路场正确且稳定，但下游读出删除候选差异。B3 已用 perfect teacher 给出第二类反例。

## 2. 反常现象与技术缺口

现有链条通常写成

\[
X\rightarrow Z_{vision}\rightarrow F_{field}\rightarrow M(\tau)\rightarrow\operatorname{argmax}_\tau.
\]

其中任一上游量稳定，都不推出最终排序稳定。AORTA 已观察到：

1. 全局表示可高度相似而行为失败；
2. learned road field 在 held-out development 上正确、稳定且依赖正确图像；
3. 但即使把 learned field 替换成 perfect teacher，global/fixed/temporal scalar minimum 仍因共同
   minimum 大量丢失候选差异，B3 的 temporal fork 只恢复 170/2,033 个 common-min pairs。

因此，缺口不是“再做一个更准的 field”，而是：

> 如何直接学习外观干预下应保持不变、又足以决定候选排序的任务关系，并在证据不足时不伤害
> 冻结 planner？

## 3. 唯一数学对象

令物理场景为 \(S\)，外观为 \(A\)，观测 \(X=g(S,A)\)，冻结 planner \(\pi\) 产生 native
轨迹 \(\tau_0^\pi\) 和候选面板 \(\mathcal T=\{\tau_0,\ldots,\tau_{K-1}\}\)。当前 NAVSIM
evaluator 的 progress normalization 依赖 native-candidate 评分面板，因此合法效用必须写成

\[
U_i^*=U(S;\tau_0^\pi,\tau_i),
\qquad
\Delta_{ij}^*=U_j^*-U_i^*.
\]

方法估计的是固定 native context 下的 **appearance-quotient preference operator**，而不是跨
planner 无条件成立的绝对 candidate utility。绝对节点分数只在等价类

\[
[s]=\{s+c(S,\pi)\mathbf 1:c(S,\pi)\in\mathbb R\}
\]

中可辨识，而 \(\Delta_{ij}\) 对共同 gauge 完全不变。

## 4. 从对象必然推出的架构

\[
Z=E_{vision}(X),\quad a_0=E_{traj}(\tau_0),\quad a_i=E_{traj}(\tau_i),
\]

\[
h_i=H_\theta(Z,a_0,a_i),\quad s_i=W_sh_i,
\quad \widehat\Delta_{ij}=s_j-s_i.
\]

| 设计 | 为什么必须存在 | 它解决的挑战 | 必须由什么消融证明 |
|---|---|---|---|
| 共享视觉编码 | 外观证据对同一候选面板共享，且需要实时 | 避免每候选重复视觉网络 | 相同 backbone 下的延迟与候选数缩放 |
| 完整 4 s trajectory encoder | 2.5 s 前缀可删除晚期分叉 | 保留候选完整决策差异 | prefix、fixed-grid 与完整时域对照 |
| native-conditioned interaction | evaluator 标签本身依赖 native-candidate panel | 与真实 estimand 对齐 | 去掉 native context 后的 label/regret 退化 |
| shared scalar node score | 真值若来自标量 utility，其 pair preference 应可积 | 精确反对称与 cycle consistency | 容量匹配 direct-pair head 的 cycle residual、regret、迁移 |
| permutation-equivariant candidate processing | 候选列表顺序不是语义 | 防止位置记忆 | 候选置乱等变性测试 |
| selective lower-bound arbitration | 预测正确性不等于可靠替换 | 证据不足时保持 native | coverage、sharpness、false improvement、abstention、clean harm |

共享 score 使

\[
\widehat\Delta_{ij}=-\widehat\Delta_{ji},\qquad
\widehat\Delta_{ij}+\widehat\Delta_{jk}+\widehat\Delta_{ki}=0
\]

成为参数化恒等式，而非软正则。Hodge/coboundary 只用于说明这个结构；若 matched direct-pair
head 没有结构性失败，必须删除该叙事，不能把数学术语当作创新本身。

## 5. 训练目标与其职责边界

对冻结 evaluator 的合法 source-only 标签 \(y_i\)，任务差分损失为

\[
\mathcal L_{task}=\sum_{i\ne j}w_{ij}\,
\rho\!\left(\widehat\Delta_{ij}(X,\tau_0)-(y_j-y_i)\right).
\]

同一物理场景的外观干预 \(T_a\) 只约束 preference，而不是全部 feature：

\[
\mathcal L_{app}=\sum_a\sum_{i\ne j}w_{ij}\,
\rho\!\left(\widehat\Delta_{ij}(T_aX,\tau_0)-
\operatorname{sg}[\widehat\Delta_{ij}(X,\tau_0)]\right).
\]

候选总目标是

\[
\mathcal L=\mathcal L_{task}+\lambda_{app}\mathcal L_{app}
+\lambda_{factor}\mathcal L_{factor}.
\]

职责必须分清：

- \(\mathcal L_{task}\) 排除“稳定但无信息”的常数解；
- \(\mathcal L_{app}\) 只约束同场景外观扰动下的关系稳定；
- \(\mathcal L_{factor}\) 只有在 factor label 合法且带来匹配容量优势时才保留；
- 不加入 cycle loss，因为架构已经精确满足 cycle；
- mismatch/removed-image 是视觉必要性反证，不应被偷换成训练技巧；
- \(w_{ij}\)、稳健函数和温度必须在 source protocol 中冻结，不能按结果挑选。

## 6. 从预测到干预的条件闭环

令 \(\Delta_{i0}^*=U_i^*-U_0^*\)。若同时误差事件

\[
\mathcal E=\left\{|\widehat\Delta_{i0}-\Delta_{i0}^*|\le r_i,
\ \forall i=1,\ldots,K-1\right\}
\]

成立，定义 \(L_i=\widehat\Delta_{i0}-r_i\)，只在

\[
\max_i L_i>0
\]

时选择最大 lower bound 对应候选，否则保留 native。在 \(\mathcal E\) 上，被选候选严格优于
native。该结论是带假设的 model-relative 保证，不是现实驾驶安全证书；source calibration 也不能
外推为任意 NavDream、planner 或 candidate-policy shift 下的分布自由保证。

## 7. 三项潜在贡献及其生死门

### C1：重新定义外观鲁棒性的学习对象

从 feature/field invariance 转向 native-conditioned preference invariance。

**生死门：**B4 必须先证明标签接口合法、当前候选面板有非平凡 oracle ceiling；随后 correct image
必须在未见场景和外观下优于 mismatch、removed 与无图像先验。

### C2：候选图上的 exact preference architecture

用 score coboundary 把 antisymmetry、cycle consistency 和 gauge invariance 变为硬结构，同时保持
\(O(K)\) 推理。

**生死门：**在相同输入、参数量、训练数据、计算和 seeds 下，必须优于 direct-pair、普通
candidate scorer、GTRS-style 和 DA-WAM-style controls。否则这只是“scores 相减”。

### C3：选择性 native-preserving arbitration

把 simultaneous uncertainty 用于“是否允许干预”，而不是为预测附加一个 confidence 数字。

**生死门：**独立 calibration 上同时满足 coverage 与非真空 sharpness，并在 held-out appearance、
跨 planner、clean harm 和 false-improvement 上成立；否则保留为条件定理，不作为实证贡献。

## 8. 与近邻工作的最短区别

- NMP/MP3 已覆盖 field/cost-volume 接规划，因此“视觉场 + planner”不是贡献；
- GTRS 已覆盖 trajectory tokenizer、scorer、augmentation 与 refinement，因此“普通候选 scorer”
  不是贡献；
- DA-WAM 已覆盖 candidate-conditioned future 与 factor/utility head，因此 cross-attention 和多头本身
  不是贡献；
- TOAD 已覆盖 scorer-as-reward 与 test-time search，因此 rerank/search 本身不是贡献；
- decision-focused learning、CVaR/EVaR 和分布鲁棒风险图均是可用工具或基线，不是新颖性来源。

AORTA 只有在 **appearance intervention 上的合法 preference estimand + exact relational structure +
selective native-relative decision rule** 三者由同一证据链支持时，才与这些近邻形成实质区别。

## 9. 30 秒、2 分钟与 Poster 合同

### 30 秒

> 外观鲁棒驾驶通常稳定图像特征或道路场，但我们发现，即使使用正确的道路场，一个不充分的
> 读出也会删除几乎全部候选决策差异。AORTA 因而直接学习冻结 planner 候选图上的外观商偏好：
> 它以 native 为上下文，用一个可积的 score potential 精确保持偏好结构，并只在同时误差下界证明
> challenger 优于 native 时干预。我们稳定的不是代理表征，而是真正决定轨迹选择的关系。

在 B4、结构消融、NavDream、跨 planner 和时延门全部通过前，上述后两句只能称方法假设。

### 2 分钟

1. **矛盾：**同一路只换天气，feature 仍相似，候选排序却翻转；稳定也可能只是常数先验。
2. **反例：**perfect road field 经 scalar minimum 后仍删除候选差异，证明问题位于学习对象与读出，
   不只是视觉精度。
3. **对象：**真实应保持的是 fixed native context 下的 \(\Delta_{ij}^*\)，绝对 score 只在共同
   gauge 等价类中可辨识。
4. **方法：**共享视觉编码 + 完整轨迹条件 + exact score coboundary + selective lower bound。
5. **证据：**oracle ceiling、correct-vs-mismatch、matched structure、NavDream/跨 planner decision
   gain、clean harm/abstention 与端到端 P50/P95/P99。

### Poster 第一眼结构

1. 左栏只展示“stable representation / perfect field, wrong decision relation”的反例；
2. 中栏只展示 \(\Delta_{ij}=s_j-s_i\) 及 native-conditioned network；
3. 右栏只展示 ranking/regret、选择性干预与 latency，不堆砌历史几何模块；
4. 每个主结果旁直接放最强竞争解释的控制，而不是把消融藏到补充材料。

## 10. Oral 级证据标准

Poster 竞争力至少要求：

- B4 非平凡 oracle ceiling；
- 未见组与 NavDream 上同时 task-correct、appearance-stable、correct-image necessary；
- exact head 对容量匹配自由 pair head 的结构优势；
- 至少两个冻结 planner 上的排序/官方行为收益；
- clean harm、abstention、最差组、候选规模和 P50/P95/P99 完整报告。

Oral 候选还要求：

- “perfect field 仍会被错误 readout 删除决策差异”的现象跨 planner 重复；
- 一张主图同时排除“只是 DINO 更强”“只是普通 scorer”“只是更多参数”三个解释；
- 理论、网络和实验始终使用同一个 preference 单位；
- 核心结果能被一句话复述，并能改变社区对视觉鲁棒性评价对象的理解。

## 11. 禁止进入主故事的内容

- 圆盘/矩形证书、ROI 工程史和所有失败模块清单；它们进入 failure analysis 或 appendix；
- 没有 matched ablation 支持的 Hodge、信息瓶颈、conformal、CVaR 等术语；
- 把 development 结果写成 untouched test，把稳定性写成安全，把机械审计写成同行评审；
- 以公式数量、网络复杂度或模块数量替代贡献必要性；
- 在 B4 结果之后移动 gate，或只展示有利 candidate/appearance/planner。

## 12. 当前裁决与下一步

当前故事是连贯且有潜力的，但只到“问题与方法假设”层：

- 问题真实性与旧 scalar readout 反例：`SUPPORTED`；
- held-out development 视觉正确性/必要性：`SUPPORTED_WITH_SCOPE`；
- preference label 合法性与 oracle ceiling：`B4_RUNNING`；
- exact architecture 必要性、NavDream、跨 planner、决策收益、时延：`UNPROVEN`。

严格顺序只有三步：

1. B4 完成独立重放和机械审计，裁决 preference object 是否值得学习；
2. 若 B4 通过，只预注册 matched-structure study，不直接训练；
3. 只有结构必要性成立，才进入外观泛化、选择性决策和部署时延闭环。

