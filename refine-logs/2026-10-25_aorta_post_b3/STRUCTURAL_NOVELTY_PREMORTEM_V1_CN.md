# AORTA preference structure 创新性预死亡审查 v1

状态：`CORE_STRUCTURE_AT_RISK__STAR_GRAPH_VACUITY_PROVED__NO_TRAINING`。

目的：在 B4R1 之前先用最强 reviewer 反驳检查 exact coboundary/Hodge 是否真的构成网络创新。
结论是：**若部署只使用 native–challenger comparisons，则该结构在星图上没有约束力，不能作为
核心贡献。**

## 1. Reviewer 最短拒稿意见

> 你们的模型只是为每条候选输出一个 scalar score，再做差。由于系统最终只比较 challenger 与
> native，任意 \(K-1\) 维相对分数都能由这种参数化表示；所谓 Hodge consistency 在星图上是
> vacuous。除非完整 candidate–candidate cycle 具有独立任务语义，并且结构化模型在容量、数据和
> 计算匹配下显著改善泛化或 regret，否则这不是方法创新。

这条批评成立，必须在实现前进入理论与实验合同。

## 2. 星图空泛性定理

固定 context 的合法候选域为 \(V=\{0,1,\ldots,K-1\}\)，其中 0 是 native。部署只需要

\[
q_i=\Delta_{i0},\qquad i=1,\ldots,K-1.
\]

令 native-centered star incidence operator 为

\[
D_\star s=(s_1-s_0,\ldots,s_{K-1}-s_0)^\top.
\]

### 命题 S1：native-star 上 exactness 不限制假设空间

对任意 \(q\in\mathbb R^{K-1}\)，均存在 \(s\in\mathbb R^K\) 使 \(D_\star s=q\)。因此

\[
\operatorname{Im}(D_\star)=\mathbb R^{K-1}.
\]

**证明。** 取 \(s_0=0,s_i=q_i\) 即得 \(s_i-s_0=q_i\)。证毕。

### 推论 S1.1

在只有 native-relative labels 和 decisions 的接口上：

- 自由输出 \(K-1\) 个 \(\widehat\Delta_{i0}\)；
- 输出 \(K\) 个 node scores 再计算 \(s_i-s_0\)；

具有相同的可表示集合。星图没有闭合 cycle，`cycle consistency` 不产生可检验约束。把第二种写成
Hodge/coboundary 并不会自动产生新的归纳偏置、样本效率或泛化保证。

命题 S1 本身只是用于排除错误创新主张的初等线性代数事实，**不作为 AORTA 的理论创新**。
它的价值在于把可发表命题收紧到真正可证伪的机制，而不是为方法增加新的数学包装。

当 \(K=1\) 时更为彻底：没有 edge、preference 或谱隙。

## 3. 完整图为何也不能自动挽救创新性

如果定义所有 challenger–challenger edges，complete graph 上确实存在 cycle，并且 node scorer
强制 edge field 可积。但当前直接观测的是 native-conditioned scalar

\[
y_i=U(S;\tau_0,\tau_i),
\]

而 challenger–challenger labels \(y_j-y_i\) 是从这些 scalar 代数派生，不是另一个 official
candidate-pair evaluator 的独立观测。因此 complete-graph consistency 首先是标签构造的恒等式，
不是已证明的驾驶世界性质。

它仍可能作为有用的多任务归纳偏置，但必须证明：

1. 对未见场景/外观的 ranking regret 有改善；
2. 改善不是因为 node head 参数更少、loss edge 更多或计算预算不同；
3. challenger–challenger edges 提供超出 native-star supervision 的统计信息；
4. 若真实 preference 出现 context-dependent non-transitivity，强制 exactness 不会产生系统偏差。

否则主文只能称其为高效参数化，不能称核心算法创新。

## 4. 对现有谱 regret 界的收紧

现有命题

\[
\mathrm{Regret}\le 2\sqrt{\mathcal L_Q/\lambda_2(L_w)}
\]

是正确的代数界，但不是自动的新理论：图排序与 Laplacian 谱界已有成熟先例。对实际 native
仲裁，更直接的对象是同时 native-relative error：

\[
\max_{i\in V\setminus\{0\}}
|\widehat\Delta_{i0}-\Delta^*_{i0}|\le\varepsilon.
\]

若 \(\hat i=\arg\max_i\widehat\Delta_{i0}\)，则

\[
y_{i^*}-y_{\hat i}\le2\varepsilon.
\]

而 selective lower bound \(L_i=\widehat\Delta_{i0}-r_i>0\) 已足以推出条件性 native
improvement。该接口与部署动作完全对齐，理论更紧，也暴露出 Hodge 并非必要条件。

## 5. 因此真正可能成立的贡献是什么

当前只能保留一个候选主贡献组合，且仍待实验：

> **appearance intervention 下的 task-sufficient native-relative relation learning，加上
> simultaneous uncertainty 驱动的 selective arbitration。**

这里的潜在新意来自三件事的闭环，而非 Hodge 术语：

1. **学习对象：**稳定的是合法候选域上的任务关系，不是 feature/BEV/SDF；
2. **视觉机制：**正确图像必须对未见场景候选关系提供不可由 prior/mismatch 解释的信息；
3. **行为规则：**只在同时 lower bound 支持时改变 frozen planner，否则 native fallback。

共享 scalar scorer、set equivariance、full-trajectory encoder、DINO backbone 本身都只能算必要实现
或强基线，除非 matched experiment 证明不可替代。

## 6. B4R1 后的结构生死实验

只有 B4R1 合法 oracle 通过后，才允许另行预注册以下容量/计算匹配实验：

| Arm | 输出接口 | 目的 |
|---|---|---|
| A | ordinary native-relative vector head | 最强简单基线；直接输出 \(K-1\) deltas |
| B | shared node scorer + star differences | 检验“score difference”本身是否只是重参数化 |
| C | shared node scorer + complete-graph quotient loss | 检验额外 pair structure 是否改善泛化/regret |
| D | free antisymmetric complete-edge head | 检验 exact cycle restriction的作用 |
| E | candidate-independent visual prior / removed-image | 排除数据集和轨迹先验 |

所有 arms 必须匹配 backbone、trajectory inputs、train labels、optimizer、seeds、参数/FLOP budget 与
训练步数。主要 endpoint 必须是 held-out group ranking regret/native-selection error，而不是 train
cycle residual。若 B≈A 且 C/D 无可靠差异，则删除 Hodge/结构创新；方法应退回 relation target +
selective arbitration，或承认当前网络创新不足。

## 7. 对 ICCV 故事的影响

主故事应避免：

> “我们使用 Hodge theory，因此结构更严谨。”

只有证据通过后才允许写：

> “我们发现外观不变的代理场会删除决策差异，因此把学习目标移到合法候选关系；进一步，完整
> pair structure/选择性仲裁中的某一项经公平消融被证明是实现未见外观决策收益的必要机制。”

这会让故事由失败现象和不可替代性推动，而不是由数学名词推动。

## 8. 顶会故事的唯一允许骨架

一篇主论文只保留一项主贡献和一项从属机制：

1. **主贡献：decision-quotient learning。** 外观干预下不强迫完整 feature、BEV 或道路场不变，
   而只学习合法候选域上足以决定 frozen planner 选择的 native-relative task relation；
2. **从属机制：selective arbitration。** 同时误差界不足以证明 challenger 优于 native 时拒绝干预。

网络结构、loss 和实验都必须由这条骨架推出：

- 完整轨迹条件用于避免前缀截断删除晚期分叉；
- correct/mismatch/removed image 用于证明 relation 确实依赖场景视觉；
- appearance intervention loss 只约束同场景关系，不约束无关 feature；
- ragged legal-domain mask 保证训练对象与 evaluator/oracle 一致；
- simultaneous lower bound 把预测误差转换为条件性干预规则；
- NavDream、跨 planner、clean harm 和 latency 是行为终点，不是额外贡献。

如果 matched experiment 不能证明 complete-graph structure 的增量，它必须从标题、摘要和主图中
删除；论文仍可由 decision quotient + selective arbitration 成立，但不能再声称图/Hodge 架构创新。
