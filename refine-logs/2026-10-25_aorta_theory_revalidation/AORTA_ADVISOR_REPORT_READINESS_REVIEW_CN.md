# AORTA 导师汇报就绪审查

日期：2026-10-25  
状态：`RATIONALE_DEFENSIBLE__CORE_INNOVATION_STILL_EMPIRICAL`

## 1. 总裁决

当前研究思路已经可以向导师进行有把握的汇报，因为问题、失败证据、数学对象、网络输入、训练目标和
下一项判别实验已经对齐。可以自信陈述的是“为什么必须按这个顺序研究”；尚不能自信陈述的是
“Differential Query 已经优于最近方法”或“AORTA 已经提高了驾驶安全”。后两项仍是后续硬门。

当前最准确的一句话是：

> AORTA 研究的不是恢复一个外观不变的完整场景，而是在冻结规划器的合法候选集合中，恢复足以判断
> challenger 相对 native 任务效用的视觉证据；先证明正确视觉具有条件增量，再检验由完整时空候选
> 差异引导视觉读取，是否比容量与预算匹配的通用 scorer 更稳定、更高效。

## 2. 为什么问题从道路场转向决策关系

这不是叙事重包装，而是连续反例的结果：

1. 旧 quotient field 在 held-out development 上确实包含非平凡道路信息，但道路场误差不能直接推出
   候选排序或驾驶收益；
2. scalar set-min 被候选的共同路径支配，导致几何上不同的候选得到相同读出；
3. 2.5 秒短前缀会遗漏大量晚期分叉，因此不能用支持域便利性替代任务充分性；
4. 一般道路裕度的 temporal first-fork 恢复率和 spatial-exclusive 适用率均未达到资格门，旧
   road-margin reranker 因而在训练前停止；
5. B4R1 与 R005E1B 随后证明：合法候选域中确实存在可改善机会，但 raw one-frame、filtered
   one-frame、`U_ctx` 与策略级价值并不等价。

所以研究对象必须从“稳定地恢复整个道路”收紧为“在明确 evaluator scope 内恢复候选间的相对任务
关系”。这一步减少了不必要的场景重建任务，也避免把道路 proxy 冒充完整驾驶目标。

## 3. 当前数学对象为什么这样定义

对候选集合、因果历史和合法 mask 记为 `B=(C,H,M)`，冻结视觉表示记为 `Z`，监督对象是

\[
Y_i=U_{ctx}(\tau_i;h^-)-U_{ctx}(\tau_0;h^-),\qquad Y_0=0.
\]

选择 native-relative margin 有四个理由：

- 部署动作本身就是“保留 native”或“用 challenger 替换 native”；
- 绝对 utility 的共同偏置不影响选择，直接学习相对量更贴近 estimand；
- ragged candidate panel 可以用同一个 native star 表示，不需要伪造 complete graph 结构；
- 预测误差与选择 regret 存在清楚的桥：若合法候选上的同时误差不超过 `epsilon`，按预测最大值选择
  的真实 regret 不超过 `2 epsilon`。

`U_ctx` 不是最终策略价值。它被采用，是因为当前 panel 上它具有合法、可复算、非平凡的 oracle
headroom，而且 previous trajectory 是该 evaluator 的真实条件变量。官方 two-stage mapping 对当前
panel 覆盖为零，因此项目明确 fail-closed，没有把 `U_ctx` 偷换成完整 pseudo-closed-loop value。

## 4. 为什么必须先做 R005E2，而不是直接训练主网络

R005E2 检验最基础的必要条件：给定候选与历史后，正确视觉是否仍改变 margin 的条件均值。平方损失下：

\[
\mathcal R^*(B)-\mathcal R^*(B,Z)
=\mathbb E\!\left[\sum_i w_i(M)
(\mathbb E[Y_i\mid B,Z]-\mathbb E[Y_i\mid B])^2\right]\ge 0.
\]

这个恒等式不是 AORTA 的创新定理；它的职责是证明当前实验问题合法。如果 correct visual 没有可恢复
增量，直接训练 Differential Query 会把三种失败混在一起：视觉本身没有信号、通用函数类没有学到、
差异归纳偏置没有价值。

因此 R005E2 使用同参数、同 FLOPs、同 token、同初始化与同训练预算的三个通用 scorer：

- `V0 removed`：删除 anchor-specific 场景视觉，检验视觉是否胜过轨迹/历史与稳定先验；
- `V1 mismatch`：保持训练视觉边际但破坏场景绑定，检验正确场景对应是否重要；
- `V2 correct`：使用正确视觉，检验当前函数类能否恢复条件增量。

同一 V2 checkpoint 上的 mismatch/removed 干预再检查网络是否实际使用视觉。只有 proper risk、decision
regret、场景绑定干预和最小实践效应同时通过，才允许进入主架构比较。

## 5. 主网络为什么可能需要 Differential Query

若视觉增量成立，AORTA 的唯一主创新候选是：用完整 4 秒 native–challenger signed occupancy
difference

\[
\nu_{i0}=\mu_i-\mu_0=\nu^+_{i0}-\nu^-_{i0}
\]

构造候选关系 query，再从全部全局视觉 token 中读取证据：

\[
\nu_{i0}
\longrightarrow Q(\nu_{i0})
\longrightarrow \operatorname{Attn}(Q,K(Z),V(Z))
\longrightarrow \widehat\delta_{i0}.
\]

这样设计的原因不是“差分注意力更高级”，而是三个已观察事实：

- shared prefix 会淹没普通 set-min 的候选差异；
- late fork 要求完整时域，而不是 2.5 秒前缀；
- 只看差异区域也不充分，因为交通参与者、信号和全局布局可能产生非局部、非加性结果。

因此差异支持只负责提出“在哪里、以什么符号比较”，全局视觉仍负责回答“哪些上下文决定完整任务
结果”。它必须在相同 backbone、数据、参数、FLOPs、token 和 seed 下胜过 D-JEPA/
World4Scorer-style 通用 scorer；否则删除网络创新主张。

## 6. 与最近工作的边界

不能把下列内容单独称为创新：候选关系学习、native-relative correction、previous-plan context、risk
gate、native fallback、cross-attention、冻结 DINO 或 conformal calibration。D-JEPA、World4Scorer
及相关 driving scorer 已覆盖其中多项。

可防守的创新边界只有：

> 在 appearance intervention 下，完整时空 signed candidate difference 是否能以更低的决策 regret、
> 更少的 appearance-induced flips 或更高的样本效率，恢复 evaluator-aligned relation，并最终减少
> 错误替换而不伤害 clean。

这是一项待证假设，不是当前结论。

## 7. 导师最可能追问的问题与回答

### 为什么冻结 DINO？

为了先隔离“视觉信息是否可读”，避免 backbone 适配、表示学习和 scorer 结构同时变化。冻结 DINO
降低混淆，但不保证信息完整；若它丢失关键动态证据，R005E2 可以失败。

### 为什么不用 BEV、SDF 或道路风险场？

它们仍是合法基线和诊断对象，但既有反例已经证明道路几何 proxy 对完整候选任务排序并不充分。
AORTA 不需要恢复所有场景内容，只需恢复会改变合法候选相对效用的证据。

### 为什么不是直接 pairwise ranking loss？

当前先用 squared margin loss，因为它对应条件均值与 proper Bayes risk，便于判断视觉是否提供信息。
平均 MSE 不等于决策有效，因此裁决同时要求 group-level decision regret。ranking/orbit loss只有在已观察
到明确边界问题后才有资格引入。

### 为什么当前结果还不能称 ICCV 方法成功？

因为尚未证明 correct visual 在 R005E2 中胜过两类控制，也未证明 Differential Query 胜过 matched
generic scorer，更没有 NavDream、actual-history rollout、三类 planner、clean no-harm 和 batch-1
latency 的完整证据。

### 如果 R005E2 失败怎么办？

停止当前视觉主张，并区分信号缺失、冻结 backbone 信息损失、有限函数类、优化不稳定和统计功效不足；
不能直接增加复杂模块挽救。若只改善 risk 而不改善 regret，则裁决为 MIXED，问题转向 decision
boundary，而不是包装成成功。

## 8. 当前能够汇报与不能汇报的结论

能够汇报：

- 合法 source-fit `U_ctx` panel 存在非平凡、可复算的候选选择 headroom；
- previous trajectory 对该 scoped target 确实影响候选排序；
- R005E2 的 estimand、三臂控制、交叉拟合、统计单位和停止规则互相对齐；
- 当前理论允许一次严格、可证伪的视觉增量实验。

不能汇报：

- Differential Query 已有效或具有顶会级增量；
- AORTA 已经外观鲁棒；
- 离线 `U_ctx` 提升等于完整 pseudo-closed-loop 或驾驶安全提升；
- source group 置信区间可以外推到 NavDream 或新 planner。

## 9. 90 秒导师汇报版本

> 我们最初尝试用稳定道路场约束冻结规划器，但连续实验发现：共同路径会支配 scalar set-min，短时域
> 会遗漏晚期分叉，而道路几何差异也不能充分决定完整任务排序。因此我们把问题收紧为合法候选域上的
> native-relative contextual utility。R005E1B 已证明这个 scoped target 中存在可复算的 oracle
> headroom，并证明 previous trajectory 不能忽略。当前 R005E2 不验证主创新，而先用严格匹配的
> removed、mismatched 和 correct 三臂，回答正确视觉在候选与历史之外是否提供可学习的条件增量。
> 如果这一步通过，我们才比较通用 scorer 与由完整时空 signed candidate difference 引导的视觉
> query；后者的动机来自 shared-prefix、late-fork 和非局部结果三个反例。它只有在 matched baseline、
> NavDream、sequential history、跨 planner、clean no-harm 和时延上产生稳定增量，才构成论文方法。
> 所以目前我们能自信说的是研究链条和淘汰规则已经扎实，核心架构创新仍由接下来的实验决定。

## 10. 最终判断

可以自信向导师解释当前每个关键选择的原因，而且逻辑链经得起“为什么不用更简单方法”“为什么现在
不能直接宣称创新”“失败后怎么办”三类追问。当前最大的优点是主张边界清楚；最大的风险是核心网络
增量仍未被实验支持。最专业的汇报方式不是承诺成功，而是展示每个实验如何排除一个竞争解释，并说明
什么结果会让我们停止或重构。
