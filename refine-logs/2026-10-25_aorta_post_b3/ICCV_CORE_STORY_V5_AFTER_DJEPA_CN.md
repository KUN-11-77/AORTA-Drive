# AORTA ICCV 核心故事 v5：D-JEPA 之后的收紧版本

状态：`SHARP_HYPOTHESIS__NOVELTY_NARROWED__EMPIRICAL_CORE_UNPROVEN`

这不是宣传稿，而是论文资格合同。D-JEPA、World4Scorer、iDriveVLA、ManeuverFlow、PlanWAM、
GeoWAM 与 Expand-and-Prune 已经占据大量邻近空间。AORTA 只有在一个更窄的因果链上赢得 matched
证据，才可能成为 ICCV 方法论文。

## 1. 一句话问题

外观变化不应改变同一物理场景的驾驶决定；但稳定 feature、geometry 或 candidate score 并不保证
native 与 challenger 的相对优劣稳定，因为决定排序的信息只存在于两条轨迹产生不同后果的位置。

## 2. 一句话方法命题

> AORTA 不学习“整幅世界是否稳定”，而学习“外观干预下，native 与 challenger 的完整时空差异
> 由哪些视觉证据支持、其 full-task margin 是多少”；证据不可识别或同时下界非正时保持 native。

英文候选表述：

> **Reliable driving under appearance shift requires invariant evidence about where candidate decisions
> differ, not merely invariant perception or a stronger generic scorer.**

### 2.1 一个容易混淆但决定架构的区分

AORTA 不是“候选无关地预测最终决策量”，也不是“为每个候选重复运行整套视觉网络”。冻结定义为：

- **candidate-independent visual encoding：**同一场景的视觉 token 只编码一次，保证共享证据与实时性；
- **candidate-conditioned differential query：**每个 native–challenger pair 用完整时空差异提出不同问题，保证决策充分性；
- **candidate-relative output：**输出 full-task margin，而不是独立绝对安全分数。

这三者缺一不可。完全候选无关的 readout 无法一般地区分候选；完全候选专属的图像编码会令计算随
候选数爆炸，也更容易把同一场景的外观噪声重复注入比较。

### 2.2 审稿人的 15 秒理解路径

1. **反常现象：**完美道路场也会因 common-min 与短前缀丢失候选排序信息；
2. **核心 insight：**外观鲁棒性的基本对象不是“稳定场景”，而是“稳定且正确的候选决策差异”；
3. **最小方法：**共享视觉编码 + 完整时空差异查询 + full-margin 输出 + 证据不足保持 native；
4. **决定性证据：**相对 matched generic scorer，在 NavDream 和三类 planner 上降低 regret、错误替换与
   hard failure，同时 clean 非劣且满足 P95 时延。

若 Figure 1、摘要第一页或 poster 顶部不能在这四步内传达论文，就说明故事仍然过度依赖模块细节。

## 3. 顶会故事的四步闭环

### 3.1 Tension：两个成熟方向之间仍缺一座桥

- appearance-robust representation / geometry 方法试图让场景表征稳定；
- decision-aligned scorer / world model 方法试图让候选排序符合 outcome。

前者没有保证稳定信息足以区分候选，后者没有专门约束 driving relation 在纯外观干预下保持正确，
也没有显式说明哪些视觉区域对 native-relative 决定具有辨识力。

### 3.2 Surprise：我们的失败不是 backbone 不够强，而是 estimand/readout 错了

现有审计给出两个直接反例：

1. B3 中 2,033 个 pair 被共同 scalar minimum 压成相同证据；perfect field 也无法恢复被 readout
   删除的候选差异；
2. BS2 中 128 个 pair 在 2.5 s 前缀 footprint 完全相同、4 s 完整轨迹不同；短前缀在信息论上无法
   区分晚期分叉。

因此，继续扩大 backbone、优化 field MAE 或换一个普通 scalar 聚合器都没有回答问题。

### 3.3 Insight：鲁棒性的最小任务对象是 appearance-conditioned decision difference

对同一物理场景 `S`、外观 `a`、native `tau_0`、challenger `tau_i` 与 previous selected trajectory
`h_(t-1)`，最终估计对象为

\[
\Delta^*_{i0}(S,h_{t-1})
=F_{\rm full}(S,\tau_i,h_{t-1})-F_{\rm full}(S,\tau_0,h_{t-1}).
\]

外观只改变观测 `X^a=g_a(S)`，不应改变 `Delta*`。网络需要同时满足：

\[
\widehat\Delta_{i0}(X^a,\tau_i,\tau_0,h_{t-1})\approx\Delta^*_{i0},
\qquad
\widehat\Delta_{i0}(X^a)\approx\widehat\Delta_{i0}(X^{a'}).
\]

第二式单独成立会产生常数坍缩；第一式单独成立会产生外观脆弱。因此 correctness 与 invariance 是
两个不可互换的约束。

### 3.4 Mechanism：一个 differential-evidence operator，而不是模块清单

完整 4 s 候选产生 soft occupancy measures `mu_i, mu_0`。其 signed contrast

\[
\nu_{i0}=\mu_i-\mu_0=\nu^+_{i0}-\nu^-_{i0}
\]

生成 challenger-exclusive、native-exclusive 与 common-context queries。所有冻结视觉 token 仍
作为全局 key/value，以允许交通灯、遮挡和道路连续性等非局部证据回答局部差异问题。共享比较器
以交换反对称化输出

\[
\widehat\Delta_{i0}
=\tfrac12\left[G(E^a,Q_{i0},h_{t-1})-G(E^a,Q_{0i},h_{t-1})\right].
\]

这不是因为 antisymmetry 本身新，而是它精确编码了部署 estimand 的方向性。最终训练目标默认使用
augmented-context full official margin；one-frame score 只作 B4R1 oracle 资格门与 `1/7` sign-certified
子域。

因此网络的计算形态是一次视觉编码、\(K\) 个轻量差异查询，而不是 \(K\) 次视觉前向。最终 profiling
必须验证候选数 \(K\) 增长时 query 部分近似线性、视觉主干成本保持常数；否则“实时通用接口”不成立。

点预测与不确定尺度分开。对未来独立 calibration groups，在固定模型与候选机制下对每组全部
appearance/anchor/challenger normalized residual 取最大值，得到 simultaneous radius `q`。只在

\[
\widehat\Delta_{i0}-q\widehat\sigma_{i0}>0
\]

时替换 native。该校准是支持机制，不是主创新；其保证严格限定于所声明的 exchangeable source
population，不外推为任意 NavDream OOD 保证。

## 4. 与 D-JEPA 的一句话差别

D-JEPA 学习候选集合中哪些 future 应被选择；AORTA 要检验的是：

> 在驾驶外观干预下，显式以 native–challenger 完整时空差异为查询，是否比通用 relational scorer
> 更能找到保持 candidate ordering 所需的视觉证据，并在该证据不足时少做有害替换。

这只是可检验差别。若容量、FLOP、token、数据与 seed 匹配的 D-JEPA-style baseline 达到相同或更好
结果，AORTA 的网络创新主张必须删除。

## 5. 为什么每个设计存在

| 设计 | 被什么强制 | 删除后的可证伪后果 |
|---|---|---|
| full 4 s support | BS2 晚期分叉反例 | 相同前缀候选不可区分 |
| signed differential query | B3 common-min 折叠 | 公共路径继续淹没分叉证据 |
| separate positive/negative branches | signed evidence early cancellation | 新增风险与放弃区域在编码前抵消 |
| common-context query | 差异后果依赖周围道路/参与者 | 只见差异位置而失去语境 |
| global visual key/value | direct projection 缺失不等于全局不可观测 | 交通灯、遮挡、连续道路证据被硬裁掉 |
| antisymmetric shared comparator | `Delta_i0=-Delta_0i` | 候选交换不能精确翻转符号 |
| task loss + appearance constraint | 正确与稳定互不推出 | 分别产生稳定错误或正确但脆弱 |
| previous-action geometry token | full target 含 two-frame comfort | 当前输入不足以一般表示目标 |
| simultaneous scale + fallback | 有限样本误差与不可识别世界 | 模型在无证据时仍强制干预 |
| one shared visual encoding | batch-1 在线预算 | 每候选重复编码导致延迟随 `K` 爆炸 |

## 6. Loss 只保留三个职责

主目标不做装饰性堆叠：

\[
\mathcal L_{\rm task}
=\operatorname{Huber}(\widehat\Delta_{i0},\Delta^*_{i0})
+\lambda_{\rm ord}\,\ell_{\rm ord}(\widehat\Delta_{i0},\Delta^*_{i0}),
\]

其中 regression 校准 margin 大小，ordering 项只强调会改变选择的符号/边界。paired appearance 约束

\[
\mathcal L_{\rm orbit}
=\rho\!\left(\widehat\Delta_{i0}(X^a)-
\widehat\Delta_{i0}(X^{a'})\right)
\]

控制同场景外观漂移；scale objective 只拟合 residual scale，并在独立 calibration 上冻结。任何
factor、cycle、smoothness、depth、TTA 或 auxiliary reconstruction loss 必须由新的可复算失败授权，
且必须有删除消融。

## 7. 30 秒、90 秒与标题级表达

### 30 秒版本

> 最近方法已经能学习候选关系和 outcome scorer，但它们没有回答纯外观变化下应该稳定什么。
> 我们发现，稳定甚至完美的道路场仍会因共同路径聚合和短时域截断而删除真正改变候选排序的
> 信息。AORTA 因此以 native–challenger 的完整时空差异提出视觉问题，直接预测 full-task relative
> margin；证据不足时保持原 planner。核心检验不是它能否拟合分数，而是这种 differential evidence
> 是否在 NavDream 与不同冻结 planner 上，比 D-JEPA/World4Scorer-style scorer 更少产生错误替换，
> 同时不伤害 clean 行为。

### 90 秒版本

1. 外观改变了 planner 的轨迹，但物理任务关系没有改变；
2. 稳定感知不是充分条件：我们的 perfect-field 与 prefix 反例精确指出信息被 readout 删除；
3. 最近 scorer 学“候选的 outcome”，我们进一步问“外观变化下，哪部分视觉证据决定候选差异”；
4. signed full-horizon support 只负责提问，全局视觉 token 负责回答，反对称 head 输出 full margin；
5. task/orbit 双约束防止稳定坍缩，previous-action token使目标函数闭合；
6. simultaneous lower bound 不为正则保持 native；
7. matched D-JEPA/World4Scorer、NavDream、三 planner、clean no-harm 与 P95 latency 决定主张成立与否。

### 候选标题

**AORTA: Appearance-Robust Trajectory Arbitration from Differential Decision Evidence**

标题中的每个词都必须由结果支持：没有 NavDream 则删除 `Appearance-Robust`；没有 matched scorer
优势则删除 `Differential Decision Evidence`；没有行为净收益则删除 `Arbitration`。

## 8. 三张主图

1. **Paradox figure：**同场景不同外观；perfect field；common-min collapse；late-fork counterexample；
   最后显出 `nu+ / nu- / common`。
2. **Operator figure：**一次视觉编码；differential queries；global KV；antisymmetric full-margin head；
   lower-bound fallback。每条箭头只标一个对应反例或理论条件。
3. **Causal evidence figure：**appearance severity / planner 为横轴，纵向依次展示 correct-image
   necessity、relation error、ranking regret、false intervention、hard-failure net rescue、clean harm、
   P50/P95/P99 latency。

Figure 1 的阅读顺序冻结为：左侧反例说明“稳定 perception 不充分”；中部只画一次共享视觉编码和
pair-conditioned differential query；右侧画 positive lower bound 才替换 native。主图禁止加入训练
细节、校准次序统计量、全部 baseline 名称或历史 road-field 分支，这些属于方法/实验图而不是核心
insight。

## 9. 摘要最多两项贡献

1. **主贡献候选：appearance-interventional differential evidence。** 一个由实证反例强制出的
   estimand 与视觉查询接口，用完整时空 candidate difference 学习 full-task native-relative margin。
2. **从属机制：evidence-qualified intervention。** 将不可识别性与有限样本 simultaneous error
   显式转成 native fallback，并报告风险—覆盖权衡。

不得把 DINO、geometry、JEPA、set attention、antisymmetry、Jordan decomposition、conformal、
previous-plan token 或 frozen planner 单列为贡献。

严格说，“共享视觉编码”与“候选条件查询”也不是两项贡献；它们共同实现同一个 differential-
evidence operator。若拆成多个 bullet，会重新退化为模块拼装叙事。

## 10. 论文资格门

论文主张按以下顺序获得资格：

1. B4R1 证明合法候选域存在非平凡 one-frame oracle；
2. augmented-context full official target 可复算且具有足够非平凡 margin；
3. correct image 显著优于 geometry-matched mismatch、removed 与 trajectory-only；
4. differential operator 在 matched D-JEPA-style、World4Scorer-style 与 ordinary scorer 上有稳定增量；
5. 未见 source groups 与 NavDream 上 relation correctness 和 appearance stability 同时成立；
6. 至少三类冻结 planner 上 OOD hard-failure 净减少且 clean non-inferiority；
7. simultaneous calibration 不是近乎全拒判，false intervention 明显下降；
8. batch-1 P50/P95/P99、峰值显存和 `K` scaling 满足冻结预算；
9. 独立机械审计、paper claim audit 与强反方 review 无致命问题。

## 11. 当前裁决

当前设计在“问题—反例—对象—架构—loss—行为”上已形成比此前更紧的理论闭环；但最新文献使
创新边界明显变窄。准确裁决是：

`REVISE / PLAUSIBLE ICCV POSTER THESIS / NOT YET ICCV-READY / NOT ORAL-READY`

今天能够严谨主张的是：研究问题清楚、失败证据真实、数学对象与网络接口一一对应、最强近邻和
决定性 falsifier 已明确。不能主张的是：differential evidence 必然优于 D-JEPA/World4Scorer、能够在
NavDream 改善驾驶、clean no-harm、跨 planner 泛化或满足实时预算。

下一科学动作仍是 B4R1；在用户明确授权正式运行前保持 HOLD。B4R1 即使通过，也只允许进入
full-target materialization 与 matched architecture preregistration，不直接宣称论文成功。
