# AORTA 理论与系统再核验 V1

日期：2026-10-25  
状态：`THEORY_COHERENT__R005E2_IS_A_VALID_FALSIFICATION_GATE__POSITIVE_RESULT_NOT_GUARANTEED`

## 0. 结论先行

AORTA 当前已经形成一条作用域明确、可以被实验推翻的理论—网络—决策链，足以在资格构建、完整
replay 和机械审计通过后执行 R005E2。这里的“足以训练”只表示：输入、目标、函数类、对照、损失、
统计单位和停止规则对准同一个科学问题；不表示理论能够预先保证 correct-visual arm 一定优于控制。

训练前能够保证的是：

1. 若实验实现失败，结果归类为 `INVALID`，不进入科学解释；
2. 若视觉只改善平均拟合而不改善选择，结果归类为 `MIXED`，不进入主架构；
3. 若正确视觉没有稳定胜过 removed 与 mismatch，当前视觉主张停止；
4. 只有正确视觉同时降低 proper risk、decision regret，并通过场景绑定干预和最小效应门，才解锁
   Differential Query 的公平结构实验。

因此，“结果符合预期”的严谨含义是任何结果都能映射到预注册裁决，而不是只接受正结果。

## 1. 整条研究链的复盘

### 1.1 失败如何收敛成当前问题

1. 旧道路场在受控外观上显示过非平凡视觉任务信息，但不能直接推出候选选择或驾驶收益；
2. scalar set-min 被共同路径支配，短前缀又会丢失晚期分叉，因此旧 road-margin reranker 退役；
3. B4R1 证明合法候选 panel 在 raw one-frame 分数上存在 oracle 空间，但 E0 随后证明 raw、filtered、
   contextual 与 two-stage value 不是同一对象；
4. R005E1B 最终在 42 个 source-fit groups 上资格化了 scoped、native-history、teacher-forced 的
   `U_ctx`，并证明 one-frame 忽略上下文会带来显著 top-set、pair-sign 与 regret 差异；
5. 官方 two-stage mapping 在当前 source panel 上覆盖为 0/672，因此 `Q_pi_ref` fail-closed，禁止用
   `U_ctx` 冒充完整 pseudo-closed-loop value；
6. R005E2 不提前验证主创新，而先回答更基础的问题：在轨迹、合法 mask 和真实因果历史已知后，
   正确场景视觉是否仍提供可学习的条件增量？

这条顺序是合理的。若第 6 步失败，直接训练复杂 Differential Query 只会把“视觉没有增量”、
“函数类不适合”和“结构归纳偏置无效”混在一起。

### 1.2 当前唯一合法的分层对象

必须始终区分：

\[
F_{raw}^{1f}\neq \widetilde F^{1f}\neq U_{ctx}
\neq Q_{\pi_{ref}}\neq J(\pi).
\]

R005E2 的监督仅为

\[
Y_i=U_{ctx}(\tau_i;h^-)-U_{ctx}(\tau_0;h^-),\qquad Y_0=0,
\]

其中 (h^-) 是上一周期 native 计划经官方 simulator 得到的 41 点状态轨迹，删除已过期的前 5 点，
变换到当前 ego rear-axle 坐标系后形成的 (36\times3) 因果重叠轨迹。该定义与 two-frame comfort
实际比较的时间索引一致；禁止把 9 点 planner 输出插值成 41 点，禁止分别把 36/41 点时间轴拉伸
到 `[0,1]`。

## 2. R005E2 的数学地基

### 2.1 带 ragged mask 的视觉条件价值定理

令 (B=(C,H,M)) 包含候选轨迹、因果历史和合法 mask，(Z) 为冻结视觉表示。对有 challenger 的
anchor 定义 (w_i(M)=\mathbf1\{i>0,M_i=1\}/n_M)，无 challenger 时全部权重为零。实际 proper
risk 为

\[
\mathcal R(f;X)=
\mathbb E\!\left[\sum_iw_i(M)(Y_i-f_i(X))^2\right].
\]

因为 (w_i(M)) 对 (B) 可测，平方损失的 Bayes 解为相应条件均值。记

\[
\mu_0=\mathbb E[Y\mid B],\qquad
\mu_1=\mathbb E[Y\mid B,Z],
\]

则由条件期望的正交投影恒等式：

\[
\boxed{
\mathcal R^*(B)-\mathcal R^*(B,Z)
=\mathbb E\!\left[\sum_iw_i(M)(\mu_{1,i}-\mu_{0,i})^2\right]\ge0.}
\]

证明只需把 (Y-\mu_0=(Y-\mu_1)+(\mu_1-\mu_0)) 展开；交叉项在给定 ((B,Z)) 后期望为零。

该式精确说明了 R005E2 检验什么：视觉是否改变给定候选与历史后的**条件均值**。它不等同于检验
所有形式的条件互信息；视觉若只改变条件方差而不改变均值，平方损失 Bayes risk 可以不下降。

### 2.2 从总体定理到有限网络仍存在三项差额

对 arm (a\in\{V0,V1,V2\})，有限训练器的风险可写为

\[
\mathcal R(\widehat f_a)
=\mathcal R_a^*+
\mathcal E_{app,a}+
\mathcal E_{est,a}+
\mathcal E_{opt,a},
\]

分别对应 Bayes risk、函数类逼近误差、有限样本估计误差和优化误差。因此

\[
\mathcal R(\widehat f_{V0})-\mathcal R(\widehat f_{V2})
=\underbrace{\mathcal R^*_{V0}-\mathcal R^*_{V2}}_{\text{视觉条件增量}}
+\underbrace{\Delta\mathcal E_{app}+\Delta\mathcal E_{est}+\Delta\mathcal E_{opt}}_{\text{有限训练器差额}}.
\]

参数、FLOP、token、初始化、样本顺序和训练预算匹配，只能尽量控制后三项，不能从数学上把它们
设为零。这是无法在训练前保证正结果的根本原因。

### 2.3 平均预测误差不能替代决策有效性

若所有合法候选同时满足

\[
\|\widehat Y-Y\|_\infty\le\varepsilon,
\]

按 \(\widehat Y\) 选择候选的真实 regret 至多 \(2\varepsilon\)。但 MSE 变小不推出
\(\ell_\infty\) 误差变小，也不保证 near-tie 候选的排序正确。因此 R005E2 同时要求：

- V2 相对 V0/V1 的 group-level proper risk CI 上界小于 0；
- V2 相对 V0/V1 的 decision regret CI 上界小于 0；
- 同一 V2 checkpoint 被 mismatch/removed 干预后的 regret CI 下界大于 0。

这组 intersection-union 门比只看 loss 或 AUROC 更贴近最终决策命题。

### 2.4 `rho_vis` 的量纲澄清

正式审计实现中的机会恢复率是

\[
\boxed{
\rho_{vis}=
\frac{\bar r_{V0}-\bar r_{V2}}
{G^{ctx}_{oracle}}},
\qquad G^{ctx}_{oracle}=0.0166503660.}
\]

分子与分母均为 `U_ctx` utility/regret 单位，因此无量纲。预注册正文原先用未定义的 (R) 容易被
误读为 relation MSE；实际代码、审计器和冻结裁决使用的是 decision regret，不是 MSE。10% 是
训练前冻结的实践效应阈值，不是理论常数。

## 3. 为什么当前 probe 必须这样设计

| 结构 | 必要原因 | 它解决什么 | 它不证明什么 |
|---|---|---|---|
| 冻结 DINO | 隔离视觉信息可读性，避免 backbone 适配混淆 | 固定视觉信息源与计算成本 | 不保证 `U_ctx` 信息完整 |
| 41 点 current trajectory | 完整覆盖当前 0--4 s proposal | 保留 late-fork | 不提供环境状态 |
| 36 点因果 previous overlap | 与 `U_ctx` two-frame comfort 的自变量一致 | 修复历史语义与坐标 | 不证明复杂 history encoder 必要 |
| 共享 trajectory encoder | 控制参数并让 current/history 同尺度 | 学习时序几何摘要 | 不是主创新 |
| `[candidate, history, difference]` fusion | 显式提供相对历史关系 | 降低由网络自行做差的负担 | 不保证所有 contextual interaction 可辨识 |
| permutation-equivariant set block | 候选顺序不是物理变量 | 建模 panel interaction、避免顺序捷径 | 不构成新颖性 |
| candidate-to-global visual attention | 检验候选条件下能否读出视觉增量 | 比 pooled feature 更强的通用 probe | 不是 Differential Query |
| scalar score 后减 native、`tanh` | native 精确为零、范围与 label 一致、固定 panel 内排序可积 | 提供合法 native-star margin | reroot 后的函数级交换反对称并未由本 probe 保证 |
| squared margin loss | 对应条件均值 Bayes 恒等式，是 proper loss | 让视觉增量有清楚统计解释 | 不直接优化 regret |
| V0/V1/V2 同架构重训 | 控制容量与计算 | 区分无场景视觉、错配视觉和正确视觉 | V1 不是精确条件随机化 |

R005E2 的网络是一个**科学探针**。即使通过，它也只证明普通 full-context scorer 能利用正确视觉；
主方法的 signed differential-support query 仍必须在后续参数、FLOP、token、数据和 seed 匹配实验
中单独胜过这个 probe。

## 4. 三种视觉控制能排除什么

### V0 removed

outer-train 内按 camera/history slot 求均值并复制到相同 token 网格。它保留计算图、slot 和空间位置
编码，但删除 anchor-specific 视觉内容。因此 V2 胜 V0 支持“场景视觉有增量”，而不是只胜过零向量。

### V1 geometry-matched mismatch

它保持训练视觉边际分布，并用不含 outcome 的轨迹描述子跨组匹配，破坏正确场景绑定。它是较强的
scene-binding negative control，但不等于从精确 (P(Z\mid B)) 抽样：道路布局、动态体、信号状态和
记录域仍可能失配。故 V2 胜 V1 必须与 V2 胜 V0、within-checkpoint 干预、匹配距离和后续多-donor
敏感性共同解释，不能单独称为因果识别。

### V2-M / V2-R

同一 V2 checkpoint 的输入干预检验网络是否实际使用视觉。性能下降可能同时包含“删除正确信息”和
“制造输入分布变化”，所以它们只作依赖性三角验证，不替代公平重训控制。

## 5. 训练前可证与不可证账本

| 命题 | 当前状态 | 依据/缺口 |
|---|---|---|
| `U_ctx` target 有合法候选 headroom | 支持 | 42/42 groups；oracle gain CI 下界为正 |
| previous context 是 target 自变量 | 支持 | 37.50% top-set、12.48% pair-sign 改变 |
| 历史张量与 evaluator 时间语义一致 | 实现已修复，待 replay/audit 完成 | 41 状态去前 5 点，36 点 current-frame overlap |
| scorer 存在视觉梯度与学习通路 | 支持为实现 sanity | 合成视觉关系 MSE `0.1160 -> 0.0051`；移除视觉后 `0.6592` |
| 冻结 DINO 含 `U_ctx` 条件均值信息 | 未知 | 这正是 R005E2 科学问题 |
| 42 groups 足以稳定检测最小效应 | 未完全保证 | 只能暴露 group CI 和 seed 波动；小效应可能功效不足 |
| V1 是严格条件随机化 | 否 | 仅为 geometry-matched scene-binding control |
| 通用 scorer 足以逼近 Bayes 条件均值 | 未知 | 有限函数类/优化/样本误差仍存在 |
| Differential Query 优于通用 scorer | 未知 | R005E2 明确禁止该结构 |
| 外观鲁棒、NavDream、行为收益、安全 | 未知 | 均属于后续独立门 |

合成 sanity 不读取任何真实 endpoint，也不构成科学证据；它只排除了“视觉分支完全断路或模型无法
表达最简单视觉—候选交互”这一实现解释。

## 6. 预期结果与唯一分流

| 观察结果 | 最强合法解释 | 后续动作 |
|---|---|---|
| V2 的 risk、regret、两类 control 和 `rho_vis` 全通过 | 当前函数类可恢复 source-panel 的正确视觉条件增量 | GO 到 generic-vs-differential 预注册 |
| risk 通过、regret 不通过 | 视觉改善平均值，但未改善关键排序边界 | MIXED；分析 near-tie，不自动加 loss |
| V2 胜 V0、不胜 V1 | 视觉内容有用，但正确场景绑定未被证明 | HOLD；审查 donor 与场景特异性 |
| V2 胜 V1、不胜 V0 | mismatch 有害，但正确视觉未胜稳定先验 | STOP/HOLD 当前视觉增量主张 |
| 重训 V2 好，但 V2-M/V2-R 不变 | 网络可能没有实际使用视觉，或控制干预无效 | STOP；先查梯度/输入捷径 |
| seed 波动大、CI 跨零 | 估计/优化不稳定或功效不足 | 不宣称无信息；也不解锁主架构 |
| clean 通过、受控 appearance 失败 | 信息可恢复但外观脆弱 | 允许研究鲁棒机制，不宣称已鲁棒 |
| 任一 provenance/replay/hash/finite 门失败 | 实验无效 | INVALID；禁止读科学结论 |

## 7. 对“基本保证结果符合预期”的严格回答

在未知数据分布上，不存在不偷看结果又保证 V2 正向提升的理论。若提前把“必须提升”写成结论，实验
就失去可证伪性。现在能够最大化成功概率且不破坏科学性的措施已经是：

1. 目标先由官方 evaluator 语义和完整 replay 固定；
2. 输入历史按 target 的真实条件变量重建，而非便捷插值；
3. 先用 generic probe 测试视觉信息存在性，不把主架构优劣混入；
4. 三 arm 参数/FLOP/token/初始化/顺序一致；
5. 6-fold group cross-fitting 阻止同组记忆直接进入 OOF 结果；
6. synthetic learnability 排除了最基础的视觉断路；
7. proper risk 与 regret 联合门阻止“loss 涨点但决策无用”；
8. correct/mismatch/removed 三角控制阻止把稳定先验冒充视觉理解；
9. primary/replay/audit 和封存角色阻止事后调阈值；
10. GO/MIXED/STOP/INVALID 为所有结果预先定义了科学意义。

仍不能消除的风险是：真实视觉增量很小、冻结 DINO 丢失关键动态信息、标签包含不可观测未来、
42 groups 对小效应功效不足、或通用 probe 的有限优化没有达到可用解。

## 8. 对 AORTA 主方法理论的最终核验

若 R005E2 通过，后续主方法仍应保持一主一从：

1. **主贡献候选：appearance-interventional differential decision evidence。** 用完整时域 signed
   occupancy difference \(\nu_{i0}=\mu_i-\mu_0\) 生成 query，但保留全局视觉 token 回答
   非加性、非局部结果；该结构必须胜过当前通用 scorer 才成立。
2. **从属机制候选：evidence-qualified native-preserving arbitration。** 独立 group calibration
   后，只在同时 lower bound 为正时干预；它提供条件可靠性，不是任意 OOD 安全保证。

未来核心公式的逻辑责任分别是：

\[
\underbrace{\nu_{i0}=\nu^+_{i0}-\nu^-_{i0}}_{\text{在哪里比较}}
\quad\longrightarrow\quad
\underbrace{\operatorname{Attn}(Q(\nu),K(Z),V(Z))}_{\text{读取什么视觉证据}}
\quad\longrightarrow\quad
\underbrace{\widehat\delta_{i0}}_{\text{预测任务关系}}
\quad\longrightarrow\quad
\underbrace{\widehat\delta_{i0}-r_i>0}_{\text{何时允许干预}}.
\]

Jordan 分解、cross-attention、native-star 和 conformal 本身都不是创新。创新只可能来自这一接口在
外观干预、late-fork、common-path 与 frozen-planner arbitration 上相对最近 matched scorer 的稳定
增量。若该增量不存在，必须删除“差异证据网络创新”，不能靠增加数学名词挽救。

## 9. 审稿人式最终裁决

- **理论自洽性：通过。** 目标、条件变量、网络输入、proper loss、regret 和行为边界已经对齐；
- **训练资格：条件 GO。** 仅待当前 scientific bundle replay 与机械 audit 完成；GPU 训练仍需资源空闲；
- **正结果保证：不存在，也不应伪造。** 当前协议能保证的是结果可解释，不是结果必为正；
- **算法创新：尚未验证。** R005E2 只检验视觉条件增量，Differential Query 仍处于可证伪假设；
- **ICCV 竞争力：有清晰入口，未达到完成态。** 必须依次通过视觉增量、matched 结构增量、受控与
  NavDream 外观、sequential history、三 planner 行为、clean no-harm 和 batch-1 时延。

当前最优研究动作不是再加理论模块，而是完成 replay/audit 后执行冻结 R005E2；它是决定后续所有
复杂网络是否有科学许可的最高信息价值实验。

## 10. 两轮反方审查记录

### Round 1：最强拒稿意见

1. **目标过窄。** `U_ctx` 只是 row-level teacher-forced utility，不是完整 benchmark policy value；
2. **数学结论过弱。** Bayes risk 恒等式属于标准平方损失理论，不能构成算法创新；
3. **视觉对照不具有严格因果识别。** mismatch donor 不是精确条件随机化；
4. **网络并不新。** set transformer、cross-attention、score subtraction 都已有大量先例；
5. **有限样本结论脆弱。** 42 groups、7-group inner validation 和重叠 cross-fit 训练集可能使 CI 与
   early stopping 不稳定；
6. **离线关系不等于驾驶改善。** teacher-forced history、固定候选供给和 source panel 不能替代
   sequential、跨 planner 与 NavDream 行为验证。

### 对 Round 1 的回应与修订

- 接受目标作用域批评：全文将 R005E2 限定为 `U_ctx` conditional-mean recoverability；
- 接受理论原创性边界：Bayes 恒等式只负责给 probe 合法性，创新候选留给 differential evidence；
- 将 V1 明确降格为 negative control，并以 V0、within-checkpoint intervention 和 donor sensitivity
  三角验证；
- 将当前网络明确称为 strongest generic probe/baseline，不列为贡献；
- CI 明确为冻结 cross-fit 流程条件下的 42-group 描述，不冒充包含全部训练重抽样的不确定性；
- 把 sequential history、三 planner、clean no-harm、NavDream 和 batch-1 latency 保留为不可跳过的
  后续门。

### Round 2：收敛后的剩余致命问题

修订后没有发现阻止 R005E2 运行的数学矛盾，但仍有三个无法靠纸面消除的致命经验问题：

1. (Z\) 是否真的改变 (\mathbb E[Y\mid B,Z]\)；
2. 若改变，当前冻结 DINO 与 generic scorer 是否能在 42-group 样本量下稳定恢复；
3. 若 generic probe 通过，signed differential support 是否能在 matched 条件下提供额外增量。

三个问题分别由 R005E2、其 seed/group 诊断和后续 generic-vs-differential 实验裁决。任何一项失败都
已有停止路径，因此审查结论为：**理论允许实验，实验决定主张；不再用新增公式替代证据。**

### 优先级 TODO

1. `P0`：完成当前 bundle replay、exact audit 与 provenance 核验；
2. `P0`：GPU 空闲后执行冻结 54 primary + 54 replay，不读取部分科学指标；
3. `P0`：只按预注册审计器发布 GO/MIXED/STOP/INVALID；
4. `P1`：若 GO，先冻结 generic-vs-differential 的 matched 协议，不直接扩展网络；
5. `P1`：若 MIXED/STOP，先按 approximation/estimation/optimization/absent-signal 四类归因；
6. `P2`：只有结构增量通过后才进入 appearance、sequential、跨 planner、calibration 与 latency。
