# AORTA 研究宪法 V1

状态：`ACTIVE_RESEARCH_CONTRACT__METHOD_HYPOTHESIS_NOT_YET_VALIDATED`

本文件固定 AORTA 的长期科学目标、理论对象、证据链、成功标准与停止规则。具体网络是可替换的候选机制；项目目标不能随着某次实验结果漂移。

## 1. 不可变的研究使命

在不修改冻结端到端驾驶规划器的前提下，研究一种 source-only、外观鲁棒、任务充分、选择性干预且满足实时预算的视觉决策接口，使同一物理场景在 NavDream 等未见外观变化下产生更稳定、更优的候选选择，同时对 clean 行为不造成不可接受伤害。

最终论文必须回答同一条因果链：

\[
\text{appearance intervention}
\rightarrow
\text{task-relevant visual evidence}
\rightarrow
\text{native-relative decision relation}
\rightarrow
\text{selective intervention}
\rightarrow
\text{cross-planner behavioral gain}.
\]

任何只证明特征稳定、道路场拟合、几何覆盖或离线相关性的结果，均不是项目终点。

## 2. 五层理论建筑

### L1：外观等价类与合法干预

设潜在物理场景为 \(S\)，驾驶上下文为 \(h\)，外观算子为 \(g_a\)，观测为

\[
X^a=g_a(S), \qquad a\in\mathcal A.
\]

合法外观干预只改变成像 nuisance，不改变道路几何、交通参与者状态、驾驶意图或用于定义任务标签的物理量。若算子改变了任务状态，它不属于同一外观等价类，不能用于证明外观鲁棒性。

### L2：唯一学习对象是作用域闭合的任务关系

冻结规划器在上下文 \(h\) 中提供 native \(\tau_0\) 与合法 challengers \(\tau_i\)。最终行为目标由
完整官方 evaluator 单独定义；训练估计对象必须先限定到冻结代码逐项闭合的 contextual candidate
utility \(U^{ctx}\)，其 native-relative margin 为：

\[
\Delta^*_{i0}(S,h)=U_{\mathrm{ctx}}(S,\tau_i,h)-U_{\mathrm{ctx}}(S,\tau_0,h).
\]

当前冻结实现提出的 `7/8 one-frame + 1/8 two-frame comfort` 分解只在其代码层级、乘性门、过滤、
归一化与聚合边界逐项重放后使用；在此之前不得称为完整 EPDMS 恒等式。previous action 属于
contextual estimand 的输入，并且在部署中由 AORTA 之前的干预内生地产生。

道路场、SDF、BEV、风险热图和 one-frame score 都只能是先验、辅助头或诊断变量；除非证明它们对 \(\Delta^*\) 决策充分，否则不得替代最终对象。

### L3：决策充分的外观干预表示

当前候选机制以完整时域 occupancy measure \(\mu_i\) 构造 signed differential support：

\[
\nu_{i0}=\mu_i-\mu_0=\nu^+_{i0}-\nu^-_{i0}.
\]

它只负责提出“native 与 challenger 在哪里、何时不同”的查询；全局视觉 token 负责回答该差异对应的场景证据。关系模型满足反对称接口：

\[
\widehat\Delta_{i0}=R_\theta(Z(X),\nu_{i0},h),\qquad
R_\theta(Z,-\nu,h)=-R_\theta(Z,\nu,h).
\]

这是待证架构假设，不是预先成立的创新。D-JEPA-style set relation、World4Scorer-style candidate state、普通 matched set scorer 和 independent scorer 是强制对照。如果差异支持不能在相同数据、参数、FLOP、token、优化器与 seed 预算下产生稳定增益，删除该网络创新主张。

这里必须区分三个层次：视觉编码对候选无关并在同一场景只执行一次；差异查询由候选 pair 条件化；最终输出是 candidate-relative margin。若连 readout 也候选无关，则不能一般地区分决策；若为每个候选重复视觉编码，则破坏共享证据解释和实时预算。

信息论只承担一个明确的设计职责：寻找对任务关系充分、对合法外观 nuisance 尽可能不敏感的表示 \(R\)。理想条件写为

\[
I(\Delta^*;X\mid R,\tau_{0:K},h)=0,
\qquad
I(R;A\mid S,\tau_{0:K},h)=0.
\]

第一式是决策充分性，第二式是条件外观不变性。由于潜变量 \(S\) 不可直接观测，实验不能声称已经估计这两个互信息；只能用 full-target prediction、correct/mismatch/removed、paired appearance orbit、NavDream regret 与反例共同检验其可观测后果。

### L4：误差—后悔—干预桥

若同一合法候选域上所有 margin 满足同时误差界

\[
\max_i\left|\widehat\Delta_{i0}-\Delta^*_{i0}\right|\le \varepsilon,
\]

则基于估计 margin 选择的候选相对最优候选的任务后悔满足条件界

\[
\operatorname{Regret}\le 2\varepsilon.
\]

若同时下置信界

\[
L_{i0}=\widehat\Delta_{i0}-r_{i0}
\]

在同一覆盖事件上满足 \(L_{i0}>0\)，则该次 challenger 相对 native 的真实任务 margin 为正。否则保留 native。该结论只在明确的覆盖、候选机制、分组交换性和目标定义下成立，不外推为任意 OOD 或真实道路安全保证。

### L5：跨规划器行为与实时闭环

最终方法必须在至少三类冻结规划器上使用同一方法原则：

1. 回归式 planner；
2. diffusion / generative planner；
3. generate-and-select / scoring planner。

方法必须在 batch-1 端到端部署路径中报告 P50/P95 latency、显存、参数量、候选数敏感性和最坏拒判率。离线缓存吞吐不能替代在线时延。

## 3. 三个条件命题与边界

完整定义、证明、最小反例及“标准理论/项目贡献”边界见 refine-logs/2026-10-25_aorta_post_b3/FORMAL_THEORY_APPENDIX_V1_CN.md。本节只保留项目级摘要。

### 命题 A：任务关系的外观不变性

若 \(g_a\) 保持潜在任务状态 \((S,h)\)，候选域相同，且 \(F_{\mathrm{full}}\) 只依赖该任务状态，则

\[
\Delta^*_{i0}(g_a(S),h)=\Delta^*_{i0}(S,h).
\]

这定义了应学习的不变量，但不保证视觉网络能从像素识别它。正确图像相对 mismatch/removed 的优势、任务误差与非坍缩测试负责验证可识别性。

### 命题 B：关系误差控制选择后悔

在候选集合、目标函数和上下文一致时，同时 \(\ell_\infty\) margin 误差至多 \(\varepsilon\) 推出选择后悔至多 \(2\varepsilon\)。因此主评估必须包含 relation error、ranking regret、top-choice error 和 false intervention，而不能只报告 field MAE 或 feature distance。

### 命题 C：选择性仲裁的条件真实性

在 simultaneous coverage event 上，只有 lower bound 严格为正的 challenger 才获授权，因此每次已授权替换相对 native 改善。覆盖事件之外的行为是拒判或经验风险，不是理论保证。NavDream 用于经验压力测试和失覆盖诊断，不能被 source conformal 定理自动覆盖。

## 4. 网络与损失设计责任表

| 组件 | 已观察问题或官方依赖 | 数学责任 | 必须消融 |
|---|---|---|---|
| 冻结视觉 backbone | 外观变化引起 planner 行为漂移 | 提供可迁移视觉证据，不单独算创新 | backbone/feature controls |
| 完整时域 signed support query | prefix 丢失晚分叉；global min 被共同区域支配 | 定位候选差异而非共同路径 | 无 support、prefix support、unsigned/path-only |
| 全局视觉 key/value | direct-projective evidence 局部缺失不等于全局不可观测 | 允许上下文补全但不得凭空外推不可见区域 | local-only、pooled、mismatch/removed |
| 反对称 relation head | 最终对象是 pairwise native-relative margin | 固化换序符号与零自比较 | 普通 set scorer、independent scalar scorer |
| previous-action context | full target 含 two-frame comfort | 使输入覆盖 estimand 自变量 | minimal token、history removed、复杂 temporal arm |
| task loss | 关系必须保留完整官方任务排序 | 直接控制 margin/regret，而非代理场误差 | one-frame、factorized、field-only loss |
| appearance/orbit constraint | 同场景合法外观应共享任务关系 | 降低 nuisance sensitivity，同时防止常数坍缩 | 无约束、过强约束、correct/mismatch/removed |
| uncertainty + native fallback | OOD 证据可能不足 | 将未知转化为拒判并限制错误替换 | point estimate、独立非同时区间、always-intervene |

任何组件如果没有“失败证据—理论责任—可区分预测—删除实验—运行代价”五项记录，不进入主方法。

### 4.1 核心训练目标

在合法样本与候选 mask 上，主目标预定义为

\[
\mathcal L
=\mathcal L_{\mathrm{task}}
+\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{rank}}
+\lambda_{\mathrm{orbit}}\mathcal L_{\mathrm{orbit}}
+\lambda_{\mathrm{unc}}\mathcal L_{\mathrm{unc}}.
\]

- \(\mathcal L_{\mathrm{task}}\)：对 augmented-context full margin 的稳健回归，保证表示保留任务量纲；
- \(\mathcal L_{\mathrm{rank}}\)：在预先定义的非 tie pairs 上约束符号/列表排序，直接对应 selection error；
- \(\mathcal L_{\mathrm{orbit}}\)：同一场景合法 appearance orbit 上的 relation 一致性，而不是无条件 feature collapse；
- \(\mathcal L_{\mathrm{unc}}\)：proper likelihood 或 interval score，使预测半径可由独立 calibration 资格化。

反对称性优先通过参数化精确满足，而不是依赖软惩罚。\(\lambda\) 只在 source train/development 内按预注册规则选择；必须分别比较 task-only、task+orbit、task+uncertainty 与完整目标。若 orbit 降低漂移却损害 full-target regret，它不能因“更稳定”而保留。

## 5. 最终验收指标

### 5.1 视觉与关系层

- 未见 recording groups 上 correct image 优于 mismatch、removed 与 source-only prior；
- NavDream 各 appearance 上 relation error、pair-sign、top-choice 与 regret 不出现系统性崩溃；
- 稳定性与任务正确性同时成立，常数场、数据集空间先验和场景记忆被排除；
- differential-support 模型优于容量与计算匹配的最近 relation/scorer 对照，否则删除主架构创新。

### 5.2 行为层

在预冻结 baseline-hard 集合上报告净困难案例改善：

\[
G_{\mathrm{net}}
=
\frac{N_{\mathrm{rescued}}-N_{\mathrm{new\mbox{-}hard}}}
{N_{\mathrm{baseline\mbox{-}hard}}}.
\]

必须同时报告总体 score、组级配对差异、false intervention、coverage/abstention、candidate supply 与 selection loss。只展示 rescued cases 或删除零 challenger/失败组无效。

最终成功要求 \(G_{\mathrm{net}}\) 的预注册组级置信下界高于零或超过预先定义的最小有意义效应，而不是只要求点估计为正。

### 5.3 clean no-harm

clean 使用预注册的非劣效边界；边界必须由独立 baseline 方差、实际任务量纲和统计功效决定，不能在看结果后取值。均值不下降不足以代替 group-level non-inferiority 和 hard-case harm 统计。

### 5.4 实时性

报告同一硬件、batch-1、完整在线链路的 P50/P95、峰值显存与候选数扩展曲线。若无法满足预注册部署预算，则只能作为离线分析方法，不能宣称实时驾驶接口。

### 5.5 完整性

训练、development、calibration、test、NavDream/target 角色分离；所有阈值和主指标在开封前冻结；结果由机械重算与独立科学审查区分处理；历史失败不得覆盖或改名为成功。

## 6. 三档成功定义

| 层级 | 必须满足 | 允许主张 |
|---|---|---|
| 基础成功 | 合法 full target、正确图像必要性、held-out relation 泛化、实现/审计闭合 | 建立了可信的视觉—任务关系机制 |
| 方法成功 | matched closest baselines 上 relation/regret/false intervention 有稳定增益；clean 非劣；实时可行 | 提出有效的选择性候选仲裁方法 |
| 顶会目标 | NavDream + 三类 planner 上行为增益、\(G_{\mathrm{net}}\) 的预注册置信下界为正或超过最小有意义效应、关键消融支持因果链、限制诚实且故事单一 | ICCV/CVPR 级方法论文候选 |

“达到顶会标准”不是先验保证，而是第三档全部证据通过后的裁决。

## 7. 论文的一句话故事

> 稳定视觉特征并不自动产生稳定驾驶决策；AORTA 学习外观干预下、由完整时域 native–challenger 差异支持所查询的任务关系，并只在同时证据足以证明相对改善时替换冻结规划器的原决策。

口头报告按五步展开：现象反例 → 决策对象 → 差异证据架构 → 条件性仲裁 → NavDream/跨规划器行为证据。不能用模块清单代替故事。

## 8. 明确的 non-goals 与禁用表述

- 不从 NAVSIM 离线结果宣称真实道路安全；
- 不从 source calibration 宣称任意 OOD coverage；
- 不把 DINO、SDF、BEV、cross-attention、conformal 或完整时域轨迹本身单独列为创新；
- 不把 candidate supply 的改善归因于 scorer；
- 不把 development 当作最终未触碰测试集；
- 不为通过门槛事后更换候选、阈值、appearance、统计单位或子集；
- 不因当前候选架构失败而降低长期任务目标。

## 9. 研究治理与当前门

最终 claim-driven 路线与锁定状态由 refine-logs/2026-10-25_aorta_final_validation/EXPERIMENT_PLAN.md 和 EXPERIMENT_TRACKER.md 管理。

1. 每轮先写可证伪命题、竞争解释和停止规则，再读取科学结果。
2. 失败先区分实现错误、协议错误、统计功效不足与理论对象错误；只在证据指向后两者时改方法。
3. 新复杂模块必须由新反例、官方目标依赖或 matched baseline 缺口授权。
4. 当前最近门仍是 B4R1：只验证合法候选域上的 one-frame 接口与 oracle ceiling；它不训练网络、
   不证明 contextual target、不证明 NavDream 或驾驶收益。
5. B4R1 需要用户明确授权才可正式运行；接口有效后才依次进入 evaluator-scope audit/contextual
   target、matched scorer、NavDream、actual-history/sequential evaluation、三 planner、clean
   no-harm 与 latency。其 one-frame 科学 GO/HOLD/STOP 不替代 contextual oracle 裁决。
6. D-JEPA-style 与 World4Scorer-style 是强制最近基线；FlowDrive、READ、PEA/MG-style controls 按传感器、数据、backbone 与适配权限对齐后使用，不以不公平 headline 数字比较。

## 10. 当前裁决

- 长期问题：重要且可形成顶会级研究问题；
- 数学对象与条件链：已形成可证伪合同；
- 当前差异支持架构：有反例驱动的动机，但新颖性与有效性均未证明；
- 视觉旧场机制：held-out development 有正证据，但不能回填新 relation head；
- 行为、NavDream、跨 planner、clean no-harm、实时性：均未验证；
- 投稿资格：`NOT_ICCV_READY / NOT_ORAL_READY`；
- 下一步：保持 B4R1 formal HOLD，先等待明确运行授权，不用新模块或新公式绕过该证据门。
