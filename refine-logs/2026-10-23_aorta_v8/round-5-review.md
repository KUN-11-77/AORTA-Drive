# Round 5 Review

<details><summary>Raw reviewer response</summary>

# AORTA-v8 最终轮审稿：V0 Freeze Candidate

完整读取了 `round-4-refinement.md`。本轮只核查上一轮剩余修复，没有运行实验或修改方法文件。既有完整理论与证明沿用 v8.3 及前轮记录，本轮冻结合同不应被误读为已通过 Gate A/B。

## 1. 最终裁定

**V0 理论规范是否可冻结：YES。** 该版本已形成一个足够清楚、条件闭合的最小系统规范，可以冻结核心机制并进入 experiment-plan 与 Gate A/B。无需再增加新模型、图算法或不确定性组件。

**顶会方法主张是否可授权：NO。** 当前仍不能授权“顶会原创”“首次”“优于 CDF/DriveReferee/scalar referee”，也不能授权已校准、有效或实时的系统结论。

**最终 Verdict：REVISE。** 这是 research-refine 顶会方法门槛下的 verdict；不是要求继续改造已可冻结的 V0 核心。

**Drift Warning：NONE。** 原问题保持为冻结 planner 的 appearance-driven nominal-road arbitration，范围没有扩展到真实执行、碰撞、交通规则或任意 OOD 安全。

## 2. 七项评分

| 项目 | 权重 | 分数 | 加权贡献 |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9/10 | 1.35 |
| Method Specificity | 25% | 9/10 | 2.25 |
| Contribution Quality | 25% | 6/10 | 1.50 |
| Frontier Leverage | 15% | 7/10 | 1.05 |
| Feasibility | 10% | 7/10 | 0.70 |
| Validation Focus | 5% | 9/10 | 0.45 |
| Venue Readiness | 5% | 6/10 | 0.30 |
| **OVERALL SCORE** | **100%** | **7.60/10** | **7.60** |

Problem Fidelity 高分来自任务与风险范围保持明确。Method Specificity 高分来自 geometry、calibration、solver 和 selection 合同已闭合。Validation Focus 的提升来自两层公平对照和数值优化方向的补齐。Contribution Quality 与 Venue Readiness 没有因措辞更严谨而自动提高：本轮没有增加原创性证据，也没有改变标准工具归约的事实。

## 3. Geometry proof-carrying contract 与 alpha/beta 风险

### 3.1 核心几何合同可接受

`valid=true` 的条件现在包含 native 与 challenger 的内部采样方向、challenger 的内部 h-net、完整 ROI、非空、协议一致性和 GT signed-distance 的 Lipschitz 性。由这些条件得到

\[
\Delta_{j0}^{cont}\ge L_G(A_j,A_0)-h_j
\]

是正确的。解析/保守半径与有限测试点经验最大值已被明确区分，上一轮的关键缺口已修复。

`valid` 和 `protocol_hash` 是接口输出，不是其自身的 soundness 证明。V0 冻结的是“通过检查必定意味着这些条件成立”的规范；实际 checker 是否满足它，要在实现及 Gate A 中证明或验证其构造。把 \(\beta\) 设为零不是获得确定性 soundness 的替代方式。

这一点不阻止冻结理论规范，只阻止现在就宣称已得到一个确定性可靠的几何实现。

### 3.2 alpha/beta 的事件包含关系正确

定义 \(\mathcal F\) 为该 session 所有声明时刻、所有边的 field coverage，\(\mathcal G\) 为该 session 所有实际授权 pair 的几何界真实有效。则

\[
H\subseteq\mathcal F^c\cup\mathcal G^c
\]

正确。

若 nominal geometry checker 对所有通过的 pair 都具有确定性 soundness，\(P(\mathcal G^c)=0\)，所以原 \(\alpha\) 界可保留。几何资格被拒绝不会产生干预，也不需要追加风险预算。

若未来采用概率几何 checker，\(\alpha+\beta\) 上界也正确，但必须将

\[
\beta\ge P(\mathcal G^c)
\]

定义为**整个 session 至少一次错误几何接受的概率上界**。它不能只是单 pair 或单调用的错误率。多次、自适应调用时，需要有效的 session 聚合保证或相应联合上界，才能继续使用同一个 \(\beta\)。V0 的确定性版本没有这一累计问题。

禁止通过条件于 \(\mathcal G\) 重解释 conformal coverage 的文字也正确。

## 4. H/I/T 量词已统一

\(I_t\)、\(I(Z)=\max_{t\in T(Z)}I_t\) 与 H 中的 existential time 均使用同一个、calibration 前冻结的有限 \(T(Z)\)。授权同时包含动力学/意图资格、有效 geometry certificate 和下界阈值。

因此，在同一个 field event 与 sound geometry 协议上，任意自适应选择的候选均满足对应 nominal continuous margin 不下降，从而 H 为零。候选数、最终选择和新 frozen planner 的查询身份不需要另加多重比较修正；source observation 分布和既定图/时间协议仍然是保证前提。

条件于发生干预的 \(\min\{1,(\alpha+\beta)/P(I=1)\}\) 上界也与这个 session 事件一致。它不是每次干预风险，也不是固定 calibration realization 的条件风险。当前稿件没有混淆这些层级。

这一部分无剩余实质性数学问题。

## 5. Super-source 标签恢复已修复

当前明确写出

\[
c=\min_Bu,\quad r_b=u_b-c,
\quad\widetilde\psi_B=\psi_B-c,
\quad\psi_B=\widetilde\psi_B+c.
\]

恢复方向正确，也保留了非负 super-source 边和一次多源求解。Gate A 已包含非零标签、共同 shift、相等/包含集合及零权边，足以使这一易错点成为明确验证对象。

数值合同中“误差计入 \(\gamma\)”应按以下含义落实：如果最终计算证书可能比数学证书乐观至多 \(\varepsilon_{num}\)，则用于授权的总阈值至少为

\[
\gamma_{total}\ge\gamma_{task}+\varepsilon_{num}.
\]

这里的上界必须覆盖最终 bound 的相关数值误差，而不仅是挑出的几个测试误差。当前合同已经要求误差上界和事前冻结，因此属于实施事项，不构成新的核心机制修复。

## 6. 两层 baseline 公平性已实质修复

Layer 1 从合法 absolute-trained predictor 出发，冻结同一输出后只改变 uncertainty geometry。这消除了让 absolute band 人为承担任意 gauge 的不公平性，也能清楚隔离局部图与全域集合的差异。

Layer 2 允许不同方法使用各自合法训练目标，同时匹配 backbone、field architecture、source data、参数预算和选择规则。这可以检查完整方法收益，而不是强迫 absolute baseline 使用 edge-only predictor。

所有结构化 baseline 直接读取 relative set-min，CDF 的具体 rank、region、score、solver 和误差预算将事前冻结；一般 joint-field 类别可以包含本方法，因此不被虚构成一个必须击败的无限类别。上述处理是公平的。

数值 infimum 的 primal feasible objective 不能当安全下界，必须使用解析、dual 或有方向的优化误差预算，这一点也修复正确。

当前已经有足够清楚的公平比较规范。具体 baseline 能否实现、优化误差是否足够小，要由后续计划和实现落实；不能把“合同中已要求公平”写成“已经完成公平比较”。

## 7. 唯一需要在执行前澄清的 handoff 顺序

Handoff 目前先要求用 frozen feature cache 做 Gate B，随后才在 Gate B 通过后训练最小 field decoder。这里必须指明 Gate B 实际使用哪个合法预测器。

feature cache 本身不产生 \(u\)，GT field 与自己对齐也不能代表视觉预测 residual。可接受的两种最小澄清是：

1. Gate B 使用一个已存在、独立于该 held-out source panel 的合法 \(u_{abs}\) 或其他明确 predictor；或者
2. 允许在 Gate B 内训练一个小规模 source-only probe/最小 decoder，将“Gate B 后才训练”改为“Gate B 后才开展正式或较大规模训练”。

两者都不改变核心理论，也不授权本轮运行实验。不得在没有预测器的情况下将理论 oracle headroom 当成实际视觉证书的非平凡性。

这是 handoff 的执行依赖澄清，不是要求再创造一轮新方法。可以连同冻结配置由 experiment-plan 一次落实。

## 8. V0 与论文原创主张的分离已清楚

Authorized now 列出的条件链、公式和进入 Gate A/B 的规范资格，均与当前材料相符。Not authorized now 明确排除了已校准、实时、有效、OOD 统计保证、优于最近邻和“首次”，没有再用理论设计审查冒充系统验收。

Innovation Decision Rule 也是一个可失败的后续裁定规则，而不是已经获得的方法原创资格。它不改变当前的事实：核心工具和精确 min-plus 归约均是标准的，尚无证据表明这套组合超出了一个严谨的结构化道路应用。

系统规范可以先冻结，之后由资格、对照和查新决定论文地位。无需为了顶会评分继续更换数学对象。

## 9. 低于 7 分项目的方法级行动

### Contribution Quality：6/10

弱点：一个清楚且严谨的组合，仍不等于新的通用建模或校准机制；本轮主要完成合同修复。

**CRITICAL 行动：** 保持当前小机制不扩张，用已写明的 Layer 1/2 和具体最近邻验证剩余差异。若没有不可替代的锐度、迁移或成本收益，执行应用定位/停止条件，不追加模块维持原创标题。

### Venue Readiness：6/10

弱点：尚未完成独立查新或任何系统资格与强对照，不足以授权顶会方法主张。

**CRITICAL 行动：** 将“V0 规范冻结”作为当前完成事项；将论文级“首次”和方法优越性留给独立证据裁定。不能仅因为最终合同闭合而把 verdict 改成 READY。

## 10. 剩余必须落实事项

### 执行前

1. 明确 Gate B 的合法预测器与训练依赖，消除 feature-cache-before-predictor 的顺序歧义。
2. 固定实际 ROI、resolution、图尺寸、几何 checker 构造与最终数值误差预算；这些在首次运行前进入不可变配置。
3. 保留 \(\beta\) 的 session-level 定义。V0 使用确定性 nominal checker；若未来改为概率 checker，不能直接使用单调用错误率。

### 声称系统或论文结果前

1. 由实际 Gate A/B 结果决定数学实现、几何资格和非平凡性是否通过。
2. 完成两层合法 baseline 与具体 CDF 对照，按同一风险和数值标准判定。
3. 完成独立查新后再授权任何原创措辞；失败时降级，不修改已冻结门槛来制造通过。

上述事项不要求本轮继续修改核心方法或执行实验。

## 11. Simplification / Modernization

**Simplification Opportunities：NONE。** 当前 fixed-scale、single-native、exact solver 的主版本已经足够小。不要在 V0 引入 learned scale 或近似路径算法。

**Modernization Opportunities：NONE。** 冻结基础模型与解析求解的角色已经自然；当前需要的是落实资格，不是新模型。

## 12. 最终状态

- **V0 条件理论规范：可冻结。** 允许进入专门 experiment-plan 与最小资格实现。
- **已验证系统：未建立。** 本轮没有实验或运行验收。
- **顶会方法主张：不授权。** 总分 7.60/10，最终 verdict 为 REVISE。
- **问题漂移：NONE。**

结束本轮方法细化是合理的。后续应以已经冻结的协议接受或否证这个小系统，而不是继续通过改名、加模块或更换理论对象追逐审稿分数。

</details>
