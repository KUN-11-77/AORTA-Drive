# AORTA Best-Paper-Level First-Principles Blueprint v1

状态：`BEST_PAPER_NORTH_STAR__ONE_DOMINANT_THESIS__EMPIRICAL_QUALIFICATION_PENDING`

本文件不是获奖承诺，也不把待验证假设写成贡献。它定义：若 AORTA 最终要具备 Best Paper 候选
资格，论文应围绕什么领域级洞见、数学对象、最小方法和决定性证据展开。

## 1. Problem Anchor

- **Bottom-line problem：** 同一物理驾驶场景只改变视觉外观时，冻结 planner 的候选供给或候选
  选择可能发生有害变化；需要在不重训 planner 的条件下，恢复外观稳定且任务充分的决策证据。
- **Must-solve bottleneck：** 现有“稳定特征/稳定场景表示”目标没有说明哪些视觉信息是候选决策所
  必需，也没有区分信息不可见、表示丢失、scorer 无法读取和决策证据不足。
- **Non-goals：** 不重建完整世界作为必要目标；不学习 candidate generator；不宣称完整道路安全；
  不靠堆叠 world model、RL、diffusion、SDF 或额外大模型制造贡献。
- **Constraints：** source-only 训练；NavDream/target 不参与选择；冻结 planner 与候选协议；真实
  batch-1 时延受限；任何保证必须陈述 exchangeability、作用域和有限样本条件。
- **Success condition：** 新的决策条件表示在 matched strongest scorer 上产生稳定、可解释、跨外观、
  跨 planner 的行为增量，并且 clean no-harm、sequential history 和时延闭合。

## 2. 一句话论文命题

> Robust driving does not require appearance-invariant features; it requires calibrated inference over a latent
> decision quotient. AORTA queries global visual evidence with the full-horizon signed difference between a
> challenger and the native plan, and intervenes only when the relative utility remains identifiable.

中文版本：

> 鲁棒驾驶需要的不是外观不变特征，而是对潜在决策商空间的校准推断；AORTA 用 challenger 与 native
> 的完整时空 signed difference 查询全局视觉证据来推断相对效用条件分布，并只在相对效用可辨识时干预。

这是全文唯一主 thesis。

## 3. 领域级问题重述：Decision Quotient

### 3.1 基本随机对象

令潜在物理状态为 \(S\)，外观干预为 \(A\)，观测为

\[
I=g(S,A,\epsilon).
\]

令 \(B=(\mathcal T,h)\) 为合法候选集合与因果历史，native 为 \(\tau_0\)。相对上下文效用向量为

\[
Y_B=(\Delta_{10},\ldots,\Delta_{K0}),
\qquad
\Delta_{i0}=U(\tau_i;S,h)-U(\tau_0;S,h).
\]

### 3.2 潜在决策等价与观测预测等价

对固定候选上下文 \(B\)，先定义物理状态的任务等价：

\[
S\equiv_B^{\mathrm{lat}}S'
\iff
P(Y_B\mid S,B)=P(Y_B\mid S',B),
\qquad Q_B=q_B(S).
\]

有限视觉观测上的预测等价为：

\[
I\equiv_B^{\mathrm{obs}} I'
\iff
P(Y_B\mid I,B)=P(Y_B\mid I',B).
\]

潜在商变量删除所有不改变候选相对效用分布的物理细节；观测预测商描述图像在当前数据分布下提供的
Bayes 决策信息。论文不应预设某一 BEV、SDF、risk map 或 backbone feature 为正确表示，也不能假定
每个外观图像都能确定性恢复潜在商变量。

### 3.3 决策充分表示

表示 \(T(I,B)\) 对目标 \(Y_B\) 决策充分，当且仅当

\[
Y_B\perp I\mid T(I,B),B,
\]

等价地，

\[
P(Y_B\mid I,B)=P(Y_B\mid T(I,B),B).
\]

在适当正则条件下，条件分布 \(P(Y_B\mid I,B)\) 本身是该任务的规范充分对象。网络目标是逼近它，
而不是保持全部视觉信息。

### 3.4 外观鲁棒性的正确含义

合法外观首先必须保持真实任务对象 \(q_B(S)\)，但未必保持可辨识信息。忽略观测噪声时，存在映射

\[
r_B(g(S,a))=q_B(S),\qquad\forall S,a,
\]

当且仅当任意外观轨道间的观测碰撞只发生在同一潜在商类内：

\[
g(S_1,a_1)=g(S_2,a_2)\Rightarrow q_B(S_1)=q_B(S_2).
\]

这是确定性外观不变、决策充分表示存在的充要条件。若低照度、遮挡或伪影破坏该条件，鲁棒系统应
扩大后验不确定性并拒绝干预，而不是强迫point prediction相同。全局 feature invariance 仍既非充分
也非必要：常数表示完全不变却不决策充分；不同视觉特征也可以诱导相同的目标条件分布。

### 3.5 Best Paper 级洞见候选

如果实验支持，上述对象允许论文提出一个可推广命题：

> 视觉鲁棒性的适当等价关系由潜在决策商与其可辨识性共同决定，而不是由像素、特征或完整场景表示
> 的距离预先规定；外观销毁信息时，鲁棒性的表现应是校准的不确定性与拒判，而非虚假不变性。

该命题可以迁移到机器人操作、导航、视觉控制和其他候选决策系统，因而比 NAVSIM 专用 reranker
更有领域影响力。

## 4. 理论链条

### 4.1 定理 A：视觉条件价值分解

令 \(X_0=(C,H)\)，\(X_1=(C,H,Z)\)，在合法 mask 的平方空间中：

\[
\mathcal R^\star(X_0)-\mathcal R^\star(X_1)
=\mathbb E\left[
\left\|
\mathbb E[Y_B\mid X_1]-\mathbb E[Y_B\mid X_0]
\right\|^2
\right]\ge0.
\]

**作用：** 定义何时视觉在候选与历史之外具有价值。  
**边界：** 标准条件期望恒等式，不单独构成原创；有限网络失败不等价于信息论零增量。

### 4.2 定理 B：不可辨识性下界

给定观测 \(O=(I,B)\)，若后验中

\[
p_O=P(\Delta_{i0}>0\mid O)\in(0,1),
\]

则任何确定性符号决策器在该观测上的最小条件错误率至少为

\[
\min(p_O,1-p_O).
\]

若两个观测不可区分的潜在世界具有相反 margin 符号，完美确定性选择不可能。  
**作用：** 从第一性原理推出 distributional prediction 与 abstention，而不是额外加一个 uncertainty
head。  
**边界：** 需要通过可观测冲突、重复条件或特权上界诊断证明该现象实际存在。

### 4.3 定理 C：加性任务的 signed-support 恒等式

候选 \(i\) 的完整时空占据测度为 \(\mu_i\)，定义

\[
\nu_{i0}=\mu_i-\mu_0=\nu^+_{i0}-\nu^-_{i0}.
\]

若任务效用的相关部分是局部可加泛函

\[
J_S(\tau)=\int c_S(x,t)\,d\mu_\tau(x,t),
\]

则

\[
J_S(\tau_i)-J_S(\tau_0)
=\int c_S(x,t)\,d\nu_{i0}(x,t).
\]

**作用：** 说明为何正负完整时空差异是候选比较的自然 query geometry。  
**边界：** signed integral 和 path cost 已有先例；完整 NAVSIM utility 含非加性交互，故 \(\nu\) 只能
提出 query，不能替代全局视觉与历史。

### 4.4 命题 D：跨外观误差与决策 regret

若真实物理 margin 对合法外观干预保持不变，clean 预测误差满足

\[
\max_i|\widehat\Delta_i^{clean}-\Delta_i|\le\epsilon,
\]

且同场景预测漂移满足

\[
\max_i|\widehat\Delta_i^{a}-\widehat\Delta_i^{clean}|\le\delta_a,
\]

则 shifted appearance 上

\[
\max_i|\widehat\Delta_i^{a}-\Delta_i|
\le\epsilon+\delta_a.
\]

对预测最优候选，标准 argmax stability 给出条件 regret 上界

\[
\operatorname{Regret}_a\le2(\epsilon+\delta_a).
\]

**作用：** 把 task accuracy 与 appearance relation drift 连到决策指标。  
**边界：** 只有当真实目标在外观干预下保持相同、候选域一致且误差事件成立时有效；不能从经验
平均 drift 推出最坏情况保证。

### 4.5 定理 E：同时下界仲裁

若独立 calibration 支持同时事件

\[
\forall i\in V_B:\quad
|\widehat\Delta_{i0}-\Delta_{i0}|\le\epsilon_i,
\]

定义 \(L_{i0}=\widehat\Delta_{i0}-\epsilon_i\)。若系统只在

\[
i^\star=\arg\max_iL_{i0},
\qquad L_{i^\star0}>\gamma
\]

时替换 native，则在该同时事件及 estimand 作用域内有

\[
\Delta_{i^\star0}>\gamma.
\]

**作用：** 将不可辨识性转化为条件性保守执行。  
**边界：** 必须证明 coverage、sharpness 与 non-vacuity；source calibration 不保证 NavDream。

### 4.6 恒等式 F：候选供给与选择的 regret 分解

令可执行轨迹全集上的最优效用为 \(U^\star\)，外观相关 candidate generator 给出的合法集合为
\(B(I)\)，集合内 oracle 为

\[
U_B^\star=\max_{\tau\in B(I)}U(\tau),
\]

AORTA 的选择为 \(\widehat\tau\in B(I)\)。则总 regret 精确分解为

\[
U^\star-U(\widehat\tau)
=
\underbrace{U^\star-U_B^\star}_{\text{candidate-supply regret}}
+
\underbrace{U_B^\star-U(\widehat\tau)}_{\text{selection regret}}.
\]

**作用：** 固定 panel 实验只隔离 selection term；AORTA 不能恢复未被 planner 生成的轨迹。最终
NavDream 与跨 planner 行为评估必须同时报告 supply headroom 和 selection loss，避免把候选生成收益
或失败归因给 reranker。  
**边界：** 本工作不学习 generator，故主理论与算法主张限于第二项；整体系统收益由两项共同决定。

## 5. 从理论到最小架构

### 5.1 复杂度预算

- **冻结/复用：** DINOv3 或经资格化的视觉 backbone、planner、candidate generator、official scorer；
- **新 trainable component 1：** full-horizon signed differential query encoder；
- **新 trainable component 2：** global-evidence native-relative distribution head；
- **非训练支持步骤：** group-simultaneous calibration 与 selective arbitration；
- **明确不用：** 新 world model、RL policy、diffusion planner、SDF/field decoder、LLM、并列风险网络。

### 5.2 系统图

```text
multi-camera temporal RGB
        |
        v
shared frozen/qualified visual backbone ----> global visual tokens Z
                                                ^
                                                |
native trajectory ---- occupancy measure mu_0  |
challenger tau_i ---- occupancy measure mu_i    |
        |                                       |
        +--> nu_i0 = mu_i - mu_0 --> signed differential query q_i0
                                                |
                                                v
                             geometry-biased global cross-attention
                                                |
                                                v
                         permutation-equivariant candidate relation block
                                                |
                                                v
                           p_theta(Delta_i0 | I, candidates, history)
                                                |
                                                v
                             simultaneous lower-bound arbitration
                              /                         \
                     evidence sufficient          insufficient
                          choose i                   keep native
```

### 5.3 为什么这不是普通 trajectory query

普通 query 问“这条轨迹经过什么”；AORTA query 问“相对当前 native，这个行为新增与删除了什么时空
暴露，这些变化如何改变 contextual utility”。真正要证明的不是文字区别，而是 matched scorer 在
near-tie、late-fork 和 appearance shift 上无法同样高效地学到该关系。

### 5.4 为什么保留全局视觉

局部差异区域不包含全部交通灯状态、遮挡来源、其他参与者意图或道路拓扑。差异支持只形成 attention
bias；key/value 仍是全部视觉 token。这样方法对加性局部任务有正确归纳偏置，又不声称完整任务只由
局部路径积分决定。

### 5.5 输出空间

模型输出 native-star 条件分布

\[
p_\theta(Y_B\mid I,B),
\]

而不是独立 complete-pair edges。任意 challenger pair 由 native-star 差分得到，以保证可积性。

## 6. 训练目标

### 6.1 阶段 P0：信息资格

当前 R005E2 使用 squared native-relative margin loss，回答视觉条件均值是否可恢复。该阶段不训练
主创新。

### 6.2 阶段 P1：核心 matched 架构实验

若 P0 GO，generic scorer 与 AORTA 使用相同 backbone、候选、history、数据、参数/FLOP/token 预算、
optimizer、early stop 与 seeds。主 loss 首先使用相同 proper relation objective，避免靠不同 loss
制造优势。

### 6.3 阶段 P2：只加入被失败证明必要的目标

最终候选形式为

\[
\mathcal L
=\mathcal L_{proper}
+\lambda_{orb}\mathcal L_{decision\text{-}orbit}
+\lambda_{rank}\mathcal L_{boundary},
\]

但只有在以下条件下加入：

- source paired appearance 显示 correct task 保留而 relation drift 仍显著，才加入 orbit term；
- proper risk 改善但 regret 不改善且错误集中于 near-tie，才加入 boundary term；
- 每个新增项必须有删除消融、预注册权重规则和独立确认。

calibration 不与模型 loss 联合调参，避免 coverage 数据参与 checkpoint 选择。

## 7. Best Paper 级核心贡献

### 7.1 Dominant contribution

**Decision-Quotient Visual Learning via Differential Evidence**：提出由目标条件分布定义的视觉决策
商空间，并用完整时空 signed native–challenger difference 作为查询全局视觉证据的归纳偏置。

该贡献成立必须同时满足：

1. decision quotient framing 能解释现有稳定表示的失败；
2. differential query 在 matched closest baselines 上提供不可替代增量；
3. 增量在未见外观和行为层仍存在。

### 7.2 Supporting contribution

**Identifiability-Aware Selective Arbitration**：输出条件 relation 分布，在同时证据不足时保留 native。

它只能作为从属机制；若区间空泛或全拒判，则删除该贡献。

### 7.3 Explicit non-contributions

- 条件期望/Bayes risk 恒等式；
- information bottleneck 或 sufficiency 概念本身；
- signed measure/path integral；
- cross-attention、set encoder、relative scorer；
- conformal prediction、uncertainty 或 fallback；
- frozen DINO 或 source augmentation。

## 8. 三个决定性实验块

### E1：现象与第一性对象

**Claim：** 外观下的 feature stability 和场景重建误差不能充分解释候选决策稳定性，而 correct visual
scene binding 对 contextual relation 有条件增量。

**最小证据：**

- R005E2 correct / mismatch / removed；
- global feature/field stability 与 relation regret 的脱钩反例；
- near-tie、late-fork、context-flip 分层；
- recording-group paired statistics。

**Falsifier：** correct visual 不同时优于 removed 与 mismatch，或只改善 MSE 不改善 regret。

### E2：核心归纳偏置

**Claim：** full-horizon signed differential query 比 matched general-purpose scorer 更有效地恢复外观
稳定的决策关系。

**强制基线：** independent scorer、generic set scorer、D-JEPA-style relation、World4Scorer-style
candidate state、DA-WAM/GTRS-style candidate-conditioned scorer，以及旧 field readout。

**决定指标：** group-level decision regret、sign/boundary error、appearance-induced flip、sample
efficiency、near-tie/late-fork strata、P95 latency。

**删除消融：** signed→unsigned、full-horizon→prefix、differential→absolute、global KV→local-only、
history removal。

**Falsifier：** matched generic scorer 相当或更优；删除 differential support 不造成稳定退化。

### E3：从表示到行为

**Claim：** relation 机制在未见外观和多个 planner 上产生净行为收益，并在证据不足时减少错误替换。

**最小证据：**

- NavDream 一次性零样本开封；
- regression、generative/diffusion、generate-and-select 三类 planner；
- native-history、actual-selected-history、sequential rollout；
- clean non-inferiority；
- oracle supply、selection loss、false intervention、abstention 与 net gain 分解；
- batch-1 P50/P95、显存、候选 K scaling。

**Falsifier：** 收益只来自某一 planner、只来自 candidate supply、clean 显著受损、sequence 崩溃、
校准全拒判或时延超预算。

## 9. 最关键的论文图

1. **Phenomenon figure：** 外观变了，feature/field 看似稳定，但候选关系翻转；差异证据指出真正缺失
   的决策信息。
2. **One-method figure：** 一次视觉编码、signed difference query、global evidence、relation
   distribution、selective action；不画历史失败模块。
3. **Decisive comparison：** matched generic vs differential，在 regret、flip、near-tie/late-fork 与
   sample efficiency 上的组级结果。
4. **Behavior closure：** NavDream × planner、clean no-harm、sequential、coverage–intervention–gain、
   batch-1 latency。

## 10. 论文结构

### Introduction

1. 外观鲁棒通常被定义在像素/feature/scene space；
2. 驾驶系统真正执行的是候选决策；
3. 相同的稳定表示可以产生不稳定关系，常数稳定更是反例；
4. 提出 decision quotient 与 AORTA；
5. 三个实验发现形成从现象、机制到行为的闭环。

### Related Work

只围绕四组边界：appearance robustness、decision/candidate scoring、trajectory-conditioned visual
evidence、selective prediction。逐项说明继承内容与剩余空缺，不写百科全书。

### Theory

依次给出 decision equivalence、visual value、identifiability、signed-support factorization 与 selective
bound。每个定理后立即写网络或实验后果。

### Method

只描述 differential query、global evidence relation head 和 selective inference。R005E2 probe、旧
field 路线与历史资格过程不进入主方法。

### Experiments

按 E1–E3 三个 discovery 组织，不按数据集或几十个模块罗列。

### Limitations

明确候选供给上限、`U_ctx` 与 policy value 区别、appearance intervention 支持、calibration 假设、
不可观测动态和 closed-loop 边界。

## 11. 一分钟 Oral 故事

> 现代视觉表示可以跨天气保持相似，但驾驶决策仍会翻转，因为稳定表示没有说明必须保留哪些决策
> 信息。我们把视觉鲁棒性重新定义为：给定候选行为后，保持其相对效用的条件分布。由此得到一个
> 决策商空间，并推出 AORTA：用 challenger 相对 native 的完整时空差异向全局视觉提问，而不是先
> 重建整个场景；当视觉不能辨识优劣时保留 native。结果若最终成立，应显示该最小归纳偏置比容量
> 匹配的通用 scorer 更准确、更跨外观，并真正改善多个冻结 planner 的行为。

## 12. 当前机会判断

### 为什么有机会

- 问题不是一个 benchmark trick，而是“视觉鲁棒性应定义在哪个空间”的基础问题；
- 已有失败不是噪声，而是逐步排除了 field、scalar-min、短前缀和纯几何解释；
- decision quotient 能把数学对象、网络、loss、仲裁和实验统一；
- 方法可保持极简：一个差异查询机制和一个分布 relation head；
- 若跨 planner 成立，影响范围超过单一 NAVSIM 模型。

### 为什么当前还不是 Best Paper 方法

- R005E2 尚未证明视觉条件增量；
- differential query 尚未运行 matched closest-baseline study；
- decision quotient 目前主要是问题重述，尚无新的经验定律支持；
- NavDream、actual-history、cross-planner、clean no-harm 与时延均未闭合；
- 多个理论环节基于标准统计与决策工具，原创性必须由新组合产生的新发现证明。

### 条件性裁决

\[
\boxed{
\text{Best Paper opportunity}
=
\text{field-level reframing}
\times
\text{indispensable mechanism}
\times
\text{behavior-level evidence}
}
\]

任一项接近零，整体机会即消失。当前第一项有潜力，后两项仍为零证据状态。

## 13. 失败降级路径

| 失败位置 | 科学结论 | 允许的路线 |
|---|---|---|
| R005E2 STOP | 当前视觉表示无稳定可恢复增量 | 可辨识性/representation bottleneck 研究；不训练主架构 |
| R005E2 MIXED | 条件均值或任务边界错配 | near-tie/distributional 诊断；不直接上复杂系统 |
| Differential 不胜 generic | 主网络创新不成立 | 保留 decision quotient 分析或最强简单 scorer，不包装架构 |
| Source 胜、NavDream 败 | 非外观鲁棒 | 停止 OOD 主张，分析 nuisance support |
| Offline 胜、sequential 败 | history shift 未解决 | 收缩为离线 scorer 或重建因果状态分布 |
| Selective 全拒判 | 可靠性无实用价值 | 删除从属贡献 |

## 14. 研究纪律

1. 不以 Best Paper 为理由增加模块；
2. 不在看见 NavDream 结果后返回选择架构或增强；
3. 不把标准数学工具重命名为原创理论；
4. 不把 oracle、离线 margin 或视觉稳定性写成驾驶收益；
5. 每个保留模块都必须回答失败来源、数学依据、简单替代、否定消融和删除预测；
6. 主文只保留最终获胜机制和决定性失败，不复述完整项目考古；
7. Best Paper 是结果出现后的资格判断，不是实验前可以保证的目标标签。

## 15. 下一步

当前不修改 R005E2。待其 primary、replay 与审计完成后：

- GO：冻结 E2 的 generic-vs-differential 最小研究，不提前加入 orbit/ranking/calibration；
- MIXED：先定位 conditional-mean、decision-boundary 或函数类问题；
- STOP：按信息存在性→表示保留性→函数逼近性顺序诊断；
- INVALID：只修复实现，禁止科学解释。

在 E2 机制胜出前，不开封 NavDream，不训练额外复杂模块，不宣称 Best Paper readiness。
