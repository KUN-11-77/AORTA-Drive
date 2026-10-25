# AORTA 正式理论附录 V1

状态：**CONDITIONAL_THEOREMS_CLOSED__EMPIRICAL_PREMISES_UNPROVEN**

本文只建立定义、条件命题、证明与反例。它不把标准数学定理包装成 AORTA 的原创贡献，也不把尚未验证的经验前提写成事实。项目特有价值若成立，应来自：由真实失败反例选择了正确 estimand 与证据接口，并在强 matched baselines、NavDream 和跨 planner 行为上证明其必要性。

## 1. 形式化对象

### 1.1 场景、外观与合法干预

令 \((\mathcal S,\Sigma_S)\) 为潜在物理场景空间，\(A\in\mathcal A\) 为外观变量，\(X=g(S,A,\xi)\) 为多相机观测，\(h\) 为包括上一周期实际 selected trajectory 的任务上下文。冻结 planner 在上下文 \(c=(S,h,\pi)\) 中产生有限合法候选域

\[
V_c=\{\tau_0,\tau_1,\ldots,\tau_K\},
\]

其中 \(\tau_0\) 是 native。一个变换 \(T_a\) 只有在保持 \(S,h,V_c\) 与任务标签不变时才是合法 appearance intervention：

\[
X^a=T_aX,\qquad
(S,h,V_c,F_{\mathrm{full}})\ \text{unchanged}.
\]

如果雨雪算子改变了可见交通参与者的物理状态、候选 membership、时间上下文或 evaluator label，它不属于此等价类。

### 1.2 最终 estimand

对 \(i\in\{1,\ldots,K\}\)，定义

\[
\Delta^*_{i0}(c)
=F_{\mathrm{full}}(S,\tau_i,h)-F_{\mathrm{full}}(S,\tau_0,h),
\qquad \Delta^*_{00}=0.
\]

候选 mask、标签、loss、argmax、uncertainty、oracle 与 regret 必须使用同一个 \(V_c\)。否则“预测正确”与“选择正确”并非同一统计问题。

### 1.3 表示与选择器

共享视觉表示为 \(Z_\theta(X)\)。pair query 为 \(Q_{i0}=Q(\tau_i,\tau_0,h)\)。关系预测与半径分别为

\[
\widehat\Delta_{i0}=R_\theta(Z_\theta(X),Q_{i0}),
\qquad r_{i0}\ge0.
\]

点选择器为 \(\widehat i\in\arg\max_{i\in V_c}\widehat\Delta_{i0}\)。选择性仲裁器只在某个 challenger 的下界 \(L_{i0}=\widehat\Delta_{i0}-r_{i0}\) 严格为正时替换 native。

## 2. 命题 A：真实任务关系的外观不变性

### 定理 A1（合法干预下的 estimand 不变性）

若 \(T_a\) 保持 \((S,h,V_c,F_{\mathrm{full}})\) 不变，则对全部合法候选

\[
\Delta^*_{i0}(T_aX,c)=\Delta^*_{i0}(X,c).
\]

**证明。** \(\Delta^*_{i0}\) 的两个被减项只依赖未改变的 \(S,\tau_i,\tau_0,h\) 与同一个 \(F_{\mathrm{full}}\)，故相等。\(\square\)

**解释。** 这是任务定义的一致性，不是视觉泛化定理。它只规定“应该保持什么”，不推出像素中可恢复该量，也不推出任何网络会学到它。

### 引理 A2（候选共同 nuisance 的精确消除）

若某候选证据可分解为

\[
f_i(S,A,h)=u_i(S,h)+b(S,A,h),
\]

且 \(b\) 对同一 panel 的全部候选共同，则

\[
f_i-f_0=u_i-u_0.
\]

**证明。** 两式直接相减，共同项消失。\(\square\)

**最小反例。** 若 \(f_i=u_i+b_i(A)\)，则差分仍含 \(b_i(A)-b_0(A)\)。因此 relative output 不能自动保证外观鲁棒；paired appearance loss、correct-image necessity 与 NavDream 仍是必需实验。

### 网络与实验后果

- 输出 native-relative relation 有消除共同 nuisance 的结构机会，但不构成充分保证；
- clean/transformed task error 与 relation drift必须同时报告；
- appearance 算子的合法性必须由物理状态、候选域和标签 exactness 审计。

## 3. 命题 B：表示充分性与不可识别性

### 定理 B1（表示纤维上的充要条件）

给定表示 \(R:\mathcal X\to\mathcal R\) 与目标 \(\Delta^*:\mathcal X\to\mathbb R^K\)，存在函数 \(q\) 使 \(\Delta^*=q\circ R\)，当且仅当

\[
R(x)=R(x')\Longrightarrow\Delta^*(x)=\Delta^*(x').
\]

**证明。** 必要性由函数复合直接得到。充分性下，对任意 \(r\in R(\mathcal X)\)，选择一个 \(x_r\in R^{-1}(r)\)，定义 \(q(r)=\Delta^*(x_r)\)。纤维常值性保证定义与代表元无关。\(\square\)

该结果是标准 factorization 条件，不是 AORTA 原创定理。

### 推论 B2（符号不可识别的二世界反例）

若存在两个 admissible worlds \(x,x'\) 满足

\[
R(x)=R(x'),\qquad
\operatorname{sign}\Delta^*_{i0}(x)\ne
\operatorname{sign}\Delta^*_{i0}(x'),
\]

则任何只读取 \(R\) 的确定预测器至少在其中一个世界上符号错误；若两世界等概率，最小符号错误率至少为 \(1/2\)。

**证明。** 预测器接收相同输入，必须输出同一符号；真实符号相反，故至少一个错误。等概率时至少承担一个世界的概率质量。\(\square\)

**重要边界。** 某个查询节点没有直接相机投影不等于所有可访问表示相同。只有构造了全局视觉输入相同而真值符号相反的世界，才得到不可识别性结论。

### 网络与实验后果

- differential support 只负责“问哪里不同”，全局 visual key/value 保留可能消除纤维冲突的上下文；
- mismatch/removed-image、trajectory-only 与 correct-image 的比较检验视觉信息是否缩小表示纤维；
- 若仍存在不可区分的相反符号世界，增加 point-head 容量无效，只能增加观测或拒判。

## 4. 命题 C：差异支持何时决策充分

令 \(\mathcal Z=\Omega\times[0,T]\)，\(\mu_i\) 为候选 \(\tau_i\) 的完整时空软占据测度，\(\nu_{i0}=\mu_i-\mu_0=\nu^+_{i0}-\nu^-_{i0}\)。

### 定理 C1（加性任务泛函的精确差分恒等式）

若

\[
F_c(\tau)=\int_{\mathcal Z} c(z)\,d\mu_\tau(z)+b(c),
\]

则

\[
F_c(\tau_i)-F_c(\tau_0)
=\int c\,d\nu^+_{i0}-\int c\,d\nu^-_{i0}.
\]

**证明。** 利用积分对 signed measure 的线性性与 Jordan decomposition。\(\square\)

该恒等式解释为什么正负差异支持分别进入查询，但它不是对完整 NAVSIM utility 的充分性证明。

### 反例 C2（非加性交互）

设任务效用含有候选与公共场景状态的交互 \(H(\mu_i,\chi_S)\)，且两个场景具有相同 \(\nu_{i0}\) 但不同交通灯或交互车辆状态 \(\chi_S\)。则仅由 \(\nu_{i0}\) 不能决定 margin。因而公共上下文与全局视觉证据不能被删掉。

### 反例 C3（前缀不足）

若 \(\mu_i|_{[0,T_0]}=\mu_0|_{[0,T_0]}\) 而 \(\mu_i|_{(T_0,T]}\ne\mu_0|_{(T_0,T]}\)，任何只读取 \([0,T_0]\) 的算子对该 pair 输出相同输入，不能一般恢复完整时域 margin。这对应 BS2 的 late-fork 实证反例。

### 网络与实验后果

- full-horizon signed support 是由反例授权的 query，不是新定理本身；
- common-context/global-KV 是非加性任务所需的条件信息；
- 是否优于通用 set scorer 必须由 matched regret/false-intervention 实验决定。

## 5. 命题 D：正确性、稳定性与坍缩

### 定理 D1（任务误差推出 appearance drift 上界）

若同一合法外观轨道上的两个观测均满足

\[
\|\widehat\Delta(X^a)-\Delta^*\|_\infty\le\varepsilon,
\qquad
\|\widehat\Delta(X^{a'})-\Delta^*\|_\infty\le\varepsilon,
\]

则

\[
\|\widehat\Delta(X^a)-\widehat\Delta(X^{a'})\|_\infty\le2\varepsilon.
\]

**证明。** 三角不等式。\(\square\)

### 反命题不成立

零 drift 不推出任务正确：\(\widehat\Delta\equiv0\) 对所有外观完全稳定，但当任一 \(\Delta^*\ne0\) 时不充分。因此 appearance consistency 只能作为约束，不能替代 task supervision。

### Loss 后果

\[
\min_\theta\ \mathbb E\mathcal L_{task}
\quad\text{s.t.}\quad
\mathbb E\mathcal L_{orbit}\le\delta_A
\]

比“只最小化一致性”更准确地表达研究目标。实践可用拉格朗日或冻结权重近似，但必须报告 task-only、orbit-only 反例和完整目标；orbit 降低漂移却增加 regret 时应被判失败。

## 6. 命题 E：关系误差到选择后悔

### 定理 E1（\(2\varepsilon\) regret bound）

在同一候选域上令 \(i^*\in\arg\max_i\Delta^*_{i0}\)，\(\widehat i\in\arg\max_i\widehat\Delta_{i0}\)。若

\[
\max_i|\widehat\Delta_{i0}-\Delta^*_{i0}|\le\varepsilon,
\]

则

\[
\Delta^*_{i^*0}-\Delta^*_{\widehat i0}\le2\varepsilon.
\]

**证明。** 由最优性 \(\widehat\Delta_{\widehat i0}\ge\widehat\Delta_{i^*0}\)，故

\[
\Delta^*_{i^*0}-\Delta^*_{\widehat i0}
\le(\widehat\Delta_{i^*0}+\varepsilon)
-(\widehat\Delta_{\widehat i0}-\varepsilon)
\le2\varepsilon.
\]

\(\square\)

### 推论 E2（top-choice 保持）

若真实最优与所有次优候选的 margin gap 严格大于 \(2\varepsilon\)，则估计 argmax 必须包含真实最优；若再冻结 tie rule，则 top choice 唯一保持。

### 反例 E3（平均误差不足）

一个 panel 含大量容易候选和两个接近决策边界的候选。模型在前者误差为零、后两者符号互换，则平均 MAE 可随候选数增加而趋近零，但 top choice 与 regret 仍错误。因此主报告必须包含 panel-level \(\ell_\infty\)、top gap、selection error 与 regret。

## 7. 命题 F：选择性仲裁的条件正确性

### 定理 F1（同时下界上的 native improvement）

定义同时事件

\[
\mathcal E_c=
\bigcap_{i\in V_c}
\left\{|\widehat\Delta_{i0}-\Delta^*_{i0}|\le r_{i0}\right\}.
\]

若 \(\mathcal E_c\) 成立，且仲裁器只选择满足

\[
L_{i0}=\widehat\Delta_{i0}-r_{i0}>0
\]

的 challenger，则每次已授权替换都有 \(\Delta^*_{i0}>0\)。

**证明。** 在 \(\mathcal E_c\) 上，\(\Delta^*_{i0}\ge\widehat\Delta_{i0}-r_{i0}=L_{i0}>0\)。\(\square\)

### 定理 F2（group-max split calibration 的有限样本条件）

冻结模型、候选机制和 normalization。对每个 calibration recording group \(g\)，令

\[
S_g=\max_{a,t,i}
\frac{|\widehat\Delta_{gati0}-\Delta^*_{gati0}|}{\widehat\sigma_{gati0}}.
\]

若 calibration groups 与一个新 source group 的 \(S_g\) 可交换，使用标准 split-conformal 次序统计量 \(q=S_{(\lceil(n+1)(1-\alpha)\rceil)}\)；当该次序超过 \(n\) 时取 \(+\infty\)。则新 group 内全部被最大值覆盖的 appearance/anchor/challenger 同时事件具有至少 \(1-\alpha\) 的有限样本 marginal group coverage。

这是 split conformal 的标准秩论证；AORTA 特有的是把统计单位与部署 panel 对齐，而不是 conformal 定理本身。

### 必要反例

1. **边际区间不同时。** 每个候选各有 95% coverage 不推出 \(K\) 个候选同时 95%；独立情形仅为 \(0.95^K\)。
2. **anchor 伪重复。** 把同一 recording group 的 anchors 当独立样本会夸大有效样本量。
3. **post-selection 污染。** 用 calibration outcome 选择模型、阈值或候选机制后复用同一 split，秩保证失效。
4. **OOD 不交换。** NavDream、新 planner 或新 candidate policy 下 \(S_g\) 分布可改变；source coverage 不自动外推。
5. **真空保证。** \(q=+\infty\) 时理论仍可通过但系统全拒判，必须单独报告 sharpness 与 intervention coverage。

## 8. 反对称结构合同

若部署对象是有向差值，则应满足

\[
\Delta^*_{i0}=-\Delta^*_{0i},\qquad \Delta^*_{ii}=0.
\]

使用

\[
R^{anti}(z,i,j)=\frac12\{G(z,i,j)-G(z,j,i)\}
\]

可精确满足上述性质。这只是一种结构参数化，不是足以提升泛化的定理。其价值必须通过 swap metamorphic test、参数/FLOP 匹配和 held-out regret 验证；普通 set scorer 若同样有效，则不把反对称化列为贡献。

## 9. 信息论表述的严格边界

理想表示希望同时满足

\[
I(\Delta^*;X\mid R,V_c,h)=0,
\qquad
I(R;A\mid S,V_c,h)=0.
\]

第一项是任务充分性，第二项是条件 nuisance 不变性。由于 \(S\) 不可观测、\(R\) 高维且 intervention 有限，当前实验不能一致估计并证明这两个互信息等式。它们只能指导可观测检验：

| 信息论条件 | 可观测必要后果 | 不能替代的证据 |
|---|---|---|
| task sufficiency | full-target relation error、sign、regret、correct-image necessity | feature CKA、field MAE |
| appearance invariance | paired orbit drift、NavDream stability | 单一弱外观算子 |
| non-collapse | correct image 优于 mismatch/removed/prior | 小 drift 本身 |
| decision relevance | selection error、false intervention、behavior gain | 表示可视化 |

因此不得把 MI 目标写成已经取得的理论保证，也不以神经网络容量代替可识别性论证。

## 10. 命题 G：视觉条件增量的 proper-risk 身份

令 \(Y\) 为同一合法 ragged panel 上的 native-relative `U_ctx` margin，
\(X_0=(C,H)\) 包含候选、geometry 与因果 previous history，\(Z\) 为冻结视觉表示。将ragged
margin用合法mask嵌入对应的加权平方可积空间，定义平方损失的 Bayes 风险：

\[
\mathcal R^*(X)=\inf_f\mathbb E\|Y-f(X)\|_2^2
=\mathbb E\|Y-\mathbb E[Y\mid X]\|_2^2.
\]

### 定理 G1（视觉条件均值信息的风险分解）

\[
\mathcal R^*(X_0)-\mathcal R^*(X_0,Z)
=\mathbb E\left\|\mathbb E[Y\mid X_0,Z]-\mathbb E[Y\mid X_0]\right\|_2^2\ge0.
\]

**证明。** 条件期望是平方可积随机变量空间中的正交投影。将
\(Y-\mathbb E[Y\mid X_0]\) 分解为
\(Y-\mathbb E[Y\mid X_0,Z]\) 与
\(\mathbb E[Y\mid X_0,Z]-\mathbb E[Y\mid X_0]\)，两项正交；取平方范数期望即得。
\(\square\)

### 解释与严格边界

- 该恒等式说明：在理想 Bayes 函数空间内，视觉只有在改变目标条件均值时才降低平方风险；
- R005E2 用 group cross-fitting、correct/mismatch/removed 与参数匹配 scorer 操作化这一条件；
- 有限网络上观察到稳定正增量，支持“当前函数类可利用scene-specific视觉信息”；
- 未观察到增量不能证明 \(I(Y;Z\mid X_0)=0\)，因为函数类、优化和样本量也可能失败；
- 风险下降也不自动推出低 decision regret，故必须同时检验定理 E 对应的 panel-level selection 指标。

该命题把视觉必要性从模糊的“模型看图”收紧为 conditional proper-risk comparison，但它是标准条件
期望几何，不作为AORTA原创定理。项目贡献仍取决于后续 differential-support 对 matched scorer 的增量。

## 11. 标准理论与项目贡献的边界

| 内容 | 性质 | 论文中允许的角色 |
|---|---|---|
| quotient/factorization、Jordan decomposition、三角不等式、\(2\varepsilon\) regret | 标准数学 | 解释并约束接口 |
| split conformal 次序统计量 | 已有统计理论 | 资格化选择性动作 |
| 反对称化参数化 | 标准结构技巧 | 与 estimand 对齐的实现 |
| B3 common-min、BS2 late-fork、full-target context 反例 | AORTA 实证发现/任务分析 | 推导设计必要性 |
| appearance-interventional differential query 是否优于 matched scorer | AORTA 核心待证假设 | 只有实验支持后才是主贡献 |
| NavDream/三 planner 的净行为改善且 clean 非劣、实时 | AORTA 最终待证系统结果 | 决定方法论文资格 |

## 12. 定理—网络—实验映射

| 理论责任 | 网络/算法选择 | 决定性 falsifier |
|---|---|---|
| A：合法 orbit 上目标不变 | paired appearance relation constraint | 变换改变标签；OOD relation 崩溃 |
| B：表示纤维必须 task-constant | global visual KV + correct-image control | mismatch/removed 与 correct 相当 |
| C：完整差异支持保留候选变化 | full-horizon signed query | matched generic scorer 相当或更优 |
| D：稳定不能牺牲正确性 | task-first constrained orbit loss | drift 降低但 regret 增加 |
| E：控制同时 margin error 才控制 regret | full-margin regression/ranking | MAE好但 top-choice/regret不改善 |
| F：只有同时下界支持干预 | group-max calibration + native fallback | undercoverage、全拒判或 false intervention 不降 |
| G：正确视觉若含可用条件信息应降低proper risk | correct/mismatch/removed参数匹配cross-fit | correct不降低risk或降低risk但regret不改善 |

## 13. 当前结论

上述命题在其假设下成立，且明确给出了不成立的反例；这使数学地基“可审查”，但没有证明经验假设满足。R005E1B已支持native-history `U_ctx`的source-fit非平凡headroom，但仍缺：视觉条件增量、差异查询相对matched scorer的增量、NavDream外观泛化、三planner行为净收益、clean non-inferiority与batch-1 P95。故状态保持：

**THEORY_CONTRACT_READY__METHOD_AND_SYSTEM_NOT_VALIDATED__NOT_ICCV_READY**。

数据无关软件合同复核：relation/regret/arbitration 与 decision-quotient 的 17 项性质测试通过；calibration/risk 的 9 项测试通过。它们只验证实现与上述有限接口一致，不把单元测试冒充统计 coverage 或方法效果。
