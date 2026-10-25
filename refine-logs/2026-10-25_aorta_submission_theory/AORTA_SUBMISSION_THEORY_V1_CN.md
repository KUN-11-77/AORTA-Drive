# AORTA 投稿版理论—算法—架构闭环 V1

状态：`SUBMISSION_THEORY_CANDIDATE__EMPIRICAL_CORE_UNPROVEN__NOT_ICCV_READY`

本文是当前 AORTA 的规范方法版本。它统一研究问题、数学对象、网络、loss、推理规则、时序状态、
时延合同与可证伪实验。文中定理只在所列假设下成立；网络有效性、NavDream 泛化、跨 planner 行为
收益、clean no-harm 与实时性仍须由冻结实验获得，不得由理论形式代替。

## 0. 审核裁决

- **研究价值：有。** “外观稳定感知不推出稳定决策”是明确且重要的视觉—决策缺口。
- **数学地基：条件性扎实。** estimand、表示充分性、signed support、regret 与选择性仲裁能够形成
  自洽链条，但标准数学本身不构成创新。
- **架构价值：有潜力但未证明。** 唯一主创新候选是 appearance-interventional differential
  evidence operator；它必须胜过 matched D-JEPA/World4Scorer-style scorer。
- **投稿状态：`REVISE / plausible ICCV poster thesis / not submission-ready`。** 关键原因不是公式不够，
  而是 full-target 作用域、matched novelty、rollout history、NavDream 行为与时延尚无决定性证据。

## 1. 不可变问题锚点

### 1.1 Bottom-line problem

在不修改冻结端到端 planner 的前提下，学习一种 source-only、外观鲁棒、任务充分、选择性且实时的
视觉决策接口，使同一物理场景在外观变化下保持正确的候选排序，并在 NavDream 与至少三类 planner
上减少困难失败，同时保持 clean 非劣。

### 1.2 必须解决的瓶颈

稳定 feature、道路场或单候选 score 不保证 native 与 challenger 的相对优劣稳定；已有 B3
common-min 与 BS2 late-fork 反例表明，错误 readout 会在视觉场完美时仍删除决策信息。

### 1.3 Non-goals

- 不生成新候选、不训练原 planner；
- 不把道路可行性等同于 collision、交通规则或真实道路安全；
- 不把 source calibration 外推为任意 OOD 保证；
- 不把 DINO、cross-attention、Jordan decomposition、antisymmetry、conformal 或 fallback 单列创新；
- 不通过更多模块替代 matched scorer 上的决定性比较。

## 2. 估计对象：先限定官方作用域

令潜在物理场景为 \(S_t\)，合法外观为 \(A_t\)，观测为

\[
X_t^a=g(S_t,A_t=a,\xi_t).
\]

冻结 planner 给出合法 ragged 候选域

\[
V_t=\{\tau_{0,t},\tau_{1,t},\ldots,\tau_{K_t,t}\},
\]

其中 \(\tau_{0,t}\) 为 native。上一周期系统实际选择的轨迹记为 \(h_t=\tau^{sel}_{t-1}\)。

### 2.1 局部训练 estimand

在冻结 NAVSIM evaluator 经代码与逐项重放确认的层级上，定义候选条件的两帧局部效用

\[
U_t^{ctx}(\tau_i;S_t,h_t).
\]

主监督是 native-relative margin

\[
\Delta^*_{i0,t}
=U_t^{ctx}(\tau_{i,t};S_t,h_t)
-U_t^{ctx}(\tau_{0,t};S_t,h_t).
\]

若冻结实现确实满足

\[
U_t^{ctx}=\frac78 U_t^{1f}+\frac18 H_tC_{2,t},
\]

则 \(1/7\) 符号界只适用于这个**已核验的局部评分层**。在未逐项证明乘性门、过滤、阶段聚合与
归一化均被包含前，禁止把它称为完整 EPDMS 恒等式。最终行为主张由官方完整评估器单独给出，
不能由局部 margin 代理回填。

### 2.2 合法外观干预

外观变换 \(T_a\) 只有在保持

\[
(S_t,h_t,V_t,U_t^{ctx})
\]

不变时才属于同一 appearance orbit。若天气生成同时改变可见交通状态、候选 membership、上一周期
上下文或标签，它不是纯外观实验。

### 2.3 历史是内生状态

部署时 \(h_t\) 由此前 AORTA 决策产生，而非永远等于 native history。因此训练分布
\(P_{train}(h_t)\) 与 rollout 分布 \(P_{roll}(h_t)\) 必须分别记账。对任意有界损失
\(0\le\ell\le1\)，标准分布差异界给出

\[
|R_{roll}-R_{train}|
\le \operatorname{TV}(P_{roll}(X,V,h),P_{train}(X,V,h)).
\]

它不提供小风险保证，只说明 teacher-forced one-step 准确率不能替代 sequential evaluation。
训练至少使用合法 previous context；若 native-history 与 actual-selected-history 的 gap 显著，必须
使用 evaluator 支持的 counterfactual-history materialization 或冻结的 rollout/data-aggregation
方案，不能靠复杂 temporal encoder 掩盖输入分布错误。

## 3. 核心表示：完整时空差异不是最终答案，而是提问方式

在时空域 \(\mathcal Z=\Omega\times[0,T]\) 上，以 \(\mu_i\) 表示轨迹 \(\tau_i\) 的软 swept-footprint
占据测度。定义

\[
\nu_{i0}=\mu_i-\mu_0
=\nu^+_{i0}-\nu^-_{i0},
\qquad
\mu^{\cap}_{i0}=\mu_i\wedge\mu_0.
\]

- \(\nu^+\)：challenger 新增的时空支持；
- \(\nu^-\)：native 放弃的时空支持；
- \(\mu^\cap\)：两者共享但可能影响非加性结果的上下文。

对加性任务泛函

\[
U_c(\tau)=\int_{\mathcal Z}c(z)d\mu_\tau(z)+b(S),
\]

有精确恒等式

\[
U_c(\tau_i)-U_c(\tau_0)
=\int c\,d\nu^+_{i0}-\int c\,d\nu^-_{i0}.
\]

该式解释 signed support 的自然性，但 NAVSIM utility 非纯加性。因此 AORTA 不声称差异区域单独
充分；它用差异支持提出 query，同时允许 query 读取全部全局视觉 token 与历史上下文。

## 4. 投稿版网络：Differential Evidence Transport Operator

### 4.1 一次共享视觉编码

\[
Z_t=E_{vis}(X_t)=\{(z_p,\ell_p,m_p)\}_{p=1}^{P}.
\]

`E_vis` 每个场景只执行一次。\(m_p\) 是 evidence provenance，不是“可见即真实”的保证。

### 4.2 三类几何 query

冻结几何将 \(\nu^+,\nu^-,\mu^\cap\) 变成带时间、位置、朝向和 footprint 权重的稀疏 query：

\[
Q^r_{i0}=Q_{geo}(\nu^r_{i0},\tau_i\ominus\tau_0),
\qquad r\in\{+,-,\cap\}.
\]

它们只决定“问什么”，不硬裁剪视觉 key/value：

\[
E^r_{i0}=\operatorname{CrossAttn}(Q^r_{i0},K(Z_t),V(Z_t);b_{geo}).
\]

因此交通灯、遮挡、道路延续等非局部证据仍可回答局部轨迹差异。

### 4.3 反对称但全局可积的 native-star 输出

共享比较器 \(G_\theta\) 同时读取 \(E^+,E^-,E^\cap\)、相对运动与 previous-action token。对每个
challenger 定义

\[
q_{i0}
=\frac12\left[
G_\theta(E^+,E^-,E^\cap,Q_{i0},h_t)
-G_\theta(E^-,E^+,E^\cap,Q_{0i},h_t)
\right],
\qquad q_{00}=0.
\]

部署只独立预测 native-star 向量 \(q=(0,q_{10},\ldots,q_{K0})\)。任意 challenger 间关系**不再
独立预测**，而定义为

\[
\widehat\Delta_{ij}=q_{i0}-q_{j0}.
\]

于是

\[
\widehat\Delta_{ij}=-\widehat\Delta_{ji},\qquad
\widehat\Delta_{ij}+\widehat\Delta_{jk}=\widehat\Delta_{ik},
\]

精确成立，不会产生循环偏好。这里的可积性是输出类型约束，不作为创新；在 native-star 上任意
margin 向量本来都可表示，真正要检验的是 differential query 是否改善泛化与 regret。

### 4.4 最小历史接口

历史编码器只接收实际上一周期 selected trajectory 的低成本几何 token、合法 adjacency flag 与
时间间隔。复杂 transition-support attention 只有在最小 token 对 actual-selected-history 的
held-out regret 明显不足时才获准加入。

### 4.5 不确定性分支

模型同时输出正尺度 \(\sigma_{i0}>0\)。它只描述 residual sharpness；任何 coverage 都来自模型、
候选机制和规则冻结后的独立 recording-group calibration，不能由 point head 或 softmax confidence
替代。

## 5. Loss：正确性优先，稳定性受约束，选择边界可见

令 \(i^*=\arg\max_{i\in V_t}\Delta^*_{i0,t}\)，所有损失使用同一 ragged mask。

### 5.1 数值正确性

\[
\mathcal L_{val}
=\frac1{K_t}\sum_{i=1}^{K_t}
\operatorname{Huber}(q_{i0}-\Delta^*_{i0}).
\]

### 5.2 决策一致性

\[
\mathcal L_{dec}
=\max_{i\in V_t}
\left[
q_{i0}-q_{i^*0}
+\Delta^*_{i^*0}-\Delta^*_{i0}
\right]_+.
\]

它直接惩罚会改变 top choice 的结构化错误，而不是把所有近 tie pair 当作同等分类样本。

### 5.3 外观轨道约束

对保持 \(S,h,V,\Delta^*\) 不变的合法 source orbit：

\[
\mathcal L_{orb}
=\max_{a\in\mathcal A_{train}}
\|q(X^a)-\operatorname{sg}[q(X^{clean})]\|_\infty.
\]

训练合同为

\[
\min_\theta\ \mathbb E[\mathcal L_{val}+\lambda_{dec}\mathcal L_{dec}]
\quad\text{s.t.}\quad
\mathbb E[\mathcal L_{orb}]\le\delta_A.
\]

\(\delta_A\) 与 primal-dual 更新规则必须在 source development 内预注册。orbit-only 可由常数输出达到，
所以永远不能替代 task loss。

### 5.4 残差尺度

尺度使用 proper likelihood/interval score 在 point model 冻结后训练；若 learned scale 不改善独立
calibration 的 sharpness、risk–coverage 与 false intervention，则删除它并使用更简单的固定/分层尺度。

禁止默认增加 cycle、factor、depth、reconstruction、TTA 或 smoothness loss。每个新项必须由独立
反例授权，并有删除实验。

## 6. 条件理论闭环

### 定理 1：合法外观下真实 relation 不变

若 \(T_a\) 保持 \(S,h,V,U^{ctx}\) 不变，则 \(\Delta^*_{i0}(T_aX)=\Delta^*_{i0}(X)\)。这定义学习目标，
不保证像素可识别。

### 定理 2：表示充分性的充要条件

存在确定函数从表示 \(R\) 恢复 \(\Delta^*\)，当且仅当 \(R(x)=R(x')\Rightarrow
\Delta^*(x)=\Delta^*(x')\)。因此 correct/mismatch/removed 与二世界反例比 feature similarity 更关键。

### 定理 3：signed support 对加性任务精确

第 3 节恒等式由 signed measure 的线性性成立；对非加性任务只提供归纳偏置，不提供充分性保证。

### 定理 4：输出一致性

第 4.3 节构造精确满足反对称、零自比较与传递一致性。它防止非法循环，但不提高预测准确率。

### 定理 5：同时 margin 误差控制选择 regret

若

\[
\max_{i\in V_t}|q_{i0}-\Delta^*_{i0}|\le\varepsilon,
\]

则按 \(q\) 选择的候选相对真实最优候选的 regret 不超过 \(2\varepsilon\)。因此平均 MAE 不能替代
group-level maximum error、top-choice 与 regret。

### 定理 6：选择性干预的条件真实性

若同时覆盖事件

\[
\mathcal E=\{|q_{i0}-\Delta^*_{i0}|\le r_{i0},\forall i\in V_t\}
\]

成立，则任何满足 \(q_{i0}-r_{i0}>0\) 的 challenger 均真实优于 native。source calibration 只对其
声明的 exchangeable population 有效；NavDream 只报告 empirical coverage 与风险，不称保证。

## 7. 推理算法与失败闭合

1. 冻结 planner 产生 native 与合法 challengers；
2. 图像只编码一次；
3. 为每个 challenger 构造完整 4 s signed support 与 common context query；
4. 用实际 previous selected trajectory 生成最小历史 token；
5. 向量化得到 \(q_i,\sigma_i\)；
6. 独立校准给出 \(r_i=q_{cal}\sigma_i\)；
7. 仅当所有输入/数值/时延合同有效且 \(\max_i(q_i-r_i)>0\) 时选择最大 lower bound，否则保持 native；
8. 将本次实际选择写回下一周期历史。

fail-closed 条件包括：候选域不一致、历史非法、非有限输出、超时、校准不可用、全部 lower bound
非正。fallback 不是“安全动作”，只是不做未经证据支持的替换。

## 8. 时延合同

\[
C_{total}=C_{vis}+C_{support}(K,T)+C_{query}(K,M,d)+C_{head}(K,d).
\]

设计目标是一次 \(C_{vis}\) 加向量化 \(O(K)\) query，而不是 \(K C_{vis}\)。正式 profile 必须包含
图像解码、冻结 backbone、4 s support、history token、query、calibration 与选择，使用同一硬件、
batch-1，报告 P50/P95/P99、峰值显存和 \(K\) scaling。删除困难候选、缩短时域或使用离线 token
cache 的数字不能支持实时主张。

## 9. 每项设计为什么存在

| 设计 | 依据 | 预期效果 | 决定性删除实验 |
|---|---|---|---|
| full 4 s support | BS2 late-fork | 不丢 2.5–4 s 分叉 | full vs prefix |
| signed query | B3 common-min collapse | 聚焦改变候选关系的支持 | differential vs generic scorer |
| positive/negative 分开 | 早期相减会抵消高维证据 | 区分新增与放弃区域 | separated vs early subtraction |
| common + global KV | utility 非加性；D0 不等于全局不可观测 | 保留交通灯/遮挡/道路连续性 | global vs local-only |
| native-star 可积输出 | 真值来自标量效用 | 消除循环偏好 | independent pair head |
| actual history token | evaluator 依赖 previous action | estimand 输入闭合 | native-history vs actual-history |
| task + constrained orbit | 稳定与正确互不推出 | 避免稳定错误和外观脆弱 | task-only/orbit-only/full |
| simultaneous lower bound | 多候选选择与有限样本误差 | 证据不足时拒判 | point/independent intervals |
| shared visual encoding | batch-1 预算 | backbone 成本不随 \(K\) 重复 | repeated encoding profile |

## 10. 新颖性边界与最强替代解释

现有工作已覆盖 relation learning、candidate-conditioned cross-attention、outcome scorer、previous-plan
reranking、margin gate、native fallback 和 conformal monitor。AORTA 只保留一个主贡献候选：

> 在纯外观干预下，以 native–challenger 的完整时空 signed support 显式提出视觉问题，是否比拥有
> 相同信息、监督和预算的通用 scorer 更稳定地恢复候选关系。

必须对比：

1. absolute score + subtraction；
2. independent candidate-conditioned scorer；
3. D-JEPA-style set relation；
4. World4Scorer-style full-context scorer；
5. AORTA differential evidence operator。

所有 arm 匹配视觉 token、候选、history、full labels、参数、FLOP、optimizer search、seed 与在线预算。
若第 5 项不稳定胜过最强 matched baseline，删除 differential-evidence 网络创新，不以理论复杂性
挽救主张。

## 11. 证据门与停止规则

### Gate 0：接口合法性

B4R1 implementation INVALID 时先修接口。B4R1 的 one-frame GO/HOLD/STOP 只裁决 one-frame
监督，不裁决 contextual target：因为 temporal factor 可产生 one-frame 中不存在的排序信号。
只要接口有效，就另行预注册并 materialize \(U^{ctx}\)；只有 contextual oracle 也缺乏非平凡机会时，
才停止当前 candidate panel。

### Gate 1：目标作用域

逐项复算冻结 evaluator，明确 \(7/8+1/8\) 的代码层级、门控、归一化、过滤与聚合边界。未闭合前
只称 scoped two-frame candidate utility，不称完整 EPDMS。

### Gate 2：视觉必要性与架构增量

correct image 必须优于 geometry-matched mismatch、removed、trajectory-only；AORTA 必须优于
matched full-context scorer。否则分别停止视觉机制或架构创新主张。

### Gate 3：未见外观与连续决策

NavDream 报告 relation、regret、false intervention 与失覆盖；同时比较 teacher-forced native-history、
actual-selected-history 和 sequential rollout。one-step 成功不能替代连续决策。

### Gate 4：行为与部署

至少三类 planner 分别报告 hard-failure 净改善、candidate supply、selection loss、clean
non-inferiority、risk–coverage 与完整 batch-1 latency。pooled 平均不得掩盖任何一类系统性失败。

## 12. 论文故事与允许的贡献

一句话故事：

> Stable perception is not sufficient for stable driving decisions; AORTA transports globally available
> visual evidence through full-horizon candidate-difference queries, predicts a history-conditioned and
> globally consistent native-relative utility, and intervenes only when simultaneous evidence supports
> improvement.

摘要最多两项贡献：

1. **主贡献候选：**appearance-interventional differential evidence operator；
2. **从属机制候选：**history-consistent、simultaneous-evidence-qualified arbitration。

第二项只有在 sequential history 与非真空 risk–coverage 实验通过时保留。定理、DINO、antisymmetry、
conformal 与 fallback 均只作为严谨地基。

## 13. 当前允许与禁止的结论

### 允许

- 研究问题明确且有真实反例支撑；
- 条件理论、网络接口、loss、推理和 falsifier 已形成闭环；
- 当前方法具备成为 ICCV/CVPR 方法论文的可检验潜力。

### 禁止

- 声称 differential evidence 已优于最近工作；
- 声称 \(7/8+1/8\) 是未经作用域核验的完整 EPDMS 恒等式；
- 用 one-frame B4R1 失败直接否定 contextual target；
- 用 teacher-forced 离线结果声称连续决策改善；
- 用 source interval 声称 NavDream 或真实道路安全保证；
- 在 NavDream、三 planner、clean no-harm、P95 前宣称 ICCV-ready。

最终裁决：理论与设计已经达到“值得做严格验证”的门槛，但尚未达到“值得投稿”的证据门槛。
