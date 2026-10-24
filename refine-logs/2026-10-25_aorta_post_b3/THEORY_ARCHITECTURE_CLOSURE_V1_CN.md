# AORTA 理论—网络—行为闭环合同 v1

状态：`FORMAL_CLOSURE_CANDIDATE__EMPIRICAL_CORE_UNPROVEN__B4R1_REQUIRED`。

本文把分散的数学命题、网络设计、loss、选择规则、时延和论文故事压到同一条可反驳链上。它不是
用更多公式包装方法，也不授权训练；任何结构只有在对应失败已经出现、数学接口明确且 matched
删除实验能够检验时才保留。

## 1. 长期问题锚点与一句话论文命题

给定冻结 planner 产生的 native 轨迹 \(\tau_0\) 与合法 challenger 集合
\(V\setminus\{0\}\)，在只改变天气、光照等外观而保持物理场景和任务不变时，学习可靠的
native-relative task relation，并只在证据充分时覆盖 native。

一句话方法命题为：

> **可靠的外观鲁棒驾驶需要稳定“候选在哪里改变决策的证据”，而不只是稳定整幅感知表示。**

论文标题级候选为：

> **AORTA: Learning Appearance-Robust Decision-Difference Evidence for Frozen Driving Planners**

当前这仍是假设，不是结果。DINOv3、cross-attention、Jordan/Hodge、conformal、候选生成和
uncertainty head 均不是单独贡献。

## 2. 五层理论建筑

### L1：外观等价类与合法干预

令 \(S\) 为物理场景，\(A\) 为外观，\(X=g(S,A,\xi)\) 为观测，\(C\) 为冻结 planner、native、
合法候选域与评估上下文。合法外观变换 \(T_a\) 必须保持 \((S,C,Y)\) 不变。若几何、候选、标签
或时间上下文随外观改变，该样本不是合法 appearance intervention。

目标不是无条件删除全部外观信息，而是在保持任务充分性时最小化无关外观信息：

\[
\min_R I(R;A\mid S,C)
\quad\text{s.t.}\quad
I(R;Y\mid C)=I(X;Y\mid C).
\]

该式是设计原则而非可直接估计的保证。实际只能分别检验 task correctness、paired-appearance
drift、correct-image necessity 和 downstream regret。

### L2：唯一部署估计对象

令完整 official 行为效用为 \(F_C(\tau)\)，其中
\(C_t=(\pi,\tau^{sel}_{t-1},\tau_{0,t},V_t,a_t)\)。唯一主估计对象为合法 ragged 域上的
native-relative margin：

\[
\delta^*_{i0}=F_C(\tau_i)-F_C(\tau_0),\qquad
i\in V_C\setminus\{0\}.
\]

所有 label、mask、loss、uncertainty、argmax、oracle 和 regret 必须使用同一个 \(V_C\)。绝对
candidate score、完整图 potential、factor head 或道路场只是可比较参数化，不是预设真理。

### L3：完整时域差异支持与视觉可观测性

在 \(\mathcal Z=\Omega\times[0,T]\) 上令 \(\mu_i\) 为轨迹 \(\tau_i\) 的完整时空软占据测度，
定义

\[
\nu_{i0}=\mu_i-\mu_0
=\nu^+_{i0}-\nu^-_{i0},\qquad
\mu^\cap_{i0}=\mu_i\wedge\mu_0.
\]

\(\nu^+\) 是 challenger-exclusive support，\(\nu^-\) 是 native-exclusive support；公共支持只
作为非线性上下文存在。该对象由 B3 common-min 与 BS2 late-fork 反例推出，而不是为了引入测度论。

### L4：关系估计与同时不确定集合

视觉编码器只运行一次。差异支持决定 query，全部视觉 token 保持为 key/value：

\[
E^r_{i0}=\operatorname{Attn}_\theta
\bigl(Q(\nu^r_{i0}),K(Z),V(Z);b^r_{geo}\bigr),
\quad r\in\{+,-,\cap\}.
\]

共享比较器经交换反对称化得到

\[
\widehat\delta_{i0}=\frac12\{
G_\theta(E^+,E^-,E^\cap,Q_{i0})-
G_\theta(E^-,E^+,E^\cap,Q_{0i})\}.
\]

同时输出正尺度 \(\sigma_{i0}\)，但 point uncertainty 不是 coverage。冻结模型后，必须用独立
recording-group calibration 构造对一组 appearances、anchors 和 challengers 同时成立的半径。

### L5：native-preserving 行为与实时执行

令 \(r_i=q\sigma_i\)，\(L_i=\widehat\delta_{i0}-r_i\)。只有全部合同通过且
\(\max_iL_i>0\) 时才选择最大 lower bound 的 challenger，否则保留 native。

在线复杂度合同为

\[
C_{AORTA}=C_{vis}+C_{support}(K,T)+C_{query}(K,M,d)+C_{cmp}(K,d),
\]

而不是每个候选重新编码图像的 \(K C_{vis}\)。完整 4 s 时域、困难候选、输入解码和同步开销均
不得从时延统计中删除。

## 3. 六个命题及其架构后果

### 命题 A：共同外观干扰的相对消除

若候选证据在合法外观变化下满足

\[
f_i(S,A,C)=u_i(S,C)+c(S,A,C),
\]

且 \(c\) 对同一面板全部候选共同，则

\[
f_i-f_0=u_i-u_0
\]

对该干扰精确不变。

- **网络后果：**输出 native-relative relation，而不是跨场景绝对 score。
- **边界：**共享参数不推出 shared nuisance；candidate-dependent distortion \(c_i-c_0\) 仍存在。
- **反证：**paired appearance relation drift、correct/mismatch/removed image、NavDream。

### 命题 B：加性任务泛函的差异支持充分性

若

\[
F_c(\tau)=\int_{\mathcal Z}c(z)\,d\mu_\tau(z)+b(S),
\]

则

\[
F_c(\tau_i)-F_c(\tau_0)
=\int c\,d\nu^+_{i0}-\int c\,d\nu^-_{i0}.
\]

- **网络后果：**正负支持分开编码，避免高维相减后的抵消；使用完整时域。
- **边界：**official NAVSIM utility 不是纯加性泛函，所以必须保留公共上下文与非线性比较器。
- **反证：**differential-support 与 ordinary candidate scorer、field readout 的 matched study。

### 命题 C：表示纤维与不可识别性

存在确定函数 \(h\) 使 \(\delta^*=h(R)\) 当且仅当真实 margin 在每个表示纤维上为常数：

\[
R(x)=R(x')\Longrightarrow\delta^*(x)=\delta^*(x').
\]

若存在相同 \(R\) 但 margin 符号相反的两个 admissible worlds，则任何只读取 \(R\) 的确定预测器
至少在其中一个世界出错。

- **网络后果：**差异支持只决定“问什么”，全局 token 仍可提供交通灯、遮挡和道路连续性。
- **拒判后果：**真正不可识别时不能靠更大的 point head 修复，只能增加信息或拒判。
- **边界：**直接投影率 \(\kappa=0\) 不等于整图不可识别，不能自动设 hard gate。

### 命题 D：任务正确性与外观稳定性的双向约束

若真实关系在合法外观轨道上不变，并且 clean 与 transformed 输入均满足

\[
\|\widehat\delta(X)-\delta^*\|_\infty\le\varepsilon,
\quad
\|\widehat\delta(T_aX)-\delta^*\|_\infty\le\varepsilon,
\]

则

\[
\|\widehat\delta(T_aX)-\widehat\delta(X)\|_\infty\le2\varepsilon.
\]

反之，零漂移不推出正确性，因为常数关系同样零漂移。

- **loss 后果：**task loss 是主目标，appearance orbit 只能是约束；不能只优化一致性。
- **实验后果：**必须同时报告 relation accuracy、drift、correct-image necessity 和 non-collapse。

### 命题 E：同时关系误差到选择 regret

把 native 记为 \(\delta^*_{00}=\widehat\delta_{00}=0\)。若

\[
\max_{i\in V_C}|\widehat\delta_{i0}-\delta^*_{i0}|\le\varepsilon,
\]

并按 \(\widehat\delta\) 选取 \(\widehat i\)，则

\[
\max_i\delta^*_{i0}-\delta^*_{\widehat i0}\le2\varepsilon.
\]

若真实最优与次优间隔大于 \(2\varepsilon\)，top choice 保持。

- **训练后果：**主 loss 与 native-relative margin 同型；平均 field MAE 不能替代 simultaneous error。
- **实验后果：**报告 group-level regret、top-margin、selection error，不能只报训练 MAE。

### 命题 F：同时下界上的条件性无伤害

若事件

\[
\mathcal E=\{ |\widehat\delta_{i0}-\delta^*_{i0}|\le r_i,\ \forall i\}
\]

成立，则任何满足 \(L_i=\widehat\delta_{i0}-r_i>0\) 的被接受 challenger 都严格优于 native。
若在固定模型与候选机制的可交换 calibration groups 上
\(\Pr(\mathcal E)\ge1-\alpha\)，错误改善声明概率至多为 \(\alpha\)。

- **行为后果：**evidence insufficient、数值失败、超时或 quantile 无穷时统一回退 native。
- **边界：**source exchangeability 不覆盖 NavDream、新 planner 或新 candidate policy；这些只报告
  empirical coverage、risk–coverage 和 failure strata。

## 4. Loss 不是拼盘：每一项只承担一种职责

主目标为

\[
\mathcal L_{task}=
\frac1{N_{legal}}\sum_{i\in V\setminus\{0\}}
\rho(\widehat\delta_{i0}-\delta^*_{i0}).
\]

外观轨道约束为

\[
\mathcal L_{orbit}=
\frac1{N_{paired}}\sum_{a,i}
\rho\!\left(\widehat\delta_{i0}(T_aX)-
\operatorname{sg}[\widehat\delta_{i0}(X)]\right).
\]

优化合同是

\[
\min_\theta\ \mathbb E[\mathcal L_{task}]
\quad\text{s.t.}\quad
\mathbb E[\mathcal L_{orbit}]\le\delta_A,
\]

而不是凭结果调 \(\lambda\) 的任意加权和。scale 使用预注册 proper residual score，只学习相对
sharpness；最终半径来自独立 group calibration。若 scale arm 不改善 sharpness/risk–coverage，则
删除 learned scale，使用更简单的固定或分层尺度。

禁止默认加入 contrastive、cycle、smoothness、factor、depth、domain-adaptation 等辅助 loss。只有
独立失败表明主合同缺少相应信息时，才能注册新项，并必须有删除消融。

## 5. 为什么网络必须这样设计

| 结构 | 直接原因 | 预期作用 | 删除/替代检验 |
|---|---|---|---|
| full-horizon soft footprint | BS2 有晚期分叉 | 保留 2.5–4 s 决策差异 | full vs prefix |
| signed differential support | B3 common-min 折叠 | 让 query 聚焦改变候选关系的区域 | differential vs independent scorer |
| positive/negative 分支 | 正负高维证据可抵消 | 区分 challenger 新增风险与 native 放弃区域 | separated vs early subtraction |
| global visual key/value | D0 只证明直接投影缺失 | 允许上下文桥接不可直接投影区域 | global KV vs hard crop |
| shared antisymmetric comparator | 输出是有向 relation | 交换候选时符号严格翻转 | swap metamorphic test / free pair head |
| task + orbit constraint | 稳定可坍缩、正确可脆弱 | 同时保留任务信息并控制外观漂移 | remove task / remove orbit |
| point/scale 分离 | point accuracy 不等于 coverage | 允许独立校准和风险—覆盖权衡 | fixed vs learned scale |
| native fallback | 存在不可识别世界与有限样本误差 | 证据不足时避免无根据干预 | forced replace vs selective |
| shared backbone + vectorized queries | 在线预算 | 避免 \(K\) 次视觉编码 | latency profile / K scaling |
| previous-action context | official full target 函数上依赖 two-frame comfort | 最小geometry token始终属于估计上下文；复杂temporal branch按需 | current-only vs token vs temporal encoder |

该表是复杂度许可证：没有“直接原因 + 删除检验”的模块不得进入主图。

## 6. 与最近工作的精确边界

| 工作族 | 已覆盖内容 | AORTA 不得声称 | AORTA 必须额外证明 |
|---|---|---|---|
| The Constant Eye | appearance-only shift、冻结 DINO 接口、跨 planner NavDream | 使用 DINO、零样本外观鲁棒、plug-and-play | 稳定感知之后，decision-difference relation 的必要性与行为增益 |
| NMP/MP3/FlowDrive/READ | field/cost/occupancy 接规划 | 视觉场、轨迹积分或 planner-aware representation 本身 | field readout 无法保留的候选差异，以及 relation operator 的增量 |
| GTRS/DA-WAM | trajectory token、candidate-conditioned attention、factor/utility scorer | cross-attention、candidate scorer、hard negative | signed differential query 相对普通 scorer 的 matched advantage |
| RiskWorld/Expand-and-Prune | candidate reuse、selective replacement、confidence fallback | 冻结 planner 辅助或拒判本身 | relation-level simultaneous error 与不可识别性驱动的可靠仲裁 |

文献边界以 `references/2026-10-25_relation_story_sources.md`、
`references/2026-10-25_b3_decision_operator_sources.md` 和
`references/2026-10-25_preference_quotient_novelty_boundary.md` 为可追溯入口。

## 7. 一篇顶会论文只保留一主一从贡献

1. **主贡献候选：appearance-interventional differential-support relation learning。** 在完整时域
   native–challenger 差异支持上查询全局视觉证据，直接预测合法任务 margin。
2. **从属机制候选：evidence-qualified native-preserving arbitration。** 使用表示 provenance 与
   group-simultaneous residual 分开记账，只在 lower bound 为正时干预。

若 matched ordinary/DA-WAM-style scorer 与主方法相当，删除“差异支持网络创新”，保留最强简单
scorer 并降低论文定位；若 selective arm 近乎全拒判或不降低 false intervention，删除从属机制。

## 8. 90 秒 Oral / Poster 讲述顺序

1. **现象：**天气只改变外观，却能改变冻结 planner 的轨迹。
2. **反常：**稳定甚至 perfect 的道路场仍会把决策不同的候选压成相同 scalar；短前缀还删除晚期
   分叉。
3. **洞见：**需要稳定的不是整幅感知，而是 native 与 challenger 在哪里产生决策差异的证据。
4. **方法：**完整时空 signed support 产生 query，全局视觉 token 回答，反对称比较器输出相对
   official-task margin。
5. **可信动作：**同时 lower bound 不为正就不干预。
6. **证据：**correct image、matched scorer、NavDream、三 planner、hard-failure net reduction、clean
   no-harm 和 batch-1 latency 共同闭环。
7. **边界：**非反应式 PDMS 与 source calibration 不被描述为真实道路安全保证。

如果无法在 90 秒内用这七步解释一个模块，它不属于主方法。

## 9. 决定论文生死的证据链

按顺序执行，前一关失败不得用后一关平均分掩盖：

1. B4R1 合法候选域存在非平凡 one-frame oracle；
2. 以 augmented-context full official target 冻结最终 label；报告 one-frame `1/7` 符号证书覆盖与
   temporal counterexamples；
3. correct image 优于 geometry-matched mismatch、removed 和 trajectory-only；
4. differential-support 在参数、FLOP、token、数据、seed 匹配下胜过 field、ordinary scorer 与
   DA-WAM/GTRS-style scorer；
5. 未见场景与 NavDream 上 relation accuracy、appearance drift、ranking regret 同时成立；
6. 三类冻结 planner 上 \(G_{net}>0\)、OOD official metric 改善、clean non-inferiority；
7. batch-1 端到端 P50/P95/P99、峰值显存和候选数缩放通过冻结预算；
8. 独立机械审计、claim audit 与强反方 review 无致命未解决问题。

## 10. 当前结论

数学接口已经形成层次清楚的条件闭环，但经验前提仍未闭合。当前可以严谨地说：

- 为什么要学习 relation 而不是绝对场已有反例与条件命题；
- 为什么要使用 full-horizon differential support 已有失败驱动依据；
- 为什么需要拒判已有不可识别性与同时误差定理；
- 为什么共享视觉编码与向量化 query 是实时性所需已有复杂度合同。

当前不能说：该网络比最近方法更好、能在 NavDream 泛化、能改善 official driving outcome、满足
clean no-harm 或具备实时部署资格。下一步仍是 B4R1；未经明确授权不得运行。

数据无关参考实现位于 `src/aorta_drive/theory/relation_guarantees.py`。它与历史绝对安全区间仲裁
严格分离，并由 `tests/gate_b/test_relation_closure_theory.py` 验证严格正 lower-bound 选择、同时
事件上的真实改善、task correctness 到 `2 epsilon` appearance drift、稳定不推出正确的坍缩反例、
group maximum 以及42组95%有限/99%无穷的样本量边界。这些测试只验证合同，不增加经验主张。
