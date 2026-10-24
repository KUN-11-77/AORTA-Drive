# AORTA 差异支持关系架构候选 v2

状态：`ADVERSARIAL_REDESIGN_CANDIDATE__NO_TRAINING_AUTHORIZED__B4R1_REQUIRED`。

本文不是最终方法，也不授权训练。它先否定 v1 中无法兑现的机制解释，再给出一个由 B3、BS2 与
BS4A-D0 反例共同推出的最小候选架构。若 B4R1 不存在合法且非平凡的 preference oracle，本文整体
退役。

## 1. v1 的两个结构漏洞

v1 写成

\[
e_i=\Psi(Z,A_i),\quad d_{i0}=e_i-e_0,\quad
\widehat\delta_{i0}=w^\top\operatorname{LN}(d_{i0}+\Omega(Z,A_0,A_i,Q_{i0})).
\]

它有两个无法靠措辞解决的问题。

### 1.1 共享参数不推出共同干扰

即使图像 token 的外观变化是共享的 \(Z'=Z+b\)，非线性候选查询一般满足

\[
\Psi(Z+b,A_i)-\Psi(Z,A_i)
\ne
\Psi(Z+b,A_0)-\Psi(Z,A_0).
\]

因此 \(\Psi(Z,A_i)-\Psi(Z,A_0)\) 只在**查询输出层已经具有候选共同加性偏置**时才精确消除
外观项。参数共享本身不提供该性质。v1 的 common-mode theorem 只能作为数据生成假设，不能被写成
网络保证。

### 1.2 自由 interaction 可以绕过 difference

若 \(\Omega\) 不受结构限制，它可以单独表达整个 margin；此时即使 \(d_{i0}=0\)，模型仍可完成
任务。于是 “difference 负责鲁棒、interaction 负责残差” 不可识别，分支消融以外没有数学依据。
为避免用人为 norm penalty 掩盖问题，v2 删除该双分支解释，而不是再加一个权重。

## 2. 从已观察反例推出新的输入对象

三条证据共同指向候选对的**差异支持域**：

1. B3：共同 scalar minimum 支配 2,033 个 common-min pairs，temporal fork 只恢复 170 个；
2. BS2：2.5 s 前缀对 128 pairs 完全不可区分，36/42 groups 存在晚期分叉；
3. BS4A-D0：8 个几何不同 pair 的非空对称差区域没有直接投影视觉证据。

因此网络不应先构造全局 field 再压成一个 minimum，也不应只对每条候选独立 pooling。它应直接
回答：**challenger 相对 native 新增/移除了哪些完整时空支持，这些支持上有什么可见证据？**

## 3. 数学对象：轨迹占据测度的 Jordan 分解

令 \(\mathcal Z=\Omega\times[0,T]\) 为空间—时间域，\(\mu_i\) 为车辆沿完整轨迹 \(\tau_i\)
形成的有限非负软占据测度。定义 signed contrast measure

\[
\nu_{i0}=\mu_i-\mu_0.
\]

它具有唯一的 Jordan 分解

\[
\nu_{i0}=\nu_{i0}^{+}-\nu_{i0}^{-},\qquad
\nu_{i0}^{+}\perp\nu_{i0}^{-},
\]

其中 \(\nu^+\) 表示 challenger 独占支持，\(\nu^-\) 表示 native 独占支持。另定义公共测度

\[
\mu_{i0}^{\cap}=\mu_i\wedge\mu_0.
\]

离散实现使用冻结时空 lattice 上的软 footprint 权重：

\[
\nu^+=[\mu_i-\mu_0]_+,
\quad
\nu^-=[\mu_0-\mu_i]_+,
\quad
\mu^\cap=\min(\mu_i,\mu_0).
\]

该对象与候选顺序无关：交换 \(i,0\) 会交换 \(\nu^+,\nu^-\)，并令
\(\nu_{0i}=-\nu_{i0}\)。它保留完整 4 s 分叉，不允许共同前缀通过 global minimum 支配关系。

### 命题 1：加性任务泛函的精确差分充分性

若某类任务效用可写成

\[
U_c(\tau)=\int_{\mathcal Z}c(z)\,d\mu_\tau(z)+b(S),
\]

则

\[
U_c(\tau_i)-U_c(\tau_0)
=\int c\,d\nu_{i0}
=\int c\,d\nu^+_{i0}-\int c\,d\nu^-_{i0}.
\]

所以公共支持与场景共同 offset 精确抵消。该结论是测度线性恒等式，不是新的一般定理；它只证明
差异支持是加性 cost/risk 类的自然充分统计量。NAVSIM PDM 不是纯线性泛函，因此网络仍需公共
上下文与非线性交互，且必须用 oracle/regret 实验检验充分性。

## 4. Differential-Support Relation Operator

冻结视觉编码器只运行一次：

\[
Z=E_{vis}(X)=\{(z_p,\ell_p,m_p)\}_{p=1}^{P},
\]

其中 \(\ell_p\) 是 token 的冻结几何定位，\(m_p\in\{0,1\}\) 是直接投影视觉证据 mask。对每个
候选对，将 \(\nu^+,\nu^-,\mu^\cap\) 通过冻结相机/ego 几何映射为带时间编码的 token query，得到

\[
E^+_{i0}=\operatorname{Agg}_\theta(Z;\nu^+_{i0}),\quad
E^-_{i0}=\operatorname{Agg}_\theta(Z;\nu^-_{i0}),\quad
E^\cap_{i0}=\operatorname{Agg}_\theta(Z;\mu^\cap_{i0}).
\]

正负支持必须分开聚合，禁止先相减高维 feature 后因符号抵消丢失信息。完整相对运动编码为
\(Q_{i0}=E_{rel}(\tau_i\ominus\tau_0)\)。用一个共享有序比较器 \(G_\theta\) 构造

\[
g_{i0}=G_\theta(E^+_{i0},E^-_{i0},E^\cap_{i0},Q_{i0}),
\]

并以反对称化输出

\[
\boxed{
\widehat\delta_{i0}
=\frac12\left[
G_\theta(E^+,E^-,E^\cap,Q_{i0})
-G_\theta(E^-,E^+,E^\cap,Q_{0i})
\right].
}
\]

由构造直接得到

\[
\widehat\delta_{0i}=-\widehat\delta_{i0},\qquad
\widehat\delta_{00}=0.
\]

这只提供 relation type-safety，不自动提供准确性或新颖性。它与 DA-WAM 的关键区别候选不是“也做
candidate-conditioned attention”，而是查询对象由单候选 future latent 改为 native–challenger 的
signed differential support，并把不可观测差异显式接入拒判。该区别必须由 matched DA-WAM/GTRS
style scorer 对照赢得资格。

## 5. 可观测性与拒判不是附加模块

令 \(|\nu|=\nu^++\nu^-\)，定义差异支持的直接证据率

\[
\kappa_{i0}
=\frac{\int m(z)\,d|\nu_{i0}|(z)}{\int d|\nu_{i0}|(z)}.
\]

若 \(|\nu|=0\)，两条轨迹在该支持定义下相同，关系只能由相对运动/非占据任务因素决定；该类样本
单独记账。若 \(|\nu|>0\) 且 \(\kappa=0\)，系统必须拒绝基于视觉差异作正向干预。

### 命题 2：不可观测差异上的符号不可识别

设观测只决定集合 \(O\subset\mathcal Z\) 上的任务场，而
\(|\nu_{i0}|(\mathcal Z\setminus O)>0\)。若允许的任务场类包含一个支撑在
\(\mathcal Z\setminus O\) 上、且 \(\int h\,d\nu_{i0}\ne0\) 的有界函数 \(h\)，则存在两个场
\(c_+=c_0+\lambda h\) 与 \(c_-=c_0-\lambda h\)，它们在全部已观测证据上完全相同，却对
\(U_c(\tau_i)-U_c(\tau_0)\) 给出相反符号。因此任何只读取观测证据的确定性模型都不能在该函数类
上统一保证关系符号正确。

**含义。** D0 的无证据差异不是靠更大 MLP 可以解决的统计误差，而是可识别性缺失。v2 必须把
support mask、\(\kappa\) 与 missing pattern 输入 scale/拒判路径；完全无差异证据时硬拒判。部分可见
时不得宣称分布自由保证，只能依赖独立 group-level calibration 与 OOD empirical risk–coverage。

## 6. 训练目标：正确、同轨道稳定、非平凡

主任务损失仍与合法 ragged estimand 同型：

\[
\mathcal L_{task}
=\frac1{|V|-1}\sum_{i\in V\setminus\{0\}}
\rho(\widehat\delta_{i0}-\delta^*_{i0}).
\]

同场景外观轨道只约束 relation：

\[
\mathcal L_{orbit}
=\frac1{|\mathcal A|(|V|-1)}
\sum_{a,i}\rho\bigl(
\widehat\delta_{i0}(T_aX)-
\operatorname{sg}[\widehat\delta_{i0}(X)]
\bigr).
\]

优化仍采用

\[
\min_\theta\;\mathbb E[\mathcal L_{task}]
\quad\text{s.t.}\quad
\mathbb E[\mathcal L_{orbit}]\le\delta_A.
\]

为排除 trajectory-only shortcut，correct-image、geometry-matched mismatch 与 removed-image 必须作为
预注册必要性检验。是否把 mismatch 用作训练负样本必须另做 source-only matched study；v2 默认不
加入，避免把容易识别的跨场景分布差异变成捷径。primal–dual、Huber 和反对称化本身都不是创新。

## 7. 不确定性与选择规则

point relation 与 positive scale 分开输出。独立 calibration 对每个 recording group 取

\[
R_g=\max_{a,i}
\frac{|\widehat\delta_{gai0}-\delta^*_{gai0}|}{\sigma_{gai}},
\]

再使用已冻结的 group-level split quantile 得到 \(r_i=q\sigma_i\)。最终 lower bound 为

\[
L_i=\widehat\delta_{i0}-r_i.
\]

只有合法域、差异支持、数值、时延与 calibration contract 全部通过且 \(\max_iL_i>0\) 时才覆盖
native。support rejection 与 statistical rejection 必须分别报告：前者是可观测性资格，后者是误差
覆盖资格。

## 8. 实时复杂度

视觉 backbone 共享一次；\(\mu_i\) 为每个候选预计算一次，和 native 做稀疏差分。若每个候选差异
支持保留 \(M_i\) 个稀疏 query，则

\[
C_{total}=C_{DINO}+O(KT)+O\!\left(\sum_iM_i d\right)+O(Kd^2).
\]

不得用截断 4 s、删除困难候选或离线 cache 伪造在线时延。必须与普通 DA-WAM/GTRS-style
candidate scorer 在同一 backbone、token 数、参数/FLOP 与 batch-1 环境报告 P50/P95/P99、显存和
候选数缩放。

## 9. 决定架构生死的匹配实验

| Arm | 唯一变化 | 被检验的命题 |
|---|---|---|
| A | trajectory-only native-relative MLP | 视觉是否必要 |
| B | 普通 candidate-independent scorer 后做差 | 差异支持是否优于“每候选独立评分” |
| C | DA-WAM/GTRS-style candidate-conditioned scorer | 是否只是普通 cross-attention |
| D | v2 differential-support operator | 主候选 |
| E | D 去掉公共上下文 | 非线性任务是否需要 shared context |
| F | D 用 2.5 s prefix | 完整时域是否必要 |
| G | D 去掉 orbit constraint | relation-level appearance constraint 是否必要 |
| H | D 忽略 support mask | 可观测性拒判是否降低 false intervention/clean harm |

所有 arms 共享合法候选、标签、backbone、训练数据、optimizer、seeds 与近似参数/FLOP。primary
endpoint 是 held-out/NavDream group-level regret、selection error、false intervention 与
hard-failure net reduction；训练 loss、feature drift 与 attention 图只作诊断。

## 10. 当前裁决

v2 比 v1 更接近“由失败推出设计”：

- Jordan/signed support 对应共同路径消除；
- 完整时空 measure 对应晚期分叉；
- support mask 与不可识别性命题对应 D0 无证据反例；
- antisymmetric operator 对应 relation 类型；
- group-simultaneous lower bound 对应 native-preserving action。

但它仍然不是已成立创新。NMP 已覆盖 cost-volume trajectory integral，DA-WAM 已覆盖
candidate-conditioned visual scoring，RiskWorld 已覆盖 selective replacement。只有 D 在严格匹配下
同时胜过 B/C，并在 NavDream、跨 planner、clean no-harm 与时延上闭环，才能把“差异支持关系算子”
写入标题和摘要；否则回退为最强简单 scorer，承认网络创新不足。

