# AORTA-v8.3.2 Formal Theory Contract

> 本文只陈述条件定理。它不证明视觉 predictor 满足假设，也不提供 arbitrary-OOD coverage。

## 1. Assumptions and notation

- \(G=(\Omega,E)\) 是有限、连通、无向图；边权有限且非负。
- \(D\) 是任意定向 incidence matrix，绝对 edge constraints 与方向无关。
- \(u:\Omega\to\mathbb R\) 是预测节点势。
- \(\widetilde d:\Omega\to\mathbb R\) 是用于训练和校准的 node-label field。
- \(\phi:\mathcal R\to\mathbb R\) 是连续 nominal-road signed distance，在认证 ROI 上为 1-Lipschitz。
- 对参与 pair 的节点，

\[
|\widetilde d(q)-\phi(q)|\le\varepsilon_{map}(q).
\]

- \(A_j,A_0\subseteq\Omega\) 是非空 graph-node sets，\(F_j,F_0\subseteq\mathcal R\) 是连续 nominal footprints。
- \(\overline F_j\) 是 challenger footprint 的确定性外包络，满足 \(F_j\subseteq\overline F_j\)。
- 网络、图、candidate interface、时刻集、阈值和所有选择规则在 calibration 前冻结。

## 2. Quotient invariance

### Lemma 1

若 \(G\) 连通，则

\[
\ker D=\operatorname{span}\{\mathbf1\}.
\]

对任意 \(c\in\mathbb R\) 和非空 \(A,B\subseteq\Omega\)：

\[
[\min_A(d+c\mathbf1)-\min_B(d+c\mathbf1)]
=\min_A d-\min_B d.
\]

因此 graph gradients 确定了当前相对 set-min task 所需的场信息，至多缺少一个不影响该 task 的共同 offset。该结论不表示绝对安全零点对其他任务不重要。

## 3. Exact robust set-min theorem

令 edge budgets \(w_e\ge0\)，并定义

\[
\mathcal U(u,w)
=\{d:\ |D(d-u)|_e\le w_e,\ \forall e\in E\}.
\]

### Theorem 1

对任意非空 \(A,B\subseteq\Omega\)：

\[
\inf_{d\in\mathcal U(u,w)}
[\min_A d-\min_B d]
=\min_{a\in A}\max_{b\in B}
[u_a-u_b-\operatorname{dist}_w(a,b)].
\]

### Proof

写 \(d=u+r\)。对固定 \(r\)：

\[
\min_A d-\min_B d
=\min_{a\in A}\max_{b\in B}(d_a-d_b).
\]

因为 \(A\) 有限，且 \(r\) 的可行集合不依赖 \(a\)：

\[
\inf_r\min_{a\in A}g_a(r)
=\min_{a\in A}\inf_r g_a(r).
\]

固定 \(a\)，利用共同 shift 令 \(r_a=0\)。edge constraints 沿任意路径求和得到

\[
r_b\le\operatorname{dist}_w(a,b),
\]

故

\[
u_a-u_b+r_a-r_b
\ge u_a-u_b-\operatorname{dist}_w(a,b).
\]

取 distance potential

\[
r_v=\operatorname{dist}_w(a,v).
\]

由三角不等式，\(|r_v-r_z|\le w_{vz}\) 对所有边成立；且它对所有 \(b\) 同时达到上述下界。因此固定 \(a\) 的 infimum 等于相应 maximum，再对 \(a\) 取 minimum 即得证。∎

### Corollary 1 — min-plus form

定义

\[
\psi_B(v)=\min_{b\in B}
[u_b+\operatorname{dist}_w(v,b)].
\]

则

\[
L_G(A,B)=\min_{a\in A}[u_a-\psi_B(a)].
\]

每个 native set 只需一次 multi-source shortest-path transform。

### Corollary 2 — inclusion

若 \(A\subseteq B\)，则 \(L_G(A,B)\ge0\)；若 \(A=B\)，则 \(L_G(A,B)=0\)。该收益是 geometry-only law，不是视觉算法贡献。

## 4. Continuous lifting with an asymmetric geometry contract

旧的“challenger 节点既位于真实 footprint 内、又构成其内部 \(h\)-net”不是生产接口所必需，且难以从固定栅格确定性建立。V8.3.2 改用可验证的非对称合同：challenger 使用外覆盖，native 使用内部物理点。

### Assumption G2

- \(F_j\subseteq\overline F_j\)，其中 \(\overline F_j\) 是由冻结轨迹插值、车体与姿态误差协议构造的外包络；
- 对每个 \(x\in\overline F_j\)，存在 \(a\in A_j\) 使 \(\lVert x-a\rVert_2\le h_j\)；不要求 \(A_j\subseteq F_j\)；
- \(A_0\subseteq F_0\)，即每个 native 节点都是至少一个真实 native footprint 内的物理点；
- footprints 完全位于认证 ROI；
- \(\phi\) 为 1-Lipschitz。

定义

\[
\varepsilon_{map,j}=\max_{q\in A_j}\varepsilon_{map}(q),
\qquad
\varepsilon_{map,0}=\max_{q\in A_0}\varepsilon_{map}(q).
\]

### Theorem 2 — asymmetric continuous lifting

若 \(L_G(A_j,A_0)\) 是 node-label contrast 的有效 lower bound，则

\[
\Delta_{j0}^{cont}
\ge L_G(A_j,A_0)
-h_j
-\varepsilon_{map,j}
-\varepsilon_{map,0}.
\]

### Proof

由 \(F_j\subseteq\overline F_j\)、外覆盖性质与 1-Lipschitz：

\[
\inf_{x\in F_j}\phi(x)
\ge\inf_{x\in\overline F_j}\phi(x)
\ge\min_{a\in A_j}\phi(a)-h_j
\ge\min_{a\in A_j}\widetilde d(a)
-\varepsilon_{map,j}-h_j.
\]

由 \(A_0\subset F_0\)：

\[
\inf_{x\in F_0}\phi(x)
\le\min_{b\in A_0}\phi(b)
\le\min_{b\in A_0}\widetilde d(b)
+\varepsilon_{map,0}.
\]

两式相减并代入 discrete lower bound 得证。∎

注意，challenger 外包络可能引入额外低势节点，只会使 lower bound 更保守；native 若改用外包络节点则可能错误降低被减数，不能使用同一构造。

### Proposition 1 — interval swept-footprint outer cover

令车体 \(K\subset\mathbb R^2\) 以 rear axle 为原点，且 \(R_K=\sup_{k\in K}\lVert k\rVert_2\)。对相邻姿态使用

\[
p(s)=(1-s)p_0+sp_1,
\qquad
\theta(s)=(1-s)\theta_0+s\theta_1,
\qquad s\in[0,1],
\]

其中 yaw 已连续展开且 \(|\theta_1-\theta_0|\le\pi\)。令中点姿态为 \((p_m,\theta_m)\)，并定义

\[
\delta
=\frac12\lVert p_1-p_0\rVert_2
+2R_K\sin\frac{|\theta_1-\theta_0|}{4}
+\varepsilon_{pose}.
\]

则包含姿态误差半径 \(\varepsilon_{pose}\) 的整段真实扫掠体满足

\[
F_{[0,1]}\subseteq R(\theta_m)K+p_m\oplus B_2(\delta).
\]

证明：任意 \(s,k\) 相对中点变换的平移偏差不超过 \(\lVert p_1-p_0\rVert_2/2\)。旋转偏差满足

\[
\lVert R(\theta(s))k-R(\theta_m)k\rVert_2
=2\lVert k\rVert_2\sin\frac{|\theta(s)-\theta_m|}{2}
\le2R_K\sin\frac{|\theta_1-\theta_0|}{4}.
\]

再用三角不等式并加入姿态误差球即得证。∎

### Proposition 2 — fixed-grid cover

令边长为 \(r\) 的闭方格完全分割认证 ROI，格心为固定图节点，\(h=r/\sqrt2\)。对每个 Proposition 1 外包络，选择所有满足

\[
\operatorname{dist}(a,R(\theta_m)K+p_m)\le\delta+h
\]

的格心 \(a\)。若外包络完整位于 ROI，则所选格心是该外包络的 \(h\)-cover：任意外包络点 \(x\) 所在方格的中心 \(a_x\) 满足 \(\lVert x-a_x\rVert_2\le h\)，且

\[
\operatorname{dist}(a_x,R(\theta_m)K+p_m)
\le\operatorname{dist}(x,R(\theta_m)K+p_m)+h
\le\delta+h.
\]

因此 \(a_x\) 必被选择。多段轨迹取各段节点并集；任何外包络越出 ROI、节点集为空、yaw 未展开或协议不一致时必须拒绝认证，而不是裁剪或经验补点。

若一般数值实现输出 \(\widehat L_G\)，且已证明

\[
L_G\ge\widehat L_G-\varepsilon_{num},
\]

则进一步扣除 \(\varepsilon_{num}\)。

### Lemma 2 — outward-safe binary64 implementation

对有限 binary64 输入，若每次 Dijkstra path addition 都向 \(+\infty\) 外舍入、最终 \(u_a-\widehat\psi_B(a)\) 向 \(-\infty\) 外舍入，并在任何非有限结果时拒绝认证，则

\[
\widehat\psi_B(v)\ge\psi_B(v),
\qquad
\widehat L_G(A,B)\le L_G(A,B).
\]

因此该特定实现相对于“以输入 binary64 数值作为精确实数”的数学问题没有乐观误差，可取 \(\varepsilon_{num}=0\)。证明由每次上舍入保持 path-length upper bound、minimum 保持 upper bound，以及最终下舍入保持 lower bound 直接得到。此结论不覆盖输入量化、坐标误差、编译器违反 IEEE-754 语义或其他 solver；这些误差必须另行记账。

## 5. Source-session calibration

对独立 source calibration sessions \(Z_1,\ldots,Z_n\)：

\[
E_i=\max_{t\in T(Z_i)}\max_{e\in E}
\frac{|(Du_{it}-D\widetilde d_{it})_e|}{s_e}.
\]

取

\[
k=\lceil(n+1)(1-\alpha)\rceil,
\qquad q_\alpha=E_{(k)}.
\]

### Theorem 3

若 \(Z_1,\ldots,Z_n,Z_{n+1}\) 交换，且 \(k\le n\)，则

\[
P\left(
\forall t\in T(Z_{n+1}),e\in E:
|(Du-D\widetilde d)_e|\le q_\alpha s_e
\right)\ge1-\alpha.
\]

这是 calibration sessions 与新 source session 联合随机性的边际保证。

### Lemma 3 — finite-threshold sample size

\(k\le n\) 的最低样本量为

\[
n\ge\left\lceil\frac1\alpha\right\rceil-1.
\]

该条件只保证非随机阈值存在，不保证阈值稳定或有用。

## 6. Relative-intervention risk theorem

令 computed certificate 为

\[
L_{j0}^{cert}
=\widehat L_G
-h_j
-\varepsilon_{map,j}
-\varepsilon_{map,0}
-\varepsilon_{num}.
\]

冻结 \(\gamma_{task}\ge0\)，且只有 \(L_{j0}^{cert}\ge\gamma_{task}\) 的 pair 可被授权。定义 session failure

\[
R_\gamma(Z)=
\mathbf1\{\exists t\in T(Z):
I_t=1,
\Delta_{sel,0,t}^{cont}<\gamma_{task}\}.
\]

### Theorem 4

若 Theorem 3 的 field event 成立，且同一 session 所有授权 pair 的 geometry、teacher-error 与 numerical contracts 都有效，则 \(R_\gamma=0\)。

若这些非 field contracts 的 session-level 错误接受事件概率至多为 \(\beta\)，则

\[
P(R_\gamma=1)
\le\min\{1,\alpha+\beta\}.
\]

### Proof

在 field event 上，\(\widetilde d\in\mathcal U(u,q_\alpha s)\)，故 Theorem 1 给出有效 discrete lower bound。非 field contracts 有效时，Theorem 2 与 numerical budget 给出

\[
\Delta_{j0}^{cont}\ge L_{j0}^{cert}\ge\gamma_{task}
\]

对所有授权 pair 同时成立。因此

\[
\{R_\gamma=1\}
\subseteq\mathcal F^c\cup\mathcal G^c,
\]

应用 union bound 得证。∎

### Corollary 3 — safe-to-unsafe event

因为 \(\gamma_{task}\ge0\)，若 native continuous nominal margin 非负，则授权 challenger 的 margin 也非负。因此原 safe-to-unsafe failure 是 \(R_\gamma\) 的推论。

## 7. Paired appearance diagnostic

对共享同一 teacher 的 paired observations \(X,X'\)，定义

\[
r_e(X)=\frac{(Du(X)-D\widetilde d)_e}{s_e},
\]

\[
\rho(X,X')
=\max_e\frac{|D[u(X')-u(X)]_e|}{s_e}.
\]

### Lemma 4

\[
\|r(X')\|_\infty
\le\|r(X)\|_\infty+\rho(X,X').
\]

证明是逐边三角不等式。对 session 时刻再取 maximum 得到 session 版本。

该引理只确定一个可测的 appearance-gradient stability 机制；经验小 \(\rho\) 不提供 target/OOD coverage。若要从 source 推导 target guarantee，必须另加可审计的分布关系与 session-level \(\rho\) 上界。

## 8. Explicit limits

- 定理不说明 edge-trained neural network 会得到小的 session maximum。
- 定理不说明 source 与 target appearance sessions 交换。
- 定理不覆盖 collision、dynamic objects、traffic rules、comfort 或 executed footprint。
- 定理不保证 candidate provider 会产生更优候选。
- 标准 difference constraints 与 min-plus solver 不是独立原创性依据。
