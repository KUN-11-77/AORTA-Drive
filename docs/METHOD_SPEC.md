# AORTA-Drive Method Specification v0.4

> Canonical method version: AORTA-v8.3.2. Formal proofs and assumptions are in [`THEORY.md`](THEORY.md).

## 1. Task objects

令 \(G=(\Omega,E)\) 为有限、连通、无向的 4-neighbor ego-centric graph。连续 nominal-road signed-distance teacher 为 \(\phi_S:\mathcal R\to\mathbb R\)，在认证 ROI 上为 1-Lipschitz。用于训练与校准的 node labels 为 \(\widetilde d_S(q)\)，并要求

\[
|\widetilde d_S(q)-\phi_S(q)|
\le\varepsilon_{map}(q).
\]

冻结 tracker 将轨迹变为非空 swept-node set \(A(\tau)\)。连续 nominal-road margin 为

\[
M_S^{cont}(\tau)
=\inf_{x\in F(\tau)}\phi_S(x)-m_{buf}.
\]

它不等价于 collision、traffic-rule、comfort、tracking 或完整闭环安全。

## 2. Visual node evidence protocol

冻结 DINOv3-H+ 产生多相机历史 token fields。对当前 ego-frame 节点 \(q=(x,y)\)：

1. 使用预注册参考道路表面 \(z_{ref}(q)\) 将节点提升到三维；
2. 使用冻结历史 ego poses 与 camera extrinsics 转换坐标；
3. 使用 intrinsics 与实际 crop/resize transform 投影到 token grid；
4. 仅对正深度、frustum-valid 投影做 bilinear sampling；
5. 无有效投影时使用 missing-evidence token，并保留有效性 mask。

frustum-valid 不代表已知 occlusion；V0 没有真实深度时不得声称知道遮挡。

固定 evidence pack 为

\[
\mathcal F(X,q)
=\operatorname{Pack}
\{\operatorname{bilinear}(F_{c,k},\pi_{c,k}(q)),m_{c,k}(q)\}_{c,k}.
\]

唯一 trainable component 为共享 decoder

\[
u_\theta(X,q)
=f_\theta(\mathcal F(X,q),\operatorname{PE}(q),m(q)).
\]

相机顺序、历史长度、token layers、reference surface、projection convention、missing token 和聚合方式在 calibration 前冻结。

## 3. Graph-gradient training

V0 使用固定物理 edge scale \(s_e=\ell_e/\ell_{ref}>0\)。令 \(D\) 为 incidence matrix：

\[
\mathcal L_{edge}
=|E|^{-1}\sum_e
\operatorname{Huber}
\left(
\frac{(Du_\theta-D\widetilde d_S)_e}{s_e}
\right).
\]

共同 offset 位于 \(\ker D=\operatorname{span}\{\mathbf1\}\) 中，不影响 relative set-min task。average edge loss 不推出 session maximum 很小，也不推出 appearance-OOD stability；Gate B 直接审计 tail residual 和证书锐度。

## 4. Frozen candidate interface

AORTA 只做 arbitration。冻结接口为

\[
\mathcal C_\xi(O_t)
=\{\tau_0,\tau_1,\ldots,\tau_{K_t}\},
\qquad K_t\le K_{max}.
\]

- \(\tau_0\) 始终为 native；
- challenger 可来自预声明 frozen planners 或 deterministic kinematic library；
- 只能使用部署可得 observation/state；
- 禁止 GT map、target labels 与 paired clean target image；
- horizon、budget、seed、ordering、eligibility 与 provenance 在 calibration 前冻结；
- 所有 methods 使用相同 candidates 与 compute budget。

## 5. Data partition and calibration

source train、development、calibration、test sessions 严格分离。模型、checkpoint、图、\(\alpha\)、\(\gamma_{task}\)、candidate interface 和所有选择规则在首次读取 calibration scores 前冻结。

\[
E_i=\max_{t\in T(Z_i)}\max_{e\in E}
\frac{|(Du_{it}-D\widetilde d_{it})_e|}{s_e}.
\]

\[
k=\lceil(n+1)(1-\alpha)\rceil,
\qquad q_\alpha=E_{(k)}.
\]

若 \(k>n\)，返回不可部署。有限阈值至少需要

\[
n\ge\left\lceil\frac1\alpha\right\rceil-1
\]

个独立 calibration sessions。帧和同 route corruption copies 不是独立 sessions。

## 6. Exact robust readout

令 \(w_e=q_\alpha s_e\)，

\[
\mathcal U(u,w)=\{d:|D(d-u)|_e\le w_e,\forall e\}.
\]

对 challenger set \(A\) 与 native set \(B\)：

\[
L_G(A,B)
=\inf_{d\in\mathcal U}
[\min_A d-\min_B d]
\]

且

\[
L_G(A,B)
=\min_{a\in A}\max_{b\in B}
[u_a-u_b-\operatorname{dist}_w(a,b)].
\]

定义

\[
\psi_B(v)=\min_{b\in B}
[u_b+\operatorname{dist}_w(v,b)],
\]

则

\[
L_G(A,B)=\min_{a\in A}[u_a-\psi_B(a)].
\]

每个 native 一次 exact multi-source Dijkstra。V0 binary64 实现对 path additions 向 \(+\infty\) 外舍入、对最终差值向 \(-\infty\) 外舍入，并拒绝非有限结果；因此对表示后的输入方向保守，当前实现取 \(\varepsilon_{num}=0\)。输入量化与 map/geometry error 仍单独扣除。difference constraints 与 min-plus solver 是标准工具，不作为独立创新。

## 7. Continuous certificate

geometry checker 必须确定性证明非对称合同：

- challenger 连续扫掠体 \(F_j\) 被解析外包络 \(\overline F_j\) 包含；
- \(A_j\) 是 \(\overline F_j\) 的 \(h_j\)-cover，允许节点位于真实 challenger footprint 外；
- \(A_0\subset F_0\)，每个 native 节点被证明位于至少一个真实 native footprint 内；
- footprints 完全位于认证 ROI，集合非空；
- reference surface、pose interpolation、vehicle dilation、coordinate transforms 与 protocol hash 一致。

V0.4 的默认证明路径是 rotation-free disk fallback。每段以 rear-axle 平移中点为圆心，
使用

\[
r_i=\frac12\lVert p_{i+1}-p_i\rVert_2+R_K+\varepsilon_{pose}
\]

外包任意 yaw 变化下的整段车体，再以 \(r_i+h\) 选择固定格心；native 只选择
rear axle 内圆盘

\[
r_{in}=\min\{l_{rear},l_{front},w/2\}-\varepsilon_{pose}>0
\]

中的格心。中点、squared distance 和比较按 binary64 输入对应的精确有理数计算，
平方根与加法向外舍入，因此正式证书不依赖 scorer 四角点或 libm 三角函数。

解析矩形路径可作为更锐利的研究实现：冻结相邻姿态间的线性平移与连续展开 yaw
的线性插值。每段使用中点车体的膨胀外包络

\[
\delta_i=\frac12\lVert p_{i+1}-p_i\rVert_2
+2R_K\sin\frac{|\Delta\theta_i|}{4}
+\varepsilon_{pose}.
\]

边长为 \(r\) 的固定方格以 \(h_j=r/\sqrt2\) 覆盖外包络。任何 ROI clipping、未展开 yaw、空 challenger/native set 或协议不匹配均返回 reject。

令 \(\varepsilon_{map,j}=\max_{q\in A_j}\varepsilon_{map}(q)\)，native 同理。若 computed lower bound 可能乐观至多 \(\varepsilon_{num}\)，则

\[
L_{j0}^{cert}
=\widehat L_G
-h_j
-\varepsilon_{map,j}
-\varepsilon_{map,0}
-\varepsilon_{num}.
\]

无法证明 map、geometry 或 numerical budget 的 pair 不得认证 continuous task。

continuous teacher 必须来自冻结且逐 polygon 拓扑有效的 raw polygon cover，而不是
raster EDT，也不要求先做数值 overlay。成员关系取输入 polygons 的并集，距离幅值取到
全部 raw rings 的最近距离；重叠产生的内部 seam 按定义为零，因而相对于 union-boundary
clearance 保守。离线 node-label builder 使用 exact-rational segment squared distance 和
binary64 outward sqrt interval，输出 \(\widetilde d(q)\) 与
\(\varepsilon_{dist}(q)\)。部署账本保持

\[
\varepsilon_{map}
=\varepsilon_{transform}
+\varepsilon_{polygon}
+\varepsilon_{dist}
+\varepsilon_{round}.
\]

ego-local node 到 global raw polygons 的坐标变换不直接调用 libm 作为证明步骤。
对 \(|\theta|\le4\) 的 binary64 heading，使用 40 阶 exact-rational Taylor
sin/cos enclosure 和 41 阶 Lagrange remainder；完整 affine transform 用有理区间
传播。存储点到所得二维区间四角的最大距离向外舍入，逐节点写入
\(\varepsilon_{transform}(q)\)。超出 heading domain、非有限输入或区间/输出溢出
均拒绝生成标签。32-ULP libm comparison 只作接口诊断，不进入 certificate。

synthetic teacher 资格通过不自动令真实 NAVSIM 的 transform/polygon 两项为零；它们
必须由实际 preprocessing 与 protocol hash 单独闭合。

## 8. Arbitration and primary risk event

冻结 \(\gamma_{task}\ge0\)。只有 \(L_{j0}^{cert}\ge\gamma_{task}\) 的 challenger 可被授权；否则保留 native。若多个合格，选择 deviation cost 最小者。

主失败事件是所有授权干预的 relative-margin failure：

\[
R_\gamma(Z)=
\mathbf1\{\exists t\in T(Z):
I_t=1,
\Delta_{sel,0,t}^{cont}<\gamma_{task}\}.
\]

在 source exchangeability、calibration 前冻结和 deterministic sound contracts 下：

\[
P_{D_{cal},Z}(R_\gamma=1)\le\alpha.
\]

若 \(\beta\) 上界整个 session 至少一次授权 pair 的非 field contract 错误接受，则

\[
P(R_\gamma=1)\le\min\{1,\alpha+\beta\}.
\]

safe-native-to-unsafe-challenger event 在 \(\gamma_{task}\ge0\) 时只是推论。该保证不是固定 calibration realization 的条件保证，也不是 arbitrary-OOD guarantee。

## 9. Appearance mechanism diagnostic

对同一 scene 的 paired observations \(X,X'\)：

\[
\rho(X,X')
=\max_e
\frac{|D[u_\theta(X')-u_\theta(X)]_e|}{s_e}.
\]

则

\[
\|r(X')\|_\infty
\le\|r(X)\|_\infty+\rho(X,X').
\]

因此应检验 graph potential differences 的 appearance stability，而不是 token 或 node values 完全不变。经验小 \(\rho\) 不产生 target coverage；target 只报告 empirical results。

## 10. Usefulness decomposition

评估单位 \(U=(Z,t)\) 由“均匀抽 session，再均匀抽冻结时刻”定义，推断按 session 聚类。

- \(B(U)\)：存在 eligible、meaningful-non-inclusion 且 GT improvement \(\ge\delta_M\) 的 challenger；
- \(C(U)\)：存在同一个 challenger 同时满足 \(B\) 与 certificate；
- \(D(U)\)：最终 selected challenger 满足上述 useful 条件。

报告

\[
S_{better}=P(B),
\quad
R_{cert|better}=P(C\mid B),
\quad
R_{opportunity}=P(C),
\quad
R_{selected}=P(D).
\]

meaningful non-inclusion 必须使用连续 footprints \(F_j\setminus F_0\) 与 \(F_0\setminus F_j\) 的双向差面积，或方向正确的 conservative cell-union lower bounds；有限 node sets 没有连续面积。

## 11. V0 contract and stop rules

- one node-potential decoder；single native；fixed physical scale；exact solver；
- frozen DINOv3-H+、planners、candidate provider；
- deterministic nominal geometry route；source-only train/development/calibration；
- Gate A 验证 theorem、map/geometry/numerical ledger；
- Gate B 验证 tail residual、\(S_{better}\)、\(R_{cert|better}\)、\(R_{opportunity}\)；
- Gate A/B 失败则不进入完整 NAVSIM 训练；
- local edge 不优于 exact global-gauge/具体 joint-field，或证书近乎全空，则停止方法原创路线。
