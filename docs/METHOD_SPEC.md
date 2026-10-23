# AORTA-Drive Method Specification v0.2

## 1. Task graph and margin

令 (G=(\Omega,E)) 为有限、连通、无向的 4-neighbor ego-centric 图。GT signed-distance field 为 (d_S:\Omega\to\mathbb R)。冻结 tracker 将轨迹转为非空 swept-node set (A(\tau))。

\[
M_S^\Omega(\tau)=\min_{q\in A(\tau)}d_S(q)-m_{buf}.
\]

该 margin 只表示建模的 nominal road constraint，不等价于通用安全。

## 2. Candidate-free field model

冻结 DINOv3-H+ 提供多尺度视觉 tokens。固定物理节点经部署可用的相机/ego geometry 投影并采样 tokens；一个共享 decoder 输出节点势 (u_\theta(X,q))。候选不进入 decoder，因此同一物理 cell 跨候选共享语义。

V0 使用固定 edge scale

\[
s_e=\ell_e/\ell_{ref};
\]

learned scale 不是 V0 依赖。

## 3. Training

令 (D) 为任意定向 incidence matrix：

\[
\mathcal L_{edge}
=|E|^{-1}\sum_e
\operatorname{Huber}
\left(\frac{(Du_\theta-Dd_S)_e}{s_e}\right).
\]

共同 offset 位于 (\ker D=\operatorname{span}\{\mathbf1\}) 中。可用 (\sum_q u_q=0) 作数值 gauge；它不改变 trajectory contrast。

## 4. Source-session calibration

对 calibration 前完全冻结的模型、图、尺度与时刻协议：

\[
E_i=\max_{t\in T(Z_i)}\max_{e\in E}
\frac{|(Du_{it}-Dd_{it})_e|}{s_e}.
\]

令 (k=\lceil(n+1)(1-\alpha)\rceil)。若 (k>n)，返回不可部署；否则 (q_\alpha=E_{(k)})。交换性下：

\[
P(\forall t\in T(Z),e\in E:
|(Du-Dd)_e|\le q_\alpha s_e)
\ge1-\alpha.
\]

这是 source-session 边际保证，不覆盖任意 OOD。

## 5. Uncertainty set and exact readout

令 (w_e=q_\alpha s_e)，

\[
\mathcal U(u,w)=\{d:|D(d-u)|_e\le w_e,\forall e\}.
\]

对非空 challenger (A) 与 native (B)：

\[
L_G(A,B)=\inf_{d\in\mathcal U}
[\min_A d-\min_B d].
\]

精确式为

\[
L_G(A,B)
=\min_{a\in A}\max_{b\in B}
[u_a-u_b-\operatorname{dist}_w(a,b)].
\]

定义

\[
\psi_B(v)=\min_{b\in B}[u_b+\operatorname{dist}_w(v,b)],
\]

则

\[
L_G(A,B)=\min_{a\in A}[u_a-\psi_B(a)].
\]

每个 native 一次 multi-source Dijkstra；Dijkstra/min-plus 是标准工具，不作为创新。若将初始标签统一减去 (c=\min_Bu)，最终 envelope 必须恢复 (+c)。

## 6. Geometry certificate

V0 checker 只在以下条件可被确定性证明时返回 valid：

- native/challenger samples 都位于各自连续 nominal footprint 内；
- challenger samples 是 footprint 的内部 (h_j)-net，(h_j) 是解析或保守上界；
- 完整 footprints 在认证 ROI 内且集合非空；
- footprint、pose interpolation、车辆膨胀、grid 和数值协议一致；
- teacher signed distance 在 ROI 上为 1-Lipschitz。

于是

\[
\Delta_{j0}^{cont}\ge L_G(A_j,A_0)-h_j.
\]

无法证明条件的 pair 拒绝。经验测试点最大误差不能替代证明半径。

## 7. Arbitration

候选先通过冻结的动力学、意图、进度和偏离预算。只有当

\[
L_G-h_j\ge\gamma_{task}+\varepsilon_{num}
\]

时才可替换，其中 (\varepsilon_{num}) 上界最终 bound 的乐观数值误差。若无候选合格则保留 native；否则选择 deviation cost 最小者。

## 8. Risk contract

令 (T(Z)) 为冻结 session 时刻集，

\[
I(Z)=\max_{t\in T(Z)}\mathbf1\{\text{replace at }t\},
\]

\[
H(Z)=\mathbf1\{\exists t\in T(Z):I_t=1,
M_{0,t}^{cont}\ge0,M_{sel,t}^{cont}<0\}.
\]

若 geometry checker 对 nominal footprints 确定性 sound，则

\[
P_{D_{cal},Z}(H=1)\le\alpha.
\]

若未来采用概率 checker，(\beta) 必须上界整个 session 至少一次授权 pair 几何误接受事件，风险上界变为 (\min\{1,\alpha+\beta\})。

## 9. V0 implementation contract

- single native；connected 4-neighbor graph；fixed physical scale；
- exact multi-source solver；one node-potential decoder；
- deterministic internal geometry route；
- frozen backbone/planners；source-only train/development/calibration；
- target domain 禁止选模型、阈值或配置。

首次运行前必须由 experiment-plan 冻结 ROI、resolution、(|\Omega|)、(|E|)、(T(Z))、checker 构造和数值误差预算。

## 10. Qualification and stop rules

- Gate A：theorem oracle、geometry checker、数值方向；
- Gate B：合法 source-only predictor 的米制 bound 非平凡性；允许训练小型 probe，不得用 GT 自拟合替代；
- Gate C：meaningful continuous non-inclusion；
- Gate D：held-out candidate policy/planner。

若 local graph 在 exact-readout 公平比较下不优于 global gauge/CDF，若证书几乎总为空，或收益只来自 geometry inclusion，则停止方法原创路线。
