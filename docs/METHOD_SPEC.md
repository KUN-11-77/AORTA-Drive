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

V0 主图冻结为 current rear-axle frame 上的 \([-32,32]\times[-32,32]\) m
方形 support、1 m cells、64×64 cell centers、4096 nodes 和 8064 条无向
4-neighbor edges。连续覆盖半径为向外舍入的

\[
h=\sqrt{(1/2)^2+(1/2)^2}=1/\sqrt2\ \text{m}.
\]

该选择来自 source-only 协议：NAVSIM 4 s horizon、现有冻结模型族共同的 64 m BEV
support，以及 0.25 m 输入 BEV 的 factor-4 decoder stride；没有读取 target score。
若 Gate B 显示该分辨率使证书近乎全空，按停止规则报告或执行预注册敏感性分析，
不得根据 target 结果事后替换主图。

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

R004D 的最小合法 probe 将 12 个槽位固定为 history-major、每个历史时刻内
`left/front/right`。每个有效槽位经共享的 (1280\to64\to64) 投影，并加入
camera/history embedding；masked soft attention 只在有效槽位上归一化，不允许
hard best-view。全缺失节点使用单一 learned missing token，且有效槽位比例作为显式
输入。融合后的 64×64 场经过宽度为 96/128/192 的两尺度 residual U-Net。该结构
不接收 candidate、map、planner output、certificate 或 score。

输出 raw potential 后执行 gauge projection

\[
u_\theta(q)\leftarrow u_\theta(q)
-|\Omega|^{-1}\sum_{v\in\Omega}u_\theta(v).
\]

这只选择商空间中的一个稳定代表，不增加任务假设，也不把绝对零点重新引入监督。

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

实际 probe 的优化目标同时包含平均项与 tail-aligned surrogate。令

\[
r_{t,e}=\frac{|(Du_\theta-D\widetilde d_S)_{t,e}|}{s_e},
\]

则

\[
\mathcal L_{train}
=\operatorname{mean}_{t,e}\operatorname{Huber}_{0.25}(r_{t,e})
+0.25\operatorname{mean}_{t}
\left[
0.10\log\sum_e\exp(r_{t,e}/0.10)-0.10\log|E|
\right].
\]

第二项与 smooth maximum 具有相同梯度，但减去了与参数无关的常数；它只是训练
surrogate，不是 coverage bound。每个 train recording group 的 16 个 anchors 每 epoch
各使用一次，并将 recording groups 交错排序。anchor 是组内重复观测，不作为独立
统计样本。

每个 epoch 后，在全部 24 个 development recording groups 上计算

\[
E_z^{dev}=\max_{t\in T(z),e\in E}r_{t,e}.
\]

checkpoint 按以下预冻结字典序选择：五个最大 (E_z^{dev}) 的平均、全部 24 个的
平均、最早 epoch。candidate utility、certificate rate、planner score、calibration/test
数据均不得进入该规则。

共同 offset 位于 \(\ker D=\operatorname{span}\{\mathbf1\}\) 中，不影响 relative set-min task。average edge loss 或上述 smooth-tail surrogate 均不推出 session maximum 很小，也不推出 appearance-OOD stability；Gate B 仍直接审计独立 session residual 与证书锐度。

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

R002G node-label builder 对每个冻结 cache 和全部 4096 nodes 保存
`label_m`、`transform_error_m`、`distance_error_m` 与 `total_error_m`。raw cache
polygon coordinates 本身定义 paper teacher，因此只在该明确 scope 内取
\(\varepsilon_{polygon}=0\)；这不等于真实道路或另一个 map release 的 polygon
semantic error 为零。float64 `.npy` payload 使用固定 ZIP metadata，逐文件和逻辑
manifest 均做 SHA-256；任一 node、cache、hash 或 replay 失败即拒绝 Gate A。

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

## 12. AORTA-v8.4 candidate amendment after R004E-G

本节不删除 V0 fixed-scale control，而是定义下一项可证伪候选。完整 R004H 注册前不得执行。

### 12.1 Architecture

Stage P 仅用冻结的 42 fit groups 重新训练既有 node-potential/trunk，并在 8 selection
groups 选择一次 checkpoint。随后永久冻结 DINO、projection、trunk 与 potential。Stage S
只训练一个输出 canonical right/down 8064 edges 的 bounded positive scale head：

\[
s_e(X)=2^{-10}\mathrm m+(2^4-2^{-10})\mathrm m\,\sigma(a_e(X)).
\]

Stage S 只最小化 bounded Laplace loss

\[
\mathcal L_{scale}=\mathbb E\left[
\frac{|(Du-D\widetilde d)_e|}{s_e(X)}+
\log\frac{s_e(X)}{1\mathrm m}
\right].
\]

不得把 group maximum surrogate、candidate、planner score 或 certificate outcome 反传到
scale head。旧 66-group checkpoint 不具备 proxy/official calibration 独立性，只可作历史
诊断。

### 12.2 Data roles and coverage boundary

66 个旧 train groups 以注册 SHA-256 顺序固定为 42 fit、8 selection、16 proxy-cal。
proxy-cal 全部带历史曝光标签，只用于一次内部流程排练，不产生 coverage claim。正式
split-conformal 只允许在最终模型、score 和决策规则冻结后首次读取 official calibration；
其 calibration 与 future source session 必须条件联合可交换。任一 calibration group
无效则整个 variant calibration 失败，不允许过滤。

### 12.3 Required controls and primary endpoint

所有控制共享同一冻结 potential 并分别校准：learned edge scale、把 learned scale 每帧
几何平均后广播的 pooled scale、uniform 1 m scale、evidence-only scale；zero field 单独
作为历史参考。主终点使用独立于方法的共同 opportunity set：冻结候选同时满足 geometry
资格、meaningful continuous non-inclusion 与 teacher improvement \(\ge\delta_M\)。报告
同一分母上的 \(R_{cert|better}\)，invalid/fallback 留在分母且记未认证。

learned 只优于 uniform 支持一般 scale adaptation；同时优于 pooled 才支持 edge-wise
allocation。两者均不自动证明正式 coverage、OOD robustness 或算法原创性。

### 12.4 Qualification status

R004E-G 已通过 role、four-term interval residual、outward division/multiplication、joint-rank
oracle、scale gauge 与 heterogeneous graph readout。R004H0 随后通过旧势场 exact replay、
194 参数 right/down scale head、bounded initialization、candidate-free signature 与
scale-only gradient isolation。R004H1 再用 fresh initialization、42 fit/8 selection roles
完成 20 epochs，并在冻结 primary 上严格优于 zero field；但其 exact group maximum 仍大，
因此只授权注册 Stage S。首个 R004H2 Stage S 配置虽保持 Stage P 零更新并通过硬 cap 门，
但 learned proper loss 为 uniform 的 `6.76×`，正式 STOP。该负结果否定当前 raw-trunk
1x1-head + `3e-4` AdamW 实例，不否定 heterogeneous-envelope 定理；下一步必须先区分
activation scale、parameterization 与 optimizer conditioning。proxy-cal、development one-read
和 official calibration 仍由后续独立门禁控制。
