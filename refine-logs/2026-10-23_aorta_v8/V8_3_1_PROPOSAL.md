# AORTA-v8.3.1 — Conditional Theory and System Specification

## Problem Anchor

- **Bottom-line problem:** 同一物理场景只发生外观变化时，完整冻结的端到端驾驶 planner 可能输出显著不同甚至违反道路约束的轨迹；我们要从视觉中学习一个可跨冻结 planner 使用的任务接口，使干预依据与物理道路约束一致，而不是依赖全局特征不变性。
- **Must-solve bottleneck:** 绝对道路裕度的安全零点在视觉退化下可能不可辨识，但候选之间的相对道路可行性仍可能保留；方法必须明确回答在不知道绝对安全锚点时，哪些替换仍能被合法授权，并对局部残差、选择后风险、候选自适应性和几何离散误差负责。
- **Non-goals:** 不重建 HD map；不修改或联合训练原 planner；不宣称道路可行性等于碰撞、交通规则或闭环安全；不以 source conformal 冒充任意 OOD 保证；不把 DINO、SDF、split conformal、difference constraints、Dijkstra 或 min-plus transform 单独宣称为创新。
- **Constraints:** NAVSIM v1/v2；冻结 DINOv3-H+ 与 planner；source-only 学习、选择和校准；候选域与几何协议预先冻结；主模型只有一个共享场 decoder；Gate A/B 之前不运行完整训练或目标域长跑。
- **Success condition:** 在严格同源对照下，局部差分场接口必须在预声明 source-session 风险对象上有效，并在具有连续几何意义的非包含候选上，比 scalar、absolute field、global-gauge exact readout 和具体 joint-field baseline 产生更紧且有用的 relative-margin certificates；否则降级或停止。

## Anchor and Simplicity Check

- 原问题仍是 appearance-driven frozen-planner road arbitration，没有扩展到完整安全或地图重建。
- v8.3 的 graph quotient、exact set-min theorem、fixed-scale decoder 和 exact multi-source solver 保持不变。
- v8.3.1 不新增 trainable module；只补齐风险定理、连续 teacher 误差、视觉节点协议、候选接口和资格指标。
- learned edge scale、时空 transformer、VLM、OOD detector 与 candidate generator learning 继续排除。

## 1. Scientific Thesis

AORTA 学习 candidate-free 的道路相对节点势，对 source session 内所有图边的预测梯度误差同时校准；在一个 field event 上，使用标准 min-plus envelope 精确计算任意自适应 swept-set pair 的最坏相对 margin，并只授权经过 sound nominal-geometry correction 后达到非负改进阈值的 challenger。

理论只说明“覆盖事件成立时证书正确”。视觉网络能否产生足够小的 session-tail residual，以及冻结 candidate interface 能否提供可改善候选，是两个必须独立否证的科学假设。

## 2. Continuous Teacher, Node Labels, and Error Ledger

令连续 nominal road signed-distance teacher 为

\[
\phi_S:\mathcal R\to\mathbb R,
\]

并假设它在认证 ROI \(\mathcal R\) 上为 1-Lipschitz。固定物理图 \(G=(\Omega,E)\) 的 node label 为 \(\widetilde d_S(q)\)。标签生成、地图栅格化、坐标变换和插值误差必须给出确定性上界

\[
|\widetilde d_S(q)-\phi_S(q)|\le\varepsilon_{map}(q).
\]

对一个 pair，可使用

\[
\varepsilon_{map,j}=\max_{q\in A_j}\varepsilon_{map}(q),
\qquad
\varepsilon_{map,0}=\max_{q\in A_0}\varepsilon_{map}(q).
\]

若无法给出该上界，系统只能认证离散 teacher task，不能声称连续 nominal-road certificate。

## 3. Candidate-Free Visual Node Potential

冻结 backbone 产生相机 \(c\)、历史时刻 \(k\) 的 token field \(F_{c,k}\)。对当前 ego-frame 节点 \(q=(x,y)\)：

1. 使用预注册参考道路表面 \(z_{ref}(q)\) 将 \(q\) 提升到三维；V0 可采用局部 ego ground plane，但必须记录该模型误差；
2. 用冻结历史 ego poses 与相机外参变换到 camera frame；
3. 用相机内参和实际裁剪/缩放变换得到 token coordinate \(\pi_{c,k}(q)\)；
4. 仅对正深度且位于图像/token grid 内的投影做双线性采样；
5. 输出 frustum-valid mask。该 mask 不代表真实 occlusion visibility；没有深度时不得声称知道遮挡；
6. 对无有效投影的节点使用固定 missing-evidence token，并保留 mask 作为输入。

定义固定证据构造

\[
\mathcal F(X,q)
=\operatorname{Pack}
\left(
\{\operatorname{bilinear}(F_{c,k},\pi_{c,k}(q)),m_{c,k}(q)\}_{c,k}
\right).
\]

唯一 trainable component 为共享 decoder

\[
u_\theta(X,q)
=f_\theta(\mathcal F(X,q),\operatorname{PE}(q),m(q)).
\]

相机顺序、历史长度、token layers、projection convention、missing-token 处理和聚合方式必须在 calibration 前冻结。DINOv3-H+ 只是冻结精度/算力折中，不声称是官方最大或必然最优 backbone。

## 4. Graph-Gradient Training

令 \(D\) 为有限连通无向图的 incidence matrix。V0 edge scale 固定为 \(s_e=\ell_e/\ell_{ref}>0\)。训练目标保持

\[
\mathcal L_{edge}
=\frac1{|E|}\sum_{e\in E}
\operatorname{Huber}
\left(
\frac{(Du_\theta-D\widetilde d_S)_e}{s_e}
\right).
\]

该 average loss 不推出 session maximum 很小，也不推出 appearance-OOD stability。v8.3.1 不用新 loss 掩盖这一缺口；Gate B 必须直接测量 tail residual、session maximum 和最终证书锐度。

## 5. Calibration Protocol and Minimum Sample Size

数据严格分为 source train、source development、source calibration 和 source test sessions。模型、checkpoint、图、尺度、\(\alpha\)、\(\gamma_{task}\)、candidate interface 和所有选择规则在首次读取 calibration scores 前冻结。calibration sessions 不得反复用于选模型、阈值或架构。

定义

\[
E_i=\max_{t\in T(Z_i)}\max_{e\in E}
\frac{|(Du_{it}-D\widetilde d_{it})_e|}{s_e}.
\]

令

\[
k=\lceil(n+1)(1-\alpha)\rceil,
\qquad q_\alpha=E_{(k)}.
\]

非随机有限阈值要求

\[
n\ge\left\lceil\frac1\alpha\right\rceil-1.
\]

因此 \(\alpha=10\%,5\%,1\%,0.5\%\) 至少分别需要 9、19、99、199 个真正独立 calibration sessions。这只是阈值存在条件，不是稳定性或锐度充分条件；帧和同 route corruption copies 不能充当独立 sessions。

## 6. Exact Discrete Robust Bound

令 \(w_e=q_\alpha s_e\)，

\[
\mathcal U(u,w)
=\{d:|D(d-u)|_e\le w_e,\forall e\in E\}.
\]

对非空 swept-node sets \(A,B\)：

\[
L_G(A,B)
=\inf_{d\in\mathcal U(u,w)}
[\min_A d-\min_B d]
\]

精确等于

\[
\min_{a\in A}\max_{b\in B}
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

每个 native 只需一次 exact multi-source Dijkstra。该定理与 McShane–Whitney 型 Lipschitz envelope 有直接结构联系，作为计算引理而非独立新算子。

## 7. Continuous Relative-Margin Bound

令连续 footprints 为 \(F_j,F_0\)，node samples 为 \(A_j,A_0\)。geometry certificate 必须确定性验证：

- \(A_j\subset F_j\) 与 \(A_0\subset F_0\) 的节点坐标确为 footprint 内物理点，不是可能落在外部的任意 rounded centers；
- \(A_j\) 是 \(F_j\) 的内部 \(h_j\)-net；
- footprints 位于认证 ROI，集合非空；
- pose interpolation、vehicle dilation、reference surface 和 coordinate transforms 与 protocol hash 一致。

由 1-Lipschitz teacher 和 label-error ledger：

\[
\Delta_{j0}^{cont}
\ge L_G(A_j,A_0)
-h_j
-\varepsilon_{map,j}
-\varepsilon_{map,0}.
\]

若只知道统一上界 \(\varepsilon_{map}\)，则扣除 \(2\varepsilon_{map}\)。

若数值实现可能比数学 lower bound 乐观至多 \(\varepsilon_{num}\)，最终可认证量为

\[
L_{j0}^{cert}
=\widehat L_G
-h_j
-\varepsilon_{map,j}
-\varepsilon_{map,0}
-\varepsilon_{num}.
\]

## 8. Strengthened Relative-Intervention Theorem

冻结非负任务阈值

\[
\gamma_{task}\ge0.
\]

只有 \(L_{j0}^{cert}\ge\gamma_{task}\) 的 challenger 才可授权。定义

\[
R_\gamma(Z)=
\mathbf1\left\{
\exists t\in T(Z):
I_t=1,
\Delta_{sel,0,t}^{cont}<\gamma_{task}
\right\}.
\]

若 calibration/test source sessions 交换、所有选择规则在 calibration 前冻结、field event 同时覆盖所有声明边和时刻，且 geometry/teacher/numerical contracts 对所有授权 pair 都 sound，则

\[
\boxed{
P_{D_{cal},Z}(R_\gamma=1)\le\alpha
}.
\]

若 \(\beta\) 上界整个 session 至少一次授权 pair 的 geometry/teacher contract 错误接受概率，则

\[
\boxed{
P(R_\gamma=1)\le\min\{1,\alpha+\beta\}
}.
\]

这里 \(\beta\) 不是单 pair 错误率。safe-native 被替换为 unsafe challenger 的原事件 \(H\) 在 \(\gamma_{task}\ge0\) 时只是该定理的推论。

该保证仍是 calibration set 与新 source session 联合随机性的边际保证，不是固定 calibration realization 的条件保证，也不是“发生干预时有 \(1-\alpha\) 成功率”。

## 9. Appearance-Gradient Stability as a Testable Mechanism

对同一物理场景的 paired clean/appearance-shifted observations \(X,X'\)，共享同一 teacher。定义

\[
r_e(X)=\frac{(Du_\theta(X)-D\widetilde d_S)_e}{s_e}
\]

和

\[
\rho(X,X')
=\max_e
\frac{|D[u_\theta(X')-u_\theta(X)]_e|}{s_e}.
\]

由三角不等式：

\[
\|r(X')\|_\infty
\le\|r(X)\|_\infty+\rho(X,X').
\]

对 session 所有时刻同时取 maximum 可得 session 版本。这表明真正需要检验的神经性质不是 token 或节点势完全不变，而是 graph potential differences 跨外观变化小。

经验上的小 \(\rho\) 只支持机制诊断，不产生 target/OOD coverage。除非未来额外给出配对分布关系和可靠的 session-level \(\rho\) 上界，否则 target 只报告 empirical coverage。

## 10. Frozen Candidate Interface

AORTA 是 arbiter，不是 candidate generator。定义部署可用的冻结接口

\[
\mathcal C_\xi(O_t)
=\{\tau_0,\tau_1,\ldots,\tau_{K_t}\},
\qquad K_t\le K_{max},
\]

其中 \(\xi\)、候选来源、时域、预算、随机种子和排序在 calibration 前冻结。

候选可来自预先声明的 frozen alternative planners 或 deterministic kinematic perturbation library，但必须满足：

- 只使用部署时可获得的 observation/state；
- 不使用 GT map、target labels 或 paired clean target image；
- native 始终为 \(\tau_0\)；
- 所有方法接收完全相同的候选集、eligibility filters 和 compute budget；
- 每个 candidate 带可复现 provenance；
- calibration 后不得因 target 结果改变候选数量或生成规则。

最终 utility 分解为

\[
\text{useful intervention}
=\text{better candidate available}
\cap\text{certified}
\cap\text{eligible}.
\]

因此实验必须把 candidate supply failure 与 certificate failure 分开报告。

## 11. Usefulness and Anti-Inclusion Metrics

所有 usefulness 指标使用同一预注册评估单位 \(U=(Z,t)\)：先从评估 sessions 均匀抽取 \(Z\)，再从其冻结时刻集 \(T(Z)\) 均匀抽取 \(t\)。候选集由同一个冻结 provider 产生；置信区间与显著性分析按 session 聚类，不把时刻当作独立 session。

令 \(B(U)\) 表示该时刻存在至少一个 eligible、meaningful-non-inclusion 且满足 \(\Delta^{GT}_{j0}\ge\delta_M\) 的 challenger。令 \(C(U)\) 表示存在同一个 challenger 同时满足 \(B\) 的条件与 \(L_{j0}^{cert}\ge\gamma_{task}\)。定义：

\[
S_{better}=P_U(B),
\]

\[
R_{cert|better}=P_U(C\mid B),
\]

\[
R_{opportunity}=P_U(C).
\]

再令 \(D(U)\) 表示 arbiter 的最终 selected challenger 本身 eligible、meaningful-non-inclusion、\(\Delta^{GT}_{sel,0}\ge\delta_M\)，并已通过证书。最终单独报告

\[
R_{selected}=P_U(D)
\]

以及条件于发生干预的 selected-useful precision。这样 candidate supply、certificate recall 和 selection policy 的损失不会混在一个比例中。

Gate C 的 meaningful non-inclusion pair 必须同时满足：

- 连续 footprints \(F_j\setminus F_0\) 与 \(F_0\setminus F_j\) 的面积均超过预注册阈值；若使用 cell unions，必须给出对连续双向差面积方向正确的保守下界；有限 node sets 本身不被赋予连续面积；
- 相同预测时域和 progress budget；
- footprint Hausdorff/mean displacement 超过网格容差阈值；
- GT relative margin improvement 至少为 \(\delta_M>0\)；
- 不是只增加或删除一个边界 raster cell。

若 \(S_{better}\) 近零，问题在 candidate supply；若 \(S_{better}\) 足够而 \(R_{cert|better}\) 近零，问题在 predictor/calibration/certificate；若 \(R_{opportunity}\) 足够但 \(R_{selected}\) 明显较低，问题在 selection policy。三者不得混为“方法没提升”。

## 12. Minimal Architecture and Inference

```text
multi-camera history + frozen calibration/ego poses
                     │
                     ▼
             frozen DINOv3-H+ tokens
                     │ fixed node projection + mask
                     ▼
       shared node-potential decoder uθ(X,q)
                     │ once per observation
                     ▼
 source-calibrated fixed graph-difference envelope
                     │
 frozen candidate interface ──► swept-node sets
                     │
                     ▼
 one exact native-seeded min-plus transform
                     │
 proof-carrying continuous correction
                     │
                     ▼
       least-deviating certified challenger
       or keep native
```

## 13. Gate A/B Revisions

### Gate A — proof and implementation

- exact theorem vs LP/dual oracle；
- super-source label recovery、common shift、zero edges、chain/cycle；
- physical sample coordinates truly inside footprints；
- \(h_j\)、\(\varepsilon_{map}\)、\(\varepsilon_{num}\) directions；
- relative-intervention event \(R_\gamma\) and safe-to-unsafe corollary；
- calibration sample-size eligibility。

### Gate B — learned certificate feasibility

使用合法 source-only predictor 或最小 probe，直接报告：

- average edge loss 与 session-max residual 的差距；
- \(q_\alpha\)、tail concentration、q-dominating edge/frames；
- \(S_{better}\)、\(R_{cert|better}\)、\(R_{opportunity}\)、\(R_{selected}\)；
- meaningful non-inclusion 分层；
- paired appearance \(\rho\) 及 empirical target coverage，明确无 OOD theorem；
- absolute、global gauge、local edge、具体 joint-field 的公平 exact-readout comparison。

若 coverage 正确但 \(R_{cert|better}\approx0\)，停止正式训练或回退更简单 uncertainty geometry；不增加模块维持方案。

## 14. Contribution Boundary

当前可冻结的贡献候选是：一个针对 frozen planners 的 task-relative field interface，把 source-session calibrated graph-gradient evidence、uncertainty-set-exact set-min contrast、sound continuous correction 和 adaptive native-preserving selection 对齐。

当前不授权：新 min-plus 算子、任意 OOD coverage、完整驾驶安全、优于 DriveReferee/CDF/scalar referee、H+ 最优或“首次”。

只有公平基线证明 local relative field 在实际证书 recall、candidate transfer 或成本上不可替代，且独立 novelty-check 未发现等价接口，才能升级为顶会方法主张。

## 15. Immediate Handoff

1. 将 v8.3.1 写入正式 method/theory docs 与 immutable config schema；
2. 冻结 candidate provider、projection protocol、continuous teacher 与误差账本；
3. 执行 Gate A 的 CPU oracle；
4. Gate A 全通过后，用合法 source-only predictor 执行 Gate B；
5. Gate B 证明证书非平凡后，才进入正式 NAVSIM training/evaluation。

本版本没有运行实验，也没有把外部审查意见当作实证结果。
