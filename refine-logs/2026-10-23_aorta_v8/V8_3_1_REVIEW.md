# AORTA-v8.3.1 Review

<details><summary>Raw reviewer response</summary>

## 裁定

**冻结：NO（针对当前文档原样冻结）。** 条件理论核心通过；唯一实质阻塞集中在第 11 节的统计单位和反包含面积定义。修复这两处后，无需增加模块或改变核心定理，即可冻结规范并进入 Gate A/B。

这不撤销 v8.3 已获得的条件理论规范资格，也不授权顶会原创、已校准、有效、实时或 OOD 保证。本轮完整读取了 proposal，没有运行实验或修改 proposal。

## 1. Node-label/map error 与连续扣项：正确

令 \(\widetilde m_i=\min_{A_i}\widetilde d\)，\(m_i^\phi=\min_{A_i}\phi\)。节点标签误差给出

\[
|\widetilde m_i-m_i^\phi|\le\varepsilon_{map,i}.
\]

由内部采样方向与 challenger h-net：

\[
M_j^{cont}\ge\widetilde m_j-\varepsilon_{map,j}-h_j-m_{buf},
\qquad
M_0^{cont}\le\widetilde m_0+\varepsilon_{map,0}-m_{buf}.
\]

因此

\[
\Delta_{j0}^{cont}
\ge L_G-h_j-\varepsilon_{map,j}-\varepsilon_{map,0}.
\]

符号和双侧 map 扣项正确，统一上界时减 \(2\varepsilon_{map}\) 也正确。field calibration 直接覆盖的是 \(\widetilde d\)，所以 label-to-continuous-teacher 误差是独立的一层，并非重复扣除。减去最终数值乐观误差上界后再与非负任务阈值比较，同样正确。

若无法取得 map bound，只能保留离散 teacher 结论；该 fallback 不继承连续 \(R_\gamma\) 或 H 保证。reference-surface 误差若仅影响图像特征投影，应进入 predictor residual；若改变物理采样点或连续几何对应关系，才需要相应位置误差账本。实现时应明确分类，避免重复计账或遗漏。

## 2. Strengthened intervention theorem：正确

在 field、geometry、teacher 和数值合同都有效时，任意授权 pair 满足

\[
\Delta_{sel,0,t}^{cont}\ge L_{sel,0,t}^{cert}\ge\gamma_{task}.
\]

因此 \(R_\gamma\) 不发生，session 事件包含关系及 \(\alpha\) 上界成立。\(\beta\) 已明确是整个 session 至少一次授权 pair 的错误接受概率，不是单 pair 错误率，\(\alpha+\beta\) 的量词正确。数值合同若未来也改为概率资格，其失败应同样纳入 session 联合预算；V0 的确定性数值前提下不需另加一项。

当 \(\gamma_{task}\ge0\) 时，safe-native 被替换成 unsafe challenger 必有 \(\Delta<0\le\gamma_{task}\)，所以 H 是 \(R_\gamma\) 的子事件。这一推论正确。全部事件使用同一个有限 \(T(Z)\)，自适应选择没有额外的多重比较漏洞。

## 3. Appearance-gradient stability：正确，未偷渡 OOD 保证

相同物理场景、相同节点标签和固定正尺度下，逐边有

\[
r_e(X')=r_e(X)+D[u(X')-u(X)]_e/s_e.
\]

取 sup norm 得到稿中的三角不等式；对成对对齐的 session 时刻再取最大值也成立。该式不推出 \(\rho\) 小，不推出 clean residual 小，更不推出 OOD coverage。稿件明确将它作为机制诊断，并要求额外分布联系与可靠 session 上界才能升级保证，边界合格。

## 4. Decoder 协议：足够作为 V0 规范

正深度、视锥有效、真实 crop/resize 到 token coordinate、固定 missing-evidence token 以及 calibration 前冻结聚合规则，已经给出了实现入口。尤其是 frustum-valid 不等于 occlusion-visible 的区分正确，没有假设被遮挡道路已被观测。

无有效投影的节点仍属于完整场 target，不能因为输入 missing 而从 calibration maximum 中删除。完全 missing 节点能否由当前 decoder 产生有用场预测，由 Gate B 决定；不需追加网络来预先保证成功。

token layer、坐标 convention、相机顺序和聚合实现仍需进入首次运行的 immutable config。这属于既定执行资格，不构成新方法缺口。

## 5. Frozen candidate interface：正确；新增 usefulness 定义未闭合

候选接口明确只用部署信息、不用 GT 或 paired clean target image，所有方法共用候选与 eligibility，规则和 provenance 事前冻结。这正确拆开了 arbiter 与 generator 的职责。

但第 11 节的三个概率目前不是同一统计对象：

- \(S_{better}\) 是“存在某个候选”的事件；
- \(R_{cert|better}\) 与 \(R_{useful}\) 中的 \(j\) 没有定义成随机候选、存在量词或实际选中 index；
- 因而它们不能构成当前宣称的供给—认证—实际干预分解。

**必须修改 1：统一抽样单位与量词，并区分可认证机会和实际选中效用。** 一个最小修复如下。

先固定评估分布 \(W\)：例如 session 等权、在每个 session 的预注册 \(T(Z)\) 内取固定权重的 \((Z,t)\)。该分布与 session 风险 H 不混用。令

\[
J_{better}(Z,t)=\{j:j\text{ eligible},\ \Delta_{j0}^{GT}\ge\delta_M\},
\]

其中 GT margin 使用与连续 nominal theorem 一致的 teacher。定义

\[
B=\{J_{better}\ne\varnothing\},\qquad
C=\{\exists j\in J_{better}:L_{j0}^{cert}\ge\gamma_{task}\}.
\]

那么可以严谨报告

\[
S_{better}=P_W(B),\quad
R_{cert|better}=P_W(C\mid B),\quad
R_{available,useful}=P_W(C)=S_{better}R_{cert|better}.
\]

若 \(P_W(B)=0\)，conditional recall 未定义，应标 N/A。

再单独报告实际选择规则的

\[
R_{selected,useful}
=P_W(I_t=1,\ \Delta_{sel,0}^{GT}\ge\delta_M).
\]

后者不等于 \(P_W(C)\)。例如有一个改善 0.2 的已认证候选，但 least-deviating 规则选择了另一个改善为零、也通过 \(\gamma_{task}=0\) 的候选：存在可认证机会，却没有实际有用干预。

这也意味着实际效用除了供给和证书，还受 selection rule 影响。应同时报告 eligibility yield；\(S_{better}\) 近零说明 eligible pool 缺少改善机会，可能来自生成器，也可能来自 geometry/progress filters，不能唯一归因于候选生成器。

## 6. Anti-inclusion：意图正确，面积对象写错

**必须修改 2：面积使用连续 footprint 差集，而非有限节点集差集。**

当前 \(A_j,A_0\) 是有限节点集合，其连续面积为零。不能直接对 \(A_j\setminus A_0\) 写连续面积门槛。应改为

\[
\operatorname{area}(F_j\setminus F_0)>a_{min},\qquad
\operatorname{area}(F_0\setminus F_j)>a_{min},
\]

或给出针对这两个连续差集的、具有几何保证的 cell-union 面积下界。简单将不同节点数乘 cell 面积，不自动得到保守下界。

相同时域/progress budget、超过网格容差的连续 displacement、严格 GT 改善与排除单个边界 cell 的其余条件合理。mean displacement 与 Hausdorff 二选一或如何组合，需要在配置中预注册；无需增加新指标体系。

## 7. 最小性与冻结条件

当前仍是一个 frozen backbone、一个共享节点 decoder、固定 edge scale、一次 native-seeded exact transform 和一个几何资格接口。新增 label-error ledger、风险事件与诊断指标没有增加 trainable module。保持最小 V0 的判断成立。

Gate B 使用合法 source-only predictor 或最小 probe，已经修复上一轮 predictor-before-training 的顺序歧义。

**冻结前只需修复第 11 节两项定义问题：**

1. 统一 supply、conditional certification、available opportunity 与 selected utility 的统计单位和量词；
2. 将反包含面积改到连续 footprints 或其可靠面积下界上。

其余理论与实现合同可保留。完成这两处后，可以冻结 v8.3.1 并进入 Gate A/B，不需要再增加方法组件或修改核心证明。

本审稿不授权任何已验证系统、性能提升、实时性、OOD guarantee 或顶会原创主张。

</details>

## Post-fix verification

已复核修订后的第 11 节，仅检查原审稿的两项冻结阻塞；未运行实验或修改 proposal。

1. **统计单位与量词：已解决。** 所有 usefulness 指标统一使用先均匀抽 session、再均匀抽冻结时刻的 \(U=(Z,t)\)。B 是 eligible、meaningful 且 GT-better 候选存在事件，C 要求同一候选还通过认证，D 则考察实际 selected challenger。因而 \(D\subseteq C\subseteq B\)，并在 \(P_U(B)>0\) 时满足

   \[
   R_{opportunity}=S_{better}R_{cert|better},\qquad
   R_{selected}\le R_{opportunity}\le S_{better}.
   \]

   供给、证书召回和实际选择效用已不再混用。实现时 \(P_U(B)=0\) 的条件召回应标 N/A；这里的 supply 指通过 eligibility 与 meaningful 条件后的候选供给，不等同于裸生成器产出。

2. **连续反包含面积：已解决。** 面积明确作用于 \(F_j\setminus F_0\) 与 \(F_0\setminus F_j\)。cell union 只能提供方向正确的保守下界，有限节点集不再被赋予连续面积。

**最终冻结：YES。** 原两项阻塞均已解除，可将 v8.3.1 冻结为条件理论与系统规范，进入 Gate A/B。此结论取代前文因这两项定义问题给出的冻结 NO；原始审稿保留不改。

该冻结不代表 Gate A/B 已通过，也不授权已验证系统、性能优势、实时性、OOD 保证或顶会原创主张。
