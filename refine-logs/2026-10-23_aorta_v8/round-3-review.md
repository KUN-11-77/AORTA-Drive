# Round 3 Review

<details><summary>Raw reviewer response</summary>

# AORTA-v8.2 Round 3 原始方法审稿

审查范围：完整阅读了 `round-2-refinement.md`，沿用此前已读取的项目问题定义、GROUNDING_NOTES 与前两轮方法记录。本轮依照 research-refine 的七项量表审查，没有修改方法文件，也没有运行实验。本审稿没有完成新的系统查新，因此不作“先行文献已经明确包含完全相同公式”的断言；下面的创新性质疑主要依据可直接验证的数学归约和最近邻对照缺口。

## 1. Problem Anchor 复述与检查

问题仍是：同一物理场景发生外观变化后，冻结 planner 可能输出道路违规轨迹；研究希望建立与物理道路约束一致的视觉任务接口，在绝对安全零点不可靠时，为 challenger 相对 native 的替换提供有明确风险边界的依据。

v8.2 将接口具体化为共享网格上的节点势和局部边差分不确定集，再对任意 swept-set pair 求最坏道路 margin 差。它仍然只承诺相对道路不退化，不承诺绝对安全、碰撞、交通规则、真实执行或任意 OOD 保证。原 planner 与视觉 backbone 继续冻结。

因此，本轮没有实质性问题漂移。图商空间、局部差分和鲁棒 set-min 读出是解决同一问题的机制变化。

需要防止的是研究价值的替代：证明一个标准差分约束集合可以用最短路求解，并不自动证明这个集合特别适合外观退化下的道路证据，也不自动形成顶会主贡献。问题锚点要求的是可靠且有用的道路干预接口。

**Drift Warning：NONE。** 当前任务目标保持一致。全域道路场的建模成本和图算法复杂度仍需接受必要性审查，但这不等于已发生问题漂移。

## 2. 评分与结论

| 项目 | 权重 | 分数 | 加权贡献 |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 8/10 | 1.20 |
| Method Specificity | 25% | 7/10 | 1.75 |
| Contribution Quality | 25% | 6/10 | 1.50 |
| Frontier Leverage | 15% | 7/10 | 1.05 |
| Feasibility | 10% | 6/10 | 0.60 |
| Validation Focus | 5% | 7/10 | 0.35 |
| Venue Readiness | 5% | 6/10 | 0.30 |
| **OVERALL SCORE** | **100%** | **6.75/10** | **6.75** |

**Verdict：REVISE。**

核心最短路等式成立，且比上一版统一扣除全域半径的读出更强。但本轮尚有一个明确的连续证书条件缺失、一个尺度归一化设计问题，以及没有解决的创新边界。评分不会因增加图对象或得到一个正确闭式解而自动上升。

阻塞 READY 的主要事项是：

1. 连续下界只声明 challenger 的 internal-net 条件，遗漏 native 的必要采样方向条件；按当前文字可以产生假证书。
2. 逐帧尺度归一化被错误地解释成消除全局 scale gauge，实际上会删除整体观测质量信息并改变 uncertainty geometry。
3. 当前精确读出可以直接化为标准 min-plus 多源最短路变换；与图 Lipschitz band、差分约束和直接 CDF 联合集合的边界仍未建立。
4. 全局 conformal maximum 仍会让远处坏边抬高所有边的半径；当前“远处坏边只在必经路径出现时影响证书”的表述不成立。

## 3. Exact graph-quotient lower bound：等式成立，但应更精确地定位贡献

### 3.1 inf/min 合并正确，没有非法交换 min 与 max

令

\[
\mathcal Z=\{z: |Dz|_e\le w_e\},\qquad d=u+z.
\]

对任意固定的 \(z\)，有

\[
\min_A d-\min_B d
=\min_{a\in A}\max_{b\in B}(d_a-d_b).
\]

又因为 \(A\) 有限且可行集合不依赖 \(a\)，

\[
\inf_{z\in\mathcal Z}\min_{a\in A}g_a(z)
=\min_{a\in A}\inf_{z\in\mathcal Z}g_a(z).
\]

这里合并的是两个 minimization，不是在没有条件时交换 inf 与 max。原证明的这一部分是正确的。

### 3.2 最短路势同时实现所有上界，等号正确

固定 \(a\) 后可用 gauge freedom 令 \(z_a=0\)。由于约束是绝对边差分，任意定向 incidence matrix 对应的是无向的对称约束，故

\[
z_b\le \operatorname{dist}_w(a,b).
\]

对非负有限边权，函数

\[
z_v=\operatorname{dist}_w(a,v)
\]

满足每条边的差分约束，并同时实现所有 \(b\) 的上界。因此

\[
\inf_{z\in\mathcal Z}\max_{b\in B}(u_a-u_b-z_b)
=\max_{b\in B}[u_a-u_b-\operatorname{dist}_w(a,b)].
\]

再对 \(a\) 取最小值得到提案的等式。连通性保证距离有限；固定 gauge 后的可行域有界闭合，也可以明确写出最优值可达到。\(q=0\) 导致零权距离时不破坏结论。

应在定理中明确：有限连通无向图、非空集合、有限非负边权；incidence 的任意方向只作记号使用。不能把这里的距离换成未经论证的有向道路可达距离。

### 3.3 该读出等价于一次标准多源距离变换

令 native 集合为 \(B\)，定义

\[
\psi_B(v)=\min_{b\in B}\left[u_b+\operatorname{dist}_w(v,b)\right].
\]

则原公式直接变成

\[
L_G(A,B)=\min_{a\in A}[u_a-\psi_B(a)].
\]

这不是为每个 \(a,b\) 运行一次 shortest path，也不是每个 challenger 都需要独立求解。可以对所有 native 节点用初始标签 \(u_b\) 做一次多源 Dijkstra；若要保持普通非负 super-source 边形式，将所有初始标签减去 \(\min_B u\) 即可。

对同一个 native，计算量为

\[
O((|\Omega|+|E|)\log|\Omega|)
+O\left(\sum_j|A_j|\right),
\]

内存可为 \(O(|\Omega|+|E|)\)。不同 native references 通常仍需分别计算 \(\psi_B\)。

这个等价形式应成为主算法，能显著澄清实时成本。同时，它也削弱“精确 robust transform 是独立新算法”的表述：该对象就是差分约束 Lipschitz 集上的 min-plus 最短路 envelope。定理可以正确且有用，但必须把新颖性定位在超出这一标准归约的地方。

## 4. Inclusion、快速近似与连续几何

### 4.1 inclusion corollary 正确

若 \(A\subseteq B\)，则每个 \(a\in A\) 的 maximum 包含 \(b=a\) 的零项，所以 \(L_G\ge0\)。当 \(A=B\) 时应得到精确的 \(L_G=0\)。这与真实 set-min 结构一致。

该结论不依赖视觉，因此 geometry-only baseline 仍必须保留。当前已经承认这一点，是正确修复。

### 4.2 快速近似的保守方向正确，但需一个实际算法合同

若对所有参与计算的节点对都有

\[
\widetilde d(a,b)\ge\operatorname{dist}_w(a,b),
\]

则逐项减去更大数，再取 max 和 min，确实得到

\[
\widetilde L_G\le L_G.
\]

但“GPU wavefront”本身不是这个方向的证明。截断的可行路径松弛、保留完整 predecessor 链的路径库可以给出上界；任意神经距离、启发式几何距离或普通插值不能自动取得该资格。

若要保留免费的 inclusion 证书，近似还应保持 \(\widetilde d(a,a)=0\)。它不是保守有效性的必要条件，却是保留该结构收益的必要实现要求。

既然精确解可化为一次多源 Dijkstra，应该先将该形式作为明确参考实现，再决定是否需要快速近似。不要先引入复杂近似来解决尚未定义的运行瓶颈。

### 4.3 连续扣项按当前文字不成立：遗漏 native 的方向条件

提案仅写：若 challenger 的 \(A_j\) 是连续 footprint 的内部 \(h_j\)-net，则使用

\[
L^{cont}_{j0}=L^\Omega_{j0}-h_j.
\]

这还不够。证明需要 native 的离散 margin 不低于其连续 margin，即

\[
M_0^\Omega-M_0^{cont}\ge0.
\]

例如要求 \(A_0\subseteq\mathcal B_0\)。原文没有在一般 rasterization 定义或该连续条件中保证它。

反例：场预测完全正确，\(q=0\)。native 连续 margin 为 \(+0.1\)，但外侧 rasterization 纳入一个不在其 footprint 中的道路外 cell，使 native 离散 margin 为 \(-1\)。challenger 的内部采样精确，连续与离散 margin 均为 \(-0.1\)，且 \(h_j=0\)。那么

\[
L^\Omega=(-0.1)-(-1)=0.9,
\]

提案规则会认证替换，但连续真实 contrast 为 \(-0.2\)，恰好把连续可行 native 换成违规 challenger。

**必须修复：** 对单侧扣项，至少声明 native \(A_0\subseteq\mathcal B_0\)，以及 challenger 的内部 \(h_j\)-net 条件。更容易统一实现的方式是让两者都满足相同的内部采样协议。若做不到，使用有效的双边 Hausdorff 预算。

另外，完整扫掠体位于认证 ROI、集合非空、几何近似资格有效，应成为仲裁入口的强制条件，而不是 failure-mode 备注。

## 5. Learned scale：conformal 有效性基本正确，归一化理由和设计需要修改

### 5.1 正尺度误设不破坏边际 source coverage

模型和尺度参数在 calibration 前冻结，尺度严格正且有限，session 非符合分数按同一规则计算。在交换性成立时，scale head 即使不准确，split conformal 的边际场事件仍然有效。

它可能严重损害锐度，但这与覆盖有效性不同。不能从 Laplace NLL 推导覆盖；提案在这点上表述正确。

### 5.2 真正的缩放等价只涉及一个全局常数的 s 与 q

如果对所有观测与所有边统一使用一个常数 \(c>0\)，且不触发新的 clipping，

\[
s'_e=c s_e,\qquad q'_\alpha=q_\alpha/c,
\]

则最终边权 \(q'_\alpha s'_e=q_\alpha s_e\) 不变。

这不是 \((u,s,q)\) 的整体缩放自由度。GT teacher 的米制单位固定，缩放 \(u\) 会改变 residual，不是 gauge。

更不是每个 frame 各有一个任意常数 \(c(X)\) 的自由度：由于 quantile 是跨 frames/sessions 的单个全局量，逐帧改变尺度不能由一个 \(q\) 精确补偿。

### 5.3 逐帧 geometric-mean 归一化删除了整体证据幅度

如果一个退化观测使所有边的原始尺度都应扩大相同倍数，逐帧约束

\[
|E|^{-1}\sum_e\log s_e=0
\]

会将这个变化完全消掉。它使尺度只能重新分配相对预算，不能表达这帧整体更不确定。

而且，提升某条原始尺度后，归一化会缩小其他尺度。例如两条边从 \((1,1)\) 变为原始 \((4,1)\)，geometric-mean 归一化后是 \((2,0.5)\)。第二条边的证书可能反而变得更乐观。因此，不能用该参数化支持“证据退化只会使证书更保守”的泛化表述。

有一个正确的单调性结论：给定 \(u,q\)，若最终尺度逐边非减，uncertainty set 扩大，\(L_G\) 不增。当前逐帧归一化不保证这一前提。

**具体修复：** 删除逐帧 mean-log 约束，保留 source-frozen 的正值范围与 NLL。若需要固定一个数值 convention，只用一个从 source train 得到的全局常数归一化，且在 calibration 前冻结；不要删除每观测的整体幅度。

若坚持逐帧约束，则必须承认它只建模相对空间尺度，不能承载全局观测质量。还须规定 bounds 与零均值约束的可行条件和执行顺序：先中心化再 clipping 一般不会同时满足两者，简单相减也可能将尺度移出声明的范围。

## 6. 局部图并没有自动消除全域极值问题

### 6.1 远处坏边仍能通过 q 影响所有证书

本轮最需要纠正的机制表述是：远处坏边只有在相关最短路必须经过它时才进入证书。

在固定 \(q,u,s\) 时，距离确实由相关路径决定。但 \(q\) 来自所有边、所有时刻的最大标准化 residual。一个远处坏边如果其尺度不足或触及上界，可以抬高 \(q\)，从而扩大图中所有边的 \(w_e=q s_e\)。

简单反例：相关区域 residual 为零、尺度为 1；远处一条无关边 residual 为 \(M\)、尺度也为 1，且这些 session 决定 calibration quantile。此时全局 \(q\) 达到 \(M\) 量级，相关区域的边权同样变大，即使任何相关最短路都不经过那条远处边。

所以图局部性解决的是给定边预算后的空间传递，不是自动解决 calibration maximum 的全域耦合。learned scale 可能缓解，但必须被视为可失败的条件，不能写成结构保证。

### 6.2 路径累积可能比 global gauge 更差

图集合只约束每条边的 residual 差分，允许误差沿长路径同向累积。链式图上，该增长可以达到路径长度乘以边预算；有环时最短路只允许绕开昂贵路径，并不保证误差会按随机噪声方式抵消。

如果 residual 是幅度有界的振荡场，global residual range 可能很小，而局部最大差分沿长路径相加会很大。若节点噪声幅度不随网格细化减小，路径边数增加也会恶化上界。因此图界并不普遍支配 global gauge 界。

“edge residual 比 absolute residual 小”也不是有效的结构资格。局部差分天然随边长缩小；常数预测器面对 1-Lipschitz teacher，其边 residual 本来就至多为相邻节点距离，但这并未恢复有用道路结构。应该比较米制集合对 bound 或按物理长度归一化的差分误差，而不是拿原始 edge MAE 与 node MAE 直接作优势证据。

### 6.3 图不是车辆通行图

这里的最短路传递的是 residual-difference 约束，不是车辆可行路径。只要每条边的覆盖事件成立，路径经过不可行驶区域不会使逻辑证明失效。

因此，“跨障碍捷径污染距离”需要改写。它可能影响模型拟合和统计锐度，但不能在同时承认所有边约束有效时，又把该路径称为不合法证据。反之，如果按预测道路拓扑删边导致图断开，就会出现多个独立常数模式；跨连通分量的 contrast 一般无有限下界，不能继续沿用单一 additive gauge 的结论。

## 7. 创新边界：图局部差分与精确读出仍是直接应用风险

当前集合

\[
|D(d-u)|_e\le q s_e
\]

就是残差上的加权图 Lipschitz／差分约束 polytope。它是 graph-gradient 的逐边 \(\ell_\infty\) 约束，不是 graph total variation 的求和预算；两者不应混称。

对该集合应用标准 split conformal，是对一个结构化非符合分数的直接校准。再对 set-min contrast 求最坏值，得到上一节的 min-plus shortest-path envelope。节点势自动 cycle-consistent 是表示正确性，不是独立创新。

本轮确实比全域单半径方法多了空间局部性和集合对专属读出。但“先行工作没有把完全相同的组件用于这一驾驶任务”仍不足以支撑方法级原创性。

与 CDF 的比较尤其不能限定为“若它最后只读取 absolute box/global norm 就不够”。最强的联合场方法应该直接对 relative set-min 目标进行鲁棒读取；它没有理由被迫丢弃相关性。当前把 CDF-style joint field 设为“可实现时加入”，与它作为最接近方法的地位不匹配。

应给出明确分界：哪些部分只是标准差分约束和 conformal，哪些部分提出了这些工具没有的道路证据建模或可验证计算收益。若不存在额外分界，应将论文定位为一个清楚的结构化应用或经验发现，而不是继续通过添加新的数学名词保住主贡献。

## 8. 最强 baseline 与可证伪性

### 8.1 global gauge baseline 也必须使用精确鲁棒 set-min 读出

若 GQCF 使用 exact readout，而 GCMF 仍统一使用 \(\widehat\Delta-2q\)，比较会混合两个因素：局部 uncertainty geometry 与更强鲁棒读出。

global gauge 集合 \(\operatorname{osc}(d-u)\le R\) 可以写成完整图上的统一 pairwise difference constraints，也可以直接计算其最坏 set-min contrast。它应同样利用 overlap 和 inclusion。

必要的同源比较是：

1. global gauge uncertainty set＋exact robust set-min；
2. local graph uncertainty set、固定尺度＋同样 exact readout；
3. local graph uncertainty set、learned scale＋同样 exact readout。

这样才能判断收益是否确实来自局部图，而不是 baseline 使用了已知更松的决策器。

### 8.2 图尺度需要有确定性几何控制

除 \(s=1\) 外，应有与物理边长相关的固定尺度方案，避免 learned scale 的增益仅来自对不同邻接长度或方向作基本单位校正。最接近的 graph-Lipschitz/difference-constraint 基线应读取相同的 \(u\)，并使用同一风险对象。

同一个 \((u,s,q)\) 下，通用差分约束优化与本文闭式式子应完全一致；这是等价实现，不是要求 GQCF 在统计表现上击败的不同方法。其角色是明确算法归约与正确性，避免把标准求解器换成闭式读出当作额外统计能力。

### 8.3 非包含干预必要，但不充分

\(A_j\nsubseteq A_0\) 可以只来自一个额外 raster cell 或一个几乎等价的轨迹，不能单靠它证明视觉方法取得了有意义的新决策。

需要同时报告：连续几何差异与网格分辨率的关系、非包含干预中的真实严格 margin 改善，以及相同资格和 progress budget 下的效用。原稿已有相关下游指标，应该把这些与“非包含成功”绑定，而不是只报告一个非包含比例。

这不要求扩展实验菜单，只要求对核心成功事件作更严格定义。

## 9. Source-session 风险量词

离散版本的统计论证成立：网络、尺度、图与时间协议在 calibration 前冻结，session score 对所有边和所有时刻取最大值；在交换性下得到一个场事件。在该事件上，Theorem 1 同时覆盖所有域内非空集合，包括看到 \((u,s)\) 后自适应产生的集合。因此任意通过下界的选择都不需要额外多重比较校正。

\(I\) 与 \(H\) 使用 session 事件，联合 calibration/test 随机性的无条件风险界，以及条件风险 \(\alpha/P(I)\) 的限制，也基本正确。

但“native 离散/约定连续 margin”必须在正式合同中选定，不能用斜线留作自由解释。连续版本只有在修复 native 几何条件后才能继承该风险界。若只取得离散资格，不能将结论写成连续道路失败不新增。

个别边的平均 coverage 不能替代 session 内所有边同时覆盖的主事件。变更 scale bounds、图邻接或输入协议后，也必须重新建立相应 calibration，不能继续沿用原阈值。

## 10. 七项评分说明及低分方法级修复

### Problem Fidelity：8/10

原道路外观鲁棒问题、冻结 planner 与 source-only 边界保持。图机制可以服务这个目标；不需要扩展到其他安全任务。扣分主要是“局部差分更容易学”仍缺少不受边长影响的清晰假设。

### Method Specificity：7/10

核心表示、目标、校准与精确读出已足够具体。但连续证书目前条件不足，尺度 bounds 与逐帧归一化也未给出自洽执行方式。它们属于方法合同问题，不只是实现细节。

### Contribution Quality：6/10

主机制集中，图局部性确有不同于全域半径的能力；但整体可归约为标准 graph-difference polytope＋conformal＋min-plus 读出。

**CRITICAL 修复：** 明确承认这些归约，加入 exact global-gauge readout 和直接联合场比较。将主贡献收敛到一个可以证明并可证伪的剩余差异；若没有，降低方法原创性定位。不要用新模块修补创新陈述。

### Frontier Leverage：7/10

冻结 DINO 提供局部证据是自然用法。没有理由追加 LLM、VLM 或 diffusion 模块。现代性不取决于是否把解析图算法替换成神经算法；后者反而可能破坏保守性。

### Feasibility：6/10

一次多源 Dijkstra 使计算可行性比原叙述更好，但 grid size、动态 edge weights、路径保守性和全局 q 的耦合仍未关闭。

**IMPORTANT 修复：** 用一次多源变换替代潜在的逐节点求解；固定实际图规模和 native 数量。删除不必要的逐帧尺度规范化，明确可失败的路径锐度条件。资格比较必须使用合法预测器产生的 held-out residual 和最终米制下界，不能用 GT 自拟合或原始 edge/node MAE 差异代替。

### Validation Focus：7/10

两项核心验证与停止条件合理，仍然聚焦。但最强 global-gauge 读出缺失，CDF 对照被设为可选，非包含事件定义也不够有力。修复这些核心控制即可，不需要扩大 benchmark 数量。

### Venue Readiness：6/10

本轮有一个可用的精确解析结果，但还没有证明它不是标准工具的直接道路应用；连续合同还存在明确反例。

**CRITICAL 修复：** 先修复几何与尺度合同，再以直接场级基线和标准 min-plus 归约重新裁定贡献。若最终只剩新应用，应据此重写论文目标；不能在当前状态标 READY。

## 11. Simplification Opportunities

1. **把主读出写成一次多源 min-plus 距离变换。** 一个 native 计算一次 \(\psi_B\)，所有 challenger 只做集合 reduction；不需要逐候选最短路或默认引入近似路径库。
2. **删除逐帧 mean-log scale 约束。** 它并非消除必要 gauge 的方法，还限制整体不确定性表达。数值规范可用一个预先冻结的全局常数完成。
3. **先保留固定尺度版本。** learned scale 只是效率辅助；如果无法提供超过物理边长尺度的必要收益，应按已有停止条件删除，而不是将双输出计为主贡献的必备部分。

## 12. Modernization Opportunities

**NONE。** 现有冻结基础模型＋稀疏共享物理查询已经足够自然。当前缺口是数学合同、校准几何和最近邻边界，不是缺少更流行的模型。不要用神经最短路、VLM 推理或额外 OOD detector 代替这些修复。

## 13. 必须修改项，按优先级排序

### CRITICAL

1. **修复连续界的 native 条件。** 单侧只扣 \(h_j\) 需要 native 离散 margin 对连续 margin 乐观；明确要求相应内部采样方向，否则使用有效双边预算。将 ROI、非空和几何资格加入仲裁入口。
2. **重写创新边界及最强对照。** 承认 graph-Lipschitz/difference-constraint 与 min-plus 距离变换归约；global gauge 与 CDF/联合场都应采用直接 relative robust readout，不能强制它们使用松的 absolute box。
3. **修复尺度规范化。** 区分全局 \(s\leftrightarrow q\) 等价与每观测的真实尺度信息；删除或严格重定义逐帧 geometric-mean 约束及 bounds 顺序。

### IMPORTANT

1. **纠正局部性的过强陈述。** 远处坏边可以通过全局 \(q\) 影响所有权重；只能承诺固定边预算下的路径局部性。
2. **明确一次多源求解的运行合同。** 给出节点/边规模、每帧 native 数量、exact 读出成本与近似的上界证明。
3. **修正结构资格与非包含成功定义。** 不以原始 edge residual 小于 absolute residual 当作证据；要求最终米制锐度，以及非包含干预的严格道路效用和连续几何意义。
4. **修正图错设解释。** 这是 uncertainty transport graph，不是车辆通行图；若断开，必须处理多个 offset 模式及跨分量无有限下界。

### MINOR

1. **明确无向约束、有限非负权重和最优值可达到。** 不要把 incidence 记号方向误作有向通行约束。
2. **快速近似保留零自距离。** 同时明确浮点容差与原本冻结的 \(\gamma\) 的关系。
3. **统一公式与术语。** 修复大量括号中缺少反斜线的数学记号，明确 \(H\) 的唯一 margin 层级，区分 graph-TV 和逐边 gradient \(\ell_\infty\) band。

当前最值得保留的是：候选无关的边事件、正确的精确 robust set-min 读出，以及清楚的离散 source-session 风险量词。当前不能接受的是：连续几何条件不足、逐帧尺度约束的错误理由，以及把标准图差分和 min-plus 归约本身当作已经成立的顶会主贡献。

</details>
