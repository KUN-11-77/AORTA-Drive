# AORTA 理论迁移审查：从数学工具到可证伪系统结构

状态：`CANDIDATE_THEORY_STACK__NOT_YET_METHOD_CLAIM`。本审查依据现有实验反例与官方
arXiv元数据，目的不是增加公式数量，而是判断哪些理论真正解决AORTA的已知痛点。

## 1. 问题锚点

在不重训冻结规划器的条件下，从受外观扰动的视觉观测中恢复稳定、决策充分且可信的任务
关系；只在证据足以可靠区分完整候选时干预，否则保留native。目标证据包括未见外观、三类
规划器、clean no-harm、行为净恢复和实时推理，不把视觉MAE写成驾驶安全。

## 2. 已知反例决定的理论需求

| 已知证据 | 被否定的简单解释 | 必须补入的理论结构 |
|---|---|---|
| H5/H6：关系场稳定且有任务信息 | 稳定性必然等于决策充分性 | 信息充分性与decision regret |
| H7BS2：2.5秒丢失128对完整时域差异 | 短前缀足够 | 完整时域候选函数 |
| H7BS4A：2个candidate零直接证据 | 空间包含等于视觉可用 | 部分可观测模型 |
| H7BS4A-D0：8个distinct pair差异区零证据 | 只看对称差即可 | 联合不确定集合与拒判 |
| set-min公共项反例 | 公共区域自然相消 | 共享变量鲁棒比较 |
| PEA/MG-select失败 | 全局特征稳定或矩对齐足以保护任务 | 任务约束的表示与决策审计 |

## 3. 立即采用的理论

### 3.1 离散外微分、Hodge结构与商空间

把96×64网格视为二维cell complex。令\(B_0\in\{-1,0,1\}^{|E|\times|V|}\)
为node-to-edge coboundary，\(B_1\)为edge-to-face coboundary，则

\[
B_1B_0=0.
\]

任务势\(u\in C^0\)产生edge关系\(z=B_0u\in\operatorname{im}B_0\)。连接图上
\(\ker B_0=\operatorname{span}(\mathbf1)\)，因此正确对象是

\[
[u]\in C^0/\operatorname{span}(\mathbf1).
\]

这给出三项实质约束：

1. **gauge不变性**：\(B_0(u+c\mathbf1)=B_0u\)；任意两个非空候选的set-min差也
   在全局常数平移下不变；
2. **可积性**：potential head保证\(B_1z=0\)。直接edge head可能产生不能由任何道路
   任务势解释的cycle circulation；
3. **结构消融**：potential、direct-edge、direct-edge+cycle penalty必须容量匹配，并报告
   cycle residual、task error、ranking regret与时延。

该迁移来自离散外微分/Hodge，而不是把局部坐标frame的gauge-equivariant mesh CNN误写成
AORTA的additive gauge。后者研究局部方向frame等变性，只作术语边界参考。

### 3.2 信息充分性：不变性与有用性缺一不可

令\(S\)为物理场景、\(A\)为外观nuisance、\(X=g(S,A)\)、\(C\)为候选panel、\(Y\)为
任务关系或最优候选。表示\(Z=T_\theta(X)\)需要同时满足：

\[
I(Z;A\mid S)\ \text{小},\qquad
\mathcal R_{decision}(Y,g(Z,C))\ \text{小}.
\]

常数表示使第一项为零却没有任务信息，因此appearance invariance不能单独成为目标。AORTA
不直接用高方差mutual-information estimator训练，而使用可审计的操作化条件：same-scene
关系漂移、correct/mismatch/removal、teacher task error、non-collapse和候选regret。信息瓶颈
只提供“最小充分而非最大不变”的理论原则。

### 3.3 图上的相关不确定集合与精确鲁棒比较

网络输出中心势\(u_\theta\)与edge预算\(w_\phi\)，定义

\[
\mathcal U_\theta(X,O)=
\{d\in C^0/\mathbf1:\ |B_0(d-u_\theta)|_e\le w_{\phi,e}(X,O),\ \forall e\}.
\]

\(O\)是投影support与缺失模式。对候选\(F_i,F_j\)使用同一个\(d\)定义

\[
L_{ij}=
\inf_{d\in\mathcal U_\theta}
\left[\min_{q\in F_i}d_q-\min_{q\in F_j}d_q\right].
\]

现有difference-constraint solver已能对固定edge budgets精确计算该量。关键不是公式更长，
而是共同节点、路径相关性和gauge由同一个可行集保存；两个独立candidate interval相减会丢失
这种结构。未观测区不删除，必须通过更宽预算降低可裁决性。

需要新增的理论义务：support模式如何给出预算下界；learned scale如何防止在missing区域虚假
收窄；source calibration在NavDream shift下只能提供经验覆盖，不能宣称任意OOD保证。

### 3.4 选择性预测与Simplex运行时保障

AORTA不是替代planner，而是supervisor。native是fallback，challenger是advanced action。
只有\(L_{j0}\ge\gamma\)且所有非field合同通过时才替换，否则abstain。若真实任务场
\(d^*\in\mathcal U_\theta\)，则

\[
L_{j0}\ge\gamma\quad\Longrightarrow\quad
R(d^*,F_j)-R(d^*,F_0)\ge\gamma.
\]

这是条件性排序命题，不是车辆安全定理。部署评价必须同时报告coverage与conditional risk，
并把新增hard failures作为一级指标。该结构与selective classification、conformal risk control
及Black-Box Simplex一致，但source exchangeability不能自动延伸到未知NavDream分布。

## 4. 条件采用的理论

### 4.1 Cellular sheaf / sheaf Laplacian

多相机、多历史帧在不同坐标系下提供局部证据，restriction maps可描述它们到BEV任务节点的
兼容关系；global section对应一致融合，sheaf Laplacian可度量冲突。它可能比普通masked
attention更自然地处理非对称、带符号和缺失关系。

但当前证据只证明“存在未直接观测节点”，没有证明主要瓶颈是跨视角restriction冲突。因此
sheaf模块暂不实现。只有当view-conflict审计显示masked fusion无法区分一致证据与冲突证据，
且轻量固定restriction baseline失败时，才授权sheaf消融。

### 4.2 连续坐标场 / neural operator

连续查询可避免固定网格尺寸与分辨率绑定，支持速度自适应范围和近细远粗采样。但H7BS3已经
证明96×64包含完整候选，D0失败来自可观测性而非前向空间不足；连续坐标不会凭空产生视觉
证据。因此它只在1m离散aliasing被teacher-criticality证明影响排序，或96×64时延失败时进入。

### 4.3 Decision-focused learning

pairwise/listwise ranking loss可以直接优化候选顺序，但会把当前candidate provider的偏差写入
表示，并削弱跨planner的candidate-free主张。因此它应作为结构对照或最终轻量fine-tune，不能
取代任务关系teacher。必须在相同候选、相同容量下比较edge supervision、direct ranking与联合
训练，并在未见planner上验证。

### 4.4 Distributionally robust / weighted conformal

若能明确定义并估计source到NavDream的density ratio或Wasserstein半径，可研究shift-aware
risk bound；当前没有这些可验证条件。现阶段只能把NavDream作为冻结经验OOD评价，禁止用
“DRO/conformal”术语暗示任意天气保证。

## 5. 明确不迁移的装饰性理论

- 不因名称高级引入双曲几何、黎曼优化或因果发现；当前没有对应结构性证据；
- 不把local-frame gauge equivariance与additive task-potential gauge混为一谈；
- 不以信息瓶颈数值估计替代task error和regret；
- 不以evidential softmax、entropy或单个confidence head冒充联合不确定集合；
- 不以更多attention层、diffusion planner或RL重训原planner掩盖视觉任务接口失败。

## 6. 文献与迁移边界

| 文献 | 状态 | 可迁移内容 | 不可直接主张 |
|---|---|---|---|
| Desbrun et al., *Discrete Exterior Calculus*, 2005 | 基础理论/preprint | cochain、coboundary、离散微积分 | 自动产生视觉鲁棒性 |
| Roddenberry & Segarra, *HodgeNet: GNNs for Edge Data*, 2019 | preprint | edge signal与Hodge Laplacian | 与AORTA方法等价 |
| Achille & Soatto, *Emergence of Invariance...*, 2017 | preprint | minimal sufficient representation原则 | MI可可靠直接估计 |
| Ahuja et al., *Invariance Principle Meets IB*, 2021 | preprint | invariance alone不足 | 任意OOD可解 |
| Mandi et al., *Decision-Focused Learning...Ranking*, ICML 2022 | peer-reviewed | ranking/regret对照 | candidate-free泛化 |
| Angelopoulos et al., *Conformal Risk Control*, 2022 | preprint | monotone risk under stated assumptions | arbitrary NavDream coverage |
| Geifman & El-Yaniv, *Selective Classification...*, 2017 | preprint | risk–coverage与reject option | 自动驾驶安全保证 |
| Mehmood et al., *Black-Box Simplex*, NFM 2022 | peer-reviewed | supervisor/fallback系统结构 | native planner本身安全 |
| Hansen & Gebhart, *Sheaf Neural Networks*, 2020 workshop | workshop | relation-dependent diffusion | 当前必须使用sheaf |
| de Haan et al., *Gauge Equivariant Mesh CNNs*, ICLR 2021 | peer-reviewed | 局部frame等变性参考 | 等同于additive gauge |

官方元数据与摘要保存在本目录`sources/arxiv_*.xml`及相应`.meta.json`，没有下载或解析PDF，
因此方法细节结论限于摘要与项目已有理论；进入实现前仍需针对最终选中的3–5篇读取正文。

## 7. 当前裁决

立即冻结为候选核心：**Hodge-consistent quotient representation + support-conditioned correlated
graph uncertainty + exact shared-variable robust readout + selective Simplex arbitration**。

它不是四个并列创新，而是一条单一机制链：视觉只提供部分任务关系；Hodge结构保证关系可积；
联合不确定集合保存缺失与共享相关性；精确readout判断排序是否可识别；不可识别就回退。

下一步只允许teacher-criticality和oracle coverage实验判断这条链是否非平凡。在这两个门通过前，
不新增sheaf、连续operator、PEA升级或planner训练。
