# Round 1 Review

<details><summary>Raw reviewer response</summary>

# AORTA-v8 原始方法审稿

审查范围：完整阅读了 v8 提案、GROUNDING_NOTES，以及项目的 Research Brief、Method Specification 与 Claim–Evidence Matrix。以下判断基于这些材料；未执行实验、未修改文件，也未把已有 grounding 当作完整查新。

## 1. Problem Anchor 复述与检查

我理解的原问题是：同一物理场景发生外观变化后，冻结的端到端 planner 可能产生道路违规轨迹。研究希望利用冻结视觉基础模型建立一个可跨 planner 使用的任务接口，使干预依据与道路约束保持一致，而不是追求全局表征不变。

v8 选择解决其中更窄、也更清楚的问题：绝对道路安全零点不可靠时，能否预测 challenger 相对 native 的道路裕度差，并以受控的 source-session 风险授权替换，使原本道路可行的 native 不被替换成道路违规轨迹。它不承诺绝对安全、碰撞安全、总体驾驶性能不退化或任意 OOD 保证。

这仍然围绕原问题，没有滑向地图重建、重新训练 planner 或泛化的安全系统。候选、意图、动力学资格和 source-only 边界也基本保留。

但有两个范围变化必须明确：

1. v8 从“绝对 unsafe→safe 的修复证书”收缩为“相对 native 不新增道路失败”。这是合理收缩，但两者不能作为同一个成功条件。相对裕度改善可以发生在两条轨迹都违规时，也可能不足以改变任何道路失败。
2. 项目现有 Method Specification 和 Claim–Evidence Matrix 仍以 affine factorization、绝对 anchor 和绝对仲裁为主。v8 若成为主方法，需要同步问题合同，不能继续借用旧版本的贡献与验收标准。

我的总体判断是：**问题保真度较高，但方法主张尚未闭合到可以支撑独立顶会贡献。当前最大风险是把一个正确的任务重定义与标准校准流程，包装成尚不存在的方法新颖性。**

## 2. 评分与结论

| 项目 | 权重 | 分数 | 加权贡献 |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 8/10 | 1.20 |
| Method Specificity | 25% | 6/10 | 1.50 |
| Contribution Quality | 25% | 5/10 | 1.25 |
| Frontier Leverage | 15% | 7/10 | 1.05 |
| Feasibility | 10% | 6/10 | 0.60 |
| Validation Focus | 5% | 8/10 | 0.40 |
| Venue Readiness | 5% | 5/10 | 0.25 |
| **OVERALL SCORE** | **100%** | **6.25/10** | **6.25** |

**Verdict：REVISE。**

这里的 REVISE 不是“再补几个实验即可”。需要先修正可辨识性表述、support 的物理语义和统计合同，并正面处理与简单基线的算法等价性。若这些完成后，唯一剩余区别仍是 soft-min 和尺度头，则需要重新选择论文的主贡献。

**Drift Warning：AMBER。** 尚无明显研究方向漂移，但存在成功条件收缩及新旧规范不一致。禁止把“相对道路裕度不下降”改写为“恢复安全驾驶”。

## 3. 七项评分依据及修复要求

### Problem Fidelity：8/10

v8 对原问题的回答比 v7 更诚实：它承认不能从不可靠的绝对零点直接推出安全，又把干预目标限定到明确的道路裕度事件。删除完整 anchor 分布和 panel transformer 也使方法更集中。

扣分来自剩余的叙事跳跃：“绝对锚点可能不可辨识”并不能推出“相对算子就是可辨识对象”。这不是措辞细节，而是 Method Thesis 的关键前提。

### Method Specificity：6/10

候选差值、正尺度、session 最大残差、有限 conformal 索引和仲裁规则已经给出，明显好于概念性方案。直接对 contrast 校准，也确实避免了 v7 固定 shape 而漏掉局部误差的问题。

尚未闭合的是：

- support decoder 的局部输出到底代表什么；
- soft-min 与真实最小道路裕度的关系；
- 校准标签、离散教师、名义轨迹及部署轨迹之间的误差分工；
- session 风险事件与条件干预风险是否使用同一个统计单位；
- 跨 planner、候选插入和不同 session 长度时保证的适用范围。

**方法级修复，P0：** 写出单一 target ledger：连续真实裕度、教师计算裕度、网络预测 target、部署裕度分别是什么，以及每项误差由 residual calibration 还是确定性 \(\eta\) 覆盖。随后重写 Proposition 4 的完整随机实验。

**方法级修复，P0：** 在“潜在 support pooling”和“有物理含义的局部道路势”之间二选一。不能继续用前者的监督，声称后者的可解释性与几何保证。

### Contribution Quality：5/10

目前最主要的算法与标准 normalized split conformal 应用于候选差值预测没有本质区别。现有五个命题分别是图关联矩阵的核、infimum 的稳定性、标量单调性、同时覆盖后的选择合法性，以及 Hausdorff 稳定性。它们能使方法严谨，但不足以自动构成新的方法理论。

提案把这些对象对齐是有价值的研究设计；问题在于尚未说明这种对齐带来了简单基线没有的算法能力。

**方法级修复，P0：** 把“相对 scalar score＋session 最大残差阈值”写成完整算法，并明确 GCCO 与它的差异。如果差异只有 support pooling 和异方差尺度，则主贡献必须落在一个具体、可解释且必要的结构上，而不能继续落在商空间与同时校准本身。

**方法级修复，P0：** 将“最大可辨识对象”收回为“给定共同平移 nuisance 时的最大不变量”，除非能补上关于观测分布的识别假设。

### Frontier Leverage：7/10

冻结 DINOv3 的使用自然：它提供局部视觉证据，新增学习集中在道路决策接口；没有为了前沿感强加 VLM、语言推理或额外世界模型。

但基础模型不会自动补齐不可见道路几何。局部 support 查询能改善证据利用，不会把遮挡后的道路边界变成已知量。这里需要依靠明确的统计风险边界，而不是 foundation model 的名称。

### Feasibility：6/10

两个新增组件的预算合理，source-only 标签也具有明确来源。但计算预算目前缺少决定成本的量：历史帧数、相机数、候选数、每条轨迹 support 数、视觉 attention 范围及缓存策略。

更严重的是，统计预算可能比 GPU 预算更紧。99 个 calibration sessions 只使 99% conformal 阈值有限，不意味着阈值有用，更不意味着固定模型达到了高置信度的 1% 部署风险上界。

**方法级修复，P0：** 核实真正独立的 calibration 与 validation 单位数量；同一 recording 的帧、corruption 或候选不能增加独立 session 数。

**方法级修复，P1：** 固定一个可实现的最小接口，包括有限候选策略、时间集合、support 密度、缺失值处理和可证明的 \(\eta\) 上界。没有这些，训练时间和部署成本估算不具有可审查性。

### Validation Focus：8/10

验证草图的方向正确：同时看 coverage、sharpness、干预率、真实 contrast 和新增道路失败，也保留了简单基线与失败停止条件。没有用整体 PDMS 掩盖证书失效，这是优点。

需要避免把验证扩展为大量模块排列。先判定共同误差结构和最简单校准 baseline 是否有非零效用，再讨论 support/evidence 架构。

### Venue Readiness：5/10

尚无充分创新边界，也没有完成核心定理的统计对象定义。当前更像一份较好的研究合同，而不是已形成的 NeurIPS/ICML/ICLR 方法。

**方法级修复，P0：** 先回答“相同 backbone、候选、资格层和校准下，简单 baseline 具体缺少哪一种能力”。回答必须是数学或结构差异，不能只是“我们更完整”。

**方法级修复，P1：** 完成与 conformal selection/ranking、trajectory-conditioned implicit fields、runtime assurance 的针对性边界核对。现有三篇 grounding 还不足以支持独立方法新颖性。

## 4. 关键反例与数学检查

### 4.1 最大不变量成立，但统计可辨识性没有随之成立

Proposition 1 在其代数域上成立：对于连通候选图，

\[
\ker B=\operatorname{span}\{\mathbf 1\},
\]

所以 \(B\mathbf m\) 确实是共同平移作用下的最大不变量。

实际网络通过 \(s_j-s_0\) 输出 contrast，校准也直接使用 contrast residual，这一层面是对齐的。不能把这一点批评为“仍然漏掉 shape error”；当前校准已经可以覆盖任意预测误差，只要其统计条件成立。

但以下推论不成立：

> 绝对安全锚点不可辨识，所以候选差值是视觉下可辨识的最大有用对象。

反例：两种物理场景在观测中完全相同，区别是被遮挡的右侧路肩。

\[
\mathbf m(S_1)=(0.2,0.3),\qquad
\mathbf m(S_2)=(0.2,-0.3).
\]

native 均走可见道路，challenger 进入不可见区域。两种场景的 contrast 分别为 \(0.1\) 与 \(-0.5\)，不是共同平移可以消除的歧义。绝对不可辨识与相对不可辨识可以同时发生。

因此，商空间是**选定 nuisance 模型下的任务表示**，不是由当前观测模型推导出的可辨识结果。

最低限度的修复是撤回“可辨识”断言，把经验假设写成：

> 在指定候选域与观测条件下，contrast 的条件不确定性比绝对 margin 更小，或更容易由可用证据预测。

该假设可以被检验，也允许失败。

另外，Proposition 2 允许共同尺度 \(a\) 改变，但方法只消除了平移。由其假设真正得到的是

\[
|\widetilde\Delta-\Delta|
\le |a-1||\Delta|+2\varepsilon.
\]

它能支持排序符号的充分条件，不能直接支持“米制 contrast 数值稳定”。若 \(a\) 也是主要 nuisance，需要约束它接近 1，或将主张限定为符号稳定。不能把未知尺度留给网络后，继续用消除共同平移解释所有鲁棒性。

### 4.2 简单基线与 GCCO 存在直接算法等价

考虑最简单的 baseline：

\[
\widehat\Delta_{j0}=s_\theta(X,\tau_j)-s_\theta(X,\tau_0),
\]

\[
R_i=\max_{t,j}(\widehat\Delta_{j0,it}-\Delta^*_{j0,it}),
\]

再取相同的 session conformal quantile，并用

\[
\widehat\Delta_{j0}-q_\alpha-\eta_j-\eta_0\ge\gamma
\]

授权替换。

这就是 \(\widehat\sigma=1\) 时的 GCCO。加入预测尺度后，则成为标准 normalized conformal。商空间、相对不退化、选择后合法性和几何预算全部同样适用。

因此，“标准相对分数＋共享阈值＋标准联合校准”不是一个外围 baseline，而是当前方法的核心算法本身。

支持 GCCO 独立贡献所需的区别，不能是：

- 使用相对差值；
- 对最大 residual 做分位数；
- 相对 baseline 不退化；
- 不需要绝对零点；
- 在通过下界的候选里自适应选择。

这些都已经包含在该 baseline 中。

如果最终只有局部 support 架构带来差异，就应据此重写论文，而不是把标准校准对象改名当作主要算法贡献。

### 4.3 support soft-min 目前没有被监督为物理道路裕度

当前 decoder 输出 \(u_{i\ell}\)，但损失只监督候选级 contrast。这不能识别局部势，更不能识别 active constraint。

以两个等权 support 为例。对任意候选势 \(s\) 和任意 \(w\in(0,1)\)，令

\[
u_1=s-T\log(2w),\qquad
u_2=s-T\log(2(1-w)).
\]

它们产生完全相同的 soft-min \(s\)，但 activity weights 为 \((w,1-w)\)。也就是说，同一个正确的候选预测可以把“活动约束”放到任意 support 上。

因此，当前训练不足以支持：

> activity weights 明确定位决定真实最小裕度的时间—车体 support。

此外，即便 \(u\) 恰好等于真实局部距离，归一化 soft-min 也能反转最小值排序。取 \(T=1\)、等权重：

- native 的 support 距离为 \((0.1,0.1)\)，真实最小值和 soft-min 都是 \(0.1\)；
- challenger 为 \((-0.1,10)\)，真实最小值是 \(-0.1\)，但 soft-min 约为 \(0.593\)。

真实 contrast 为 \(-0.2\)，soft-min contrast 却约为 \(+0.493\)。

这不否定独立 conformal 校准的边际有效性；校准可以把这种误差计入 residual。但它否定了从该 pooling 形式直接取得物理 min 语义、活动约束解释和纯几何误差界。

这里有两个合格的修复方向：

1. **最简修复：** 把 \(u\) 和 soft-min 明确称为 latent pooling，不声称局部 clearance 或真实活动约束定位。所有预测误差通过独立 contrast calibration 处理。
2. **物理修复：** 令局部函数具有共享坐标语义，例如 \(u_\theta(X,q)\)，对同一物理点保持一致，并提供局部 contrast 或等价的物理约束监督；再明确 soft-min 温度误差及空间、时间离散误差的关系。

不要同时享受第一条的低监督成本和第二条的物理解释。

还有一个具体公式问题：若 \(\pi_{i\ell}\) 非均匀，soft-min 对局部势的活动权重应为

\[
\alpha_{i\ell}
=\frac{\pi_{i\ell}\exp(-u_{i\ell}/T)}
{\sum_r\pi_{ir}\exp(-u_{ir}/T)}.
\]

当前写法漏掉了 \(\pi\)。必须声明权重是否均匀、是否归一化，以及 missing support 如何进入计算。

### 4.4 同时下界确实支持选择，但只支持被校准的完整随机过程

对于固定网络、正尺度、固定候选生成策略和同一个 session 随机对象，

\[
R_{\rm new}\le q_\alpha
\]

确实推出该 session 内所有已纳入最大值的 contrast 同时受到下界覆盖。因此，从通过下界的候选中自适应选择是合法的，不需要再为最终 argmin 追加一次多重比较校正。

这是当前方法中成立的部分。

但以下扩展没有由此获得：

- 保持 \(K_{\max}\) 不变后，任意插入不同分布的候选；
- 换一个 native planner，而只要求它输出的轨迹仍在相同“域”内；
- 将离线 logged session 的保证用于已经受干预影响的闭环观测序列；
- 将校准时的有限时间集合扩展为任意长的运行。

候选插入不变是网络输出性质，不能推出插入后的统计覆盖不变。相同的候选预算也不能代替相同的候选生成分布。

跨 planner 若要共享一个 \(q_\alpha\)，应明确对哪个冻结 planner 集合取 session 内最大值。未纳入该策略的 planner 只能作为实证迁移，或另做 source calibration。

### 4.5 新道路失败界必须统一统计单位

建议将 Proposition 4 写成以下对象：

- \(Z\)：一个随机 source session；
- \(T(Z)\)：由固定协议定义的评估时间集合；
- \(H(Z)\)：该 session 是否至少发生一次“native 可行、被替换轨迹违规”；
- \(I(Z)\)：该 session 是否至少发生一次干预。

那么在完整覆盖事件下 \(H(Z)=0\)，可以得到

\[
P_{D_{\rm cal},Z}(H(Z)=1)\le\alpha.
\]

相应地，

\[
P(H=1\mid I=1)
\le\min\{1,\alpha/P(I=1)\}
\]

成立，前提是 \(H\) 和 \(I\) 都是上述 session 事件。

当前公式没有明确 \(I\) 的单位，容易被误读为“被干预决策中的失败率”。按干预次数汇总的风险、按帧汇总的风险和“每 session 至少一次失败”的风险不是同一个量，尤其是在 session 长度及干预次数不均衡时。

还应明确：普通 split conformal 给出的是对 calibration 数据与新 session 联合随机性的边际保证，不是对当前固定 calibration realization 的高置信度风险上界。文中 299 个独立零失败验证单位的计算，只有在独立性和风险对象都明确后才适用。

### 4.6 插入不变不必牺牲真实算子，但不能被过度解释

真实静态道路裕度本来就是单轨迹函数 \(m_S(\tau)\)。因此用共享 scalar potential 再作差，原则上不会损失真实函数的表达能力。不能仅凭没有 panel transformer，就断言方法无法表达必要关系。

但插入不变也不要求完全禁止 pair-conditioned 证据比较。固定 \((\tau_j,\tau_0)\) 的成对比较，同样可以不受其他候选插入影响。

在有限容量与部分可观测条件下，共享 support、增量扫掠区域以及两条候选对同一边界的关系，可能是高效预测 contrast uncertainty 的关键信息。当前 pair scale head 是否保留了这些信息，尚不明确。

所以正确结论是：**插入不变是值得保留的接口约束；“因此删除所有候选关系建模”不是其数学必然结果。** 应以同样插入不变的 pair-conditioned baseline 检查这项限制是否损害统计效率，而不是重新加入整个 panel transformer。

### 4.7 \(\eta\) 预算有正确方向，但尚未形成部署合同

Hausdorff 距离控制 signed-distance infimum 的结论成立，但需要指明以下轨迹：

1. 输入候选；
2. tracker 得到的名义扫掠体；
3. 教师标签所对应的扫掠体；
4. 部署执行扫掠体。

若 calibration target 已经是连续真实裕度，则 support 采样引起的预测误差已经包含在 residual 中，再扣同一项会重复保守。若 target 只是离散教师裕度，则必须补上它相对连续 target 的误差界。

名义—执行误差也不能只写成一个符号。需要给出其条件和来源；没有上界资格的 execution discrepancy，不能进入确定性证书。

道路缓冲 \(m_{\rm buf}\) 定义的失败事件也可能不同于官方 DAC。后者应继续作为下游实证指标，不能直接继承该定理。

### 4.8 负 conformal quantile 会反转尺度头的语义

当前 one-sided residual quantile 可以为负。若 \(q_\alpha<0\)，则

\[
L=\widehat\Delta+|q_\alpha|\widehat\sigma.
\]

证据尺度越大，下界反而越高。例如 \(q=-1\)、\(\widehat\Delta=-0.1\)、\(\eta=0\)，当 \(\sigma=0.05\) 时不能替换，当 \(\sigma=0.2\) 时却能替换。

这不自动破坏同分布 conformal 的边际有效性，但与“证据变差时证书更保守”的设计语义冲突。

最简单修复是使用

\[
q_\alpha^+=\max(0,q_\alpha),
\]

保持覆盖的保守性，并保证下界对尺度单调不增。若不采用，应明确尺度并不是具有该单调意义的 uncertainty。

## 5. 与三篇最近邻的创新边界

依据提供的 grounding：

- **DriveReferee** 已有同图相对 margin 和共同地图偏差部分抵消。v8 的直接 contrast target 是可辨别的实现选择，但“消除共同锚点”不能再独立承担新颖性。
- **Conformalized Distance Fields** 已涉及联合校准、残差和几何离散化。v8 避开完整场预测是有意义的任务收缩；但转为直接校准有限维 contrast，本身也是标准 conformal target 的选择。
- **SPIBB** 已确立相对 baseline 的安全改进思路。v8 的逐决策道路裕度合同不同于策略回报保证，不能借其语言暗示更强的序贯安全。

这些重叠没有证明 v8 必然无价值，但使目前的 novelty argument 不足。剩余可能有价值的贡献，是一个有明确适用条件、确有统计收益的视觉 contrast 接口；它必须超过“将相对道路评分器接上标准校准器”。

## 6. Simplification Opportunities

1. **先承认并落实最简算法。** 一个 scalar trajectory decoder、固定尺度、session 最大残差阈值和当前仲裁器，已经包含主要理论合同。其他结构必须证明必要性。
2. **删除未经识别的 active-constraint 语义。** 若不增加物理资格，就把 support soft-min 作为普通 pooling，并避免用其权重制作“解释正确”的主张。
3. **缩减理论包。** 将平移不变量、标量不退化和 Hausdorff 稳定性作为支持性引理。主定理只负责清楚表达完整 session、候选策略与选择风险，避免五个常识命题制造贡献数量感。

## 7. Modernization Opportunities

1. **把冻结基础模型用于稀疏、共享坐标的局部证据接口。** 若局部几何语义必要，优先让同一物理 support 在不同轨迹中共享含义，而不是增加 dense BEV 或更大的推理模型。
2. **使 uncertainty 直接服务一侧决策误差。** 保留正尺度与独立校准，处理负分位数，并检查 pair 共享 support 是否能解释乐观误差。目标是改善同 coverage 下的有效干预，而不是证明尺度随 corruption 强度单调增长。

这两项都应替换不合格设计，而不是追加成新的模块层。

## 8. 排序行动项

### CRITICAL

1. **P0：解除可辨识性误推。** 区分 margin 空间的代数不变量与视觉观测的统计可辨识性；加入局部遮挡反例，收缩主张。
2. **P0：给出与简单 baseline 的逐式等价关系。** 明确真正剩余的算法差异。若没有差异，立即重写 dominant contribution。
3. **P0：关闭 support 语义缺口。** 选择 latent pooling 或物理局部势；修正 \(\pi\)、soft-min bias、missing support 和 active localization 的资格。
4. **P0：重写风险合同。** 明确定义 session、时间集合、候选生成策略、失败事件、干预事件，以及边际保证的随机性。

### IMPORTANT

1. **P1：关闭几何 target ledger。** 区分教师、名义和执行裕度，确保误差预算不遗漏也不重复。
2. **P1：限定跨 planner 与插入覆盖范围。** 输出插入不变不能代替校准分布资格；明确共享阈值覆盖的 planner/候选策略。
3. **P1：处理负分位数与预算资格。** 采用 \(q^+\) 或明确替代语义；固定候选/support/时间预算，并核实独立 session 数量。

### MINOR

1. **P2：修正 Proposition 2 中的 `a,d_S` 排版；将道路缓冲记为 \(m_{\rm buf}\)，避免与 native margin \(m_0\) 重名。**
2. **P2：将“排序误差尺度”改为“米制 contrast residual 尺度”。** 当前 NLL 拟合的不是纯 ranking error。
3. **P2：同步新旧项目规范。** v8 不再负责 affine anchor set 或 unsafe→safe rescue，旧 claim matrix 不能继续作为其验收表。

当前方案值得继续的方法部分是：明确的道路裕度 target、直接 contrast 监督、固定候选下的一侧 session 同时校准，以及克制的仲裁合同。最需要撤回的部分是：由商空间代数自动取得可辨识性、由弱监督 soft-min 自动取得物理活动约束，以及把标准校准组合视为已经成立的独立主贡献。

</details>
