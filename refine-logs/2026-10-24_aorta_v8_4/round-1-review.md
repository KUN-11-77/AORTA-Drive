# Round 1 Independent Method Review

> Reviewer substitution: the skill-specified legacy reviewer model was unavailable;
> the review used the available frontier reviewer at xhigh reasoning. No files were
> modified by the reviewer and no literature novelty search was performed.

<details>
<summary>Full raw review</summary>

Verdict：**REVISE**。方案保留了原始问题，异质边预算与现有认证理论兼容，值得进入理论资格审查；但当前尚不能称为“最小充分机制”，也没有闭合顶会算法原创性。主要阻塞是尺度目标的表述、内部校准与模型选择的边界、新增数值运算，以及机制归因。

| 维度 | 分数 | 判断 |
|---|---:|---|
| Problem Fidelity | 9 | 仍解决 candidate-independent relative-margin certification，没有转向绝对地图重建或修改 planner。 |
| Method Specificity | 7 | 输出、边序、校准统计量和推理路径明确；训练冻结范围、内部划分、尺度边界与精确梯度积累仍不明确。 |
| Contribution Quality | 5 | 集中且简洁，但新增部分目前主要是 normalized conformal 的标准实例化；原定理已支持异质边权。 |
| Frontier Leverage | 9 | 冻结 DINOv3 表征与轻量 head 符合瓶颈，不需要引入 LLM、diffusion 或 RL。 |
| Feasibility | 8 | 工程增量小，已有图与几何基础可复用；组级训练的内存和梯度实现需要明确。 |
| Validation Focus | 6 | 实验规模克制，但同一 potential 下的尺度对照、校准资格和最终 graph-bound 指标尚未闭合。 |
| Venue Readiness | 5 | 可以成为有价值的方法修订，但尚缺少超出标准工具组合的不可替代贡献。 |

按指定权重计算：

\[
0.15(9)+0.25(7)+0.25(5)+0.15(9)+0.10(8)+0.05(6)+0.05(5)
=\boxed{7.05/10}.
\]

**关于六个核心问题的判断**

1. **问题锚点保留。** potential 与 scale 都不读取候选，预测对象仍是道路差分场，干预仍由相对 nominal margin 下界授权。必须继续明确：source coverage 不意味着 appearance-OOD coverage，保留 native 也不意味着 native 安全。

2. **异质 uncertainty 是合理的最小增量，但“充分”和“必要”都未证明。** R004E-F 证明平均信息存在、当前极值指标失败；它没有证明错误边能从部署观测预测，也没有排除“只改组级训练目标即可改善 uniform-scale predictor”。异质尺度允许局部化误差，但一个被严重低估尺度的边仍可抬高全局 \(q_\alpha\)，放大所有边预算。建议将“minimal sufficient”改为“minimal candidate mechanism”。

3. **coverage 与 exact min-plus 在数学上可接通；训练和数值合同未完全接通。** 对独立冻结的可测预测器 \(u(X),s(X)>0\)，每组只有一个确定性标量 score，标准 split-conformal rank argument 直接成立。现有 Theorem 1 已允许任意非负异质 \(w_e\)，不需要新的 shortest-path 定理。但有限样本条件、scale score 的数值方向以及内部 calibration 用途仍需补足。

4. **当前存在明显的“标准工具组合”风险。** observation-conditioned normalized residual、group maximum 与任意加权图上的 robust readout 可以自然组合；“组合后覆盖所有 adaptive pairs”很有用，但大部分推导来自现有 simultaneous field event 和已有确定性定理。新增一个 scale head 尚不足以构成顶会算法贡献。这里是基于材料的贡献判断，不是已完成查新后的不新颖判决。

5. **实现可启动，但还不应冻结训练合同。** 输出形状与 8064-edge order 已具体；尚需明确哪些参数更新、使用什么初始化、如何实现上界、\(\lambda,\tau\) 如何固定、内部数据角色以及 micro-batch 的精确梯度实现。

6. **最小验证接近充分，但现在缺少关键识别条件。** 不需要扩大成大规模 benchmark。现有两个验证块应先补全同一 \(u\) 的尺度对照、合法校准说明、条件 soundness 和最终 graph certificate 的判据。

**CRITICAL：必须先修改**

1. **纠正 anti-inflation 主张，把“防无界膨胀”和“尺度不被训练目标推大”分开。** proper loss 加 log 项可以阻止无界尺度降低目标，但加入 tail 项后，最优尺度一般会主动变大。单边反例：

\[
\mathcal L(s)=\frac{a}{s}+\log s+\lambda\frac{a}{s},
\qquad s^*=(1+\lambda)a.
\]

因此不能把完整目标继续称为 proper scale estimation，也不能用 properness 证明它不会通过扩大尺度降低 standardized tail。应区分正下界/有限上界、log 项抑制无界膨胀、以及同一个全局常数重标后重新校准的预算不变性。输入相关整体幅度仍可能改变预算，floor/cap 也破坏任意重标的精确不变性。

2. **消除 internal-cal 与 checkpoint selection 的歧义。** fit / selection / internal-cal 必须三路分离，且 internal-cal 不得进入 warm-start、拟合、尺度统计量或选择；否则内部分位数只能作为 proxy，不能主张有限样本 coverage。旧 checkpoint 训练过全部 66 组，不能把其子集重新命名为独立 internal-cal。

3. **补齐新增运算的数值 soundness。** 需要覆盖 residual 差分/绝对值、除以 scale、校准分位数、\(q_\alpha s\) 乘法和 correction 加法；注册溢出、下溢、非有限和边界拒绝规则。map-to-continuous teacher error 已在 lifting 扣除，不应不透明地重复混入 edge budget。

4. **先解决贡献定位。** coverage 是标准 normalized split-conformal，异质 min-plus compatibility 是现有 Theorem 1 的直接实例。论文贡献必须落到相同 predictor、相同统计合同下，对 global-gauge/joint-field 接口的任务证书优势；若查新和公平对照后只剩增量集成，应降低定位。

**IMPORTANT：进入真实 probe 前修改**

5. **主对照使用同一个 potential。** 同一最终 \(u\) 上 learned scale 与 uniform scale 必须分别合法重新校准；首轮可冻结 trunk/potential，只训练 scale head。不能只用边预算汇总替代连续几何修正后的 certificate sharpness。

6. **给 group-tail surrogate 可检查的逼近合同。** 对 \(M=16|E|=129024\)，normalized log-sum-exp 满足

\[
\max z-\tau\log M\le\widetilde R_g\le\max z.
\]

若保留 surrogate，必须冻结允许 gap、\(\tau,\lambda\) 和正确的全组梯度实现；普通独立 micro-batch backward 不等价。

7. **修正 synthetic 验证的 zero-false-acceptance 含义。** 只应在 field event 与其他合同成立的 oracle cases 上要求零错误授权；conformal 本身允许 \(\alpha\) 失覆盖。24 个反复使用的 development groups 不能承载 held-out coverage。

8. **更新输入相关 scale 的 appearance diagnostic。** 合法扩展是

\[
R(X')\le\kappa(X,X')R(X)+
\max_e\frac{|D[u(X')-u(X)]_e|}{s_e(X')},
\quad
\kappa=\max_e\frac{s_e(X)}{s_e(X')}.
\]

它只能作为诊断，不能升级为 OOD coverage。

**MINOR：表述与接口修正**

- 每个固定 native set 一次 transform，可查询所有 challenger sets；native 改变通常需要重新计算。
- 明确 scale 单位、上界实现和边序；区分 scale logits 与 residual 记号。
- candidate 输入首先违反的是本项目 candidate-independent 复用合同，不应泛称所有 conformal theorem 都失效。

**Drift Warning:** **NONE**。通过标准必须继续是米制、连续几何修正后的 relative-margin certificate。

**Simplification Opportunities**

1. 首个真实 probe 冻结 trunk 与 potential，仅训练 edge-scale head。
2. 删除 anti-inflation 作为独立算法贡献的表述。
3. 复用已有任意非负权重 robust theorem，只写统计到确定性认证的组合推论。

**Modernization Opportunities:** **NONE**。当前瓶颈不需要生成模型、LLM 或 RL。

</details>
