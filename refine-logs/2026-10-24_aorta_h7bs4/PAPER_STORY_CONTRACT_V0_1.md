# AORTA 论文故事合同 v0.1

状态：`FROZEN_CANDIDATE_STORY__NOT_A_PAPER_CLAIM`。

本文件在 H7BS4B0B 结果产生前固定“问题—洞见—方法—证据”的唯一主线。它不是宣传稿，
而是防止结果出来后拼接模块、移动目标或牵强包装创新的约束。任何不能通过下列反证门的内容，
不进入主贡献。

## 1. 一句话问题与反常现象

**反常现象：**同一物理场景只改变外观时，视觉表征可保持很高的全局相似度，驾驶候选的
安全排序仍会改变；反过来，一个跨外观几乎不变的场也可能只是稳定先验，而不是场景条件的
有效证据。

因此，论文不研究“如何让 feature 更相似”，而研究：

> 在部分可观测和外观变化下，怎样学习一个既保持候选间任务差异、又显式表达证据支持范围，
> 并且只在联合不确定集合内一致占优时才改变冻结驾驶模型决策的视觉任务算子？

这把外观鲁棒性的对象从 feature invariance 改为 **decision-sufficient operator invariance**。

## 2. 唯一数学对象

令候选轨迹面板为 \(\Pi=\{\tau_0,\ldots,\tau_K\}\)，\(F_j\subseteq V\) 是候选 \(\tau_j\)
在完整规划时域内的扫掠节点集合，\(d\in\mathbb R^{|V|}\) 是场景条件的任务裕度。候选算子为

\[
M_d(\tau_j)=\min_{q\in F_j}d_q-m_0 .
\]

算法真正需要保持的不是整幅 BEV、SDF 或风险热图，而是同一候选面板上的相对算子

\[
\Delta_{j0}(d)=M_d(\tau_j)-M_d(\tau_0).
\]

对任意常数 \(c\)，\(\Delta_{j0}(d+c\mathbf 1)=\Delta_{j0}(d)\)。因此可辨识对象属于商空间
\(\mathbb R^{|V|}/\operatorname{span}\{\mathbf 1\}\)，而不是带任意零点的绝对场。

## 3. 核心洞见

论文候选洞见不是“扩大 ROI”或“使用 DINO”，而是下列四件事必须同时成立：

1. **结构：**学习相对任务势，而不是要求不可辨识的绝对零点稳定；
2. **可积性：**边差分必须来自同一个节点势，不能由互相矛盾的独立 edge head 拼成；
3. **支持：**外观稳定性只对有视觉证据的查询有意义；未支撑且可能成为极小值的节点必须扩大
   联合不确定集合，而不是被插值成“安全”；
4. **决策：**不确定性必须在同一个潜在场上联合比较 native 与 challenger；不能把两个独立置信
   区间相减后称为可靠替换。

这四点共同解释已有失败：PEA 的全局统计对齐没有定位候选关键证据；固定 64×64 场把完整时域
压入有限支持；2.5 秒截断提高覆盖率却可能删除晚期分叉；独立阈值和 model confidence 不能控制
替换伤害。

## 4. 方法闭环

### 4.1 Hodge-consistent quotient operator

在候选扫掠联合图 \(G=(V,E)\) 上，网络输出节点 logits \(a_\theta\)，并投影为 mean-zero 代表元：

\[
u_\theta=P_\perp a_\theta,
\qquad
P_\perp=I-\frac{1}{|V|}\mathbf 1\mathbf 1^\top,
\qquad
z_\theta=B_0u_\theta.
\]

其中 \(B_0\) 是 node–edge incidence operator。若 \(B_1\) 是 edge–cycle incidence operator，
则架构恒有

\[
B_1z_\theta=B_1B_0u_\theta=0.
\]

因此 cycle consistency 是输出空间的硬性质，不是事后加权的软正则。这是 quotient field 相对
direct-edge predictor 必须通过容量匹配消融证明的结构必要性。

### 4.2 Support-conditioned joint uncertainty

令 \(O(X)\subseteq V\) 是由相机历史、投影合同和证据质量共同定义的支撑域。网络学习 edge
scale \(s_\phi\)，并构造共享潜在场集合

\[
\mathcal U_{\theta,\phi}(X,O)
=\left\{d=c\mathbf 1+u_\theta+\delta:
\left|(B_0\delta)_e\right|\le r_e(X,O),\ e\in E\right\}.
\]

\(r_e\) 必须随 evidence 缺失单调不减；联合集合中的同一个 \(d\) 同时生成所有候选分数，
保留候选间相关误差。B0B 已在 fit teacher 上发现 596/4,536 candidates、39/42 groups 存在
unsupported possible minimizer，并在 1/8 个冻结 D0 盲差异 pair 上确认盲区可控制 set-min。
因此“missing evidence 必须进入接口或不确定集合”获得机制支持；具体集合形式仍未验证。

### 4.3 Native-preserving robust arbitration

对 challenger \(j\) 与 native \(0\)，定义最坏情形相对增益

\[
L_{j0}(X)=\inf_{d\in\mathcal U_{\theta,\phi}(X,O)}
\left[\min_{q\in F_j}d_q-\min_{q\in F_0}d_q\right].
\]

只在 \(L_{j0}\ge\gamma\)、完整时域支持合同和候选合法性同时通过时替换 native；否则保留原决策。
这不是现实安全证书，而是一个可审计的 **model-relative selective arbitration rule**。

## 5. 核心网络与损失候选

网络只保留一条主干：冻结或受控微调的多相机时序视觉编码器，接 candidate-union coordinate
decoder，再接共享 quotient potential / edge-scale heads。固定 dense 64×64 场是基线，不是方法定义；
连续查询也不能为不可观测区域凭空产生证据。

主监督为 support-weighted heteroscedastic edge proper loss：

\[
\mathcal L_{\mathrm{edge}}
=\sum_{e\in E}w_e(O)
\left(
\frac{|z_{\theta,e}-z_e^\star|}{\exp s_{\phi,e}}
+s_{\phi,e}
\right).
\]

外观一致性只约束同场景 pair 的可辨识 edge operator：

\[
\mathcal L_{\mathrm{app}}
=\sum_{e\in E_{\mathrm{shared}}}
\omega_e
\rho\!\left(z_{\theta,e}(X)-z_{\theta,e}(T_aX)\right),
\]

并以 clean/task fidelity 作为硬资格门，避免常数场通过“一致性”。候选排序 loss 仅作后续消融，
不得先加入主目标后再声称 quotient structure 是收益来源。

## 6. 顶会式贡献必须形成的三层证据

### C1 — 问题与理论

证明 feature invariance、有限 ROI 覆盖率和短前缀评分均不足以保证候选可区分；给出 quotient
不变性、exactness 与 robust shared-field comparison 的精确命题和反例。

### C2 — 算法与架构

提出 candidate-conditioned、Hodge-consistent、support-aware 的 joint operator network，并给出
可计算的 native-preserving robust arbitration。创新必须在同输入、容量匹配、完整 4 秒时域下优于
dense field、direct-edge、independent uncertainty 和普通 candidate score head。

### C3 — 实证

依次证明：未见场景上正确图像优于错配/移除；受控与 NavDream 外观下 operator 稳定且任务正确；
结构消融不可替代；完整时域候选排序改善；最终冻结驾驶模型在不伤害 clean 的条件下获得官方
决策收益；推理时延满足冻结周期预算。

少一层都不能写成完整方法论文成功。

## 7. Figure 1 与口头叙事

Figure 1 只画一个矛盾和一个闭环：

1. 左：同一场景不同外观，全局 feature 相似，但 native/challenger 安全排序翻转；
2. 中：旧 dense field 在固定边界或短时域遗漏晚期 candidate-critical evidence；
3. 右：AORTA 在候选联合图上学习 exact relative operator，传播 support-conditioned joint
   uncertainty，只在最坏情形一致占优时替换 native。

30 秒版本：

> 现有外观鲁棒驾驶通常让视觉特征或稠密场保持不变，但我们发现稳定表征仍会丢失真正区分
> 驾驶候选的局部证据。AORTA 把鲁棒性对象改成候选面板上的相对任务算子：它在候选扫掠联合图
> 上学习可积的 quotient potential，用视觉支持条件化一个共享不确定集合，并仅在该集合内 challenger
> 对 native 一致占优时干预。这样，表示、可观测性和决策被同一个数学对象闭合，而不是由若干
> 独立模块事后拼接。

## 8. Rebuttal 前置反证门

- B0B 已通过 fit teacher-criticality 门，但只支持问题必要性，不支持具体架构有效；
- 若 quotient head 不优于容量匹配 direct-edge / ordinary head，则结构创新失败；
- 若 joint uncertainty 不优于 independent intervals，则联合集合贡献失败；
- 若 correct image 不优于 mismatch/removal，则视觉依赖主张失败；
- 若完整时域没有额外决策信息，则 adaptive spatial support 退回工程优化；
- 若时延超出冻结预算，则不得宣称在线可部署；
- 若只有 fit 或 historically exposed development 成立，则不得写独立泛化；
- 若视觉指标改善但官方决策收益不成立，则论文只能停留在表示研究，不能声称驾驶鲁棒性提升。

## 9. 当前真实性边界

目前已有 fit 视觉机制的正向证据、teacher-critical projective blindness 和多项失败归因；尚无
证据证明本文件中的候选架构优于匹配基线，
也没有完整时域、NavDream、正式 test 或驾驶安全收益。论文故事的连贯性已经可以冻结，但论文贡献
仍必须由 B0B、结构必要性、时延、独立泛化和决策效用逐级赢得。
