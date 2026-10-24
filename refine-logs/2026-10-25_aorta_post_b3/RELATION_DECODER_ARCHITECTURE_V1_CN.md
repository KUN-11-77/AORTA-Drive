# AORTA relation-first 网络架构候选 v1

状态：`SUPERSEDED_BY_DIFFERENTIAL_SUPPORT_V2__NO_TRAINING_AUTHORIZED`。

2026-10-25 反方审查发现：共享非线性查询不保证输出层外观干扰为候选共同加性项，自由
interaction 又可绕过 difference 分支。本文保留为被否定的候选设计记录；当前条件候选见
`DIFFERENTIAL_SUPPORT_RELATION_ARCHITECTURE_V2_CN.md`。

本文把外观—决策商理论翻译成可实现的最小网络。它不是已验证架构，也不把 DINO、cross-attention、
trajectory tokenizer、heteroscedastic head 或 calibration 工具声明为原创。任何模块只有通过对应的
删除实验后才允许进入最终方法。

## 1. 设计目标与复杂度预算

输入是共享的多相机历史图像 \(X\)、冻结 planner 的 native 轨迹 \(\tau_0\)、可变数量合法
challengers \(\{\tau_i\}_{i\in V\setminus\{0\}}\) 及 mask。输出与部署 estimand 同型：

\[
\widehat\delta_{i0}\approx F_S(\tau_i)-F_S(\tau_0),\qquad
F_S(\tau)=\operatorname{PDM}_{1f}([\tau_{PDM}(S),\tau])_{pred}.
\]

复杂度预算只允许一个新的 trainable relation decoder；scale head 是同一 decoder 的校准接口，
不是第二项论文贡献。冻结 DINOv3 只运行一次，不训练 planner，不在线适配，不生成新候选。

## 2. 张量与 type contract

设 frozen visual encoder 产生共享 token

\[
Z=E_{vis}(X)\in\mathbb R^{P\times d_v}.
\]

完整 4 s 轨迹以位置、yaw、速度、曲率、时间和相对 native 几何编码为

\[
A_i=E_{traj}(\tau_i)\in\mathbb R^{T\times d_t},\qquad
Q_{i0}=E_{rel}(\tau_i\ominus\tau_0)\in\mathbb R^{T\times d_t}.
\]

所有输出沿候选维度 ragged/permutation-equivariant。冻结 membership mask 必须在 token batching、
attention、loss、argmax、calibration 和 arbitration 中保持同一个布尔对象；masked candidate 从不
获得伪 score。

## 3. Shared–Difference–Interaction decoder

### 3.1 共享候选证据

用同一个参数共享的轨迹条件视觉查询器提取每条轨迹的证据：

\[
e_i=\Psi_\theta(Z,A_i),\qquad e_0=\Psi_\theta(Z,A_0).
\]

\(\Psi_\theta\) 可以由 trajectory queries 对共享视觉 token 的轻量 cross-attention 实现；该算子
本身已有充分先例，不是创新。参数共享和显式差分

\[
d_{i0}=e_i-e_0
\]

对应共同外观偏置消除命题：若外观只在所有候选证据上加入同一偏置，差分会精确抵消。该结构
不是完整鲁棒性保证，因为 candidate-dependent distortion 不会被消除。

### 3.2 候选相关交互残差

为处理道路分叉、交通交互和 evaluator 的 native-conditioned 效应，引入单个受限交互支路：

\[
c_{i0}=\Omega_\theta(Z,A_0,A_i,Q_{i0}).
\]

最终 margin 为

\[
\widehat\delta_{i0}
=w_r^\top\operatorname{LN}(d_{i0}+c_{i0}).
\]

该分解给两个机制不同的可证伪职责：\(d_{i0}\) 负责消除共享干扰，\(c_{i0}\) 只补偿候选相关
残差。若删除 \(c_{i0}\) 不损害 held-out regret，则移除交互支路；若删除 \(d_{i0}\) 无影响，则不能
声称显式相对化是必要架构。

### 3.3 完整时域而非固定 BEV 读出

trajectory encoder 保留每个时间步，并用时间位置编码聚合，不使用 global/fixed/temporal scalar
minimum。原因来自 B3 反例：共同 minimum 会删除候选差异，2.5 s 前缀会删除晚期分叉。完整时域
并不自动保证 decision sufficiency，但至少不在接口处确定性地丢弃已知差异。

## 4. 训练目标

### 4.1 主任务目标

\[
\mathcal L_{task}
=\frac{1}{|V|-1}\sum_{i\in V\setminus\{0\}}
w_i\,\rho(\widehat\delta_{i0}-\delta^*_{i0}).
\]

默认先用 uniform \(w_i\) 与 source-frozen Huber scale；任何 hard-negative/task-critical weighting
必须证明改善 group-held-out regret，而不是只降低训练误差。

### 4.2 外观约束

\[
\mathcal L_{app}
=\frac{1}{|\mathcal A|(|V|-1)}
\sum_{a\in\mathcal A}\sum_{i\in V\setminus\{0\}}
\rho\!\left(
\widehat\delta_{i0}(T_aX)-
\operatorname{sg}[\widehat\delta_{i0}(X)]
\right).
\]

优化写成

\[
\min_\theta\ \mathbb E[\mathcal L_{task}]
\quad\mathrm{s.t.}\quad
\mathbb E[\mathcal L_{app}]\le\delta_{app}.
\]

task loss 防止常数塌缩；appearance constraint 只约束同场景关系，不强迫全部 feature 不变。
\(\delta_{app}\) 与 dual 更新规则在 source-only fit 阶段冻结；NavDream 不能参与选择。

### 4.3 不确定性不与点预测混为一谈

同一 decoder 可输出正 scale

\[
\sigma_i=\sigma_{min}+\operatorname{softplus}(w_\sigma^\top h_{i0}),
\]

并用 proper residual score 训练 sharpness，但这不产生 coverage。独立 calibration 以 recording
group 为单位，对同时归一化残差

\[
R_g=\max_{a,i}
\frac{|\widehat\delta_{gai0}-\delta^*_{gai0}|}{\sigma_{gai}}
\]

用 recording group 作为独立单位，令

\[
k=\left\lceil(n+1)(1-\alpha)\right\rceil,
\qquad q_{1-\alpha}=R_{(k)},
\]

并取 \(r_i=q_{1-\alpha}\sigma_i\)。若 \(k>n\)，半径定义为无穷并全部拒判。该 split-calibration
规则在固定模型、候选机制和 group exchangeability 下给出 group-level simultaneous coverage；
条件之外只报告 empirical coverage，不得把 source calibration 写成任意 NavDream 保证。若独立
calibration 有 42 个 groups，则可支持有限的 95% 分位数，但不足以无假设地支持有限 99%
simultaneous radius；fit/development groups 不能回收充当 calibration。

## 5. 推理与拒判

对每个合法 challenger 计算

\[
L_i=\widehat\delta_{i0}-r_i.
\]

只有 \(\max_iL_i>0\) 且所有数据、候选、数值与时延合同通过时才替换 native，否则保持原规划器。
点估计、scale、calibration、abstention 和最终行为分别记账，避免“低误差”被偷换成“可信干预”。

## 6. 实时结构

共享视觉前向一次，所有候选的 trajectory encoding 与 cross-attention 批量并行：

\[
C_{total}=C_{DINO}+O(KT d_t^2)+O(KT P_s d),
\]

其中 \(P_s\ll P\) 时可使用冻结、预注册的稀疏视觉采样；不得通过减少完整时域或删除困难候选
获得时延。最终必须在同一硬件、batch=1、同步条件下报告 backbone/head/arbitration 的
P50/P95/P99、显存和候选数缩放。

## 7. 结构必要性实验

| Arm | 唯一变化 | 要排除的解释 |
|---|---|---|
| A | 普通 direct native-relative vector head | 主简单基线 |
| B | shared-difference，无 interaction | 共享偏置消除是否足够 |
| C | shared-difference + interaction（候选主模型） | 候选相关残差是否必要 |
| D | star node-score difference | 是否仅为重参数化；预期应与 A 可比 |
| E | complete-score / free complete-edge | complete pair structure 是否真有增量 |
| F | 去掉 appearance constraint | 收益是否来自 relation-level invariance |
| G | mismatch / removed image | 是否只依赖轨迹或空间先验 |
| H | 2.5 s prefix / field readout | 完整时域是否由 B3 机制需要 |

所有 arms 共享视觉 token、合法候选、训练标签、数据、optimizer、seeds、参数/FLOP 与步数预算。
primary endpoint 是 group-held-out ranking regret/selection error；cycle residual、train loss 和 field
MAE 只能作为诊断。

## 8. 与近邻方法的边界

- GTRS/DA-WAM 已覆盖 trajectory tokenizer、candidate-conditioned latent、factor/utility head 和
  hard negatives，因此 cross-attention 或 candidate scorer 本身不是贡献；
- NMP/MP3/FlowDrive/READ 已覆盖 field/cost representation 接规划，故“不预测 BEV”也不是贡献；
- preference/reward scorer 已有大量先例，故 scalar margin supervision 本身不是贡献；
- 可能成立的单一机制贡献只能是：**appearance intervention 的合法 decision-quotient estimand，
  由 shared-difference/interaction 实现，并通过 selective native-preserving rule 改变 OOD 行为。**

## 9. 生死门

1. B4R1 不存在合法非平凡 oracle ceiling：停止当前 estimand/panel；
2. correct image 不优于 mismatch/removed：停止视觉关系主张；
3. C 不优于 A/B：删除 interaction 或整个结构创新，只保留更简单模型；
4. appearance constraint 不降低未见外观 regret：删除 relation-invariance 机制主张；
5. selective calibration 空泛或 OOD 严重失覆盖：不得干预或宣称可信性；
6. NavDream/跨 planner 没有净 hard-failure 改善或 clean harm：不得宣称驾驶收益；
7. 时延不合格：不得宣称在线通用辅助系统。

该架构的“优美”不来自模块少或公式多，而来自每个模块只承担一个可验证职责，并且任何失败都
有明确删除路径。
