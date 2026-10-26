# Final Proposal：Causal Planner-Tangent Repair

## Problem

Constant Eye 已证明冻结 DINOv3-H+ 是有效的外观稳定视觉底座，但仍需为每类 planner 训练 adapter/head，也没有说明原生 planner 的哪部分表示介导外观失效。AORTA 的问题是：能否在 planner/head 完全冻结时，识别并只修复外观导致的 planner-sensitive 内部变化？

## Dominant contribution

用组外因果干预资格化一个低秩 planner-tangent target，并用冻结 DINO 仅预测该 target 的空间系数；这是 rank-constrained repair，不声称全局最小范数，也不是一般 adapter 或全局 feature invariance。

一句话：**不让特征处处不变，而是定位并修复外观扰动中真正改变冻结planner决策、且视觉证据能够恢复的低秩内部坐标。**

## Core definition

\[
B_r^*=\arg\min_{B^TB=I_r}\mathbb E\|J_{a,i}(I-BB^T)\delta_i\|_W^2,
\qquad \delta_i=z_{c,i}-z_{a,i}.
\]

其中 \(J_{a,i}\) 在shifted activation处求值，因为修复从 \(z_a\) 出发。单点Jacobian只产生候选
方向；有限外观位移的资格来自冻结nonlinear tail上的held-out patch、reverse、dose与错误方向控制。

对一般interface sites，投影为 \(I_{P_{repair}}\otimes BB^T\)，修复为

\[
\widehat Z_a=Z_a+A_\theta(D_a,Z_a,H)B_r^{*T}.
\]

该对象直接最小化低秩投影未覆盖的 planner-sensitive appearance effect。

真实LTF接口语义冻结为：C3是64个8×8 BEV memory tokens加一个不可修复的status token；C4是单个
trajectory-query token，不宣称空间对应。统一reader以native sites为queries、DINO patches为keys/values
做site-conditioned cross-attention；C3输出64×r并给status补零，C4输出1×r。

Basis 使用预注册 Stiefel 梯度、QR retraction 与 Armijo line search，通过显式 Jacobian synthetic oracle 后才能运行。六折需满足 layer 5/6、rank 4/6 一致性；先取满足5/6的唯一 layer 众数，再只在选择该 layer 的 folds 中取至少占全部六折4折的唯一 rank 众数，最后在全部 source groups 重估唯一部署 basis。

## Why the method follows from the theory

令真实 coefficient 为 \(A_i^*=\Delta Z_iB_r^*\)。修复后的行为误差被三项控制：reader coefficient error、rank-r basis 未覆盖的 task drift、网络非线性余项。因而 reader 只需要预测有行为意义的低秩系数，而无需恢复完整 clean feature。

## Training

planner 与 DINO 全冻结。主损失为 frozen-tail functional restoration；从属损失为 coefficient supervision、residual-energy regularization 和 clean identity。该正则只约束不必要干预，不声称求得全局最小范数解。所有超参数在 source split 冻结。

机制发现使用资格化math-SDPA计算图；最终快速图只有在完整确认集满足轨迹最大绝对差`<=1e-5`且离散
选择完全一致后才可部署，否则保留科学图并报告真实端到端时延。

## Validation chain

1. **R006V0 mechanism:** 6-fold nested groupheldout low-rank patch，双向/dose及wrong-scene/random/noise控制；
2. **R006V1 evidence:** 在容量、参数、训练预算与优化匹配的函数类中，correct DINO 相对条件匹配 mismatch、removed、native-only 降低组外 coefficient 与 functional risk；只主张指定 probe class 的增量可预测信息，不声称严格 Bayes identifiability；
3. **R006V2 method:** 分开报告 faithful Constant Eye（按原文训练adapter+planning head的绝对性能基线）和 frozen-tail matched adapter；后者与 CPTR 匹配数据、参数/FLOPs、optimizer steps。另比较 random/PCA/Jacobian-only basis、full-feature regression、direct-output residual；
4. **R006V3 closure:** NavDream、clean no-harm、三类 planner、official utility、batch-1 latency。

## Stop rules

低秩 OOF 介导失败、DINO 增量失败、matched adapter/direct-output 同样有效、或 utility/clean/runtime 失败时，分别停止对应主张；不通过增加模块、提高 rank 或调整 NavDream 阈值补救。

## Novelty boundary

不把 DINO、Jacobian、SVD、activation patching、adapter、信息瓶颈或 Lipschitz 界称为创新。候选创新只来自其联合：**causally localized, rank-constrained planner-tangent repair for fully frozen planners**。
