# Final Proposal：DINO Evidence Mechanism First

## Problem Anchor

Constant Eye 已证明 DINOv3-H+ 能为不同规划器提供外观鲁棒增益。新 AORTA 不把 DINO 接入旧 SDF、
低秩、Differential Query 或 CPTR 体系，而从 DINO 重新提出问题：哪些 DINO 证据对驾驶任务条件充分，
这些证据经过真实 planner 接口时在哪里被丢失或放大，以及如何只修复被证实的瓶颈。

## First-principles Objects

设冻结 DINO 特征为 (D^a=\Phi(g(S,a)))，任务上下文为 (B)，驾驶目标为 (Y)。研究对象是：

1. 条件充分性：(D) 在给定 (B) 后是否仍降低预测 (Y) 的风险；
2. 功能保持：信息保持 appearance orbit 上，组合任务函数是否稳定；
3. 接口实现：何种最小接口同时保留任务可分性、clean 行为与实时性。

平方损失下，视觉增量的总体对象为

\[
\mathcal R^*(Y\mid B)-\mathcal R^*(Y\mid B,D)
=\mathbb E\|\mathbb E[Y\mid B,D]-\mathbb E[Y\mid B]\|^2.
\]

接口压缩 (C) 的任务证据损失写为

\[
\Delta_{suf}(C)=\mathcal R^*(Y\mid B,C(D))-\mathcal R^*(Y\mid B,D),
\]

信息保持外观上的功能漂移写为

\[
\Delta_{orb}(F)=\mathbb E_{a\in\mathcal A_{pres}}
d_Y(F(B,D^a),F(B,D^{clean})).
\]

有限 probe 只能在指定函数类和数据合同内估计这些量，不能声称信息论充分或不可能。

## Immediate Scientific Task

不先训练新方法。先按每个模型的真实计算图执行 DINO stage interchange 与
final/last-four/intermediate/full/compressed/skip-token 组外 probe，区分：scene prior、interface
amplification、compression loss、layer complementarity、mixed 与当前接口不可恢复。
DINO-only 模型使用顺序 stage interchange；只有真实 hybrid 模型才使用 native×DINO 因子交换。
在任务信息增量成立后，再分别审计冻结tail写入目标是否失配，以及任务上下文在压缩前进入是否产生
等预算组外交互增量；二者未通过前，不实现预设reader–writer。
probe 使用 task target 与 clean functional reference 双目标；只有可读增量能经真实 planner 干预影响
输出或效用，才称为可用驾驶证据。

## Method Authorization

- 没有 compression 证据：不增加 full-token reader；
- 没有 amplification 证据：不增加功能一致性约束；
- correct DINO 不优于 mismatch/removed：停止当前视觉主张；
- 只有 compression：比较最小 full-token skip/reader 与 matched ordinary adapter；
- 只有 layer complementarity：先比较等权 normalized layer mean 与同参数 scalar mixture；
- 只有 amplification：比较原接口上的最小 task-output orbit regularization；
- 两者独立成立：才组合一个 task-conditioned DINO evidence interface，并做逐项删除消融。

任何新方法最多新增一个可训练组件。cross-attention、低秩、VJP、旧 FQB 与 planner 内部切口均无默认
资格。若相同 DINO、参数、FLOPs 和数据下的普通 adapter/consistency 等效，复杂方法主张立即删除。

## Core Validation

所有方法共享同一冻结 DINOv3-H+。faithful Constant Eye 单列为重训练 baseline；冻结 planner arms
比较 full-token ordinary reader、matched residual adapter、ordinary output consistency、direct-output
residual 与机制授权方法。最终硬门为 NavDream zero-shot、多 planner、clean non-inferiority、official
utility、hard-failure 净减少与 batch-1 P50/P95。

## One-minute Story

> Constant Eye 证明 DINOv3-H+ 能提升恶劣外观下的自动驾驶，但没有说明真正起作用的是哪些任务证据，
> 也没有说明这些证据经过 planner 接口后为什么仍会丢失。我们固定 DINO 为共同视觉底座，直接审计
> token 到驾驶输出的真实信息流：完整证据是否任务充分、哪一段压缩丢失它、哪一段放大残余外观漂移。
> 然后只针对组外确认的瓶颈设计一个最小接口，并要求它在相同 DINO 与预算下击败普通 adapter 和
> consistency，同时满足 clean 非劣和实时约束。机制不成立，方法就不出现。

当前只是可证伪的研究闭环，不是已成立的 Best-Paper 方法。
还必须明确：DINOv3-H+不是VLM。若论文最终声称面向VLM/VLA视觉鲁棒性，主行为验证至少覆盖一个
真实多模态驾驶策略，并将视觉派生caption、query与history纳入受干预路径；普通planner只承担机制控制。
