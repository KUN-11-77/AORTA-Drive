# Final Proposal：DINO Interface Mechanism First

## Problem

Constant Eye 已证明冻结 DINOv3-H+ 能为重新训练的不同规划器提供外观稳定视觉底座。AORTA 进一步研究：当 DINOv3-H+ 与已部署 planner 均被冻结时，视觉信息在接口的哪里被放大或丢失，以及怎样用一个最小模块修复，而不重训 planner、不伤害 clean。

## First-principles object

对信息保持的 appearance orbit，目标是冻结 planner 的任务功能保持，而不是所有 features 相等：

\[
D^a=\Phi(g(S,a)),\qquad
\widehat O^a=T(Z_n^a+R_\theta(Q^a,D^a,H),H).
\]

\[
\widehat O^a=\widehat O^{a'},
\quad a,a'\in\mathcal A_{pres},
\]

同时保留不同场景的任务可分性。信息损失外观不强制不变，而评价风险与拒判。

## Immediate scientific task

不先训练新方法。通过 DINO/native 双路径因子化交换和 full/compressed token 组外 probe，审查：

- interface amplification；
- compression loss；
- scene prior；
- mixed/unresolved。

low-rank、Fantope、CPTR 已停止；VJP 只是有额外必要性证据时的附录候选。

## Conditional method

- 只发现 compression：full-token query-conditioned bridge；
- 只发现 amplification：同结构 ordinary functional consistency；
- 二者都存在且消融不可替代：Functional Quotient Bridge；
- correct 不优于 mismatch/removed，或 full-token 当前合同无恢复上界：停止当前 DINO-only 接口主张。

## Core validation

所有方法使用同一冻结 DINOv3-H+。最强对照为 faithful Constant Eye、full-token ordinary reader、matched residual adapter、ordinary output consistency、direct-output residual。最终只以 NavDream zero-shot、多 planner、clean non-inferiority、official utility、hard-failure净减少与 batch-1 P95 闭环。

## One-minute story candidate

> Constant Eye 证明冻结 DINOv3-H+ 可以提供稳定视觉，但稳定视觉并不自动成为稳定决策。我们首先用同场景因子化交换证明任务证据在 DINO-to-planner 接口被放大或压缩；随后只用该发现授权一个 full-token、原生 planner-query 驱动的功能商桥接器，在完全冻结 planner 的条件下恢复信息保持外观上的任务功能。若相同 DINO 的普通 adapter 或 consistency 能做到同样效果，我们就删除这个方法主张。

当前只是可证伪闭环，不是已成立的 Best-Paper 方法。
