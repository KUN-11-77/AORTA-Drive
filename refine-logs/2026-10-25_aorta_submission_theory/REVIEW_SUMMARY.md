# AORTA 投稿版方法审查总结

日期：2026-10-25
方式：按 `research-review` 与 `research-refine` 结构进行本地双轮审查；未调用外部独立 reviewer，
因此不能标记为独立同行评审。
最终裁决：`REVISE / 7.8 out of 10 / theory-ready for falsification, not ICCV-ready`。

## Problem Anchor

在不修改冻结端到端 planner 的前提下，学习 source-only、外观鲁棒、任务充分、选择性且实时的
视觉决策接口，使同一物理场景在未见外观变化下保持正确候选排序，并在多类 planner 上减少困难
失败，同时保持 clean 非劣。

## 审查过程

| Round | 主要问题 | 修订 | 状态 |
|---|---|---|---|
| 0 | relation/fallback 新颖性过宽；full target 与行为目标混称 | D-JEPA 后收缩为 differential evidence | partial |
| 1 | evaluator scope、pair cycles、history rollout、B4R1 gate 逻辑 | scoped `U_ctx`、integrable native-star、actual-history/sequential、接口与科学 outcome 分离 | resolved in design |
| 2 | differential bias 是否真有增量、NavDream/行为/时延无证据 | 固定 matched falsifier 和五级证据门，不再加模块 | empirically unresolved |

## 最强部分

- 问题由 common-min 与 late-fork 真实反例驱动；
- 数学定理、经验假设与原创性边界被分开；
- 网络每个模块均有直接原因、删除实验与失败预测；
- 最强替代方法被放在主实验而非 related-work 文字中；
- sequential history 和 official evaluator scope 被纳入 P0。

## 剩余弱点

- 核心创新仍可能只是 generic scorer 可自行学习的 feature engineering；
- scoped contextual target 尚未 materialize；
- 训练/部署 history 分布尚未闭合；
- 没有 NavDream、跨 planner、clean no-harm 或 batch-1 latency 证据；
- 当前审查不是独立外部 reviewer。

## 规范产物

- 投稿级蓝图：`AORTA_PAPER_BLUEPRINT_V1_CN.md`
- 理论—算法—架构合同：`AORTA_SUBMISSION_THEORY_V1_CN.md`
- 评分演化：`score-history.md`
- 完整审查记录：`REFINEMENT_REPORT.md`
