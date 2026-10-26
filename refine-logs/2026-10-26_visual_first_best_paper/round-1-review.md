# Round 1 External Method Review

**Overall:** 6.8/10
**Verdict:** REVISE
**Drift warning:** NONE

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9 |
| Method Specificity | 6 |
| Contribution Quality | 6 |
| Frontier Leverage | 8 |
| Feasibility | 6 |
| Validation Focus | 8 |
| Venue Readiness | 5 |

## Blocking findings

1. 一般充分统计量存在性近似空泛；核心应改为“低维、跨场景共享、测试时可恢复的外观效应介导表示”。
2. 原最小范数定理假设已知输出修正 \(b\)，却没有推出固定全局基与 coefficient reader；存在理论—算法断裂。
3. 应以
   \[
   B^*=\arg\min_{B^TB=I}\mathbb E\|J_i(I-P_B)\delta_i\|_W^2
   \]
   直接定义基，并用 coefficient error、basis uncovered error、nonlinear remainder 闭合行为界。
4. 固定共享基可能遗漏稀有关键场景；必须报告 held-out basis coverage、rank–coverage、最差组与场景分层，严格隔离 basis discovery/selection/confirmation。
5. full clean patch 在确定性分解下接近构造性结果，只能作机械上界；科学结果必须是低秩、组外、双向、dose-response 干预。
6. “可识别性”应收紧为条件 Bayes risk 意义的经验可恢复性；finite probe 不能证明严格 identifiability。
7. 必须区分 output restoration、argmax preservation、official utility 与 safety；加入 clean margin 条件。
8. 必须加入 capacity-matched direct-output residual，否则无法证明内部 feature repair 的必要性。
9. Constant Eye 应同时作为 faithful absolute baseline 与 frozen-tail matched ablation，不能建立稻草人。

## Simplification

- 信息瓶颈与一般充分统计量降为背景；
- 暂不加入 uncertainty head；
- 用单一 task-weighted drift objective 取代多个 covariance 拼装。

## Required revised chain

```text
appearance-sensitive planner geometry
  -> qualified low-rank target
  -> DINO recoverability
  -> minimal frozen-planner repair
```

该评审认为方向有 ICCV 潜力，但在零训练机制门通过前没有 CPTR 训练资格。
