# Experiment Plan：Visual-First Causal Planner-Tangent Repair

**Problem:** 稳定视觉表示不自动带来冻结 planner 的稳定决策。
**Method Thesis:** 只修复经因果干预资格化、同时受外观扰动且对 planner 敏感的低秩内部方向。
**Date:** 2026-10-26

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：外观失效由组外可复现的低秩 planner-sensitive 子空间介导 | 没有这一点，feature repair 没有对象 | nested OOF low-rank patch 恢复，双向/dose成立且优于 wrong-scene/random/noise | B0 |
| C2：DINO 可识别该任务统计量，低秩修复优于普通 adapter | 决定算法必要性与创新性 | correct/mismatch/removed + matched CPTR/adapter/Constant Eye + behavior closure | B1–B4 |

反主张：收益只来自 DINO 更强、更多参数、重训规划头、向 clean 轨迹回归、样本筛选或 NavDream 调参。

## Paper Storyline

- Main paper must prove：causal localization → DINO recoverability → rank-constrained repair → OOD/clean/latency closure。
- Appendix：rank spectrum、更多层、更多外观、可视化、额外 planner 分解。
- Intentionally cut：决策 scorer/Differential Query、SDF/field、PEA 升级、复杂 uncertainty、diffusion/RL repair。

## B0：低秩因果介导资格（MUST-RUN）

- Dataset：42 source-fit groups；clean + 3 registered pointwise appearance operators。
- Systems：full patch只作smoke；正式比较 task-weighted basis、PCA、random rank、wrong-scene、noise、reverse与dose response，见 R006V0。
- Metrics：OOF raw restoration、basis coverage、rank–coverage、control gaps、worst group、P50/P95。
- Success：R006V0 联合门通过。
- Failure：停止 registered interfaces 上的 repair；不训练 CPTR。
- Target：主文 Fig. 2 mechanism localization 或论文停止。

## B1：任务子空间与增量可恢复性（MUST-RUN，B0 后）

- 冻结 B0 nested OOF 资格化的层/rank策略，不使用 NavDream 选层；
- DINO correct / geometry-matched mismatch / removed / native-only 预测目标系数与 frozen-tail output；
- Success：低秩 task subspace 保留大部分 full-patch restoration，且 correct DINO 对所有控制有组级可靠增量；
- Failure：若低秩失败，方法失去简洁性；若 DINO 失败，停止 DINO lift。

## B2：算法必要性（MUST-RUN，B1 后）

同数据、backbone、参数、FLOPs、optimizer steps、seeds 比较：

1. frozen native planner；
2. Constant Eye faithful interface；
3. capacity-matched plain adapter；
4. full-feature regression repair；
5. random-subspace repair；
6. direct-output residual/distillation；
7. CPTR。

主指标：group-held-out official utility、appearance-induced output drift、clean no-harm；次指标为 feature/coefficient error。必须报告参数、FLOPs、峰值显存和 batch-1 增量时延。

## B3：NavDream 与跨 planner（MUST-RUN，架构冻结后）

- 一次性解封 NavDream；不再改层、rank、loss、阈值；
- regression、diffusion、generate-and-select 三类 planner；
- 报告每种外观、最差条件、平均、clean、new failures/rescues、置信区间；
- 只在至少两类 planner 上一致成立时主张通用机制。

## B4：失败边界（MUST-RUN）

- 几何改变、严重遮挡、动态目标缺失、DINO/native observation collision；
- 检验 repair 是否应拒判；
- 不把不可识别输入强制修复。

## Run Order

| Milestone | Goal | Runs | Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | port hooks + exact identity smoke | 1–2 anchors synthetic/native-shift | exact only | <1 GPUh | hook semantics |
| M1 | R006V0 B0 | 6-fold nested basis + 42 OOF groups | C1 | 先以V0S实测推算，正式预算再冻结 | JVP/VJP与I/O |
| M2 | B1 subspace/identifiability | source nested groups, 3 seeds | C2 information gate | 10–30 GPUh | high-dimensional Jacobian |
| M3 | B2 matched methods | 6 systems × 3 seeds | architecture gate | 40–100 GPUh | baseline fairness |
| M4 | B3+B4 | NavDream + 3 planners | paper gate | 依官方评测实测 | domain/latency |

## First Three Runs

1. exact native-shift identity patch smoke；
2. one-anchor clean/controlled-shift full-layer patch smoke，只验证形状、非图像一致、tail分解与有限值；
3. 冻结实现、manifest、统计和 audit 后一次性 R006V0。

## Final Checklist

- [x] Main claims limited to two
- [x] Constant Eye is a mandatory closest baseline
- [x] DINO is not claimed as novelty
- [x] Zero-training causal gate precedes method training
- [x] Nice-to-have modules do not block the core story
- [ ] R006V0 implementation qualification
- [ ] R006V0 scientific result
- [ ] External matched-baseline validation
