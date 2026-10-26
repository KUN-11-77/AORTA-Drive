# Round 3 / Final Independent Review

最终评分：`9.0/10`
裁决：`DESIGN_READY`

独立复核确认以下理论—算法闭环已成立：

```text
appearance-induced drift
→ planner-weighted low-rank basis
→ held-out causal mediation
→ DINO incremental recoverability
→ rank-constrained repair
→ behavior / utility validation
```

已闭合的关键问题：

1. 空间 channel basis、逐位置 coefficient field 与误差界维度一致；
2. 最终 layer/rank 选择不受异层 outlier fold 污染；
3. random/noise、reverse 与 dose-response 均为机械可判定规则；
4. P2 只主张有限函数类和训练合同下的增量可预测性；
5. CPTR 不再声称全局最小范数；
6. Constant Eye faithful 与 frozen-tail matched baseline 明确分开；
7. 负结果能够停止主张，而不是触发事后扩网或阈值搜索。

`DESIGN_READY` 只表示理论和预注册有执行资格，不表示低秩机制、DINO 可恢复性、CPTR 有效性或 ICCV / Best Paper 证据已经成立。

正式执行前仍需冻结：planner checkpoint/source/hook/preprocess SHA、coverage epsilon、synthetic dense-Jacobian tolerances、bootstrap seed/CI、zero-intercept OLS 语义，以及 smoke 后的正式计算预算和异常停止条件。
