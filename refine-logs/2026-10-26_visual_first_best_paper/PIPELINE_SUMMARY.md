# Pipeline Summary

**Problem**: 冻结驾驶planner在外观变化下哪些内部变化真正影响决策，以及能否只修复这些变化。  
**Final Method Thesis**: 用组外有限干预资格化planner-sensitive低秩接口，再由冻结DINO通过site-conditioned reader仅恢复这些坐标。  
**Final Verdict**: DESIGN_READY 8.9/10（仅设计资格；等待正式R006V0科学证据）
**Date**: 2026-10-26

## Final Deliverables

- Proposal: `refine-logs/2026-10-26_visual_first_best_paper/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/2026-10-26_visual_first_best_paper/REVIEW_SUMMARY.md`
- Experiment plan: `refine-logs/2026-10-26_visual_first_best_paper/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/2026-10-26_visual_first_best_paper/EXPERIMENT_TRACKER.md`
- Theory boundary: `docs/R006_THEOREM_ASSUMPTION_EVIDENCE_MATRIX.md`

## Contribution Snapshot

- Dominant contribution: causally localized, rank-constrained planner-tangent repair for a fully frozen planner.
- Supporting contribution: mediation → recoverability → functional/utility validation chain.
- Explicitly rejected complexity: multi-layer repair、mixture basis、world model、VLM、diffusion/RL、joint planner training。

## Must-Prove Claims

- R006V0：共享低秩内部坐标在组外有限干预中具有scene-specific行为介导作用。
- R006V1/V2：DINO能增量恢复该坐标，且CPTR优于matched adapter与direct-output residual。

## First Runs to Launch

1. source-fit activation-cache smoke与完整机械审计；
2. 42-group cache primary/replay及schema/provenance冻结；
3. 一次性nested-OOF R006V0，不读取中途科学指标。

## Main Risks

- 低秩只在Jacobian局部成立：以真实nonlinear-tail有限patch、reverse和dose裁决。
- C3/C4语义错配：C3冻结status site；C4使用single-query reader，不宣称空间注册。
- 快速部署图不同：完整确认集数值+决策等价门；否则保留科学图。

## Next Action

完成cache资格与formal solver后执行R006V0；任何核心门失败则停止CPTR分支，不增加模块补救。
