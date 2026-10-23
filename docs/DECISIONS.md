# Architecture Decision Log

## ADR-0001 — Use a `src/` Python package layout

- **Status:** accepted
- **Reason:** 防止 repository-root accidental imports，并支持可测试的研究包。

## ADR-0002 — Retire affine panel factorization as the main method

- **Status:** superseded
- **Reason:** 该分解是代数恒等式；固定候选 panel、anchor set 与 scalar contrast 未建立不可替代的方法贡献。
- **Preservation:** 现有 reference code 保留为历史 baseline，不能代表当前 v8.3.1 方法已实现。

## ADR-0003 — Use a candidate-free graph-quotient field

- **Status:** accepted for V0 qualification
- **Reason:** 共享物理 field 允许一次 source-session event 同时覆盖任意域内 swept-set pairs，并显式消除共同 offset。
- **Reversal:** Gate B 证书空泛，或 exact global-gauge/CDF baseline 等价。

## ADR-0004 — Fixed physical edge scale in V0

- **Status:** accepted
- **Reason:** learned scale 不是理论闭合所必需，且会混入额外的效率与校准问题。

## ADR-0005 — Exact multi-source min-plus readout

- **Status:** accepted
- **Reason:** 它是 graph difference polytope 的标准精确读出，复杂度明确且保持证书方向。
- **Non-claim:** Dijkstra/min-plus 不是项目原创贡献。

## ADR-0006 — Native-preserving arbitration

- **Status:** accepted
- **Reason:** 只在 sound geometry gate 与 relative non-degradation bound 同时成立时干预。

## ADR-0007 — Separate V0 correctness from paper novelty

- **Status:** accepted
- **Reason:** 条件理论闭合不证明系统有效、实时、OOD-calibrated 或顶会原创。

## ADR-0008 — No license until the owner selects one

- **Status:** pending user decision
- **Reason:** 许可改变法律权限，不应推断。

## ADR-0009 — Freeze v8.3.1 as the Gate A/B specification

- **Status:** accepted
- **Reason:** v8.3.1 strengthens the primary theorem to every authorized relative intervention, adds map/geometry/numerical error accounting, and freezes visual-node and candidate-provider interfaces without adding trainable modules.
- **Boundary:** this is not a paper-novelty or empirical-success decision.

## ADR-0010 — Separate candidate supply, certification, and selection

- **Status:** accepted
- **Reason:** zero intervention can arise from no better candidate, a vacuous certificate, or the selection policy. Report \(S_{better}\)、\(R_{cert|better}\)、\(R_{opportunity}\) and \(R_{selected}\) on one predeclared session-time sampling unit.

## ADR-0011 — Asymmetric continuous geometry lifting

- **Status:** accepted for synthetic Gate A; NAVSIM ledger pending
- **Reason:** challenger 外包络 + fixed-grid \(h\)-cover 与 native internal nodes 足以推出连续 relative-margin lower bound，且避免不可生产化的 challenger internal-net 假设。
- **Boundary:** R002C 只验证解析构造与拒绝逻辑；continuous polygon teacher、coordinate transform 和 geometry arithmetic 的确定性误差预算仍阻塞 full Gate A。
