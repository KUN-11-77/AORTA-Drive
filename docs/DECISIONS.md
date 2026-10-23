# Architecture Decision Log

Record durable decisions here. Temporary observations belong in experiment result cards.

## ADR-0001 — Use a `src/` Python package layout

- **Status:** accepted
- **Reason:** prevents accidental imports from the repository root and matches packaging practice.

## ADR-0002 — Separate task shape from scale and absolute anchor

- **Status:** proposed, under empirical qualification
- **Reason:** appearance changes may preserve relative decision structure more reliably than absolute safety margin.
- **Reversal condition:** C1 or C3 is falsified under controlled comparisons.

## ADR-0003 — Preserve the native planner by default

- **Status:** accepted as algorithm policy, empirical utility unverified
- **Reason:** the learned module should intervene only when native violation is certified and a replacement is certified within the modeled constraints.

## ADR-0004 — No license until the owner selects one

- **Status:** pending user decision
- **Reason:** a public repository without a license is visible but not generally reusable; choosing a license changes legal permissions and must not be inferred.
