# AORTA-v8.4 Refinement Report

## Outcome

The initial v8.4 proposal was not accepted as written. Three review rounds converted it
from an overclaimed heteroscedastic-tail design into a narrower, auditable candidate:
scale-only proper training, historical proxy disclosure, official-only formal calibration,
four-term outward residual arithmetic, same-potential controls and a common task-opportunity
endpoint.

## Main revisions

1. **Removed invalid anti-inflation logic.** Proper loss plus a tail term is not proper;
   the tail term was deleted from Stage S.
2. **Separated data roles.** 42 fit / 8 selection / 16 proxy-cal is now explicit, and
   historical exposure prevents a proxy coverage claim.
3. **Closed arithmetic.** Residuals are enclosed from four node values before division;
   the registered cancellation counterexample prevents naive gradient subtraction.
4. **Improved attribution.** Learned, pooled and uniform scales share one frozen potential
   and separate thresholds.
5. **Aligned evaluation with the decision.** The primary endpoint is corrected-certificate
   recall on a method-independent teacher opportunity set, not edge MAE alone.
6. **Kept novelty provisional.** The proposal explicitly treats normalized conformal,
   heteroscedastic NLL and min-plus as standard.

## Review trajectory

| Round | Score | Verdict | Main unresolved issue |
|---|---:|---|---|
| 1 | 7.05 | REVISE | tail term invalidated properness; split/numerics/novelty incomplete |
| 2 | 7.75 | REVISE | historical proxy dependence; four-term cancellation |
| 3 | 8.05 | REVISE; R004E-G READY | real mechanism value and originality untested |

## Empirical follow-through

R004E-G was registered only after Round 3 declared the CPU gate ready. It passed all
registered exact and regression checks. The result changes the next action from “repair
theory primitives” to “freeze a real R004H scale-allocation probe.” It does not change the
paper verdict to READY.
