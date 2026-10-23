# Claim–Evidence Matrix

No row may be upgraded from `proposed` without a registered experiment and immutable result reference.

| ID | Intended claim | Required evidence | Falsifier | Status |
|---|---|---|---|---|
| C1 | Relative shape is more appearance-stable than direct absolute margins. | Paired same-scene appearance interventions; matched model capacity; confidence intervals across scenes and seeds. | No meaningful stability advantage or worse task error. | proposed |
| C2 | Joint set prediction is calibrated on unseen source sessions. | Session-held-out empirical joint coverage at predeclared levels; sharpness reported beside coverage. | Undercoverage outside tolerance or vacuous sets. | proposed |
| C3 | The affine split improves OOD decision quality beyond extra parameters. | Parameter-matched direct baseline and factorized ablations on frozen target domains. | Gain disappears under capacity control. | proposed |
| C4 | Native-preserving arbitration reduces modeled violations without excessive intervention. | Native failure recall, false intervention rate, replacement success, downstream NAVSIM metrics. | Safety gain requires indiscriminate replacement. | proposed |
| C5 | Candidate-conditioned tube features are necessary. | Centerline, global pooling, and swept-tube ablation with identical heads. | Equivalent result from simpler global features. | proposed |
| C6 | Improvements are not caused only by better candidate coverage. | Fixed-panel comparison and oracle-panel coverage analysis. | Method advantage disappears at matched coverage. | proposed |

## Claim language policy

- Use **shows** only for registered empirical results.
- Use **guarantees** only with an explicit theorem, assumptions, and proof.
- Use **safe** only with the modeled constraint class attached.
- Report null and negative results; do not silently redefine the primary endpoint.
