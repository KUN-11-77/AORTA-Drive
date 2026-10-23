# AORTA-Drive Research Brief

> Status: theory-first research specification; no empirical performance claim is authorized yet.

## 1. Problem anchor

AORTA-Drive studies **source-only appearance robustness for end-to-end driving**. For the same physical scene and driving intent, an appearance intervention may change pixels while leaving the feasible and safe decision structure unchanged. A robust system should preserve task-relevant decision properties, not necessarily reproduce an identical latent vector or a single reference trajectory.

The central question is:

> Can a model learn the scene-conditioned ordering and calibrated safety anchor of a canonical trajectory panel, then use uncertainty-aware arbitration to preserve a native planner unless a replacement is demonstrably safer?

## 2. Research hypothesis

For a canonical candidate panel, the signed safety-margin vector admits a useful affine decomposition into:

1. a **relative shape** describing which candidates are safer than others;
2. a **positive scale** measuring panel separation;
3. an **absolute offset** anchoring the zero-safety boundary.

Appearance robustness is expected to be easier to learn for relative shape than for an absolute safety value. Absolute safety must therefore remain set-valued and calibrated rather than collapsed to an unjustified point estimate.

## 3. Frozen scope

- Benchmark family: NAVSIM v1/v2.
- Backbone baseline: frozen DINOv3-H+ unless a documented decision supersedes it.
- Robustness setting: source-only training; target-domain labels, tuning, and threshold selection are forbidden.
- Perturbation focus: appearance changes that should preserve scene geometry and task intent.
- Decision interface: canonical candidate panel plus native-planner trajectory.
- Primary goal: explain and reduce task-level failure, not maximize feature invariance.

## 4. Non-goals

- Claiming universal safety from a finite candidate panel.
- Treating CKA/RSA similarity as a sufficient robustness certificate.
- Reconstructing a full HD map when a task operator is sufficient.
- Replacing the native trajectory by default.
- Using target-domain data for calibration while calling the method source-only.
- Adding modules or losses without a claim they uniquely test.

## 5. Evidence already motivating the project

Previous project work found that appearance shifts can cause large planning changes even when global representations remain similar, and that geometry supplied at the decision interface can repair some trajectories. These observations motivate the task-operator hypothesis, but they do **not** establish the proposed factorization, calibration, or algorithmic advantage.

## 6. Unknowns that must remain explicit

- Whether the affine factorization is statistically more stable under appearance shift.
- Whether relative shape transfers better than direct margin regression.
- Whether set-valued anchoring is calibrated on unseen source sessions and OOD appearances.
- Whether the native-preserving arbiter improves safety without excessive intervention.
- Whether gains remain after controlling candidate-panel coverage and trajectory diversity.

## 7. Acceptance criterion

The method advances only if pre-registered tests support all three links:

`factorization advantage -> calibrated joint operator -> beneficial arbitration`.

Failure of any link triggers diagnosis or scope reduction rather than narrative substitution.
