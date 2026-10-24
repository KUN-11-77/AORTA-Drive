# Experiment Audit Report

**Date:** 2026-10-24  
**Project:** AORTA-Drive / R004H4P5I  
**External reviewer:** `REVIEW_UNAVAILABLE`

## Overall verdict: REVIEW_UNAVAILABLE

## Integrity status: provisional

The required independent cross-model reviewer backend is not exposed in this execution
environment. No executor-authored review is substituted for independent acquittal.

The fail-closed mechanical audit passed. It pinned the implementation revision and frozen
config digest, required eight complete sentinel rows and six candidate ledgers per row,
recomputed all pipeline counts, checked finite nonnegative interface quantities, verified
the registered decision and rejected every prohibited positive claim. It read no metric
cache or label payload. This establishes internal consistency only.

## Files queued for independent review

- `scripts/run_r004h4p5i_realized_geometry.py`
- `scripts/audit_r004h4p5i_result.py`
- `src/aorta_drive/evaluation/navsim_geometry.py`
- `src/aorta_drive/evaluation/navsim_execution.py`
- `src/aorta_drive/certification/geometry.py`
- `src/aorta_drive/certification/opportunity.py`
- `tests/test_navsim_geometry.py`
- `tests/test_r004h4p5i_result_audit.py`
- `configs/experiment/r004h4p5i_realized_geometry_interface.yaml`
- `experiments/results/R004H4P5I/result.json`
- `experiments/results/R004H4P5I/audit.json`
- `experiments/results/R004H4P5I/run.log`
- `experiments/results/R004H4P5I.md`
- `refine-logs/2026-10-24_aorta_v8_4/EXPERIMENT_TRACKER.md`

## Claim impact

- Realized-geometry implementation closure: provisionally supported by mechanical PASS.
- Full computation path exercised: provisionally supported on six of eight sentinels.
- Opportunity prevalence and method performance: unsupported and explicitly prohibited.
- Coverage, Gate B and development access: unsupported.
