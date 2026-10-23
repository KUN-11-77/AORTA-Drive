# R004E-C1 — Single-group optimization sanity

- **Question:** Does the frozen R004D objective receive usable learning signal through the qualified real R004E feature/label path?
- **Config:** `configs/experiment/r004ec_training.yaml`, stage `C1_single_group_overfit`.
- **Scope:** lexicographically first train recording group; 16 anchors are nested measurements, not independent replicates.
- **Frozen budget:** four epochs, 16 updates/epoch, AdamW at `3e-4`, no development/calibration/test/target access.
- **Acceptance:** final 16-anchor mean objective `<=0.90` of its pre-update value; best post-epoch exact group maximum residual `<=0.98` of its pre-update value; every loss and gradient finite.
- **Status:** PASS. Both preregistered ratios passed with finite values; see `experiments/results/R004EC1.md`.
- **Claim boundary:** optimization and real-data signal sanity only. The same group is fitted and evaluated; no generalization or performance claim is permitted, and its weights cannot seed C2.
