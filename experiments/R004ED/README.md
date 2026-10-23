# R004E-D — Development visual-necessity control

- **Question:** Does the frozen epoch-1 visual checkpoint contribute more development group-tail information than trivial nonvisual controls?
- **Config:** `configs/experiment/r004ed_visual_necessity.yaml`.
- **Systems:** zero-difference field; float64 mean of zero-gauge training-label fields; frozen C2 epoch-1 visual checkpoint with no further fitting.
- **Endpoint:** exact residual maximum over 16 anchors × 8064 edges inside each of 24 development recording groups, then worst-5 and all-24 group means.
- **Acceptance:** visual checkpoint is strictly below both controls on both endpoints; ties receive no credit.
- **Status:** FAIL. The visual checkpoint beat the train-label mean prior but failed against the zero field on both endpoints. See `experiments/results/R004ED.md`; calibration/test/target remain sealed.
- **Failure action:** do not open calibration; return the predictor/objective to theory and architecture redesign.
