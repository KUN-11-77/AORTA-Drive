# R004E-C2 — Full registered source-only training

- **Question:** Can the sole frozen R004D quotient-field probe complete its registered source train/development optimization and yield one checkpoint selected at recording-group level?
- **Config:** `configs/experiment/r004ec2_full_training.yaml`.
- **Independent unit:** recording group. Train has 66 groups and development has 24; every group contains 16 nested anchors.
- **Frozen run:** seed `20261023`, 20 epochs, 1056 train anchors/epoch, AdamW, one-epoch linear warmup followed by the specified stepwise cosine schedule, no augmentation, no search.
- **Checkpoint:** after every epoch, each development group contributes exactly one maximum over 16 anchors × 8064 edges. Selection is lexicographic: worst-5 group mean, all-24 group mean, earliest epoch.
- **Status:** COMPLETE. The training/checkpoint engineering gate passed and selected epoch 1, but later epochs showed a strong negative development-tail trend. See `experiments/results/R004EC2.md`.
- **Claim boundary:** training/checkpoint qualification only. Calibration, test, target, candidate, certificate and planner metrics remain sealed.
