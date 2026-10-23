# R004E-C0 — Real training-input contract

- **Question:** Can one frozen R004E-B visual record and its exact R004E-A teacher label pass through the registered runtime sampler, R004D predictor, graph loss, and backward path without changing any frozen interface?
- **Config:** `configs/experiment/r004ec_training.yaml`, stage `C0_real_input_contract`.
- **Independent unit:** recording group. This structural audit deliberately opens one nested anchor and makes no statistical or generalization claim.
- **Registered input:** lexicographically first train recording group, anchor 0.
- **Acceptance:** NumPy/torch sampler max error `<=1e-5`; exact 8064-edge contract; zero-gauge error `<=1e-6 m`; deterministic replay and invalid-storage errors exactly zero; all values and gradients finite with at least one nonzero gradient tensor.
- **Status:** REGISTERED; not yet executed.
- **Claim boundary:** input and differentiability qualification only. No training accuracy, calibration, certificate, target-domain, or Gate B claim.
