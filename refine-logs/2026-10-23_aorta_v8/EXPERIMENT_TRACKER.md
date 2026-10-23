# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | theorem oracle | closed form vs LP vs multi-source | synthetic graphs | bound error, shift invariance | MUST | TODO | CPU only |
| R002 | M0 | geometry soundness | deterministic internal checker | synthetic footprints | false acceptance, (h_j) validity | MUST | TODO | CPU only |
| R003 | M0 | numerical contract | super-source and float adversaries | synthetic graphs | optimism bound (\varepsilon_{num}) | MUST | TODO | CPU only |
| R004 | M1 | data/split audit | source session manifest | source | leakage, independence counts | MUST | TODO | freeze before features |
| R005 | M1 | legal predictor | existing (u_{abs}) audit or minimal probe | source train/dev | field error, held-out eligibility | MUST | TODO | no teacher-as-predictor |
| R006 | M1 | calibration sanity | absolute predictor + session score | source cal/val | simultaneous coverage, sharpness | MUST | TODO | one seed first |
| R007 | M2 | uncertainty-only control | absolute band | held-out source | exact bound, certification | MUST | TODO | same (u_{abs}) |
| R008 | M2 | uncertainty-only control | global gauge exact | held-out source | exact bound, certification | MUST | TODO | same (u_{abs}) |
| R009 | M2 | Gate B decisive test | local fixed graph exact | held-out source | meaningful non-inclusion, path strata | MUST | TODO | stop/go after this run |
| R010 | M3 | scalar baseline | candidate contrast calibrated | source | coverage, transfer, latency | MUST | BLOCKED | requires Gate B |
| R011 | M3 | absolute/global E2E | legal field baselines | source | coverage, sharpness, cost | MUST | BLOCKED | 3 seeds |
| R012 | M3 | joint-field baseline | frozen CDF-style construction | source | conservative relative bound | MUST | BLOCKED | solver error budget |
| R013 | M3 | main method | AORTA fixed-scale edge field | source | C1 metrics | MUST | BLOCKED | 3 seeds |
| R014 | M3 | transfer | held-out policy/planner | held-out source | coverage/utility degradation | MUST | BLOCKED | no target tuning |
| R015 | M4 | arbitration baseline | keep-native/geometry-only | frozen eval | H, intervention, NAVSIM parts | MUST | BLOCKED | matched candidates |
| R016 | M4 | structured baseline | strongest prior system | frozen eval | C2 metrics | MUST | BLOCKED | matched selection |
| R017 | M4 | main arbitration | AORTA | frozen eval | H, meaningful interventions | MUST | BLOCKED | 3 seeds |
| R018 | M4 | empirical appearance test | frozen methods | frozen target suites | empirical robustness only | MUST | BLOCKED | no coverage claim |
| R019 | M5 | failure atlas | global vs local | held-out outputs | q domination, path accumulation | NICE | BLOCKED | reuse main outputs |
| R020 | M5 | learned-scale check | optional scale head | source only | sharpness vs fixed scale | NICE | BLOCKED | run only if fixed V0 passes |

## Status rules

- `TODO`: 可按当前依赖开始；
- `BLOCKED`: 前置 Gate 尚未通过，不得启动；
- `RUNNING`: 已注册并在执行；
- `DONE`: 结果与 artifact reference 已登记；
- `FAILED`: 执行错误，必须保留日志；
- `FALSIFIED`: 运行成功但否定目标 claim，不得改阈值后覆盖。
