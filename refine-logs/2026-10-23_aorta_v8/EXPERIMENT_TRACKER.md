# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | theorem oracle | closed form vs LP vs multi-source | synthetic graphs | bound error, shift invariance | MUST | DONE | 280 randomized LP cases plus edge cases passed |
| R002 | M0 | continuous lifting soundness | teacher/node map + deterministic geometry checker | synthetic fields/footprints | \(h_j\), \(\varepsilon_{map}\), false acceptance | MUST | DONE | Synthetic rectangle route passed; production checker still blocks full Gate A |
| R003 | M0 | risk/numerical contract | relative-event and float adversaries | synthetic graphs/sessions | \(R_\gamma\), optimism bound \(\varepsilon_{num}\) | MUST | DONE | 500 exact-fraction cases support outward-safe \(\varepsilon_{num}=0\) |
| R002B | M0 | legacy geometry falsification | corner/discrete-pose/raster shortcuts | synthetic footprints/maps | constructive false acceptance | MUST | DONE | 3 counterexamples reproduced; legacy checker cannot certify AORTA |
| R002C | M0 | production geometry construction | analytic challenger outer cover + native internal nodes | synthetic continuous sweeps | \(h_j\), false acceptance, rejection behavior | MUST | DONE | 5 tests passed on clean revision; arithmetic-safe production fallback later qualified by R002E |
| R002D | M0 | continuous teacher construction | exact-rational polygon distance interval + explicit ledger | synthetic polygons/nodes | interval coverage, \(\varepsilon_{map}\), Lipschitz checks | MUST | DONE | 7 tests passed on clean revision; overlapping raw-cover semantics later qualified by R002E |
| R002E | M0 | NAVSIM preprocessing audit | official cache polygons + rear-axle vehicle protocol | 164 local metric caches | validity, hashes, overlap, vehicle/corner agreement | MUST | DONE | 164/164 caches + 45/45 tests passed on clean revision; raw-cover/disk route qualified; transform ledger pending |
| R002F | M0 | coordinate-transform ledger | ego-centric graph nodes to global raw polygons | 164 caches + analytic adversarial cases | interval containment, \(\varepsilon_{transform}\), protocol rejection | MUST | DONE | 164/164 caches + 49/49 tests passed; exact transform primitive qualified without libm |
| R002G | M0 | complete node-label manifest | frozen ROI/grid + raw-cover teacher + certified transform | 164 registered cache scenes | all-node labels/errors, ROI support, hashes, rejects | MUST | DONE | 671,744/671,744 nodes and 164/164 deterministic artifacts passed; full Gate A PASS within frozen nominal-road contract |
| R004 | M1 | protocol freeze audit | source session manifest + projection/candidate specs | source | leakage, independence, deterministic replay | MUST | TODO | freeze before features |
| R005 | M1 | legal predictor | existing \(u_{abs}\) audit or minimal probe | source train/dev | field error, held-out eligibility | MUST | TODO | no teacher-as-predictor |
| R006 | M1 | calibration sanity | absolute predictor + session score | exclusive source cal/test | finite-threshold eligibility, simultaneous coverage, sharpness | MUST | TODO | no calibration reuse |
| R007 | M2 | uncertainty-only control | absolute band | held-out source | exact bound, certification | MUST | TODO | same \(u_{abs}\) |
| R008 | M2 | uncertainty-only control | global gauge exact | held-out source | exact bound, certification | MUST | TODO | same \(u_{abs}\) |
| R009 | M2 | Gate B decisive test | local fixed graph exact | held-out source | \(S_{better}\), \(R_{cert|better}\), \(R_{opportunity}\), path strata | MUST | TODO | split supply from certificate |
| R010 | M3 | scalar baseline | candidate contrast calibrated | source | coverage, transfer, latency | MUST | BLOCKED | requires Gate B |
| R011 | M3 | absolute/global E2E | legal field baselines | source | coverage, sharpness, cost | MUST | BLOCKED | 3 seeds |
| R012 | M3 | joint-field baseline | frozen CDF-style construction | source | conservative relative bound | MUST | BLOCKED | solver error budget |
| R013 | M3 | main method | AORTA fixed-scale edge field | source | C1 metrics | MUST | BLOCKED | 3 seeds |
| R014 | M3 | transfer | held-out policy/planner | held-out source | coverage/utility degradation | MUST | BLOCKED | no target tuning |
| R015 | M4 | arbitration baseline | keep-native/geometry-only | frozen eval | \(R_\gamma\), supply/selection, NAVSIM parts | MUST | BLOCKED | matched candidates |
| R016 | M4 | structured baseline | strongest prior system | frozen eval | C2 metrics | MUST | BLOCKED | matched selection |
| R017 | M4 | main arbitration | AORTA | frozen eval | \(R_\gamma\), \(R_{selected}\), meaningful non-inclusion | MUST | BLOCKED | 3 seeds |
| R018 | M4 | empirical appearance test | frozen methods | frozen target suites | empirical robustness, paired \(\rho\) | MUST | BLOCKED | no target coverage claim |
| R019 | M5 | failure atlas | global vs local | held-out outputs | supply/certificate/selection, q domination, \(\rho\) | NICE | BLOCKED | reuse main outputs |
| R020 | M5 | learned-scale check | optional scale head | source only | sharpness vs fixed scale | NICE | BLOCKED | run only if fixed V0 passes |

## Status rules

- `TODO`: 可按当前依赖开始；
- `BLOCKED`: 前置 Gate 尚未通过，不得启动；
- `RUNNING`: 已注册并在执行；
- `DONE`: 结果与 artifact reference 已登记；
- `FAILED`: 执行错误，必须保留日志；
- `FALSIFIED`: 运行成功但否定目标 claim，不得改阈值后覆盖。
