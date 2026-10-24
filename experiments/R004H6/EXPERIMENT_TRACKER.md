# R004H6 实验跟踪

| 阶段 | 状态 | 证据 |
|---|---|---|
| 设计与主张冻结 | COMPLETE | `448509877a98f74bfd125f605bc40e69d838321b` |
| 实现与合成测试 | COMPLETE | 266 tests PASS；mypy strict PASS |
| development 输入预检 | PASS | 24 groups / 384 anchors / 4482 unique images；0 missing / 0 hash mismatch |
| R004H6 首轮 | STOPPED_RESOURCE_CAP | 384 fields 已写出，但末端汇总前触发 90 分钟上限；未读取科学指标 |
| R004H6R1 资源恢复重放 | COMPLETE | 24 groups / 384 anchors；6061.38 s；clean replay mismatch 0；calibration/test/target 未访问；raw result SHA256 `1e33acee...b698d1` |
| legacy serialized-teacher audit | WARN | 384/384 artifact hashes 匹配；383/384 metric bundles 完全复算一致。唯一差异为 group `2021.10.01.19.16.42_veh-28`、anchor 4 的 `fit_teacher_prior.salient_edge_mae_m`：stored `0.6014474030063619`、float32-artifact replay `0.6014187536344174`、绝对差 `2.86494e-05` |
| R004H6A exact-teacher audit | PASS | 不重跑推理；384/384 hashes、384/384 float32 projections、384/384 metric bundles 全部通过，原 tolerance 与 `0.50m` 阈值不变，冻结裁决复现为 `SUPPORTED` |
| 外部 experiment-audit | REVIEW_UNAVAILABLE | 当前会话未暴露规定的独立 reviewer backend；不得以内部机械检查替代独立复核 |
| 科学裁决 | SUPPORTED_ON_HELDOUT_DEVELOPMENT | 两条预注册科学主张均通过且 exact-teacher mechanical audit PASS；仍禁止升级为 untouched test、OOD、驾驶或安全结论 |
| 后续裁决 | GO_R004H7_PREREGISTRATION_ONLY | 允许理论与结构必要性对照的预注册，不授权训练；calibration/test/target 与驾驶收益继续 HOLD |

