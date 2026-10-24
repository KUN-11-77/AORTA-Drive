# R004H6 实验跟踪

| 阶段 | 状态 | 证据 |
|---|---|---|
| 设计与主张冻结 | COMPLETE | `448509877a98f74bfd125f605bc40e69d838321b` |
| 实现与合成测试 | COMPLETE | 266 tests PASS；mypy strict PASS |
| development 输入预检 | PASS | 24 groups / 384 anchors / 4482 unique images；0 missing / 0 hash mismatch |
| R004H6 首轮 | STOPPED_RESOURCE_CAP | 384 fields 已写出，但末端汇总前触发 90 分钟上限；未读取科学指标 |
| R004H6R1 资源恢复重放 | COMPLETE | 24 groups / 384 anchors；6061.38 s；clean replay mismatch 0；calibration/test/target 未访问；raw result SHA256 `1e33acee...b698d1` |
| 冻结机械结果审计 | WARN | 384/384 artifact hashes 可读且匹配；383/384 metric bundles 完全复算一致。唯一差异为 group `2021.10.01.19.16.42_veh-28`、anchor 4 的 `fit_teacher_prior.salient_edge_mae_m`：stored `0.6014474030063619`、artifact replay `0.6014187536344174`、绝对差 `2.86494e-05`；冻结 auditor 因 `metrics=1` 未给 PASS |
| 外部 experiment-audit | REVIEW_UNAVAILABLE | 当前会话未暴露规定的独立 reviewer backend；不得以内部机械检查替代独立复核 |
| 科学裁决 | SUPPORTED_PROVISIONAL | 两条预注册科学主张均通过；但因冻结机械 audit 非 PASS，仅保留 provisional，禁止升级为完整 integrity PASS、OOD、驾驶或安全结论 |
| 后续裁决 | HOLD | 不进行第三次运行；先单独处置 teacher 序列化精度导致的 secondary salient-edge replay 差异，再决定是否解锁 R004H7 |

