# R004H6R1 冻结结果

## 裁决

- 冻结科学裁决：`SUPPORTED`。
- 完整研究状态：`SUPPORTED_ON_HELDOUT_DEVELOPMENT`；R004H6A exact-teacher mechanical audit 已 PASS。
- 后续：`GO_R004H7_PREREGISTRATION_ONLY`；不进行第三次 R004H6 运行，不访问 calibration、test 或 target，不解锁驾驶收益实验。

## 范围与 provenance

- 24 个 development recording groups，384 anchors，每组等权统计。
- 这些 groups 未参与当前 R004H1 checkpoint 的拟合或选择，但在更早 v8.3 开发中曾被查看；因此只支持当前冻结模型的 held-out development 结论，不是 untouched test。
- 实现 revision：`291655866c29ab6e2f5fe7067e8dce176ae7b5f1`。
- 远端原始结果：`/data/khsong/aorta_runs/r004h6r1_2916558/results/`。
- `result.json` SHA256：`1e33acee0c2b8a82cada772712ed9b394dad5a6f9e9f1f484fb05a5611b698d1`。
- 384 个 field artifacts；结果目录 115,042,673 bytes。
- wall time：6061.38 s；peak CUDA allocation：3,216,324,096 bytes。
- clean replay mismatch：0；calibration/test/target access：false。

## 预注册主张

### Claim 1：held-out task generalization

通过。四个 confirmatory 条件均满足 learned 优于 zero 与 fit-only teacher prior、prior-minus-learned 的 95% group-bootstrap 下界大于 0、teacher cosine 为正、appearance drift 小于 clean edge energy。

| 条件 | Learned task MAE (m) | Appearance edge MAE (m) | Teacher cosine | Prior − learned (m) | 95% CI |
|---|---:|---:|---:|---:|---:|
| clean | 0.369921 | 0.000000 | 0.691276 | 0.083257 | [0.075960, 0.090107] |
| low exposure | 0.369637 | 0.010396 | 0.691716 | 0.083541 | [0.076387, 0.090186] |
| low contrast | 0.370006 | 0.009068 | 0.691336 | 0.083172 | [0.076104, 0.089992] |
| cool cast | 0.370029 | 0.011104 | 0.690986 | 0.083149 | [0.076010, 0.089836] |

Clean 对照：fit-only teacher prior `0.453178m`，zero field `0.554261m`。四个 confirmatory 条件下，learned-not-better-than-prior 与 learned-not-better-than-zero 的 group failure count 均为 0。

双次强度阶梯只作 secondary 解释：low exposure / low contrast / cool cast 的 task MAE 分别为 `0.371035/0.370400/0.369783m`，appearance edge MAE 为 `0.024502/0.018195/0.016827m`；不得把该结果改写为真实天气或任意 OOD 保证。

### Claim 2：正确场景图像提供任务信息

通过。

- correct clean task MAE：`0.369921m`。
- mismatched clean task MAE：`0.517764m`。
- mismatch-minus-correct：`+0.147843m`，95% group-bootstrap CI `[0.139053, 0.156883]`。
- image-removed task MAE：`0.555386m`。
- removed-minus-correct：`+0.185465m`，95% CI `[0.168068, 0.202888]`。
- uniform fit-mean DINO probe：`0.512763m`。
- mismatch-not-worse 与 removed-not-worse 的 group failure count 均为 0。

这支持当前冻结模型依赖正确场景视觉信息；错配与移除仍是机制探针，不是容量匹配的重训基线。

## 审计边界

legacy auditor 使用序列化 `float32` teacher 重放 runner 的 `float64` teacher 语义，曾在一个阈值附近的 secondary salient-edge scalar 上产生 `2.86494e-05` 差异。独立登记的 R004H6A exact-teacher audit 已验证 384/384 artifact hashes、384/384 teacher projections 与 384/384 metric bundles，并精确复现 `SUPPORTED`。详见 `MECHANICAL_AUDIT_R1.md` 与 `R004H6A_AUDIT_RESULT.json`。

规定的外部 cross-model reviewer backend 当前仍不可用，因此不能把内部机械 PASS 表述为独立科学复核或完整 cross-model integrity PASS。

本结果不支持 untouched test、NavDream/真实天气、目标域、population、coverage、规划收益、驾驶安全或方法新颖性主张。
