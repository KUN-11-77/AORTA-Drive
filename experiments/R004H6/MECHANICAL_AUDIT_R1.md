# R004H6R1 冻结机械审计

## 最终处置：PASS（R004H6A exact-teacher audit）

独立登记的 R004H6A 没有重跑模型、修改结果或放宽门槛，而是按 runner 的原始语义重新读取哈希冻结的 source `float64` teacher，并同时验证序列化 teacher 必须与 `exact_teacher.astype(float32)` bit-exact。结果为：

- 384/384 field artifact hashes 匹配；
- 384/384 teacher float32 projections bit-exact；
- 384/384 metric bundles 在原 `rtol=1e-6, atol=1e-6` 下复算一致；
- macro summary 与冻结科学裁决精确复现为 `SUPPORTED`；
- config、阈值 `0.50m`、统计方案、checkpoint、数据、外观和 decision rules 均未变化。

机器可读结果：`R004H6A_AUDIT_RESULT.json`。审计脚本/合同实现 revision：`02ac19de2238cd80912e33fadd80df6779f775c4`。该 PASS 只解除 artifact fidelity HOLD，不增加科学证据或扩大主张范围。

## 历史状态：WARN（legacy serialized-teacher auditor 未 PASS）

对完整 `result.json` 与 384 个序列化 field artifacts 执行冻结 `scripts/audit_r004h6_result.py`：

- result status 与 384-anchor completeness 通过；
- config 与 runner hash 通过；
- donor derangement、唯一 group/anchor、有限值检查通过；
- artifact hash failure：0；
- 383/384 anchor metric bundles 完全复算一致；
- metric failure：1。

唯一差异：

- recording group：`2021.10.01.19.16.42_veh-28`；
- anchor index：4；
- scene token：`9c1b06bc71f356be`；
- metric：`fit_teacher_prior.salient_edge_mae_m`；
- stored：`0.6014474030063619`；
- serialized-artifact replay：`0.6014187536344174`；
- absolute difference：`2.8649371944446693e-05`；
- frozen tolerance：`rtol=1e-6, atol=1e-6`。

该差异与 runner 在写 artifact 时将 teacher 转为 `float32`、而 salient-edge mask 使用硬阈值 `abs(edge)>=0.50` 的精度边界相符；这是诊断解释，不是通过门槛的事后修改。当前未修改 metric、threshold、tolerance、result、artifact 或 checkpoint，也未重跑实验。

该 secondary salient-edge metric 不参与两条预注册主 gate，且主 gate 所依赖的 task MAE、appearance drift、teacher cosine、prior/zero paired difference、mismatch/removal paired difference均在其余机械复算中一致。但冻结 auditor 的正式结果仍为非 PASS，因此完整 integrity 只能记为 WARN，科学结论必须带 `PROVISIONAL`。

## 后续限制

- 禁止第三次实验运行；
- 禁止放宽冻结 tolerance 来制造 PASS；
- legacy WARN 必须与 R004H6A PASS 同时保留，不能删除失败历史；
- calibration/test/target 与驾驶收益继续 HOLD；R004H7 只解锁预注册与理论/结构对照设计，不授权训练或读取新数据。
