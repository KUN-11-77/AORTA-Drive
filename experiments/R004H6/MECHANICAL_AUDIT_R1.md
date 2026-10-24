# R004H6R1 冻结机械审计

## 状态：WARN（冻结 auditor 未 PASS）

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
- 若后续处置，只能新登记一个独立 audit-contract 修复，明确统一 runner 与 artifact 的 teacher dtype/threshold semantics，并保留本次失败记录；
- 在处置前，R004H7、calibration/test/target、决策收益均 HOLD。
