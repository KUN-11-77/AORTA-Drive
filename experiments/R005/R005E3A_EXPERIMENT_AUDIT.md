# R005E3A 实验完整性审计

日期：2026-10-26  
机械审计：`PASS`  
科学分类：`MIXED`  
完整性状态：`WARN`（外部cross-model reviewer backend不可用）

## 审计范围

- 冻结runner：`scripts/run_r005e3a_fold_job.py`；
- 冻结auditor：`scripts/audit_r005e3a_run.py`；
- 配置：`configs/experiment/r005e3a_objective_qualification.yaml`；
- 18 primary + 18 replay的result、prediction与checkpoint hashes；
- 审计产物：`experiments/results/R005E3A_77F37E4/formal/audit.json`；
- post-hoc只读诊断：`scripts/diagnose_r005e3a_collapse.py`及对应JSON。

## A. Ground-truth provenance：PASS（限定作用域）

监督来自R005E1B已审计的source-fit native-history `U_ctx`，不由本轮模型生成。合法ragged mask、fold、
target与父V2绑定均通过。本对象仍不是完整two-stage policy value。

## B. Score normalization：PASS，但发现验收语义缺陷

主regret、MSE和error均未按模型输出事后归一化。登记的`0.159175`比率只除以预冻结oracle gain，计算
本身正确；但它衡量的是相对parent的regret改善，不是相对native获得的绝对收益。未来门必须显式使用
`G_model/G_oracle`，本轮冻结裁决不追溯修改。

## C. Result existence and replay：PASS

- 18+18 jobs全部存在且完成；
- primary/replay predictions bit-exact、model state exact；
- finite、fold/target/mask OOF覆盖、same-GPU、source-only、sealed-role与temperature来源均通过；
- 完整远端151个文件已生成SHA-256清单；
- 本地紧凑副本119个文件逐文件验证，0 hash mismatch；大型36个checkpoint保留在远端，其一致性由
  正式auditor和每job hash证明。

## D. Phantom-result / dead-code：PASS/WARN

冻结auditor确实读取全部prediction并生成主指标、42-group bootstrap、科学门和最终字符串。新增只读
诊断独立加载18个primary checkpoint，在outer-train重算分数与选择；3个合成单元测试验证native退化、
oracle分解和fail-closed mask。外部独立reviewer工具当前不可用，故不能标记为独立科学复核。

## E. Scope：WARN，但边界明确

范围仅为42个source-fit groups、672 OOF anchors与outer-train机制诊断。development、calibration、
test、NavDream、target、跨planner和sequential均未访问。post-hoc诊断不改变正式`MIXED`。

## 总裁决

- 执行、数据、复现：`PASS`；
- 冻结科学分类：`MIXED`；
- 机制解释：`ALL_NATIVE_COLLAPSE / ZERO_ABSOLUTE_ORACLE_CAPTURE`；
- 外部独立复核：`REVIEW_UNAVAILABLE`，总完整性`WARN`；
- 下一步：仅优化/可实现性资格；禁止自动第三次相同训练和任何下游主张。

