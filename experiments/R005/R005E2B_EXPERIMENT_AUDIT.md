# R005E2B 实验完整性审计

日期：2026-10-26  
机械审计：`PASS`  
科学支持：`UNSUPPORTED`  
完整性状态：`WARN`（外部 cross-model reviewer backend 当前不可用）

## 审计范围

- 冻结执行器：`scripts/run_r005e2_fold_job.py`；
- 冻结机械审计器：`scripts/audit_r005e2_scientific_run.py`；
- 配置：`configs/experiment/r005e2_visual_recoverability.yaml`；
- 主结果：`experiments/results/R005E2B_0FDEE4B/audit.json`；
- primary/replay：54+54组 `result.json` 与 `predictions.npz`；
- target bundle：本地外部产物 `D:/AORTA-Drive-artifacts/R005E2B_0FDEE4B/bundle_primary/`；
- 预注册：`experiments/R005/R005E2_VISUAL_RECOVERABILITY_PREREGISTRATION.md`。

## A. Ground-truth provenance：PASS（限定作用域）

监督来自R005E1B冻结、可复算的source-fit native-history `U_ctx` candidate margins，不由本轮模型输出
生成；bundle manifest、target result、panel roster和cross-fit hashes均与预注册一致。它是官方 scorer
components 派生的局部 contextual target，不是完整NAVSIM two-stage policy value，后者覆盖曾为0/672。

## B. Score normalization：PASS

主指标使用未归一化 relation MSE、decision regret、sign/boundary error。`rho_vis`只除以训练前冻结的
R005E1B contextual oracle gain `0.016650366...`，不是模型自身最大值/均值；本轮负值`-0.025449`完整保留。

## C. Result existence and replay：PASS

- 54 primary和54 replay均存在且status为COMPLETED；
- 同job prediction arrays bit-exact，logical result exact；
- 服务器完整目录329个非日志文件SHA-256验证通过；
- 本地紧凑副本221个被保留文件与完整SHA manifest一致，108个未同步项均为大型checkpoint；
- 每个result声明的prediction SHA与实际文件一致；
- 本地失败解剖重新读取全部OOF prediction并再次检查primary/replay exact。

## D. Dead-code / metric execution：WARN

冻结审计器实际生成所有登记主指标、比较、bootstrap CI和GO/MIXED/STOP；结果键均存在。由于技能要求的
独立cross-model reviewer工具当前不可用，本轮没有声称完成逐行的外部dead-code审查。内部机械审计
不能冒充独立科学复核。

## E. Scope：WARN，但主张已收缩

范围为42个source-fit recording groups、672 anchors、三fold-matched arms、三seeds的严格OOF描述。
没有development、calibration、test、NavDream、跨planner或sequential evidence。因此结果只否定当前
source-fit probe的冻结主张，不能否定视觉信息的普遍存在，也不能支持/否定最终驾驶行为。

## F. Evaluation classification

`simulation_only + official_evaluator_derived_contextual_target + source_fit_group_OOF`。

## 额外协议核验

正式并行度从config文字中的每卡一个job修订为每卡最多四个独立job。修订发生在科学输出前，用户明确
授权并由`R005E2_RESOURCE_AMENDMENT_2026-10-25.md`记录；单job科学合同未变，同job primary/replay
绑定同GPU且最终bit-exact。因此不判INVALID，但属于必须显式披露的execution amendment。

## 总裁决

- 实现/数据/复现完整性：`PASS`；
- 外部独立完整性复核：`UNAVAILABLE`，故总完整性标记`WARN`；
- 科学裁决：`UNSUPPORTED / STOP_CURRENT_VISUAL_MAIN_CLAIM`；
- 禁止自动第三次相同训练；禁止用post-hoc切片挽救主结果。

## Claim impact

- “当前generic scorer能恢复正确视觉的决策增量”：`UNSUPPORTED`；
- “正确视觉含少量可学习relation信号”：`EXPLORATORY_SUPPORTED`，仅限MSE与失败解剖；
- “Differential Query优于matched scorer”：`UNTESTED`；
- “AORTA外观鲁棒/改善驾驶”：`UNTESTED`。

