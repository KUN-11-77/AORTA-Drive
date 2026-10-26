# R005E3B Experiment Audit

日期：2026-10-26  
项目：AORTA-Drive  
内部机械审计：增强版只读审计，SHA
`b84d4ea3ce479d13cf8042e101ed761563bc9b9583ec0da3f661458822a9a25d`  
外部cross-model reviewer：`REVIEW_UNAVAILABLE`

## Overall Verdict: WARN

## Integrity Status: mechanical PASS；external review unavailable

科学裁决与完整性裁决必须分开：本实验机械有效，但C30科学命题为`UNSUPPORTED`。`WARN`只表示当前环境
没有可用的独立cross-model reviewer，不能把内部审计冒充外部审计。

## A. Ground Truth Provenance: PASS（限定目标）

- `ScientificDataset`从冻结bundle的`target_margin.npy`读取目标，而不是从本轮模型输出制造目标：
  `src/aorta_drive/training/r005e2_science.py:88-91`。
- v2审计逐索引比较本轮target/mask与冻结bundle：
  `scripts/audit_r005e3b_score_realizability.py:228-246`。
- 目标只是在R005E1B登记的source-fit `U_ctx` native-relative margin；它不是官方完整two-stage policy value、
  真实闭环安全或NavDream真值。

## B. Score Normalization: PASS

- `G_model`、`G_oracle`与hard regret直接由target和hard choice计算：
  `scripts/run_r005e3b_score_realizability.py:64-124`。
- absolute oracle capture的分母是冻结target给出的oracle gain，不来自prediction的max/min/mean；原始gain与
  raw regret同步报告。最终分子为0、比率为0，没有自归一化制造高分。

## C. Result File Existence: PASS

- primary、replay、predictions、原审计、v2审计与SHA清单均已同步到
  `experiments/results/r005e3b_compact_6535975/`；compact 14/14文件本地SHA复核无不一致。
- primary/replay final state SHA、训练合同和独立复算均在`audit_v2.json`通过。
- 冻结门输出`FAIL_LINEAR_REALIZABILITY`，与结果文件中的0 gain、0 capture、0 nonnative一致。

## D. Dead Code Detection: PASS（本轮路径）

- runner的`_decision_summary`被初始化、逐epoch及三种最终条件调用；
- auditor的独立`_summary`和`_classify`被正式v2审计调用：
  `scripts/audit_r005e3b_score_realizability.py:33-96,248-280`；
- 相应输出出现在primary/replay `result.json`及`audit_v2.json`。

## E. Scope Assessment: WARN

- 仅fold0、seed 20261025、560个source-fit outer-train anchors；这是故意设计的优化/可实现性资格实验，
  不是泛化研究。
- 不能使用“robust”“generalizes”“improves driving”或ICCV方法成功等范围表述。
- 负结果足以否定“只删除tanh即可恢复训练域收益”的具体命题，但不能否定视觉信息论可恢复性。

## F. Evaluation Type: real_gt for scoped evaluator-derived proxy

数据来自真实source记录及冻结evaluator components；`U_ctx`是明确登记的局部代理目标，不是模型自生成
reference。相对于最终策略价值，它必须继续标为proxy。

## Action Items

1. 保留C30为`UNSUPPORTED`，不删除或覆盖负结果；
2. 在不更新权重的条件下审计正/负机会与初始化梯度质量；
3. 只有该诊断支持预注册机制时，才能单因素检验pairwise surrogate；
4. Differential Query、OOF确认、NavDream与行为实验继续HOLD；
5. 投稿前补做真正的独立cross-model或人工科学复核。

## Claim Impact

- C29：保持unsupported；
- C30：unsupported；
- “identity解决all-native collapse”：rejected；
- “softmax低概率尾部参与梯度吸收”：当前轨迹支持机制诊断，但尚非排他性因果结论；
- OOF、视觉绑定、外观鲁棒、驾驶收益、安全：未验证。

