# R005E3D 实验完整性审查

日期：2026-10-26。

总体裁决：`WARN`（机械完整性PASS；独立cross-model科学审查不可用）。

## A. 目标来源：PASS（限域）

目标来自R005E1B冻结source-fit NAVSIM evaluator components，并由`target_margin.npy`提供
native-relative `U_ctx` relation；不是模型输出生成的伪真值。它仍是native-history、teacher-forced、
row-level proxy，不是官方完整two-stage policy value、真实闭环行为或安全标签。

## B. 分数与归一化：PASS

正式门直接使用raw `U_ctx`差、mean gain、hard regret和oracle opportunity；没有用模型自身最大值、均值
或预测方差归一化来制造提升。absolute oracle capture以冻结R005E1B oracle opportunity为分母，主表同时
报告raw gain和regret。

## C. 结果存在性与复算：PASS

- `primary/`、`replay/`、`audit.json`、`SHA256SUMS.txt`与`exit_code.txt`完整存在；
- primary/replay最终prediction、checkpoint及结果exact；
- 本地`audit_local.json`由保存的prediction、bundle、父/控制结果独立复算并匹配服务器审计；
- R005E1B完整component ledger与bundle trajectory arrays的SHA均核验，失败解剖重构target最大误差
  `2.97e-8`。

## D. 代码路径：PASS / WARN

正式runner、auditor与结果文件均实际执行。最终endpoint可以独立复算；每个epoch只保存聚合ledger，未保存
中间prediction，因此epoch路径只能视为exact-replay self-consistency，不能作为独立复算证据。

## E. 范围：WARN

科学运行只有source-fit fold-0 outer-train、单seed，是优化资格实验，不是组外泛化研究。结果不得称为视觉
鲁棒、架构优越、NavDream收益、planner行为改善或安全提升。failure anatomy是结果读取后的探索性分析，
不能改变冻结FAIL、选择阈值或作为下一方法的确认性证据。

## F. 评价类型

`simulation_only / evaluator-derived scoped proxy`：真实source记录、冻结合法候选、官方NAVSIM仿真与评分
components；不是最终benchmark policy evaluation。

## 完整性限制

当前环境没有可用的独立cross-model reviewer backend，因此不能把执行者自己的机械复核表述为独立科学
审稿。保守标记`WARN`；该限制不改变冻结数值和FAIL裁决。

## Claim impact

- proper MSE单因素足以恢复正决策收益：`unsupported`；
- 当前网络能使用correct视觉降低平均relation误差：`descriptive on outer-train only`；
- 灾难损失由temporal hard-gate drop主导：`supported as post-hoc exact component decomposition`；
- structured factorization会解决问题：`unproven hypothesis`。
