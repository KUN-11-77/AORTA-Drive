# R005E3C 实验完整性审查

Reviewer-Model: gpt-6-astra  
审查日期：2026-10-26  
总体裁决：`WARN`

## A. Ground Truth 与任务语义

PASS（限域）。目标来自已冻结的source-fit evaluator-derived `U_ctx`，不是合成伪标签，也没有读取
development/calibration/test/NavDream/target。但它只是局部上下文效用代理，不是完整policy value、闭环行为
或安全真值。

## B. 分数归一化与比较

PASS。诊断使用冻结softmax温度和native-relative utility；没有把模型自己的范围重新缩放成好看的结果，
也没有用常数偏移冒充改善。

## C. 结果真实性与数值闭合

PASS/WARN。560 anchors、3235 challengers、623 positives、2612 nonpositives及236 opportunity anchors
的计数闭合。终点正收益score-gradient从冻结prediction独立复算并与报告精确一致。初始化参数梯度张量
未写入紧凑归档，因此只能核对报告内部的范数、比率与余弦自洽性，不能独立重放该部分。

## D. Provenance 与机械审计

WARN。原v1审计器过度依赖primary/replay自洽，联合篡改部分冻结字段或删除科学payload仍可能误通过。
现已加入v2强化审计：固定schema/fold/seed/temperature/init/input/commit/runner/config/GPU，检查完整字段、
计数、有限值和模型未更新，并独立重算终点score-gradient。剩余限制仍是初始化原始张量未归档。

## E. 范围充分性

PASS（作为诊断），FAIL（若用于方法主张）。本轮覆盖fold-0 outer-train且没有权重更新，足以描述当前
soft-regret的局部动力学，不足以证明泛化、视觉增量、架构优势、外观鲁棒或驾驶收益。

## F. 总结与行动

- 未发现伪造ground truth、phantom result或score normalization fraud。
- 将结论限定为`descriptive optimization diagnostic`。
- 后续正式实验必须归档能独立重算关键机制量的中间张量。
- R005E3D若运行，必须作为结果读取前冻结的一因素实验，不能把E3C解释为其成功保证。

Claim impact：C30继续`UNSUPPORTED`；新增的梯度动力学观察只获得`SUPPORTED_DESCRIPTIVE`，不解锁
OOF、视觉、Differential Query、NavDream或安全主张。
