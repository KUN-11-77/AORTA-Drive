# Experiment Plan

**Problem**: 正确视觉已包含donor-robust平均relation信号，但未降低`U_ctx`选择regret。

**Method Thesis**: 在完全固定generic scorer及计算预算时，直接优化soft selection regret能否把已有视觉
relation信号转化为候选决策收益。

**Date**: 2026-10-26

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| Q1 objective alignment | 排除“只需换一个简单loss”这一最强竞争解释 | 同初始化/步数下soft-regret组级regret优于父MSE，且达到10% oracle机会恢复 | B1 |
| Q2 scene binding under decision objective | 排除收益来自稳定先验或任意视觉通道 | S1/S2/S3/removed四种输入干预均可靠增加regret | B2 |
| Anti-claim | 改善来自更多参数、训练更久或重新选epoch | 参数/FLOP/token/optimizer/epoch/data order全部exact匹配 | B0 |

## Paper Storyline

- Main paper must prove：视觉信息存在、简单decision objective仍不足/足够、随后才判断结构创新必要性。
- Appendix can support：完整group分布、donor distance与seed诊断。
- Intentionally cut：多temperature搜索、多个ranking loss、DQ、appearance loss、NavDream。

## Experiment Blocks

### B0：Implementation equivalence

- Claim tested：除loss外无实现差异。
- Dataset / split：42 source-fit groups，冻结6-fold OOF。
- Compared systems：父V2 MSE与新soft-regret。
- Metrics：init hash、step count、batch order、parameter/FLOP/token、finite、exact replay。
- Success：所有机械不变量完全一致。
- Failure：INVALID，不读取科学结论。
- Priority：MUST-RUN。

### B1：Compute-matched objective qualification

- Claim tested：decision-aligned surrogate降低OOF regret。
- Compared systems：父V2 correct vs soft-regret correct。
- Primary metric：42-group paired decision regret与oracle-normalized recovery。
- Secondary：relation MSE、sign/boundary、std ratio、choice-change utility。
- Success：CI上界<0且恢复≥10%。
- Failure：停止该loss；不能否定全部decision learning。
- Table：主论文机制资格表或负结果诊断表。
- Priority：MUST-RUN。

### B2：Frozen checkpoint scene-binding intervention

- Claim tested：改善依赖正确场景视觉。
- Conditions：correct、S1、S2、S3、removed。
- Primary metric：每个intervention-minus-correct group regret CI。
- Success：四个CI下界均>0。
- Failure：MIXED并回到representation/binding诊断。
- Table：输入干预消融。
- Priority：MUST-RUN。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---:|---|---|---|
| M0 | loss、mask、tie与zero-challenger单测 | synthetic only | 全测试通过 | <1 GPUh | mask或梯度错误 |
| M1 | 单fold单seed资源/重放演练 | 1+1 | exact、finite、预算匹配 | 约父单job两倍 | 温度/IO异常 |
| M2 | 正式primary | 18 jobs | 禁止部分指标 | 约父V2 arm成本 | 资源争用 |
| M3 | exact replay | 18 jobs | 同job同GPU exact | 同M2 | 数值非确定 |
| M4 | 一次性audit | 1 | GO/MIXED/STOP/INVALID | CPU | 统计/来源错误 |

## Compute and Data Budget

- 新训练：18 primary + 18 replay；父MSE baseline不重训；
- 每job复用父selected epoch，因此optimizer steps逐job匹配；
- DINO cache、bundle、cross-fit与三套donor均复用；
- 最大瓶颈：6912视觉token的cross-attention与多文件cache IO，而非模型参数量。

## Risks and Mitigations

- exact ties使classification不稳定：主端点使用utility regret，不用top-choice accuracy。
- soft-regret不校准margin：明确只作objective qualification，不承担最终distribution claim。
- 温度被事后调优：固定为outer-train正regret-gap中位数，不搜索。
- 中位数实现歧义：偶数个正gap时固定取两个中央次序统计量的算术平均。
- parent epoch可能偏向MSE：这是保守的compute-match；失败只否定该冻结预算下的surrogate。
- donor shift强度不同：分别报告，不按距离筛选或作单调因果解释。

## Final Checklist

- [x] 主张不超过两个
- [x] 独立单位为recording group
- [x] loss之外参数、数据与预算匹配
- [x] 失败结果有冻结解释
- [x] 不提前引入主架构

## Methodological acknowledgement

实验设计流程参考 Scientific Agent Skills 的randomization/blocking与claim-driven experiment planning
规范。该参考只影响实验组织与审计流程，不构成AORTA方法或科学结论的证据：

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
arXiv:2609.00065v2. https://doi.org/10.48550/arXiv.2609.00065

来源于2026-10-26直接获取的arXiv记录；原始网页、纯文本、链接清单和获取元数据保存在本目录，
其中获取元数据SHA-256为`87e5d6378d900a1a9299aaf8c9e13ea37db5de5bebd0c7f7af29255b457402d7`。
