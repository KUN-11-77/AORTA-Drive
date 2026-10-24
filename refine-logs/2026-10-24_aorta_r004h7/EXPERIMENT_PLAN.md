# R004H7 Experiment Plan

**Problem**: 冻结驾驶系统在纯外观变化下发生候选排序翻转；当前视觉场已在 held-out development 上有效，但其结构必要性尚未证明。
**Method thesis**: 学习与校准候选 set-min contrasts 所定义的 decision quotient，比优化完整场误差更直接地保持冻结候选排序。
**Date**: 2026-10-24

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C13 decision relevance | 排除“field MAE 已足够”的解释 | 冻结模型上 decision error 对 regret/flip 的组级关联严格强于 edge/node/DINO controls | B1 |
| C14 structural necessity | 把合理表示升级为算法贡献 | 容量、数据、预算匹配时，QP-DQ 在 regret/flip 上胜过 potential、projected edge、SDF 与 direct pair | B2 |

反主张：收益来自 DINO、模型容量、候选输入、更多训练或选择偏差。

## Paper Storyline

- Main paper must prove：正确图像提供任务信息；decision quotient 是与排序失败对齐的对象；DQ supervision 在公平对照下减少 regret/flip。
- Appendix can support：小图 oracle、tie/active-set 反例、参数/FLOPs、失败组与不同 appearance 的完整分布。
- Intentionally cut：开放式几何证书重构、更多 risk-field 模块、语言/RL verifier，以及在正式 test 前扩展大量 backbone。

## Experiment Blocks

### B0：数学资格

- Claim tested：所有后续指标和网络目标是否建立在正确数学对象上。
- Dataset：无数据，小图枚举。
- Compared systems：node potential、raw edge、projected edge、exact set-min query。
- Metrics：oracle failure count 必须为0。
- Success：全部不变量、界与反例通过。
- Failure：修正理论并重新注册；禁止用真实数据绕过。
- Target：Appendix theorem/oracle table。
- Priority：MUST-RUN。

### B1：只读决策相关性

- Claim tested：C13。
- Dataset：24个已访问 development groups；冻结 R004H6 outputs；不更新模型。
- Controls：edge MAE、gauge-centered node MAE、DINO drift。
- Metrics：group-level regret association、group-bootstrap flip AUROC、validity/tie ledger。
- Success：PREREGISTRATION 中的 conjunctive gate 全部通过。
- Failure：停止 DQ training claim；保留视觉机制论文路线。
- Target：Main Figure 3 / mechanism table。
- Priority：MUST-RUN。

### B2：容量匹配结构对照

- Claim tested：C14。
- Dataset：42 fit / 8 selection / 16 one-read proxy；3 seeds。
- Systems：QP-EDGE、EDGE-PROJ、SDF-NODE、QP-DQ、EDGE-PROJ-DQ、DIRECT-PAIR。
- Metrics：teacher regret、flip、contrast MAE、appearance drift、edge MAE、params/FLOPs/latency。
- Success：QP-DQ 满足预注册 intersection gate。
- Failure：采用最简单胜者，或把论文降级为机制发现；禁止追加模块保护原设计。
- Target：Main Table 2 / ablation table。
- Priority：MUST-RUN，但仅在B0/B1通过后。

### B3：跨候选策略迁移

- Claim tested：candidate-free relation field 相比 direct pair 更可复用。
- Dataset：后续冻结 source panel；至少两个 candidate providers。
- Success：同一 field checkpoint 在第二 provider 上保持 regret/flip 优势。
- Failure：删除 candidate-independent transfer 主张。
- Target：Main/Appendix，取决于效果。
- Priority：NICE-TO-HAVE，属于 H8，不在本轮执行。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---:|---|---|---|
| M0 | 理论与oracle | CPU synthetic only | 0 failures | <1 CPU-hour | active-set/tie定义错误 |
| M1 | shadow relevance | 1 frozen read-only run | C13 gate | <4 GPU-hours，优先复用缓存 | candidate panel coverage不足 |
| M2 | matched training | 6 arms × 3 seeds | C14 gate | 待实现前profiling冻结 | 算力与容量不公平 |
| M3 | paper diagnosis | no retraining | 完整失败/最差组分析 | <2 CPU-hours | 事后挑指标 |

## Compute and Data Budget

- 当前授权：仅 M0 的 CPU oracle 与实现测试。
- H7-B/H7-C 的 GPU 时限必须在运行器 profiling 后另行冻结，不能从结果反向修改。
- 不需要人类评价。
- 最大瓶颈：公平的 candidate panel 与 direct-pair 输入预算，而不是网络深度。

## Risks and Mitigations

- 决策误差与 regret 共享定义导致相关性近似必然：H7-B 明确只作 endpoint gate，算法贡献必须来自 H7-C。
- potential 与 projected edge 表达等价：把二者胜负限定为经验归纳偏置，不作表达能力主张。
- 16 proxy groups 历史暴露：只作方法形成，最终结论必须等待一次性 test。
- 多 seed 伪重复：统计单位始终是 recording group；seed 只报告算法稳定性。
- direct pair 获得候选输入：将其作为更强基线明确记账；若胜出则简化方法。

## Methodology provenance

本预注册的 claim-driven run order、blocking、独立统计单位和 pseudoreplication 约束参考：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner. “Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.” arXiv:2609.00065, v2, 2026. DOI: https://doi.org/10.48550/arXiv.2609.00065。核验元数据保存在 `references/sources/scientific_agent_skills_2609.00065.xml`。

## Final Checklist

- [x] 主张不超过两条
- [x] novelty isolation 与 simplicity stop rule 明确
- [x] recording group 是独立单位
- [x] 参数、FLOPs、数据和选择预算要求匹配
- [x] must-run 与 later-stage 分离
- [ ] H7-A oracle 实现与审计
- [ ] H7-B implementation preregistration
- [ ] H7-C resource profiling and authorization
