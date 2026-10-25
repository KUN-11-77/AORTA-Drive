# R005E2 Experiment Plan

**Problem**：正确场景视觉是否为`U_ctx`关系提供轨迹与历史之外的可用信息？  
**Method thesis**：先证明视觉条件增量存在，再检验differential-support是否优于普通full-context scorer。  
**Date**：2026-10-25。

## Claim Map

| Claim | Minimum convincing evidence | Linked block |
|---|---|---|
| C1：视觉条件增量可恢复 | group cross-fit中correct同时胜过参数匹配removed与geometry-matched mismatch，proper risk和decision regret均通过 | B1–B3 |
| C2：该增量与受控外观稳定性可分开判断 | clean先裁决；冻结checkpoint后报告三种paired photometric stress，不用稳定性掩盖任务错误 | B4 |
| Anti-claim：只是更大网络/更多参数 | V0/V1/V2逐层同构、参数精确相同、FLOPs≤1.01×、同fold/seed/优化预算 | B2 |

## Paper Storyline

- 主文必须证明：correct-scene visual evidence确实降低contextual decision relation风险；随后B6相对B4的增量才有资格成为主创新。
- 附录支持：mismatch距离账本、worst groups、context-flip/near-boundary/hard-failure分层和受控外观压力。
- 本阶段主动切除：Differential Query、orbit loss、uncertainty、calibration、NavDream和planner行为。

## Experiment Blocks

### B1：数据与目标资格

- Claim：输入、候选域、history与`U_ctx`可被无泄漏join。
- Setup：42 groups、672 anchors、6-fold group cross-fit；生成fold/donor/provenance manifests。
- Success：全部hash、membership、history、label、fold和donor合同通过；scientific metrics尚未产生。
- Donor control：native几何描述子上求解跨组全局线性和分配；inner/outer训练映射均为visual donor
  一一置换，validation/test donor只来自对应训练池且不重复。该设计保持训练视觉边际严格不变，
  只破坏scene correspondence；匹配距离仍完整报告，不能假设残余分布偏移为零。
- Failure：`INVALID_EXPERIMENT`，不训练。
- Priority：MUST-RUN。

### B2：视觉条件增量

- Compared systems：V0 removed/trajectory-history、V1 mismatch-trained、V2 correct-trained。
- Metrics：proper relation risk、decision regret、sign/boundary error、oracle-normalized recovery。
- Success：V2对V0/V1的risk与regret paired group CI上界均<0，且`rho_vis>=0.10`。
- Failure：停止或降级视觉主张；不增加AORTA模块。
- Table：第一张机制资格表。
- Priority：MUST-RUN。

### B3：场景特异性与捷径排除

- Compared systems：同一V2 checkpoint上的correct、mismatch intervention、removed intervention。
- Success：两种干预的regret-minus-correct CI下界均>0。
- Failure：`SCENE_SPECIFICITY_UNRESOLVED`，禁止进入架构创新实验。
- Figure：group-level paired slope plot + worst-group panels。
- Priority：MUST-RUN。

### B4：受控外观压力

- Conditions：clean、low_exposure、low_contrast、cool_cast；checkpoint冻结后重复测量。
- Metrics：task risk、regret、relation drift、appearance-induced top flip、DINO feature shift。
- Success：只产生`controlled-stable`或`appearance-brittle`附加分类，不覆盖B2/B3裁决。
- Figure：accuracy–drift双轴图，明确“稳定不等于正确”。
- Priority：MUST-RUN，但在clean结果冻结后执行。

## Run Order

| Milestone | Goal | Runs | Gate | Cost / Risk |
|---|---|---|---|---|
| E2A | outcome-free preflight | folds、donors、hash、synthetic tests | PASS才可训练 | CPU；风险是cache/provenance不闭合 |
| E2B-S | single-fold resource rehearsal | 3 arms×1 seed×1 fold，不读outer scientific table | shape/runtime/memory | 低GPU；只校资源，不改scientific config |
| E2B | 6-fold clean recoverability | 3 arms×3 seeds×6 folds，含inner early-stop/refit | GO/MIXED/STOP | cached DINO小head；最大风险是训练量 |
| E2C | frozen controlled appearances | V2 OOF checkpoints×3 transforms | stable/brittle | 只推理，禁止调参 |
| E2D | mechanical + independent audit | full replay、recompute、claim audit | PASS/WARN/FAIL | 不允许第三次科学运行覆盖失败 |

## Compute and Data Budget

- Backbone必须共享并优先复用经hash验证的DINO cache；禁止每candidate重复编码。
- 预计正式head训练：54个outer final fits，另有同数量inner epoch-selection fits；正式GPU预算在E2A资源测量后只允许上调wall cap，不改模型或统计。
- development/calibration/test/NavDream/target继续封存。
- 最大瓶颈：完整视觉token I/O与三arm×三seed cross-fit，不是scorer参数量。

## Risks and Mitigations

- 伪重复：只在42个group上推断，anchor/candidate/appearance不作独立样本。
- mismatch额外OOD：同时使用公平重训V1、removed V0和V2推理干预，并记录feature shift。
- 共享实现错误：独立读取ledger复算loss、regret、fold和donor；E2新增runtime import provenance。
- old source-panel exposure：主张限定为cross-fitted source-panel，不称最终未触碰泛化。
- correct胜出仅因容量：三主arm逐层同构、参数相同、FLOPs/token匹配。

## Final Checklist

- [x] 主张数不超过2项
- [x] 独立单位与重复测量层级明确
- [x] 视觉必要性与架构创新分离
- [x] 最强竞争解释有参数匹配控制
- [x] GO/MIXED/STOP/INVALID冻结
- [ ] E2A manifests与synthetic tests完成（cross-fit manifest已PASS；donor/provenance/model gates待完成）
- [ ] 正式训练获得单独授权

## Procedural reference

本计划的claim-driven实验块、独立单位、重复测量与阻断设计参考：Timothy Kassis, Vinayak Agarwal,
Yuhuan He, Darshil Patel, and Aubrey M. Brueckner. *Scientific Agent Skills: A Library of Procedural
Knowledge for Research Agents*. arXiv:2609.00065v2, 2026. DOI: https://doi.org/10.48550/arXiv.2609.00065。
该文只作为研究流程来源，不是AORTA方法或理论依据。
