# R006V0F Experiment Audit Report

**Date:** 2026-10-27

**Auditor:** internal mechanical audit; external cross-model reviewer unavailable

**Review status:** `PROVISIONAL_INTERNAL_REVIEW`

**Overall verdict:** `WARN`

**Mechanical integrity:** `PASS`

外部reviewer backend在本会话不可用，因此本报告不得表述为独立跨模型科学复核。WARN来自review
independence不可用，而不是检测到机械篡改。

## A. Ground-truth provenance — PASS WITH SCOPE LIMIT

R006V0F不是benchmark性能评估，而是父STOP后的source-inner estimator诊断。预注册明确封存
outer-confirmation、development、calibration、test、NavDream与target，并保留28/7/7角色隔离
（`experiments/R006/R006V0F_FAILURE_DECOMPOSITION_PREREGISTRATION.md:16-24`）。runner初始化封存标志为
false（`scripts/run_r006v0f_chain.py:55-62`），聚合审计再次核验未访问
（`scripts/audit_r006v0f_chain.py:76-80`；`aggregate_audit.json` check 5）。

分类：`self_supervised_proxy / internal-mechanism diagnostic`。它不使用模型输出伪造外部ground truth，
但也不产生benchmark或安全ground-truth结论。

## B. Score normalization — PASS

coverage阈值与epsilon在运行前固定，runner拒绝被修改的`0.90/0.50/1e-12`配置
（`scripts/run_r006v0f_fold.py:150-155`）。聚合审计验证同fold全部rank/seed使用同一冻结分母
（`scripts/audit_r006v0f_chain.py:194-199`），没有按模型自身最好结果重新归一化。近零分母仅作为描述性
健康账本，不改变coverage公式或裁决。

## C. Result existence and fidelity — PASS

- `exit_code.txt=0`，`progress.json.status=COMPLETE`；
- 6个primary、6个replay与6个逐折`audit.json`存在；
- 每折643项审计、0失败、primary/replay exact；
- 聚合审计83项、0失败，最终路线为`RANK_ESTIMATION_UNSTABLE`；
- 本地`aggregate_audit.json` SHA-256与服务器完成时核验值一致。

## D. Dead-code / execution-path check — PASS FOR REGISTERED PATH

正式chain按固定fold/rank/seed顺序执行primary、same-GPU replay与逐折audit
（`scripts/run_r006v0f_chain.py:55,112-146`）。最终聚合审计实际调用
`route_failure_decomposition`（`scripts/audit_r006v0f_chain.py:208-233`），且结果中保存路线、seed最小rank与
qualification flip。有限值、正交性与basis exact检查由
`src/aorta_drive/vision/r006_failure_diagnostic.py:32-58,273-330`执行。

## E. Scope assessment — WARN

范围是42个source-fit groups的6-fold inner诊断、4个rank、8个初始化；seed和fold pair只是blocking/repeated
diagnostic factors，不能扩大科学样本量。没有outer-confirmation，也没有group-resample projector refit。
因此结果足以否定当前estimator作为稳定接口的资格，但不足以推断真实共享方向、介导、DINO、OOD、行为
或安全。`projector_geometry_inference=UNAVAILABLE_NO_GROUP_LEVEL_REFITS`且`method_unlock=false`。

## F. Claim impact

- 当前fixed-global-rank接口具备稳定资格：**unsupported**；
- 当前失败主要属于冻结estimator的初始化不稳定：**supported operationally**；
- 全局共享方向不存在：**unresolved**；
- Fantope凸诊断可被预注册：**authorized as diagnosis only**；
- adaptive reader、DINO reader、CPTR、NavDream、行为或安全收益：**not unlocked**；
- 父`STOP_GLOBAL_SHARED_INTERFACE`：**immutable**。

## Action items

1. 如继续，只运行一次预注册Fantope/group-refit estimator诊断；
2. 保存每个recording-group resample的重新拟合结果，禁止对相关projector pair做伪bootstrap；
3. 报告软解特征谱、Stiefel gap、top-r舍入gap与跨resample稳定性；
4. 外部reviewer可用时重新执行一次只读完整性复核，在此之前保持`PROVISIONAL_INTERNAL_REVIEW`。
