# R004H7-BS4B4R1 scorer semantics audit

状态：`SOURCE_SEMANTICS_CORRECTED_BEFORE_FORMAL_RUN__NO_OUTCOMES_READ`。

## 1. 审计目的

确认理论中的 utility、runner 实际调用与 NAVSIM one-frame PDM scorer 使用同一个估计对象。本审计
只读官方源码、runner 与环境变量，没有运行 formal R1、没有读取 B4 正向结果，也没有访问
development/calibration/test/NavDream/target。

## 2. 权威调用链

冻结 runner `_score_candidate` 每次只传入一条 `model_trajectory`。官方
`pdm_score_from_interpolated_trajectory` 内部执行：

1. 从 metric cache 读取固定 \(\tau_{PDM}(S)\)；
2. 构造 `trajectory_states = [pdm_states, pred_states]`；
3. 对两条 proposal 运行 simulator/scorer；
4. 返回 `score_proposals(...)[pred_idx]`，其中 `pred_idx=1`；
5. 默认 `PROGRESS_MODE=eval` 时，progress normalization 使用这两个 proposal 的 masked progress
   最大值。

因此每条轨迹的 candidate-local surrogate 是

\[
F_S(\tau)=\operatorname{PDM}_{1f}([\tau_{PDM}(S),\tau])_{pred}.
\]

R1 的 native-relative 标签是

\[
\delta_{i0}^*=F_S(\tau_i)-F_S(\tau_0),
\]

其中两项来自两次独立 official 调用。它不是一次 `[native, challenger]` 联合评分。

## 3. 科学含义

- 好处：所有合法轨迹使用同一固定 PDM baseline 规则；score 是单轨迹函数，可形成可复算差分；
- 限制：每项 progress 都在自己的 `[PDM, trajectory]` 集合中归一化，可能压缩大于 baseline 的
  差异并增加 ties；
- 限制：候选各自诱导 reactive traffic，故该 surrogate 仍是 trajectory-conditioned outcome；
- 限制：它省略 two-frame extended comfort，不是完整 official PDMS；
- 结论边界：即使 R1 oracle 为正，也只证明该 source-fit surrogate 上存在学习机会，不证明完整
  official behavior gain、安全或跨 planner 可比的绝对 utility。

## 4. 对理论和实现的修正

1. 理论把最小 estimand 改写为 candidate-local score difference，不再声称 native 与 challenger
   联合决定 progress normalization；
2. runner 在评分前强制 `PROGRESS_MODE=eval` 并记录实际值；
3. audit 同时验证 primary/replay 的 estimand string 与 progress mode；
4. config 明确 internal reference 是 metric-cache PDM baseline；
5. B4R1 的候选、membership、阈值、bootstrap、统计单位和停止门全部不变。

## 5. 裁决

旧 `[native, challenger]` 叙述为 `FALSIFIED_BY_SOURCE_INSPECTION`。当前 R1 不因该叙述错误自动
失效，因为实现一直执行 candidate-local official path；但正式运行前必须完成上述 fail-closed 修正、
重新冻结 hashes 与 outcome-free preflight。完成前 R1 继续 HOLD。
