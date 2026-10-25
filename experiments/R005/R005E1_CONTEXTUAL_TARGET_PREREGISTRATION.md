# R005E1：上下文效用与固定后续策略价值预注册

状态：`PREREGISTERED_DESIGN_ONLY__SOURCE_BINDING_PENDING__NOT_AUTHORIZED`。

日期：2026-10-25。

## 1. 唯一目标

在不读取 RGB、不训练网络、不打开 development/calibration/test/NavDream/target 的前提下，判断冻结
source-fit 合法候选 panel 是否存在定义正确、可独立复算且非平凡的 contextual decision target。

本实验不把 `U_ctx` 称为完整 official target；只有 candidate-specific two-stage counterfactual 资格
闭合后，才报告固定 continuation 的 `Q_pi_ref`。

## 2. 冻结对象

- 42 个 source-fit recording groups、672 anchors 与 B4R1 冻结 ragged candidate membership；
- native 为 index 0，15 个 zero-challenger anchors 完整保留；
- exact NAVSIM evaluator root、commit/tree hash、所有相关 source hashes；
- 过滤顺序、metric weights、tie tolerance、seed、group bootstrap 与异常停止规则；
- previous token 由 raw-log immediate predecessor 唯一解析，时间差必须满足 `0 < gap < 0.55 s`；
- primary offline history 使用冻结 native planner 的上一周期输出。它只支持 native-history
  teacher-forced estimand；actual-selected history 和 sequential rollout 留给后续独立实验。

正式执行前必须生成 frozen config 与 input manifest；本文件不构成运行授权。

## 3. 分阶段门

### E1A：来源与可定义性资格

只读、outcome-free：

1. pin 单一 evaluator source root，记录 `pdm_score.py`、`pdm_scorer.py`、`run_pdm_score.py`、
   `scene_aggregator.py`、comfort 实现及 config hashes；
2. 证明 672 anchors 的 previous token 唯一、同 log、因果在先且 gap 合法；
3. 验证每条候选可在相同 baseline proposal 与交通策略下合法评分；
4. 审计 filter 后 components、multiplicative product 与 official recomputation 的字段一致性；
5. 审计 two-stage mapping 是否覆盖当前候选反事实。若 stage-two continuation、traffic reaction 或
   mapping 对候选替换无定义，则 `Q_pi_ref` 分支 fail-closed，只运行 `U_ctx`，不得用近似填充。

任一 source/hash/history/candidate gate 失败，停止科学 materialization。

### E1B：`U_ctx` primary、replay 与机械审计

每条合法候选独立生成：

\[
\bigl(F^{1f}_{raw},\widetilde F^{1f},EC(h,\tau_i),U^{ctx}_i(h),M_i,EP_i,TTC_i,LK_i,HC_i\bigr).
\]

必须保存逐候选 components，而不是只保存最终标量。完整 primary 后，从相同 frozen inputs 独立 replay；
机械审计重新读取 ledger 计算全部分数、pairs、group summaries 与裁决。

### E1C：条件性的 `Q_pi_ref` 分支

仅在 E1A 证明反事实 mapping 合法时执行。冻结 \(\pi_{ref}\)、second-stage rows、Gaussian weights、
underflow fallback 与 previous stream，逐候选计算 `Q_pi_ref`。若只能建立当前 stream 而不能建立完整
mapping，结果只能称 current-stream diagnostic，不能称 pseudo-closed-loop action value。

## 4. 主指标与统计单位

recording group 是唯一独立单位；anchor/candidate/pair 是组内重复。报告：

1. `U_ctx` group-equal oracle gain、42-group paired bootstrap 95% CI、strict-improvement groups；
2. 零/非零 pair fraction、unique-top、zero-challenger anchors 与所有分母；
3. `D_raw_ctx = P(argmax F_raw != argmax U_ctx)`；
4. `D_filtered_ctx = P(argmax F_tilde != argmax U_ctx)`；
5. pair-sign disagreement 与忽略 context 的 regret
   `max_i U_ctx_i - U_ctx_argmax(F_tilde)`；
6. 若 E1C 合法，同样报告 `U_ctx` 对 `Q_pi_ref` 的 top/pair disagreement 与 fixed-continuation regret；
7. 最差 group、失败 group、数值非有限、filter-trigger 与 Gaussian-underflow strata。

不得把 672 anchors 当作独立样本计算置信区间。不得删除 tie、zero-challenger、filter-trigger 或
context-sensitive anchors。

## 5. 预注册裁决

### GO

仅当：

- target fidelity、history validity、candidate validity、primary/replay exactness 和独立机械复算均 PASS；
- 42/42 groups 有可定义 ledger，所有数值有限；
- `U_ctx` 至少 21/42 groups 有 strict native improvement；
- group-equal mean oracle gain 的 95% bootstrap CI 下界严格大于 0；
- 非零 relation 不由少数组或 complete-case 删除产生。

`GO` 只解锁 correct/mismatch/removed 的视觉可恢复性预注册和 matched scorer 预算冻结，不自动训练。

### HOLD

实现完整但 oracle CI 包含 0、history effect 极弱、或只在少数组有信号：保留结果，审查 candidate
supply/estimand；不得添加网络复杂度挽救。

### STOP / INVALID

- source、history、candidate 或 replay 不闭合：`INVALID_TARGET_CONSTRUCTION`，禁止科学解释；
- 合法 `U_ctx` 无非平凡 oracle opportunity：`STOP_CURRENT_PANEL_SUPERVISION`；
- `Q_pi_ref` 不可合法定义：只停止完整两阶段 action-value 主张，不否定 `U_ctx` 子问题。

## 6. 禁止事项

- 不用 B4R1 raw score代替 filter 后分数；
- 不把 native-history 写成部署 actual-selected history；
- 不把 `U_ctx` 写成完整 pseudo-closed-loop 或驾驶安全；
- 不查看 partial scientific metrics 后调整映射、门槛、候选或 continuation；
- 不自动启动第三次运行，不访问任何封存 split，不训练视觉或 scorer 网络。

