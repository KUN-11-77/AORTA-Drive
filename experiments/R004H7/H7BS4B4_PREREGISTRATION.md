# R004H7-BS4B4：Source preference oracle 预注册

状态：`PREREGISTERED_NOT_RUN`。本实验不训练网络，也不选择 loss、阈值或架构。

## 1. 问题与停止原因

B3 已否定道路场 scalar minimum 作为一般候选 reranker。新的 appearance-quotient preference
只有在两个前提成立时才值得实现：source-only 数据上存在合法的候选效用标签，而且当前冻结面板
确实包含可由 oracle 改善 native 的机会。B4 只检验这两个前提。

## 2. 冻结 estimand

对同一 source-fit 场景的 native \(\tau_0\) 与候选 \(\tau_i\)，定义

\[
U_i = \operatorname{PDM}_{1f}(S;\tau_0,\tau_i),
\qquad
\Delta_{i0}^{*}=U_i-U_0.
\]

\(\operatorname{PDM}_{1f}\) 严格复现当前 NAVSIM evaluator 的单帧 PDM score：LQR/kinematic
bicycle 执行、候选条件 reactive traffic、四项 multiplicative metrics、progress/TTC/lane-keeping/
history-comfort 以及 human penalty filter。它**不是**完整 official PDMS，因为不包含跨相邻 anchor
聚合的 two-frame extended comfort。

每个候选必须独立放入 `[native, candidate]` 二元评分面板。禁止将 7 个候选一次性送入 scorer，
因为 evaluator 的 progress normalization 取决于当前评分面板；那会改变 estimand 并制造候选集合
依赖的标签。

## 3. 数据与独立单位

- 只读 42 个 fit recording groups × 16 anchors，共 672 个 source-train anchors；
- 每 anchor 固定 native + 6 个 offset challengers；
- recording group 是唯一统计独立单位；anchor、candidate、pair 均为组内重复观测；
- development、calibration、test、NavDream、target 和任何其他 candidate provider 全部封存。

## 4. 实现资格门

1. 672 个 metric cache、panel 与 role identity 全部匹配冻结 SHA；
2. 从相同 native 重建 provider，并经官方 simulator 得到与冻结 panel bit-exact 的 realized poses；
3. 所有 4,704 个候选 outcome 有限且位于合法范围；
4. 每个组的 anchor 0 作为事前冻结 sentinel，独立重建 scorer/policy 后 candidate outcome 与 realized
   state bit-exact；
5. 主运行之后必须再完成一次完整独立 materialization；两次 672-anchor ledger 必须 exact；
6. 任一实现门失败即 `INVALID_PREFERENCE_LABEL_INTERFACE`，禁止解释科学数值。

## 5. 科学量与预注册裁决

主量为：

\[
G_g=\frac1{16}\sum_{a=1}^{16}
\left(\max_i U_{gai}-U_{ga0}\right).
\]

以 42 个 \(G_g\) 做 group bootstrap。只有同时满足以下条件才允许进入下一份容量匹配结构
实验的**预注册**，仍不授权训练：

1. 42/42 groups 至少含一个非零 candidate pair；
2. 至少 21/42 groups 含一个严格优于 native 的候选；
3. group-equal mean oracle gain 的 95% bootstrap 下界严格大于零；
4. 所有实现资格门通过。

`0, .001, .005, .01, .02, .05` 的绝对差异率全部只作预冻结敏感性报告，不能在看结果后选择
最终 intervention margin。最终最小有意义效应仍须由独立基线方差、统计功效和任务代价冻结。

同时报告 native multiplicative hard failure 及存在 challenger 解除全部 hard failures 的 anchor 数量。
它对应项目长期使命，但本轮样本量未知，因此不把它事后变成通过门槛。

## 6. 三种裁决

- 全部实现与信号门通过：
  `GO_TO_MATCHED_STRUCTURE_PREREGISTRATION__NO_TRAINING_YET`；
- 零 group 存在严格 native 改善：`STOP_CURRENT_PANEL_NO_ORACLE_IMPROVEMENT`；
- 其他信号不足：`HOLD_CURRENT_PANEL_PREFERENCE_SIGNAL_INSUFFICIENT`。

任何结果都不得改写 B3 的 STOP，也不得支持 learned model、跨 planner、NavDream、完整 PDMS、
驾驶安全、网络必要性或 ICCV readiness。

## 7. 本轮理论价值

B4 将“候选偏好”从一个好听的网络名称约束成明确随机对象：同一物理场景、同一 native、相同
官方执行与 reactive traffic 下的 candidate-specific task utility difference。若这个 oracle ceiling
不存在，任何 attention、ranking loss 或 Hodge 结构都没有继续实现的资格。
