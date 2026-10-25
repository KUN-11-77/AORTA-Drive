# AORTA 投稿版理论细化报告

## 范围与完整性

已核对提交 `e8b2fbf`、`5df0e08`、`0d5931b`、`2effe1c`、`d0c746e`、`ff2212f`、
`34d7084`，均存在于当前分支并按该顺序进入 HEAD。已阅读研究宪法、Research Brief、
Claim–Evidence Matrix、Method Spec、formal theory、architecture closure、ICCV story、official
target/context 文档、frontier novelty audit、最新实验 plan/tracker、B4R1 preregistration 与
readiness audit。

未读取或未执行：受保护 calibration/test/NavDream/target payload、B4R1 formal outcome、新模型训练。
本轮没有产生经验结果。

## Round 1：最强拒稿意见

1. 论文把条件数学的完整性误当成算法创新；D-JEPA/World4Scorer 已覆盖 relation 与 fallback。
2. `7/8+1/8` 可能只属于局部 two-frame scorer，而非完整 EPDMS。
3. antisymmetric pair head 仍可产生循环偏好，与真实标量效用不一致。
4. differential support 对非加性任务并不充分，可能只是手工 feature。
5. previous selected trajectory 是策略干预后的内生状态，teacher forcing 会产生 rollout shift。
6. B4R1 one-frame 无信号不能推出 contextual target 无信号。
7. 没有 matched、NavDream、行为或 latency 结果，当前不具投稿资格。

## Round 1 作者响应

- 接受 1：主创新只保留 differential evidence 的 matched 增量。
- 接受 2：训练对象改称 scoped contextual utility \(U^{ctx}\)，完整 EPDMS 独立评估。
- 接受 3：输出改为 native-star potential，其他 pair 由差分得到。
- 接受 4：signed support 只负责 query，全局视觉与 common/history context 保留。
- 接受 5：增加 native-history、actual-selected-history、sequential 三臂。
- 接受 6：B4R1 interface valid 才是 contextual materialization 前提，科学 outcome 只管 one-frame。
- 接受 7：状态保持 NOT_ICCV_READY。

## Round 2：复审

### 改善

- 研究问题、estimand、网络、loss、仲裁与实验使用同一 scoped relation；
- 多候选输出合法性与历史递归状态被显式处理；
- 架构候选 A/B/C/D 和单一 novelty falsifier 清楚；
- 复杂模块被限制为失败触发，而不是预先堆叠。

### 仍然阻断

- 无法通过纯理论证明 C 路线优于 matched B 路线；
- counterfactual history 是否能由官方 evaluator 合法 materialize 尚未确认；
- query discretization 与真实时延仍无测量；
- source orbit 对 NavDream 的迁移完全是经验假设。

复审结论：理论设计值得做冻结验证，但任何 ICCV-ready 判断都必须等待实验。

## Drift / Pushback Log

| 建议 | 处理 | 理由 |
|---|---|---|
| 加 depth/world model/RL | 拒绝 | 没有独立失败授权，会造成贡献发散 |
| 用 Hodge/cycle loss 增厚理论 | 拒绝 | native-star 可积性代数上已闭合，star exactness 不新 |
| 直接扩大网络 | 拒绝 | 无法解决表示不可识别或 history distribution shift |
| 把 conformal 写成安全贡献 | 拒绝 | source exchangeability 不给 target OOD safety |
| 只做弱 field baseline | 拒绝 | D-JEPA/World4Scorer-style 是决定性最近对照 |

## 最终方法快照

1. shared frozen visual encoder；
2. full-horizon signed differential support queries；
3. global visual KV 与 common context；
4. actual-history-conditioned, integrable native-star margin；
5. task value + structured decision risk，appearance orbit 作为约束；
6. independent group calibration 与 positive lower-bound arbitration；
7. sequential history update 与 batch-1 profiling。

## 最终裁决

- THEORY STATUS：条件闭合，经验前提未证。
- NOVELTY STATUS：窄而可检验，未成立。
- ARCHITECTURE STATUS：由问题自然导出，仍需 matched falsifier。
- RESEARCH READINESS：可进入冻结 scope/oracle/matched 预注册；正式 B4R1 仍 HOLD。
- PUBLICATION POTENTIAL：全部关键证据成立时为 ICCV/CVPR poster 强候选；当前不是投稿稿。
