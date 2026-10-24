# R004H7-BS4B4：Source preference oracle 结果

状态：`INVALID_PREFERENCE_LABEL_INTERFACE`；后续 learned preference、matched-structure 训练与
development/OOD 评估继续 `HOLD`。

## 冻结裁决

- primary 与完整 replay 均完成 42 个 fit recording groups × 16 anchors；两份 672-anchor
  ledger、summary 与 root contract 逐字段 exact；机械审计 `PASS`；
- primary 实际运行 672 anchors、4,704 registered candidates、14,112 unordered candidate pairs；
- `label_generation_valid=true`、42 个 sentinel replay exact，且 development、calibration、test、
  NavDream、target 均未访问；
- 但是 `panel_reconstruction_bit_exact=false`，故 `implementation_pass=false`。按预注册规则，
  本轮唯一合法科学裁决是 `INVALID_PREFERENCE_LABEL_INTERFACE`。

机械复现成功不等于科学接口正确。以下原始诊断值——42/42 groups 有非零 pair、42/42 groups
有严格 native improvement、group-equal oracle gain `0.021231`、bootstrap 95% CI
`[0.012739, 0.031082]`——全部来自已判无效的候选集合，**不得**作为正向 oracle ceiling、方法、
网络或论文证据。

## 失败归因

对全部 48 个失败锚点执行了只读 failure-attribution 复算；它不构成第三次正式实验，也不更改
原裁决：

- 48/672 锚点、27/42 groups 的冻结 `challenger_eligible` 并非全真；
- 其中 15 个锚点的 6 个 challenger 全部不合格，33 个锚点部分不合格；共 168/4,032 个
  challengers 不合格；
- 48/48 失败锚点重新生成的 `realized_local_poses` 与冻结 panel 仍逐 bit exact，最大绝对误差
  `0.0`；
- runner 将两个不同命题合并为

  ```text
  pose_reconstruction_exact AND all(challenger_eligible)
  ```

  并把该合取命名为 `panel_reconstruction_bit_exact`；与此同时，oracle summary 又纳入了 168 个
  未通过冻结 intent-match 的 challenger。

因此本轮不是浮点不确定性、缓存错位或官方 simulator 非确定性。真实失败是**候选域定义与统计域
不一致**：预注册口头上研究“当前冻结 panel”，实现却对固定 6-offset 全集合评分并汇总，同时用
冻结 eligibility 作为全局实现门。无论把它解释为额外资格门还是 panel membership，两种解释都
不能使已汇总的 oracle 数值合法。

## 修复边界

不得把本轮改写成 PASS，也不得仅删除 `np.all(frozen_eligible)` 后重新发布原数值。若继续，必须
新建 `R004H7BS4B4R1` 预注册，并在看任何新科学结果前明确二选一：

1. **frozen eligible panel estimand（建议）：** 每 anchor 的候选域为 native 加冻结 eligible
   challengers；允许可变 panel size；零 eligible 的 anchor 只保留 native、没有 pair；统计分母与
   oracle gain 必须预先定义；
2. **all-offset provider estimand：** 明确废除 intent-match 作为 panel membership，并证明为何
   task-intent 不匹配候选仍是合法干预；它将是新 estimand，不能冒充旧 panel。

两者不得用本轮正向数值选择。R1 仍只允许 source-fit label/oracle 资格，不授权网络训练、完整
official PDMS、NavDream、驾驶收益或安全主张。

## 可追溯证据

- frozen code commit：`599a2a0`；
- primary runtime：`2490.810776 s`；
- primary result SHA-256：
  `725d485d1a6bd7a76bef64b0d0afcd4133a3aa30ce68644275e92ab81fa89086`；
- replay result SHA-256：
  `e688541e0bf403954344f623d20abe47d169e38dbabd02ef0c4b55609e294d5a1`；
- mechanical audit SHA-256：
  `ddb6ba108082cd1c42cd79988d6372ba7bc71d7fbd8a15202156fbb6ed78b149`；
- failure attribution SHA-256：
  `84c09eae44d0ac1321313bb42a6a2afaf84994ad820e4a771108a6328f429896`；
- local bundle：`F:\AORTA-Drive-artifacts\gate-b\R004H7\r004h7bs4b4_599a2a0`；
- bundle SHA-256：
  `34c0467f1facd031f4a95194bd9dcf0db80c9f301f179d0903e145f970f580f0`；
- external scientific reviewer：`REVIEW_UNAVAILABLE`。

## 论文与故事边界

本轮最重要的不是一个“看起来为正”的 oracle 数字，而是暴露出候选空间本身必须成为数学对象：
偏好函数、图边、Laplacian 与 regret theorem 都只能定义在同一个合法候选域上。顶会故事若不能
清楚回答“哪些候选属于任务、为何属于、缺失候选如何处理”，后续再漂亮的 quotient/Hodge loss
也只是对错误对象做精确优化。
