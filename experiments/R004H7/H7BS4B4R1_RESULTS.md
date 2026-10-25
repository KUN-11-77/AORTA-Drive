# R004H7-BS4B4R1：合法候选域 one-frame preference oracle 结果

状态：`GO_TO_MATCHED_STRUCTURE_PREREGISTRATION__NO_TRAINING_YET`。本轮支持合法 source-fit
候选域上存在非平凡的 candidate-local one-frame 排序机会；不授权训练，也不支持视觉鲁棒性、
完整 contextual utility、NavDream、驾驶收益或安全主张。

## 冻结裁决

- 42 个 fit recording groups × 16 anchors，共 672 anchors；
- 657 anchors 至少有一个合法 challenger，15 anchors 仅有 native；
- 3,864 个合法 challengers、4,536 个 scored candidates、13,402 个合法无序 pairs；
- 42/42 groups 至少包含一个 nonzero pair；
- 42/42 groups 至少包含一个严格优于 native 的合法 challenger；对应 292/672 anchors；
- group-equal mean oracle gain 为 `0.0202745`；42-group bootstrap 95% CI 为
  `[0.0117815, 0.0301722]`，下界严格大于 0；
- group-equal mean nonzero-pair fraction 为 `0.5304493`，95% CI 为
  `[0.4927011, 0.5679152]`；
- unique-top anchors 为 179/672；native hard-failure anchors 为 60，其中 hard rescue 为 8。

以上数值按预注册统计单位和分母报告。最后两项只作描述量，不升级为新的成功门。

## 完整性审计

- primary、完整 replay 与独立本地复算均 `PASS`，failure count 为 0；
- 两次完整 672-anchor ledgers exact；两份 summary 均由 ledger 机械重算 exact；
- candidate membership、pose reconstruction、scored-domain、sentinel replay、label estimand、
  `PROGRESS_MODE=eval`、provenance 和 root contract 全部 exact；
- 全部 JSON 数值有限，退出码为 0；
- development、calibration、test、NavDream、target 均未访问；
- external scientific reviewer backend 不可用，故科学复核保持 `PROVISIONAL_INTERNAL`，不能把
  机械审计写成独立同行评审。

## 科学解释

本轮回答的唯一问题是：在预 B4 冻结的合法 ragged candidate panel 上，official candidate-local
one-frame PDM scorer 是否产生可学习的非零 preference signal 和正 oracle ceiling。答案为
`SUPPORTED_ON_SOURCE_FIT`。

它没有回答：

1. one-frame margin 是否忠实于带 previous-action context 的 `U_ctx`；
2. RGB 是否包含恢复该 margin 所需的信息；
3. differential-support query 是否优于容量、FLOP、token 和训练预算匹配的通用 scorer；
4. 未见外观、三类 planner、clean no-harm、sequential rollout 或在线时延是否成立。

因此合法下一步是单独预注册 evaluator-scope audit 与 contextual target materialization；在该门
关闭前不训练网络。即使后续 contextual oracle 通过，也必须先做 visual recoverability 与 matched
structure study，不能从本轮 GO 跳到 ICCV 方法主张。

## 可追溯证据

- frozen code commit：`7050c6ef1c40f86d8ea97deca62a02c5d1d8a27c`；
- primary result SHA-256：
  `800e2c1b8f2cbf0db5d4614e53d8d5bd1b188ed5e8a7dd3433224f4caf2a4255`；
- replay result SHA-256：
  `51c23fa7e181154c72b9bcc93e63dbd54495683dfacb1ce898d84a7c6c40d692`；
- remote mechanical audit SHA-256：
  `8dc0518583b0834f3d9e6d7297d17dd35c89afd4639bbef97c5da197e4ffb9e5`；
- local independent recomputation SHA-256：
  `8dc0518583b0834f3d9e6d7297d17dd35c89afd4639bbef97c5da197e4ffb9e5`（repository LF bytes）；
- synchronized bundle：`experiments/results/R004H7BS4B4R1_7050C6E/`。
