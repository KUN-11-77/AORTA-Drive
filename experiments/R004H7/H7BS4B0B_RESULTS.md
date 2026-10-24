# R004H7-BS4B0B：teacher-critical projective observability 结果

状态：`SUPPORTED_ON_FIT__PARENT_STOP_UNCHANGED`。

## 冻结结果

- 672/672 anchors、42/42 train fit recording groups 完整执行；
- 4,536/4,536 candidates 与 3,864/3,864 native/challenger pairs 被机械复算；
- 596/4,536 candidates（13.1393%）存在无直接 projective evidence 的 possible teacher
  minimizer；
- 39/42 recording groups（92.8571%）至少出现一个上述 candidate；
- 本次误差区间下每个 candidate 恰有一个 possible minimizer，因此 596 个 critical candidates
  对应 596 个 unsupported possible-minimizer nodes；
- D0 冻结的 8 个“非空但零支撑对称差”pair 中，1/8 的盲差异区域可控制 teacher set-min；
- 该 pair 位于 `2021.06.09.14.58.55_veh-35`，anchor 5，native index 0 对 candidate
  index 2；teacher contrast 为 -0.020322 m。该幅度很小，且没有读取 decision outcome，不能解释
  为驾驶伤害；
- primary runtime 96.659 s，audit replay runtime 75.330 s；均低于冻结 120 min 上限。

## 裁决

冻结的 projective-support interface 在 fit teacher 上 **关键不完备**。因此，H7BS4A 的
`STOP_OR_REDESIGN_VISUAL_SUPPORT_INTERFACE` 不仅是 all-node coverage 规则造成的工程假阳性；
未支撑区域确实可能包含候选 task operator 的极小值。D0 的差异盲区也至少有一个非平凡实例。

该结果支持把 missing evidence 显式纳入不确定集合或重新设计视觉支持接口；它不选择具体网络，
不证明扩大 ROI、continuous decoder、depth/occlusion 或任何 learned uncertainty 一定有效。
它也不证明 learned field、OOD/NavDream、候选排序、驾驶收益、安全或实时部署。

## 机械证据

- primary result SHA-256：
  `52b1861fbb3fd39b2a4d549a9d972f181206bfbc377fe930d702eaceaa5a876b`；
- replay result SHA-256：
  `26ad70e044b62e7898fa7577b0454a8db674d4888d3e3f6ea946f830b2210d54`；
- audit SHA-256：
  `3dce90014c9a48dba5727e9ddaf5c98b13facd991a3239441aab56f32e437afc`；
- 672 anchor ledgers exact、summary exact、failure count 0；
- forbidden payload access 为 false，parent decision preserved 为 true；
- external scientific reviewer：`REVIEW_UNAVAILABLE`。

## 下一步边界

`GO` 仅授予 B1 的 oracle uncertainty non-vacuity 设计与预注册；H7C 网络训练、development、
NavDream、正式 test 与驾驶收益继续 `HOLD`。B1 必须比较 global、independent 与 graph-correlated
集合在相同 coverage 下的 sharpness / abstention / ranking regret，不能用 B0B 结果调整样本或阈值。
