# R004H7-BS4B3：Fork-aware task operator oracle 资格结果

状态：`STOP_ROAD_MARGIN_AS_GENERAL_RERANKER`。

## 冻结主结果

- 672/672 anchors、42/42 fit recording groups、3,864/3,864 pairs 完整执行；
- B2 的 2,033 个 `common_minimum_domination` pairs 精确复现；
- temporal first-fork factorization 对 2,033/2,033 pairs 均定义有效，但只恢复 170 个 exact-nonzero
  branch contrasts：pair-pooled 8.3620%，group-equal 8.2612%，95% group bootstrap CI
  `[6.3406%, 10.2893%]`；37/42 groups 至少出现一个恢复 pair；
- 固定 2.5 s suffix 只恢复 464/2,033（22.8234%）；
- spatial-exclusive factorization 在 1,568/2,033 pairs 上有效，且这 1,568 个全部恢复非零差异；
  但 465 pairs 至少一侧独占集合为空，故总体有效率仅 77.1274%，低于预注册 95% 门；
- spatial 的 all-pair group-equal recovery 为 77.8218%，95% CI `[74.7375%,80.9727%]`，但它是
  sensitivity 且适用域失败，不能替代失败的 temporal primary；
- teacher 绝对符号仅作 oracle 描述：2,033 个 temporal shared trunks 全部为正，teacher road
  rescue opportunity 与 induction risk 均为 0。它们不是 quotient 模型可识别的部署安全量。

Temporal primary 的 bootstrap 下界远低于严格多数 50% 门；spatial sensitivity 又未达到 95%
适用率。按看结果前冻结的裁决，结论是
`STOP_ROAD_MARGIN_AS_GENERAL_RERANKER`，而不是继续选择更有利的空间定义。

## 科学解释

删除共同**时间前缀**并没有解决大多数零差异：1,863/2,033 pairs 在完整 suffix minimum 下仍为
exact zero。固定时间切点也只恢复少数差异。说明 scalar minimum 的信息压缩问题并不只发生在
早期 trunk；分叉后共享/重访空间与最低点竞争仍可湮灭候选差异。

空间独占集合在定义有效时能完全移除该湮灭，但代价是 465 个 nested-support pairs 没有双侧独占
分支。若只保留有效子集，会把一般 reranker 偷换成 selective subset evaluator，并引入明显的选择
偏差。因此该结果不能用于事后采用 spatial operator。

更重要的是，本 oracle 没有发现任何由 teacher road sign 区分的 rescue/induction pair。即使某些
独占分支具有不同 clearance，它是否对应 planner ranking 或驾驶效用仍完全未知。

## 方法裁决

1. global、fixed-suffix 和 temporal-suffix scalar minima 不再具有 AORTA 一般候选 reranker 的主线资格；
2. spatial-exclusive minimum 仅保留为 selective diagnostic / potential veto 研究对象，不是获批方法；
3. 不启动 fork-aware network、uncertainty head、NavDream 或 H7C 训练；
4. 下一步必须先重新定义决策充分的数学对象，并用文献与 outcome-blind/合法 outcome 协议审查其
   估计目标。任何新对象都需解释 common-mass cancellation、nested support、gauge invariance、
   部分可观测性和实时计算，不能只更换聚合器名称。

## 机械证据

- primary runtime：319.949 s；完整 replay runtime 见 replay artifact；
- primary result SHA-256：
  `dfcea0df1c6f90ce5bff4cb02a54bc9d4208dec6f47e01f70e624ae91ff215e1`；
- replay result SHA-256：
  `a33d0d55d125b63fe51f2ce7dc0b205f443b047cf84a8a4afd661d0369653669`；
- audit SHA-256：
  `e59e0a77fc48a46a2b3e5cffd567c1642fc3ad41057b96317b08ed1b7394400d`；
- bundle SHA-256：
  `e2a380436cbd1542569f584ea6fbc9e13df4cdb5e46ab9b509c2374c9d8c3200`；
- 672 anchor ledgers、summary 与 root contract exact；failure count 0；
- external scientific reviewer：`REVIEW_UNAVAILABLE`。

## 主张边界

这是同一 fit panel 上的 oracle operator 资格失败，不是视觉网络失败，也不是对所有道路表示、
所有风险函数或所有 candidate-conditioned 方法的否定。它不提供 independent generalization、
NavDream、decision、driving、safety 或 top-venue 证据。
