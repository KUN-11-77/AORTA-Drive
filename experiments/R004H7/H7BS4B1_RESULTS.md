# R004H7-BS4B1：oracle joint uncertainty non-vacuity 结果

状态：`UNSUPPORTED_FOR_PREREGISTERED_NONVACUITY__REDESIGN_TASK_INTERFACE`。

## 冻结结果

- 672/672 anchors、42/42 train fit recording groups、3,864/3,864 native/challenger
  pairs 完整执行；
- teacher inclusion、contrast containment、joint→box nesting、box→candidate-independent nesting
  failure 均为 0；三种 oracle 均未发生 harmful intervention；
- 最大可行势数值放宽为 `1.04361e-14 m`，低于冻结上限 `1e-9 m`；
- joint Lipschitz、shared-node box 和 candidate-independent 三种集合均识别出 1,301/3,864
  个有符号 pair；joint 相对 shared box 的 identifiability 差异为 0，42 组 bootstrap 95% CI
  为 `[0, 0]`；
- 只有 5/42 groups 达到“至少 12/16 anchors 含 joint-identifiable pair”，冻结要求为 42/42；
- 因此预注册裁决为 `ORACLE_SUPPORT_SET_NEAR_VACUOUS_REDESIGN_INTERFACE`。

## 次级 sharpness 结果

joint 结构并非数值上完全无效：

| 集合 | 全部 pair 平均 interval width |
|---|---:|
| joint Lipschitz shared field | 0.089451 m |
| shared-node marginal box | 0.132051 m |
| candidate-independent intervals | 2.172811 m |

joint 在 153/3,864 pairs 上比 shared box 严格更窄；shared box 在 2,563/3,864 pairs 上比
candidate-independent 更窄。但这些宽度改善没有使任何额外 pair 穿过冻结的严格零点可识别门，
所以不能据此保留“graph correlation 提高决策可识别性”的主贡献。

## 结果后描述性归因线索

下列数字不是预注册 primary endpoint，只用于登记下一项机制诊断的动机：

- 2,219/3,864 teacher contrasts 精确为 0；
- 2,388/3,864 的绝对 contrast 小于 0.01 m；
- 2,892/3,864 的绝对 contrast 小于 0.10 m。

这表明当前瓶颈至少部分位于 `candidate panel × global set-min road-margin operator`，而不只是
oracle 区间过宽。下一步必须区分轨迹/网格重复、共同前缀极小值、离散化 tie 与道路任务本身
对候选等价；不得直接增加 uncertainty head 或以新的阈值重跑 B1。

## 裁决

1. B1 的预注册非空洞性假设 **不支持**；
2. 当前 support-conditioned graph-correlated uncertainty 不具备进入 H7C learned-network 训练的资格；
3. H7BS4A 的 `STOP_OR_REDESIGN_VISUAL_SUPPORT_INTERFACE` 保持不变；
4. H7C、development、NavDream、test 和驾驶收益继续 `HOLD`；
5. `GO` 只授予 fit-only 的零差异结构归因与 task-operator 重新设计，不授予新模型训练。

## 机械证据

- primary runtime：1,412.852 s；audit replay runtime：1,418.347 s；均低于 120 min 上限；
- primary result SHA-256：
  `9a064fd19c452bea0ba1f11ef58ede57fc03a24f14763f51ab7be500c128f277`；
- replay result SHA-256：
  `51b29ba19c2e7711b89cb8eb33b22e2e91d1b449d2b38107d8d6e8769983a75d`；
- audit SHA-256：
  `2fc861d4258786c6b40b2864660a965d2bdb24b5e45805e1075417b1d46b99d6`；
- 672 anchor ledgers exact、summary exact、root contract exact、failure count 0；
- external scientific reviewer：`REVIEW_UNAVAILABLE`。

## 主张边界

本结果只说明：在冻结 fit panel、teacher intervals、projective support 与 SDF 1-Lipschitz oracle
条件下，当前 global set-min 接口没有获得预注册的 joint-identifiability 增益和组级覆盖。它不否定
所有联合不确定性方法，也不证明新的 branch-aware operator 有效；更不涉及 learned field、OOD、
NavDream、驾驶、安全、实时性或顶会资格。
