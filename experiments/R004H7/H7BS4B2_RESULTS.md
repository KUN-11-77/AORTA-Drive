# R004H7-BS4B2：global set-min 零差异结构归因结果

状态：`COMMON_MIN_DOMINANCE_ELIGIBLE_FOR_FORK_OPERATOR_DESIGN`。

## 冻结结果

- 672/672 anchors、42/42 fit recording groups、3,864/3,864 native/challenger pairs 完整执行；
- B1 的 2,219 个 exact-zero teacher contrasts 精确重现；
- 互斥分类结果：
  - `common_minimum_domination`：2,033/2,219（91.6178%），影响 42/42 groups；
  - `discrete_support_duplicate`：186/2,219（8.3822%），影响 36/42 groups；
  - `distinct_minimum_teacher_tie`：0/2,219；
- 186 个 discrete-support duplicate 的连续轨迹全部并不完全相同，但其最大中心距离均未超过
  0.5 m；这支持其为 1 m 查询格下的离散 aliasing，而不是相同轨迹；
- 预注册规则要求某一类别严格超过 50% 且至少影响 21/42 groups。共同极小值类别满足该门，
  裁决为 `COMMON_MIN_DOMINANCE_ELIGIBLE_FOR_FORK_OPERATOR_DESIGN`。

## 数学解释

对 (C=F_0\cap F_j)、(A=F_0\setminus C)、(B=F_j\setminus C)，原 contrast 为

\[
\Delta_{j0}=\min(c,b)-\min(c,a),
\quad
c=\min_C d,\ a=\min_A d,\ b=\min_B d.
\]

当共同区域 minimum (c\) 同时不大于两个独占分支 minimum 时，两侧都等于 (c)，所以

\[
\Delta_{j0}=0.
\]

B2 表明这一精确不可区分条件解释了 91.62% 的 observed exact-zero pairs，且在全部 42 组出现。
因此，当前主要瓶颈不是视觉 backbone、uncertainty sharpness 或 teacher 数值 tie，而是 scalar
global set-min 把共享风险与分支差异压成同一个数。

## 裁决边界

本结果只授予 **fork-aware task operator 的理论设计和 oracle 资格实验**，不授予 learned network
训练。下一版必须同时保留：

1. 共享区域/前缀的安全资格门，不能通过删除共同风险制造差异；
2. 只在两条候选确实分叉且分支查询非空时比较 branch-specific contrast；
3. spatial-overlap 与 temporal-prefix 两种分解必须通过反例裁决，不能事后选更好结果；
4. 与原 global min、普通 suffix min、direct pair score 和 dense field 做匹配对照。

若 fork-aware teacher oracle 仍不能产生非平凡、安全一致的区分，停止道路裕度作为一般 reranker，
将其收窄为 selective road-feasibility veto。

## 机械证据

- primary runtime：29.237 s；audit replay runtime：28.600 s；
- primary result SHA-256：
  `beb61ad987b5c8afcaacc01a4316bdb62ddc966f9d76d10726dd46f98998f64e`；
- replay result SHA-256：
  `fbc1bbe8614b8910f0cb02a2d3eb15caea5c5882a623b1dbcc4d1714c44e3b27`；
- audit SHA-256：
  `b45572b31f20d12ae68d13200da72f18c99c5be673c4cfdab731b3b7406d3b6d`；
- 672 anchor ledgers、summary 与 root contract exact；failure count 0；
- external scientific reviewer：`REVIEW_UNAVAILABLE`。

## 主张边界

这是同一 fit panel 上的 post-B1 机制归因，不是独立确认。它不证明 fork-aware operator 有效，
不证明 learned field、NavDream、OOD、决策收益、安全、时延或顶会竞争力。
