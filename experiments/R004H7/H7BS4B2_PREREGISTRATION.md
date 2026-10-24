# R004H7-BS4B2 Preregistration：global set-min 零差异结构归因

状态：`PREREGISTERED_POST_B1_DIAGNOSTIC__NOT_RUN`。

## 问题

B1 机械通过但 non-vacuity 假设失败；2,219/3,864 teacher contrasts 精确为零。本实验不调 B1
阈值、不训练新网络，只回答零差异来自哪种结构。

## 冻结分类

对 native support (F_0)、challenger support (F_j) 和 teacher (d)，先计算 exact binary64
contrast。若非零，记为 `nonzero_contrast`；若为零，按以下互斥优先级分类：

1. `discrete_support_duplicate`：(F_0=F_j)；
2. `common_minimum_domination`：集合不同，但交集 minimum 同时等于两侧 global minimum；
3. `distinct_minimum_teacher_tie`：集合不同、共同 minimum 不控制两侧，但两侧 minimum 相等。

不引入 epsilon。另以 0.01 m 与 0.10 m 仅报告非零 near-tie 描述，不参与 exact-zero 分类。
连续轨迹只报告 exact pose equality、最大中心距离、首次超过 `1e-9 m`、`0.5×grid` 和 `1×grid`
的索引；这些阈值不用于选择分类。

## 冻结裁决

某一 exact-zero 类别只有在占比严格大于 50%，且至少影响 21/42 recording groups 时，才获得
单一路线的后续设计资格：

- discrete duplicate → continuous/finer query study；
- common-min domination → fork-aware operator design；
- distinct-min tie → teacher/task audit；
- 其余 → mixed causes，不授权单一 redesign。

“获得设计资格”不等于方法有效，下一版仍需独立预注册和确认。

## 数据边界

只读冻结的 42 fit groups、672 anchors、3,864 pairs、完整 4 s trajectories、forward96 teacher 与
离散 node sets。禁止 learned field、DINO values、planner outcomes、development、calibration、test、
NavDream 与 target。

## 完整性

必须精确复现 B1 的 2,219 zero pairs 与所有 teacher contrasts；primary 与 audit replay 的逐 anchor
ledger、summary 和 root contract 必须 exact。外部 reviewer 不可用时明确标记
`REVIEW_UNAVAILABLE`。
