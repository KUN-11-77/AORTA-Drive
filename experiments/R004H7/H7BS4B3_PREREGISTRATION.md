# R004H7-BS4B3：Fork-aware task operator oracle 资格门

状态：`PREREGISTERED_POST_B2_ORACLE_GATE__NOT_RUN`。

## 1. 科学问题

B2 已精确证明，2,033/2,219 个 global set-min 零差异由共同 minimum 支配。B3 不训练网络，
只检验一个更窄的问题：在不删除完整 4 秒时域、不读取 driving outcome 的条件下，将共同轨迹
与分叉后轨迹分离，是否能在 teacher oracle 上恢复非平凡候选差异。

这不是“让指标变好”的设计。若连 perfect teacher 上都不能恢复差异，则更强 backbone、loss 或
uncertainty 都不能挽救该 task operator，道路裕度作为一般 reranker 应停止。

## 2. 数学对象与可辨识性修正

对两条候选的时刻 footprint (F_i^t,F_j^t)，主 factorization 的 fork 定义为

\[
t_f=\inf\{t:F_i^t\ne F_j^t\}.
\]

令共享前缀 (C=\cup_{t<t_f}F_i^t)，分叉后 suffix 为
(B_i=\cup_{t\ge t_f}F_i^t) 与 (B_j=\cup_{t\ge t_f}F_j^t)。主量为

\[
D_{ij}(d)=\min_{q\in B_j}d_q-\min_{q\in B_i}d_q.
\]

它满足 (D_{ij}(d+c\mathbf 1)=D_{ij}(d))，与 quotient potential 的可辨识对象一致。
相反，(\min_C d>0) 随 gauge 改变，不能由 quotient potential 单独识别。故 teacher 的绝对正负
只作 oracle 分层描述，不进入 B3 主 PASS，也不得写成部署安全证书。部署层当前只保留相对
native-preserving 语义：共同前缀相同，选择只比较分叉后分支；真实决策收益仍需后续 outcome 门。

## 3. Primary、sensitivity 与 control

- **Primary：temporal first-fork suffix。** 驾驶选择具有时间顺序，因此预先指定为唯一主判据；
- **Sensitivity：spatial exact factorization。** 用完整支持的交集与独占集合复核 B2 的空间解释，
  但即使更好也不能事后替代 temporal primary；
- **Control：固定 2.5 s suffix。** 使用 H7BS1 已冻结的 index 25，只判断一个任意固定切点能否
  达到相同效果，不参与方法选择。

三者同时运行，禁止只报告更有利定义。temporal suffix 允许分叉后空间重新重合；spatial
exclusive 会删除所有重合节点。两者估计对象不同，结果不得混写。

## 4. 重复测量设计与统计单位

同一 candidate pair 同时接受 global、fixed-suffix、temporal 与 spatial 四个算子，是 paired
repeated-measures 设计。anchor 与 pair 均为组内重复，42 个 recording groups 才是统计独立单位。
主恢复率先在每组内计算，再对 42 个组等权平均并以固定 seed 进行 10,000 次 group bootstrap。

Primary PASS 必须同时满足：

1. 对 2,033 个 common-min pairs，temporal factorization 有效率至少 95%；
2. 组等权 exact-nonzero 恢复率的 95% bootstrap 下界严格大于 50%；
3. 至少 21/42 groups 出现一个恢复 pair；
4. 完整 replay audit exact，且禁止数据均未访问。

50% 是“原本不可区分 pairs 的严格多数”门，不是看数后选择；95% 防止用小规模可分子集制造
高恢复率。spatial 使用同门仅产生 sensitivity 结论，不能覆盖 temporal 失败。

## 5. 裁决

- temporal PASS → `TEMPORAL_FORK_OPERATOR_NONDEGENERATE_ON_FIT__DECISION_RELEVANCE_HOLD`，
  只解锁 matched-capacity operator/network 设计；
- temporal FAIL、spatial PASS → `SPATIAL_ONLY_SIGNAL_REQUIRES_THEORY_REVIEW__NO_OPERATOR_SELECTED`；
- 两者均 FAIL → `STOP_ROAD_MARGIN_AS_GENERAL_RERANKER`。

即使 PASS，也不证明视觉网络、排序、驾驶、安全、OOD、NavDream、跨 planner 或时延收益。

## 6. 数据与反泄漏边界

只读冻结 42 fit groups、672 anchors、3,864 pairs、B2 ledger、forward96 exact teacher 与 4 秒
trajectories。禁止 learned field、DINO、planner outcome、development、calibration、test、NavDream
与 target。输出必须精确复现 2,033 个 B2 common-min identities，primary 与机械 replay 的
ledger、summary 和 root contract 必须 exact。外部 reviewer 不可用时明确标记
`REVIEW_UNAVAILABLE`。
