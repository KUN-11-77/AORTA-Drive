# R004H7-BS4B0B：teacher-critical projective observability

状态：`PREREGISTERED_NOT_RUN`。本实验不训练模型、不读取 DINO 值或 planner outcome，只回答：
H7BS4A/D0 的无直接证据节点是否可能控制冻结 teacher 的候选 set-min。

对候选节点集 \(F\)，每个 teacher 节点具有向外舍入区间
\([\ell_q,u_q]\)。定义

\[
C(F)=\{q\in F:\ell_q\le\min_{p\in F}u_p\}.
\]

该条件既必要又充分：若 \(q\) 在某个合法区间实现中是最小值，则
\(\ell_q\le d_q\le d_p\le u_p\)；反之令 \(d_q=\ell_q\)、其余节点取上端点即可构造
一个使 \(q\) 成为（允许并列的）最小值的实现。因此 ties 必须保留，不能用浮点等号或事后
epsilon 删除。

候选的严格 teacher-critical 支撑条件是

\[
C(F)\subseteq O,
\]

其中 \(O\) 是冻结的 projective-token support mask。没有比例阈值：任一未支撑 possible
minimizer 都使该严格前提失败。对 D0 的 8 个零支撑对称差 pair，另行检查未支撑关键节点位于
公共区域还是 candidate-only 差异区域。

## 固定解释

- 任一 candidate 存在未支撑 possible minimizer：当前接口在 fit teacher 上关键不完整；
- 所有 candidate 关键节点均支撑：原 all-node gate 对 fit teacher 过严；
- 8 个 D0 pair 中任一盲差异节点关键：差异盲区确实可能控制 set-min；
- 8 个均不关键：D0 的差异盲区不是当前 teacher set-min 瓶颈，但不能升级 H7BS4A。

完整 672 anchors、4536 candidates、3864 native/challenger pairs 必须机械重放；观察结果后不得
添加 epsilon、修改候选或删掉失败组。projective support 不是 occlusion-aware visibility，结论只限
fit 机制归因，不代表模型、OOD、驾驶或安全收益。
