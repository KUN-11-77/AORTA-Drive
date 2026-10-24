# R004H7-BS4B4T0 temporal-context roster preflight

状态：**FORMAL_OUTCOME_FREE_INTERFACE_PASS**。

## 目的

确认未来 one-frame→official full-target alignment 是否具有真实前一帧上下文。该审计不调用
candidate scorer，不读取 B4/B4R1 outcome、teacher、视觉场或任何 development/calibration/test/
NavDream/target 数据。

## 发现

### 冻结 anchor roster 内

- 672 anchors、42 groups；
- 满足同 log 且时间差 \(0<\Delta t\le0.55\) s 的相邻 pair：0；
- 42/42 groups的 pair count均为0；
- 同一 log 中连续抽样 anchor 的最小正间隔：1.9992671013 s；
- 逻辑 ledger SHA-256：cafd02c8ef9c853745894a775d66d36a4e25cf17761655deb15b79ea5610399e。

因此直接在稀疏 roster 上运行官方 aggregator 会把 two-frame 权重全部置零；得到的“full=one-frame”
只是一项 roster artifact，不是 surrogate alignment 证据。

### 官方 raw log 内

按每个 current token 在其原始 log 中的唯一位置，取立即前一行并检查时间差：

- unique raw log files：261；
- previous context available：672/672；
- 42/42 groups均为16/16；
- gap范围：0.498078–0.500686 s；
- 逻辑 ledger SHA-256：95d94dd15b68ce28095518d5c9a4f162c4e550c290a8b27158a48fc4c0e67ab0。

该结果说明 augmented-context alignment 可定义。冻结提交06d3ab0的正式服务器materialization与
independent replay各生成672行完整ledger，两份259,835-byte JSON逐字节相同：

- primary/replay SHA-256：
  2ae90f5ca3d566605ae3c64d6efdc61de6eb1cfa1ba7c5c9ed126647a693512c；
- source transfer bundle SHA-256：
  f8ef5d7b2290e4b22ed1ca09753cbe373854d61c118fe1c311f0289e2ad62b21；
- local summary：
  F:/AORTA-Drive-artifacts/gate-b/R004H7/r004h7bs4b4t0_25825c6_summary.json。

8项合成反例、19项相关测试与strict mypy均通过。完整ledger保留在服务器
/data/khsong/aorta_runs/r004h7bs4b4t0_25825c6；本地summary记录其哈希、输入和claim boundary。

## 对后续设计的约束

1. anchor index 永远不是时间；
2. full-target alignment必须显式加入raw-log previous context row；
3. 同一current anchor的native与全部legal challengers共享同一previous context；
4. 离线评估默认previous context为冻结planner的上一输出；若部署时AORTA上一帧已干预，则下一帧
   应读取实际已选择轨迹，系统因此可能是stateful；该差别必须单独审计；
5. B4R1未通过前，本preflight不授权full scoring或网络训练。

