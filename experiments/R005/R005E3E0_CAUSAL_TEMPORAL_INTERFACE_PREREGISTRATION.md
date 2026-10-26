# R005E3E0：因果时序算子接口资格实验

状态：`HOLD_VISUAL_FIRST__NOT_EXECUTED`。协议保留为决策线备选；视觉主线 R006 完成资格前不执行。

## 1. 唯一问题

R005E3D的4个主要灾难误选全部由`Delta(HC2)=-1`产生精确`-1/8`损失。E3E0不训练模型，只回答：

> `C2`能否仅由推理时因果可用的ego状态、当前合法候选和上一周期实际选择，经冻结车辆执行算子确定性、
> 精确且低时延地计算？

这一步若不成立，E3E factorization会构成target leakage，必须停止。

## 2. 两阶段防泄漏执行

### Stage A：outcome-blind materialization

materializer不接受R005E1B result、`U_ctx`、hard metrics或future GT路径。每个anchor只读取当前ego初态、
合法candidate poses，以及系统在上一周期已经拥有并可缓存的actual-selected realized states。它批量执行
当前候选，按真实时间间隔建立重叠区间，并输出每个候选的binary `C2`与运行时账本。

### Stage B：独立label audit

Stage A完成并封存SHA后，auditor才读取冻结R005E1B ledger，按recording group、anchor和registered
candidate index连接，要求所有合法候选逐项完全一致。auditor同时检查Stage A源码/参数不包含target路径。

## 3. 延迟合同

当前source anchor间隔约500ms。为给DINO、scorer与系统I/O保留空间，冻结：

- 已测V2 scorer-only P95：`16.12ms`；
- scorer + temporal operator P95上限：`25ms`，即控制周期的5%；
- 因而temporal operator自身P95上限：`8.88ms`。

在固定服务器CPU/GPU、batch-1完整合法panel下先20次warmup，再200次计时，报告P50/P95/P99/max；
另报告候选数`K=1..7`缩放。该数字仍不包含DINO与I/O，所以只是一项组件门，不是端到端实时性结论。

## 4. 裁决

`GO`要求：672 anchors全部处理、所有合法候选exact、primary/replay bit-exact、无非有限值、无sealed-role
访问，且operator P95不超过`8.88ms`。

若exact但时延超限，裁决`HOLD`，只允许不改变数学对象的向量化/缓存优化；不得先训练网络。

任一exactness、determinism、causal availability或泄漏门失败，裁决`STOP` structured analytic branch。

## 5. 结论边界

PASS只允许冻结E3E1的matched monolithic-vs-factorized训练实验，不证明factorization有用、视觉有增量、
Differential Query成立、NavDream鲁棒或驾驶安全。
