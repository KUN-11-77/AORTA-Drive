# R004H7-BS4B1 首次运行前资格修订

日期：2026-10-25。

首个服务器进程使用 commit `0495949` 启动后，在尚未写出 `result.json`、尚未产生或读取任何
科学终点时被人工中止。中止原因不是数值结果，而是公式—实现复核发现：代码记录
`maximum_numerical_widening`，但原 PASS 门没有显式要求该量仅处于 binary64 账本范围。

这会允许一个不一致的 observed-interval 系统被静默扩大后继续，违反预注册的“同一 teacher
包含条件”比较。因此新增 fail-closed 条件：任一 anchor 的可行势放宽超过 `1e-9 m`，整次实验
判为 `INVALID_ORACLE_SET_OR_NUMERICS`。原 42 groups、672 anchors、3864 pairs、三种集合、
12/16 门、bootstrap 和全部 claim boundary 不变。

首次目录中不存在任何 result artifact，不得把该中止运行用于科学解释。后续运行必须使用新的
commit 和 runner hash；这是一项 pre-result protocol correction，不是结果驱动调参。
