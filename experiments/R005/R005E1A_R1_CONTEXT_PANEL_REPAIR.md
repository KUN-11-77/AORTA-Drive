# R005E1A-R1：单 anchor 因果上下文可定义性修复

状态：`FROZEN_OUTCOME_FREE_REPAIR__REPLACEMENT_CACHE_PENDING`。

日期：2026-10-25。

## 失败事实

原 R005E1A 的 672 个 immediate predecessor 均存在于 raw log，但标准 NAVSIM metric-cache
构建仅成功 671 个。唯一失败项为：

- recording group：`2021.06.07.19.29.59_veh-38`；
- anchor index：`4`；
- 原 current：`2c18d1604abe52b8`；
- 原 previous：`1f0d6191a3225bb6`；
- raw-log frame index：previous=2，current=3；
- 失败原因：官方 cache processor 固定读取 1.5 s ego history；previous 在日志起点附近，触发
  `Iteration -1 out of bound`，不能生成合法 cache。

该失败在任何 contextual utility、oracle gain 或网络结果产生前发现。禁止删除该 anchor、使用
human history、padding history 或以近似状态填充。

## 唯一修复

仅将上述 anchor 沿同一 raw log 前移一个 immediate frame：

- replacement current：`e25e0f03413553f9`（raw-log frame index 4）；
- replacement previous：`2c18d1604abe52b8`（raw-log frame index 3）；
- 时间差：`0.500173 s`；
- recording group、split 与 anchor index 不变；
- 新 current 使用标准 4-history cache；replacement previous 直接复用原 current 的已冻结标准 cache。

修复不读取任何候选分数或科学结果。下一步须先：

1. 构建 replacement current 的标准 cache；
2. 用原 H7BS1 几何/意图规则重新资格化该 anchor；
3. 若该 anchor 不能在冻结 2.5 s prefix 下通过，则本次修复 `STOP`，不得继续向后搜索；
4. 若通过，冻结其合法候选集合，随后才允许生成 one-frame 与 contextual 标签；
5. 重新形成 672-anchor combined manifests、完整 primary/replay 与机械审计。

正式 test/calibration/development/NavDream/target 保持封存。原 B4R1 结果继续保留，修复后的
R005 panel 是独立派生 estimand，不覆盖原实验。
