# R006V0F 正式结果

状态：`MECHANICAL_PASS__RANK_ESTIMATION_UNSTABLE__PARENT_STOP_IMMUTABLE`。

## 结果概览

- 正式源码：`a0675d60f5093540fff3954ba6802c30b9c6a6a7`；
- 6/6 folds完成，监督链`exit_code=0`；
- 每折32个primary job与32个same-GPU replay job完成；
- 6个逐折审计均为`PASS`，每折643项检查、0失败；
- 独立聚合审计83/83 PASS，primary/replay result JSON与全部basis文件精确一致；
- checkpoint、cache、manifest、配置、runner、solver、diagnostic模块、Python、Torch、CUDA与源码提交的
  跨折provenance一致；
- 28/7/7角色隔离、冻结物理GPU映射、coverage分母不变量、finite与basis正交性全部通过；
- development、calibration、test、NavDream与target均未访问。

## 冻结科学裁决

资格翻转fold为`[0,2,4,5]`，达到预注册“至少2/6 folds出现同fold/rank seed间PASS/FAIL翻转”的首个
裁决条件，因此正式路线为：

\[
\boxed{\texttt{RANK\_ESTIMATION\_UNSTABLE}}
\]

每seed最小合格rank模式为：

- fold 0：rank 4与8各4/8；
- fold 1：8/8均无合格rank；
- fold 2：rank 4为5/8、rank 8为3/8；
- fold 3：8/8均无合格rank；
- fold 4：rank 4为3/8、rank 8为5/8；
- fold 5：rank 32为2/8，其余6/8无合格rank。

这说明冻结Stiefel估计器下的“最小合格rank”对初始化不稳定；它不证明真实子空间不存在，也不证明
某个adaptive rank可用。fold 1与3在所有初始化下均无合格rank，进一步禁止把问题简化成“多跑几次取
最好结果即可”。

## 推断边界

冻结运行没有recording-group bootstrap refit，因此projector geometry只可描述，正式标记为
`UNAVAILABLE_NO_GROUP_LEVEL_REFITS`，`method_unlock=false`。seed与fold pair不能充当独立科学样本。

本结果支持：

1. 当前非凸估计器无法为全局共享fixed-rank接口提供稳定资格；
2. 下一步可对同一冻结对象做Fantope凸松弛诊断，以区分参数化/求解器不稳定、软task metric与结构失败。

本结果不支持：

- 低秩介导成立或不成立；
- DINO可恢复性、CPTR、视觉OOD、行为收益、时延或安全收益；
- outer intervention、adaptive reader或事后采用rank 32；
- 修改父裁决`STOP_GLOBAL_SHARED_INTERFACE`。

## 下一步

只允许预注册一次最小凸estimator诊断：冻结样本、JVP、pullback metric、rank、数据角色与判据，比较
Stiefel解、Fantope软解及top-r舍入gap，并通过recording-group resample**重新拟合**评估稳定性。该诊断
是已有凸谱理论的合法借用，不是AORTA方法创新；若凸解或group-refit仍不稳定，停止全局低秩接口家族。

## 产物

- `aggregate_audit.json`：冻结聚合审计与最终路线；
- `progress.json`、`exit_code.txt`：完成状态；
- `EXPERIMENT_AUDIT.md/json`：完整性审查（外部reviewer不可用，故为provisional）；
- `SHA256SUMS.txt`：本目录紧凑结果包哈希。
