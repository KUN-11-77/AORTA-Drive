# R004H7-B 冻结 shadow-decision 结果

## 裁决

`INSUFFICIENT_PANEL_COVERAGE`。这不是 `SUPPORTED`，也不是
`UNSUPPORTED`：预注册完整性门在读取任何相关性结论之前失败，因此 decision-contrast
error 对 regret、ranking flip 及三个 controls 的 Spearman/AUROC 均未计算。H7-C 继续
`HOLD`。

## 完整性结果

- 24/24 development recording groups、384/384 anchors 均进入分母；
- 265/384 anchors 具有有效 native 与至少一个有效 challenger；
- 只有 11/24 groups 达到每组至少 12/16 有效 anchors 的冻结门槛；
- 110 anchors 为 `native_invalid`，9 anchors 为 `no_valid_challenger`；
- candidate 级失败中，770 次来自 `oriented footprint leaves the frozen ROI`，98 次来自
  `intent_ineligible`；冻结 provider 自身四类 rejection 均为 0；
- 352/384 anchors 的六个 challenger 全部通过 intent eligibility，说明主瓶颈不是
  normal-offset 供给，而是冻结 `[-32,32]^2`、64×64、1m 离散查询域无法完整承载一部分
  official realized trajectories 及车辆矩形。

完整 raw result、384 个 panel artifacts、日志与审计保存在被 Git 忽略的
`artifacts/r004h7b_run2/`。artifact bundle SHA-256 为
`5069d36ba5b841193de06f083b7acfbd1faf5667a9a0ddfe2d9e75db764ef65b`；raw result
SHA-256 为 `64e80d90124a2d88b60cd55339970d03677d04262ddb74ec3ba4b019196f7488`。

## 机械审计

冻结 auditor 独立重建全部候选、official NAVSIM execution、oriented rectangle node
sets、teacher projection 与统计摘要：384 个 panel artifacts 全部通过，artifact failure
与 metric recomputation failure 均为 0，复现裁决
`INSUFFICIENT_PANEL_COVERAGE`。机械审计文件 SHA-256 为
`32f1add860afb8e2bc122eed17f3a1c08c7d60625a6e4ed9b4b714f91bfb94c3`。

规定的跨模型 reviewer backend 在本会话不可用，因此外部完整性审查状态是
`REVIEW_UNAVAILABLE`；机械复算不能冒充独立科学复核。

## 首次启动与可追溯性

首次启动在汇总指标产生前因 orchestration 的 label-root 少一层目录而失败；它保留在
服务器 `r004h7b_c587e1b_run1` 中，只产生 5 个未评分 panel 文件，没有 result 或 progress
ledger。随后逐一核验 384/384 标签及 384/384 metric caches 的存在性和哈希，在新目录
`r004h7b_c587e1b_run2` 以相同冻结源码、数据、候选、外观、统计和阈值完成正式运行。
这次路径修复不改变科学合同，失败产物未并入结果。

## 允许的下一步

禁止直接进入 H7-C，也禁止用事后删除失败组、降低 `12/16` 门槛或挑选短轨迹来生成正
结果。只允许先注册一个只读 support-adequacy diagnosis：在不读取预测误差或 ranking
outcome 的条件下，量化 native/challenger 的最小完整查询域、越界发生的时间位置，以及
“扩大 ROI”与“缩短 shadow horizon”各自改变的数学对象和有效 estimand。只有预先选定
一个与任务定义一致、对全部方法共同且不以 outcome 选择的 support contract，才能另行
授权 H7-B2。

本结果不支持 continuous geometry、驾驶收益、安全、NavDream/target、ICCV 创新性或
最终 test 泛化主张。
