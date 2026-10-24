# R004H6R1 资源恢复修订

首轮 R004H6 在写出全部 384 个 field artifacts 后、生成 `result.json` 前触发预注册 90 分钟墙钟上限。最后一个成功登记点为 368/384、5195.899 秒；clean replay mismatch 为 0，峰值 CUDA memory 为 3216324096 bytes，输出目录约 107 MiB。首轮未生成完整 anchor ledger 与科学汇总，故状态冻结为 `STOPPED_RESOURCE_CAP`，不得从部分结果发布主张。

在未读取 task MAE、视觉错配差异、外观稳定性或任何科学 endpoint 的前提下，授权一次完整、从零开始的资源恢复重放。唯一变化是墙钟上限由 90 分钟增至 120 分钟；checkpoint、DINO、24 groups、384 anchors、donor derangement、外观条件、强扰动、zero/prior/pooled/uniform/removal 控制、指标、bootstrap 和主裁决完全不变。首轮 fields 不复用。120 分钟仅为服务器 I/O 与运行波动留出机械余量，不改变实用性判断所报告的真实耗时。

R004H6R1 若再次触发资源或实现异常，停止本阶段，不再追加第三次运行。该修订解决运行时预算估计错误，不授权任何结果驱动的方法、阈值或样本调整。
