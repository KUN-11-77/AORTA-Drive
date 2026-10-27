# R006V0 basis 实验完整性审计

审计结论：`MECHANICAL_PASS__SCIENTIFIC_STOP__PROVISIONAL_INTERNAL_REVIEW`。

## 机械完整性

- top-level exit code 为 0，progress 为 COMPLETE；
- 6/6 folds 完成，0 failed、0 running；
- 每折 primary/replay 文件集合及 SHA-256 完全一致；
- 每折 basis shape、有限值、正交性、hash、样本 roster、coverage 与选择规则均独立复算；
- 跨折 checkpoint、配置、runner、solver、cache 与 source commit provenance 一致；
- 聚合审计 93/93 checks PASS；
- 封存角色未访问，outer-confirmation endpoint 未生成。

本地首次复算曾因三个均值存在约 `1e-16` 的末位舍入差而将完整 selection dictionary 错判为不同。
审计器已修正为：layer/rank/rule 离散字段严格相等，连续统计按原协议 `1e-12` 容差比较；新增回归
测试后相关测试为 20/20 PASS。该修复不改变任何实验值、阈值或选择。

## 科学裁决忠实性

预注册要求 layer 至少 `5/6` 一致、rank 至少 `4/6` 一致。观察到 C4 layer 为 `6/6`，但最小合格
rank 分布为 `{4:2, 8:1, 32:3}`，故自动裁决 `STOP_GLOBAL_SHARED_INTERFACE`。没有以 rank32 在6折
均合格为由事后覆盖最小秩规则，也没有执行 outer finite intervention。

## 诚实度与范围

- 评价类型：模型内部、自监督式 task-drift proxy 的 inner qualification；
- 不存在外部 ground truth 替换、分数归一化冒充提升或不完整运行冒充完整结果；
- 本轮只有 basis 资格，不是视觉方法、驾驶收益或安全评价；
- post-hoc projector overlap 明确标注 descriptive，不能改变 STOP。

当前环境没有可调用的独立外部 reviewer backend，因此无法完成 cross-model blind review。依照
`experiment-audit` 流程，这一限制记录为 `REVIEW_UNAVAILABLE`，整体审计只能称
`PROVISIONAL_INTERNAL_REVIEW`，不得表述为独立科学复核。
