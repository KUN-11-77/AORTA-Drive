# R004H7-BS4A 实验完整性审查

状态：`MECHANICAL_PASS__SCIENTIFIC_STOP__EXTERNAL_REVIEW_UNAVAILABLE`。

## A. Ground-truth provenance：NOT APPLICABLE

本实验不读取teacher、field或outcome，只审查冻结相机历史对固定候选查询节点的确定性投影
支持。它不是性能评估，也没有将模型输出伪装成ground truth。

## B. Score normalization：PASS

没有性能分数，也没有用预测自身的最大值、均值或范围归一化。严格门直接检查是否存在
`zero_direct_evidence_candidate`。

## C. Result existence and replay：PASS

- 预注册提交：`0a09600`；服务器源文件与输入hash全部匹配；
- 本地新增4 tests与strict mypy通过；服务器新增4 tests通过，mypy工具不可用；
- 42 groups、672 anchors、4536 query candidates与261 source logs完整；
- mechanical replay对ledgers、summary、source log hashes复算0失败；
- 未读取teacher、field、outcome、development、calibration、test、NavDream或target；
- result SHA-256：`4671032f4d22d37d067c83e0c298e592467e13601788b55dae53f8a7055cc8cc`；
- audit SHA-256：`6e062a3bfc17d36f7824668f29473ac2d0074371a0c7a8e7da61f1f104f71919`。

## D. Dead-code check：PASS FOR EXECUTED PATH

runner产出的672个anchor ledgers由独立audit入口重新读取并复算；本裁决不声称覆盖仓库内
与本实验无关的代码路径。

## E. Scope：PASS FOR THE PREREGISTERED FIT QUALIFICATION

范围完整覆盖预注册的42个fit groups，而不是小样本pilot；但它只支持fit上的接口资格判断，
不支持development、NavDream或最终测试泛化。

## F. Evaluation type

`deterministic_projective_support_qualification`。projective support不是遮挡感知的物理可见性，
更不是道路正确性、决策收益或安全指标。

## 独立复核边界

规定的外部reviewer backend不可用，未以内部机械审计替代独立科学复核；因此科学解释仍是
provisional，但预注册硬门失败本身不因review不可用而被放宽。

预注册硬门要求零direct-evidence candidate为0，实际为2，因此科学裁决必须为STOP。42/42
组通过不能覆盖candidate级硬失败，平均支持率也不能替代最坏情况。后续只能做失败机制归因，
不能删除两个native、改变候选集合或放宽阈值后称原实验通过。
