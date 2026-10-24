# R004H7-A 理论资格结果

## 裁决

`PASS_DATA_FREE_THEORY_GATE`

该结果只验证有限图 decision-quotient 定义与已登记确定性命题，不提供视觉、OOD、驾驶、连续几何或安全证据。

## 冻结运行

- 实现 revision：`bfd69a01270fd8fab135a7918ade9f31760b728f`；
- preregistration revision：`b07f39d624287429e739431a012294e037fe3376`；
- config SHA256：`1092355cdc9e5caddb7c65b1f0966cd4f91517715c250cad921395d101b15619`；
- script SHA256：`94c707bd481e0c280e1f69fbe23e4032d2a7b66725ca50d73dce65ddfa4797b6`；
- result SHA256：`9cc3898e706a5dfd3cb51f507176ff6c72dd1c6ea806af5b3c6d9e088e850f19`；
- 数据访问：`none`。

## Oracle 覆盖

| 检查 | 案例数 | 失败 |
|---|---:|---:|
| common-offset invariance | 3,969 | 0 |
| set-min oscillation bound | 35,721 | 0 |
| \(2\eta\) selection-regret bound | 250,047 | 0 |
| integrable potential/cut-space recovery | 27 | 0 |
| global nonlinearity counterexample | 1 | 0 |
| deterministic tie policy | 2 | 0 |

工程验证：R004H7-A focused tests `6 passed`；完整仓库测试 `274 passed`；strict mypy 对新增理论模块与 runner 通过。当前环境未安装 ruff，记录为 tooling unavailable，不以其他检查冒充。

## 解释

1. node potential quotient 与 integrable/projected edge space 表示等价，potential 不能作为表达能力创新；
2. exact set-min decision quotient 全局分段线性但一般不是线性 quotient，固定矩阵 kernel 只能在 active cell 内成立；
3. decision-contrast 误差给出候选排序 regret 的确定性上界，但界的经验紧致度与可预测性仍须 H7-B 验证；
4. H7-A 通过只允许进入 H7-B 的实现和机械资格，不允许直接读取 development payload 或训练 H7-C。

