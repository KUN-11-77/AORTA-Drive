# R004H7-BS1 fit 结果物化失败记录

状态：`672_ARTIFACTS_COMPLETE__RESULT_JSON_ABSENT__RECOVERY_REGISTERED`。

冻结提交 `81129838c8035a06ee837ae6bd7c2a28fcdf84cc` 在42个fit groups的672个anchors上
完成了全部 official execution 与 panel artifact 写入，但最终结果 JSON 严格序列化失败。
原因是辅助 `intent_eligibility` 审计结构含 `NaN`，而 runner 使用
`allow_nan=False`；因此没有生成 `result.json`，也没有读取development或任何视觉场、
teacher、ranking/regret outcome。

失败后冻结：

- panel artifacts：672个；
- panel SHA-256清单：
  `eec5d09e555a822cc16ff7470fd8156096a19ecba4d23d1204430f19d08f4232`；
- 失败日志 SHA-256：
  `14d208326b108b45abb8d0d74f14d69fbc37f7f44ce67b866f3011014636ed22`；
- `result.json` 在冻结时不存在。

不允许重新生成科学panel。登记的恢复步骤只允许：逐artifact核验清单哈希；重新执行同一
official execution并要求与已存轨迹逐bit一致；重新计算完整4秒intent eligibility并要求
与artifact一致；按原配置重新计算7个prefix与唯一的最大可行prefix规则；移除不属于科学
终点且造成JSON失败的非有限辅助诊断值。恢复结果仍须由第二个脚本对672个artifact逐项
复算后才可解释。
