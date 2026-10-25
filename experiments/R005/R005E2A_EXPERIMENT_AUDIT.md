# R005E2A outcome-free资格门完整性审计

日期：2026-10-25。  
裁决：`PASS_MECHANICAL__REVIEW_UNAVAILABLE__NO_SCIENTIFIC_ENDPOINT`。

## 审计对象

- 42 groups、672 anchors的冻结DINO cache primary/replay；
- S2/S3 donor敏感性映射primary/replay；
- 单fold 28-group、448-anchor真实I/O/backward/checkpoint资源演练；
- 证据目录：`experiments/results/R005E2A_C957AA0/`。

## 完整性结论

1. **来源与角色**：只使用source-fit train roster、冻结panel几何、原始train RGB和冻结DINO；
   development、calibration、test、NavDream、target和科学label均未访问。
2. **视觉provenance**：checkpoint、151个DINO Python源码、预处理、slot顺序、raw-image内容地址、
   BF16存储与投影合同均被manifest绑定；8,064个特征及1,344个投影文件逐一重开并核验。
3. **重放**：primary/replay逻辑manifest一致，所有特征和投影产物byte-exact；退出码与登记SHA复算一致。
4. **donor控制**：14,784个S2/S3 assignments不使用label、RGB、DINO内容或模型输出；全部fold隔离、
   跨组、无放回，训练stage保持精确视觉边际置换，S2不复用S1边。
5. **训练资源**：448/448 steps、finite loss/gradient、模型和optimizer checkpoint精确恢复、恢复前向
   bit-exact；结果只包含资源测量，不含relation MSE、regret或任何scientific decision。
6. **已发现问题**：旧审计器把`P95 >= mean`错误当作普适不变量；冷启动长尾构成明确反例。旧失败
   日志被保留，修复提交`81a2cb5`新增反例测试并对同一不可变结果重审PASS，没有重跑或筛选数据。

## 防舞弊与范围判断

- fake ground truth：不适用；本阶段target严格为零资源替身，且审计禁止科学字段；
- score normalization：不适用；没有科学score或跨方法比较；
- phantom results：未发现；文件数量、hash、重放、训练step与checkpoint均有可复算产物；
- insufficient scope：本结果只资格化执行链，不能支持视觉信息、泛化、鲁棒性、决策收益或安全；
- external cross-model review：当前工具环境没有可用review backend，记为`REVIEW_UNAVAILABLE`，不得把
  机械PASS表述为独立科学同行复核。

## GO/HOLD/STOP

- `GO`：冻结R005E2B scientific runner与auditor；
- `HOLD`：在完整runner/replay/audit闭合前不读取correct/mismatch/removed结果；
- `STOP`：不启动Differential Query，不接触封存角色，不作ICCV方法有效性或驾驶安全主张。
