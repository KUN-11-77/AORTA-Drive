# R005E2-S：视觉 donor 依赖性的预注册敏感性分析

状态：`PREREGISTERED_SECONDARY__NO_PRIMARY_GATE_CHANGE__NO_OUTCOME_ACCESSED`

## 1. 为什么需要这一项

R005E2 的主问题是：给定候选集合和因果历史 (B) 后，正确场景视觉 (V) 是否为
native-relative `U_ctx` 关系 (Y) 提供有限网络可恢复的增量。平方损失下总体 Bayes 风险满足

\[
\mathcal R^*(Y\mid B)-\mathcal R^*(Y\mid B,V)
=\mathbb E\!\left[\left\|\mathbb E[Y\mid B,V]-\mathbb E[Y\mid B]\right\|_2^2\right]\ge 0.
\]

主实验中的 V0 检验有限容量网络能否利用这种增量；V1 和同 checkpoint 的视觉替换干预检验增量
是否依赖正确场景绑定。但当前 geometry-matched donor 只匹配冻结候选几何与合法候选数量。它保持
训练视觉边际为精确置换，却不保证从真实的 (P(V\mid B)) 抽样，也不保证道路拓扑、交通参与者、
信号状态等未进入 (B) 的场景因素条件独立。因此：

> V1 是严格隔离且可复算的 scene-binding negative control，不是精确 conditional randomization
> test；其结果不得单独表述为条件互信息估计或因果视觉必要性证明。

## 2. 与主实验的关系

- 主三臂、主指标、置信区间、GO/MIXED/STOP 判据完全不变；
- 本分析是非 gating 的 secondary sensitivity，不得挽救主门失败；
- 不重新训练或选择网络，不接触 development、calibration、test、NavDream 或 target；
- 只对每个 outer-test group 的冻结 V2 checkpoint 施加预先构造的输入替换；
- 所有替换保持候选、history、label、candidate mask、visual tensor shape 与计算量完全不变。

## 3. 三种预注册 donor 干预

每个 fold 的 donor 只能来自相应 outer-train pool，禁止 same-group，且不得使用 RGB、DINO 内容、
`U_ctx`、candidate outcome、模型输出或任何科学指标。

1. **S1：primary geometry assignment**  
   使用已冻结的全局线性和分配结果，作为主 mismatch intervention。
2. **S2：disjoint geometry assignment**  
   禁止 S1 已使用的 recipient-donor 边后，以同一标准化几何代价重新执行全局线性和分配。若某一
   stage 不存在完整可行匹配，则该 stage 明确记为 `NOT_IDENTIFIABLE`，不得放宽禁边或重复 donor。
3. **S3：hash-random exact assignment**  
   在同一合法二分图上，以固定 salt `aorta-r005e2-donor-sensitivity-random-v1` 产生与标签和视觉
   无关的确定性边代价，再求无放回全局分配。训练池自映射仍须为精确视觉边际置换。

S2/S3 映射必须在 R005E2 科学输出可见前一次性生成、primary/replay，并由独立脚本复算 donor pool、
same-group 禁边、无放回、fold isolation 与映射哈希。

## 4. 固定统计对象

对每个外层测试组 (g)、seed (s) 和替换规则 (k\in\{S1,S2,S3\})，在冻结 V2 checkpoint 上计算

\[
\Delta^{(k)}_{g,s}
=L_{g,s}(V^{(k)}_{\mathrm{donor}})-L_{g,s}(V_{\mathrm{correct}}),
\]

其中 (L) 分别取预注册的 equal-group relation MSE 与 decision regret。先在组内聚合 anchors，
再在组内聚合 seeds；置信区间仍以 42 个 recording groups 为 bootstrap 单位，使用主协议相同的
seed 与 10,000 次重复。

同时报告：

- 三种干预的组级差值分布、CI、最差组和符号一致率；
- S1/S2 的几何距离分布与 S3 的距离分布，但不按结果筛选 donor；
- correct 与 donor DINO 特征距离，仅作干预强度账本，不参与 donor 选择或门槛；
- 不同 donor 规则下 top-choice flip 的交集与并集。

## 5. 解释规则

- 若 V2 对 V0 的主优势成立，且三种 donor 干预均显著恶化，则支持“正确场景绑定的视觉信息被当前
  网络利用”，但仍不等价于真实 (P(V\mid B)) 下的条件随机化证明；
- 若只有 S1 恶化而 S2/S3 不恶化，scene-specificity 对 donor 构造敏感，主视觉必要性结论降级为
  `DONOR_SENSITIVE`；
- 若 V2 优于 V0、但所有 donor 干预都不恶化，则更符合稳定数据先验或视觉无关 shortcut；进入
  shortcut 归因，不进入 Differential Query；
- 若 donor 干预越不匹配、性能越差，只能说明 distribution shift 强度变化，不能把更大退化解释为
  更强的道路几何理解；
- 无论本敏感性结果如何，都不得将 R005E2 表述为外观鲁棒性、NavDream 泛化或驾驶安全收益。

## 6. 停止规则

- 任何映射使用标签、视觉内容或模型结果；
- donor 来自 outer test、same group 或封存角色；
- 在看到科学结果后修改 salt、距离、可行边或 donor 数量；
- 将不可行 assignment 通过重复 donor、放回抽样或删除困难 recipient 改造成可运行结果；
- 用本 secondary analysis 改写、替代或放宽主实验的联合通过条件。
