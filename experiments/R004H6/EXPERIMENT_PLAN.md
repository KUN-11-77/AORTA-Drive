# R004H6 实验计划

## Claim map

| ID | 主张 | 主要证据 | 主要反解释 | 裁决 |
|---|---|---|---|---|
| H6-C1 | 当前冻结模型能泛化到未参与其训练/选点的 development groups | 四个 confirmatory 条件的 group-macro task MAE、teacher cosine、edge drift | 固定 fit 空间先验、零场、常数塌缩 | 四条件合取门 |
| H6-C2 | 输出包含 recipient 场景图像条件信息 | correct clean 对 mismatched clean 与 image removed 的组级配对差 | 场景记忆、固定布局先验、对图像不敏感 | 两个对照合取门 |

结构表示本身相对“直接边预测”等结构是否必要，不属于 R004H6；它要求单独冻结训练预算和 checkpoint 规则后重训。

## Run order

1. 冻结配置、代码、测试与 commit，不读取 development payload。
2. 只做输入身份、group 完整性、原图哈希和 donor derangement 机械预检；不运行模型、不输出 task metric。
3. 进行一次 integration smoke，指标不得用于修改配置或代码。
4. 一次性完成 24 groups × 16 anchors 的科学运行。
5. 机械重算全部场文件、组级汇总和裁决；任何重算不一致直接拒绝结果。
6. 结果无论 PASS/MIXED/FAIL 均归档；禁止结果驱动重跑。

## Failure interpretation

- 输入/来源失败：实验无效，不产生科学结论。
- H6-C1 失败：fit 机制证据不能外推到当前 development panel，应回到视觉目标/架构研究，但不能读取 test 调参。
- H6-C2 失败：即使 task MAE 尚可，也不能排除主要依赖空间先验或弱图像依赖。
- 强度阶梯非单调：只限制对 stress-response 的解释，不改写 confirmatory 主裁决。
- 通过：只授权下一项独立、预注册的结构必要性实验；驾驶收益仍为 HOLD。

