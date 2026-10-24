# R004H6 — 当前冻结模型的 held-out 视觉泛化

## 问题与边界

R004H6 只回答两个问题：

1. R004H1 冻结模型在未参与本次训练或 checkpoint 选择的 24 个 development recording groups 上，是否仍保留任务相关且跨受控外观稳定的相对道路场；
2. 正确场景图像是否比确定性错配图像和图像移除探针提供更多任务信息。

这 24 个组没有进入当前模型的 42-group fit 或 8-group selection，但在更早的 v8.3 项目开发中曾被查看。因此本实验是“当前冻结模型的 held-out development 泛化”，不是从未触碰的最终 test，也不支持总体 NAVSIM、真实天气、目标域、驾驶收益或安全结论。calibration、test 与 target payload 继续封存。

## 设计

- 独立统计单位：完整 recording group，共 24 个。
- 组内重复测量：每组 16 个 anchors、同一 recipient 场景下的全部外观条件。
- confirmatory 外观：沿用 R004H5 的 clean、low exposure、low contrast、cool cast。
- secondary 强度阶梯：把同一光度算子连续作用两次；不得用于改写主裁决。
- 空间先验对照：只用 42 个 fit 组 teacher 的逐节点均值，不读取 development label 构造先验。
- 图像错配：development group 先按 group ID 的 SHA-256 排序，再循环平移一位；保持相同 anchor index 与 slot，recipient 的投影、mask、teacher 不变。
- 图像移除：原始 RGB 设为冻结归一化均值，属于 off-manifold 机制探针，不冒充公平重训基线。

主指标都在有向 1 m 网格边上计算，因此不受势函数共同常数偏移影响。先在 anchor 内汇总，再在 recording group 内平均，最后对 24 个组等权平均。置信区间只按 recording group 成簇 bootstrap，禁止把 384 个 anchors 当作独立样本。

## 预注册裁决

主张 1 要求四个 confirmatory 条件下 learned 均优于 zero 与 fit-only teacher prior；其中 prior-minus-learned 的 95% 组级 bootstrap 下界必须大于零，teacher-edge cosine 必须为正，三个非 clean 条件的 edge drift 必须小于 clean field edge energy。

主张 2 要求 mismatched-minus-correct 与 removed-minus-correct 的 95% 组级 bootstrap 下界都大于零。两条主张只有全部通过才记为 `SUPPORTED`；只通过一条记为 `MIXED`；两条都失败记为 `UNSUPPORTED`。本轮禁止根据结果修改模型、算子、阈值或 donor 配对后重跑。

完整机器可读规范：`configs/experiment/r004h6_heldout_visual_generalization.yaml`。

