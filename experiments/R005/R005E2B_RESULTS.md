# R005E2B 视觉可恢复性正式结果

日期：2026-10-26  
裁决：`MECHANICAL_PASS__UNSUPPORTED__STOP_CURRENT_VISUAL_MAIN_CLAIM`  
冻结实现：`0fdee4b464865d76f2efbc62955baa7e8f7ca8f5`  
完整远端产物：`/data/khsong/aorta_runs/r005e2b_science_0fdee4b_multislot_v2/`  
本地紧凑证据：`experiments/results/R005E2B_0FDEE4B/`

## 1. 本轮回答的问题

在冻结合法候选集合与因果 native-history `U_ctx` 条件下，正确场景的冻结 DINOv3 token 是否比：

- `V0_removed`：只保留 outer-train slot mean 的参数匹配非场景视觉控制；
- `V1_mismatch`：由冻结几何匹配规则置换的错误场景视觉；

更好地恢复 native-relative relation，并降低候选选择 regret。

本轮不是 Differential Query、外观鲁棒、完整 pseudo-closed-loop 或驾驶收益实验。

## 2. 实现与审计资格

- 6 folds × 3 arms × 3 seeds = 54 primary jobs；
- 随后完成54个 replay jobs，共108/108；
- primary/replay predictions bit-exact；
- primary/replay model state exact；
- 同 fold/seed 三臂初始化一致；
- 每个 job 的 primary/replay 绑定同一物理 GPU；
- 672/672 OOF anchors、fold isolation、target membership、输入 hash 和 finite values 全部通过；
- development/calibration/test/NavDream/target 未访问；
- `exit_code.txt=0`，完整目录329个非日志文件 SHA-256 复核通过。

资源并行从每卡1个 job 修订为每卡最多4个相互独立 job，发生在任何正式 job 完成和科学指标读取前，
只改变调度；单 job 的模型、数据、batch、优化、seed 和数值合同未改变。该修订已预先记录于
`R005E2_RESOURCE_AMENDMENT_2026-10-25.md`，且 bit-exact replay 通过。

## 3. 冻结主指标

| 条件 | Relation MSE ↓ | Decision regret ↓ | Sign error ↓ | Boundary error ↓ |
|---|---:|---:|---:|---:|
| V0 removed | 0.113980 | 0.018877 | 0.194130 | 0.413194 |
| V1 mismatch | 0.120371 | 0.018316 | 0.193337 | 0.411210 |
| V2 correct | **0.112362** | 0.019301 | 0.193758 | **0.410218** |
| V2 checkpoint + mismatch intervention | 0.129506 | 0.018869 | 0.194436 | 0.411210 |
| V2 checkpoint + removed intervention | 0.141448 | 0.033049 | 0.208697 | 0.414187 |

42-group paired bootstrap：

| 冻结比较 | Point | 95% CI | 通过？ |
|---|---:|---:|---|
| V2 − V0 relation MSE | -0.001619 | [-0.006963, 0.003673] | 否 |
| V2 − V1 relation MSE | -0.008010 | [-0.014059, -0.001854] | 是 |
| V2 − V0 regret | +0.000424 | [-0.002111, 0.003641] | 否 |
| V2 − V1 regret | +0.000985 | [-0.001365, 0.003899] | 否 |
| mismatch intervention − correct regret | -0.000432 | [-0.002760, 0.001391] | 否 |
| removed intervention − correct regret | +0.013749 | [0.007018, 0.021697] | 是 |

oracle-normalized visual recovery：`-0.025449`，未达到冻结的 `>=0.10` 最小效应门。

联合门四项均失败：proper risk、decision regret、scene-specific intervention、minimum effect。

## 4. 可以说什么

1. 当前 probe **确实使用了视觉通道**：同一 V2 checkpoint 移除视觉后 regret 明显恶化。
2. 正确视觉相对几何匹配错配视觉降低了 relation MSE，说明存在有限的场景绑定回归信号。
3. 该信号没有转化为候选选择收益：V2 regret 不优于两个训练控制，mismatch intervention 也没有
   可靠伤害 regret。
4. 因此，当前冻结 DINO + generic full-context scorer + squared-margin objective 不能支持“视觉已提供
   决策充分增量”这一主张。

## 5. 不能说什么

- 不能说 RGB 在信息论上没有任务信息；有限网络失败包含逼近、估计和优化误差。
- 不能说 Differential Query 已失败或已成功；它没有参与本轮。
- 不能把 removed intervention 的退化单独解释为道路理解，因为它也是输入分布替换。
- 不能声称外观鲁棒、NavDream 泛化、完整策略价值、驾驶收益或安全。

## 6. 登记后的 post-hoc 失败解剖

只读诊断没有改变父裁决，得到以下解释性证据：

- V2 prediction/target Pearson 为`0.3452`，高于 V0 的`0.2845`和 V1 的`0.2378`；V2完整 pair-ranking
  error为`0.2746`，低于V0的`0.3134`和V1的`0.3613`。正确视觉确有平均关系信号。
- V2预测标准差只有target的`0.5015`，最小二乘slope为`0.1731`，存在明显 shrinkage，但不完全坍缩。
- V2相对V0/V1的MSE改善与choice utility改善相关仅`0.0189/0.0675`。在2016个anchor-seed观测中，
  V2相对V1有1211次anchor MSE改善，其中1171次choice效用不变、25次有益、15次有害。
- 435/672 anchors（64.73%）存在exact top-2 utility tie，平均最优集合大小为`2.81`；虽然276/672
  anchors存在strict oracle-over-native机会，但很多候选共享最优值。V2与V0/V1只有`3.87%/4.41%`
  的choice改变，绝大多数改变在效用上仍为neutral。
- 三seed choice全一致率：V0 `97.32%`、V1 `95.83%`、V2 `91.22%`。视觉分支的优化/预测波动更高，
  但这不足以单独解释主门失败。

最符合现有证据的解释是：**正确视觉改善了平均关系回归，但当前目标拓扑、回归损失与通用读取器没有
把该信号稳定转化到稀疏的决策关键候选边界。** 这仍是解释性假设，不是新结论。

## 7. 下一步边界

- `STOP`：不得按原方案直接进入 Differential Query 或 NavDream。
- `GO`：仅允许完成已预注册、非 gating 的 S2/S3 donor sensitivity，或登记新的 source-only 最小
  判别实验，区分 backbone 可观测性、scene binding、loss alignment 与函数类限制。
- 新实验必须先比较最简单的 decision-aligned ranking/boundary objective；只有它仍不能利用正确视觉，
  且 signed differential query 给出容量匹配优势时，后者才可能成为主架构创新。

