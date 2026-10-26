# R005E2S Donor Sensitivity 正式结果

日期：2026-10-26

裁决：`MECHANICAL_PASS__RELATION_ONLY_BINDING_SECONDARY__PARENT_STOP_IMMUTABLE`

冻结实现：`73b5e46d5e06e064d1da60007d3af6e8878bf746`

完整产物：`experiments/results/R005E2S_73B5E46/`

## 1. 本轮科学问题

R005E2B 的 V2 correct checkpoint 相对 primary geometry mismatch 降低了 relation MSE，但没有降低
decision regret。本轮不训练网络，只把同一 V2 checkpoint 的视觉输入替换为两个在科学结果可见前
构造并审计的 donor 规则：

- `S2_disjoint_geometry`：禁止 S1 已用 donor edge 后重新做几何匹配；
- `S3_hash_random`：在同一合法二分图上执行冻结 salt 的确定性无放回随机分配。

候选、history、target、mask、模型状态和计算量保持不变。本轮只判断 relation/decision scene binding
是否依赖 S1 donor 构造，不能改变 R005E2B 的父裁决。

## 2. 实现与完整性

- 6 outer folds × 2 donor rules × 3 seeds，复用18个冻结 V2 checkpoints；
- 6 primary fold jobs与6个同fold同物理GPU replay全部完成；
- prediction arrays逐元素bit-exact；
- parent checkpoint file hash与tensor-state hash逐个重验；
- 672/672 OOF anchors、target/mask、fold/donor隔离、finite、配置与输入hash全部通过；
- S2/S3 donor均来自outer-train、cross-group且无放回；
- development/calibration/test/NavDream/target未访问；
- 本地同步后的全部文件再次通过SHA-256复核。

## 3. 冻结结果

| 条件 | Relation MSE ↓ | Decision regret ↓ | Top-choice boundary error ↓ |
|---|---:|---:|---:|
| V2 correct | 0.112362 | 0.019301 | 0.410218 |
| S2 disjoint geometry | 0.134302 | 0.020852 | 0.411210 |
| S3 hash random | 0.146258 | 0.021159 | 0.408730 |

42-group paired bootstrap（donor − correct，正值表示donor更差）：

| 比较 | Point | 95% CI | 冻结支持 |
|---|---:|---:|---|
| S2 relation MSE | +0.021940 | [0.015855, 0.028243] | 是 |
| S3 relation MSE | +0.033896 | [0.025969, 0.042040] | 是 |
| S2 decision regret | +0.001551 | [-0.000243, 0.003901] | 否 |
| S3 decision regret | +0.001858 | [-0.001303, 0.005236] | 否 |

两种 donor 下 relation MSE 都可靠恶化，但 decision regret 的区间均跨零。因此冻结分类为：

`RELATION_ONLY_BINDING_SECONDARY`。

## 4. 干预强度与选择变化账本

- DINO RMS：S2均值`0.30318`、S3均值`0.31128`；
- top-choice flip：S2 `48/2016=2.38%`，S3 `51/2016=2.53%`；
- 两规则flip交集`21/2016=1.04%`，并集`78/2016=3.87%`；
- S2几何距离p95为`0.29010`，S3为`5.41066`；两者均存在极端离群值，因此均值只作为账本，
  不用于因果解释或筛选 donor。

更强的 donor shift 带来更大 relation MSE，不能据此声称更强道路理解；它同时包含输入分布距离变化。

## 5. 结论与下一步资格

本轮支持一个收缩后的结论：**当前冻结视觉通道包含对 donor 规则稳健的场景绑定平均关系信号，
但这种信号仍未转化为可重复的候选选择收益。**

它与 R005E2B 的失败解剖一致：平均关系误差和完整 pair ranking 可以改善，但选择变化稀少，且大量
panel存在真实效用ties。父裁决保持：

`UNSUPPORTED / STOP_CURRENT_VISUAL_MAIN_CLAIM`。

按照结果可见前冻结的第一性决策树，本结果命中 Branch B。下一步只允许登记一个容量、数据、训练预算
完全相同的最小 objective qualification：原平方margin回归对比最简单的decision-aligned surrogate，
以组级decision regret为主端点。不得直接训练Differential Query，不得访问NavDream或封存角色。

## 6. 不能声称

- 不能把本轮当作条件互信息或严格conditional randomization test；
- 不能说视觉已提供decision-sufficient信息；
- 不能说新loss或Differential Query必然有效；
- 不能声称外观鲁棒性、NavDream、planner行为、驾驶收益或安全。
