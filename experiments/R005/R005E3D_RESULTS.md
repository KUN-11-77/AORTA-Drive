# R005E3D 结果：Proper-Mean 单因素资格实验

日期：2026-10-26。

冻结实现提交：`c72b3692173aac087f49d5793c2227dc48f4e702`。

正式目录：`/data/khsong/aorta_runs/r005e3d_c72b369`。

裁决：`MECHANICAL_PASS__SCIENTIFIC_UNSUPPORTED__FAIL_FROZEN_QUALIFICATION_GATE`。

## 1. 实验问题

R005E3D保持identity scorer、正确视觉、fold-0 outer-train roster、初始化、AdamW、学习率、BF16、
14 epochs、7,840 steps与数据顺序不变，只将soft selection-regret替换为anchor-equal、challenger-equal
的native-relative MSE。它只检验proper conditional-mean目标能否使当前有限网络在训练域获得非退化的
正决策收益，不检验OOF泛化、视觉因果增量、Differential Query、NavDream或驾驶安全。

## 2. 执行完整性

- primary与same-GPU replay均完成，退出码为0；prediction、result与checkpoint逐文件哈希一致；
- 冻结input/provenance、fold、seed、训练步数、candidate mask、target、模型参数量与sealed-role标志通过；
- 本地独立重新运行冻结机械审计，输出与服务器`audit.json`一致；
- 所有最终数值有限，development、calibration、test、NavDream与target均未访问；
- epoch ledger可exact replay，但未归档每个epoch的完整prediction，因此中间路径只作provenance-limited描述。

## 3. 冻结结果

| 指标 | Correct视觉 |
|---|---:|
| Mean gain over native | `-0.00085196` |
| Mean oracle gain | `0.01903635` |
| Absolute oracle capture | `-0.0447543` |
| Mean hard regret | `0.01988831` |
| Relation MSE | `0.08040007` |
| Native choice fraction | `0.9607143` |
| Non-native choices | `22/560` |
| Beneficial / harmful / neutral non-native | `13 / 5 / 4` |

Mismatch与removed输入的mean gain分别为`-0.00262752`与`-0.03410792`，relation MSE分别为
`0.14345944`与`0.16661122`。这说明正确视觉对平均relation拟合有信息，但冻结科学门要求的是实际
`G_model>0`及至少10%的absolute oracle capture；二者均失败，所以正式裁决只能是`UNSUPPORTED`。

## 4. 只读失败解剖

`scripts/analyze_r005e3d_failure.py`在不改checkpoint、不选择部署阈值的条件下复算最终prediction、
完整R005E1B components与轨迹几何：

- 13次有益选择总收益仅`+0.01713796`，均值`+0.0013183`；
- 5次有害选择总损失`-0.4942344`，均值`-0.0988469`；
- 其中4次灾难损失为`-0.12281`至`-0.12444`，全部选择registered candidate 6；
- 这4次的filtered one-frame relation实际略正（`+0.000635`至`+0.002500`），但
  `hard_gate × two_frame_comfort`相对native均从1降到0，精确贡献`-1/8=-0.125`；
- 第5次有害选择仅为`-0.000156`，来自轻微负的filtered one-frame relation，temporal项为0；
- 目标分解以最大绝对误差`2.97e-8`复现：

\[
\Delta U_{ctx}=\frac78\Delta F_{filtered}+\frac18\Delta(HC_2).
\]

因此本轮不是“模型从未学到任何信息”，而是**连续单帧微收益被少量离散时序门错误压倒**。当前
score与contextual relation的Pearson相关为`0.6150`，与filtered relation为`0.5964`，与temporal
hard-gate relation仅为`0.3001`。34个预测正challenger中只有18个真实为正；零阈值precision为
`0.5294`、recall为`0.0289`。

## 5. 被否定与仍未否定的解释

已否定：

1. 仅把soft-regret换成proper MSE即可通过训练域正收益资格门；
2. 延长同一训练或在同一scalar score上事后选阈值是充分修复。即使在同一outer-train样本上，所有
   产生非零干预的非负阈值仍为负收益，最优行是全拒判、收益0；
3. 平均relation拟合改善可以替代决策尾部检查。

尚未否定：

1. correct视觉包含可学习的scene-specific relation信息；本轮relation MSE明显优于mismatch/removed，
   但不能由训练域结果升级为OOF视觉增量；
2. 当前函数类在结构化目标下能否恢复正收益；
3. 候选差异查询在结构化基础模型通过后能否优于matched scorer。

## 6. 下一步边界

不自动进行第三次训练。先资格化一个由已观察失败和官方效用代数推出的最小结构修正：把可因果计算的
candidate-history two-frame comfort从未知视觉效用中分离，使用proper heads学习filtered scene outcome
与hard-gate概率，再按官方`7/8 + 1/8`结构重构条件均值。其理论与停止条件见
`R005E3E_STRUCTURED_UTILITY_FACTORIZATION_THEORY_NOTE.md`。

该方向首先必须证明temporal term能从推理时可用的当前ego、候选轨迹和上一周期实际选择中确定性复算，
并报告CPU/GPU延迟；否则停止，不允许把evaluator标签作为推理输入。即便资格通过，该结构也只是G1的
可靠学习地基，不是AORTA的ICCV核心视觉创新。
