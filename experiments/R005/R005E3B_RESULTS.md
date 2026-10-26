# R005E3B 结果：删除`tanh`不足以恢复决策可实现性

日期：2026-10-26  
冻结训练提交：`65359750091ee8905e2c74276789293e5e3a79bb`  
运行目录：`/data/khsong/aorta_runs/r005e3b_6535975`  
最终裁决：`MECHANICAL_PASS__UNSUPPORTED__FAIL_LINEAR_REALIZABILITY`

## 1. 本轮问题与唯一处理

R005E3A把全部challenger推至`tanh`下界并恒选native。R005E3B只把

\[
s_i=\tanh(r_i-r_0)
\]

替换为

\[
s_i=r_i-r_0,
\]

其余网络、参数量、初始化、视觉cache、fold0 outer-train roster、soft selection-regret、`T=0.125`、
AdamW、schedule、14 epochs、7,840 optimizer steps与数据顺序全部冻结。该实验只检验训练域优化/可实现性，
不检验OOF、视觉scene binding、Differential Query、外观鲁棒或驾驶收益。

## 2. 机械结果

- primary与same-GPU replay全部完成；
- predictions逐数组bit-exact，final tensor state SHA均为
  `45b9093508592d32432aa3b609800d05ad86120f334b7660c79dcdb65d560724`；
- 父初始化、4,749,313参数、14 epochs、7,840 steps、温度与fold0 outer-train成员独立复算一致；
- 560 anchors、35 groups×16 anchors、target与mask均与冻结bundle一致；
- 全部数值有限，development/calibration/test/NavDream/target未访问；
- 强化审计脚本SHA为
  `b84d4ea3ce479d13cf8042e101ed761563bc9b9583ec0da3f661458822a9a25d`，退出码0。

## 3. 冻结科学门

| 指标 | 冻结要求 | 结果 | 裁决 |
|---|---:|---:|---|
| mean `G_model` | `>0` | `0.000000` | FAIL |
| absolute oracle capture | `>=0.10` | `0.000000` | FAIL |
| beneficial nonnative | `>=1` | `0` | FAIL |
| nonnative choices | `>=1` | `0/560` | FAIL |
| native choice fraction | 诊断 | `1.000000` | all-native |
| mean `G_oracle` | 机会账本 | `0.01903635` | opportunity exists |
| mean hard regret | 诊断 | `0.01903635` | equals oracle gap |

correct、S1 mismatch和removed三种输入均得到相同hard choice与相同收益；因此不存在训练域scene-specific
decision binding证据。

## 4. 优化轨迹揭示的机制

初始化时challenger score位于`[-0.0623,-0.0276]`，已经全部低于native的精确0，但遗漏优候选的
score-gradient绝对均值仍为`0.142925`。第1个epoch后，score整体降至约`[-1.8125,-1.7500]`，仍然
100%选择native；第14个epoch为`[-2.40625,-2.3125]`。

与此同时，遗漏优候选的score-gradient绝对均值降至

\[
1.6981\times10^{-9},
\]

相对初始化缩小约`8.42e7`倍，最小值为`5.24e-14`。这说明：删除`tanh` Jacobian后，模型仍沿着
“压低所有challenger”的方向进入softmax低概率尾部，随后纠错梯度近乎消失。

这支持预结果理论中的两候选反例作为当前失败机制的必要组成，但尚未单独证明根因只有softmax尾部。
共享参数下，负utility challenger的数量/梯度质量、正机会稀疏性与函数类耦合仍需只读诊断。

## 5. 理论更新

本轮严格否定C30：

> `tanh`饱和不是all-native collapse的充分解释；只删除该activation不能恢复当前generic scorer与
> soft-regret组合的训练域绝对收益。

仍然成立的边界：

1. 无界soft-regret总体条件风险的最优方向没有被该有限优化失败推翻；
2. identity删除了一个饱和因子，但没有删除`p_k`对遗漏低分候选造成的指数梯度衰减；
3. 失败不证明RGB或DINO中不存在决策信息；
4. 失败禁止直接扩网、训练Differential Query或进入NavDream。

## 6. 下一步唯一授权

先做不更新权重的source-fit只读诊断：量化正/负challenger机会比例、utility gap质量、初始化处各类
score-gradient质量及共享参数梯度冲突。只有诊断证实“错排正机会的纠错梯度被多数负机会/softmax尾部
压制”，才允许另行预注册已在结果前冻结的utility-gap加权pairwise logistic单因素实验。

该候选的理由不是复杂度，而是两个可检验性质：错排pair的梯度下界与逐anchor hard-regret上界。它仍是
优化surrogate，不得冒充AORTA的最终算法创新。

## 7. 产物

- 强化机械审计：`experiments/results/r005e3b_compact_6535975/audit_v2.json`；
- primary/replay结果与预测：`experiments/results/r005e3b_compact_6535975/`；
- compact archive SHA：`a11a78aebbb201589ec750b4e7eb25ff72e8428be13942d7098e0b82c61fa841`；
- 完整性报告：`experiments/R005/R005E3B_EXPERIMENT_AUDIT.md`。

