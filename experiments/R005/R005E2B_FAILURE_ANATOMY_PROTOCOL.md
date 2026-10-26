# R005E2B 失败解剖协议

日期：2026-10-26  
状态：`POSTHOC_EXPLANATORY_DIAGNOSTIC__MAIN_STOP_IMMUTABLE`  
父实验：R005E2B，`UNSUPPORTED / STOP_CURRENT_VISUAL_MAIN_CLAIM`

## 1. 目的与边界

R005E2B 的实现与机械审计通过，但冻结视觉增量联合门失败。本诊断只使用已经产生的 source-fit
out-of-fold predictions、targets、masks 与训练 ledger，回答失败更接近以下哪一种机制：

1. 网络没有使用视觉；
2. 网络使用了视觉，但未使用正确场景绑定；
3. 正确视觉降低了平均关系误差，却没有改善候选决策边界；
4. 信号存在，但被优化/seed 不稳定性淹没；
5. 当前 DINO token 或通用 scorer 的函数类不足。

本诊断在看到父实验主汇总后登记，因此全部输出均为 **post-hoc、解释性、非确认性**。它不得：

- 改写父实验 `STOP`；
- 重新选择阈值或删除不利 group；
- 解锁 Differential Query、NavDream 或新训练；
- 被表述为条件互信息、OOD 鲁棒性、驾驶收益或安全证据。

## 2. 固定输入

- 冻结提交：`0fdee4b464865d76f2efbc62955baa7e8f7ca8f5`；
- 6 outer folds × 3 arms × 3 seeds 的 primary OOF predictions；
- primary/replay bit-exact 和 checkpoint-state exact 的父审计；
- 672 anchors、42 recording groups、同一 ragged legal panel；
- 不读取 development、calibration、test、NavDream 或 target domain。

## 3. 诊断量

### D1. 表示使用与场景绑定

分别比较 V2 correct checkpoint 在 correct、mismatch-intervention、removed-intervention 下的：

- relation MSE、decision regret、choice 改变率；
- beneficial / harmful / neutral choice changes；
- prediction displacement。

`removed` 恶化只能说明模型依赖视觉通道；只有 `mismatch` 相对 correct 在任务量上稳定恶化，才支持
正确场景绑定。二者不得混写。

### D2. 回归—决策错配

对 V2 与 V0/V1 逐 anchor 计算：

- relation-MSE 改善；
- 由两者选择的候选真实效用差；
- MSE 改善但选择不变、MSE 改善且选择有益、MSE 改善但选择有害的比例；
- challenger-to-challenger pair ranking error。

若平均 MSE 改善主要发生在不改变 top choice 的方向，则支持“回归目标没有把容量集中到决策边界”这一
解释，但不直接证明应采用某个新 loss。

### D3. 尺度与坍缩

对每个 arm 报告合法 challenger 上的 prediction/target 标准差比、均值偏差、Pearson correlation、
最小二乘 slope、sign error 和完整 pair-ranking error。若预测方差远小于 target，说明存在 shrinkage；
若方差正常但相关性低，更接近表示/函数类失败。

### D4. 决策难度与 seed 稳定性

报告：

- oracle top-2 utility gap 的分布；
- 固定描述性区间 `<=0.01`、`(0.01,0.05]`、`>0.05` 内的 regret；
- 每 arm 三个 seeds 的选择全一致率；
- seed 间 prediction dispersion 与 oracle top-2 gap 的关系。

`0.01/0.05` 只用于解释性尺度切片，没有显著性或 gate 含义。

### D5. 优化账本

报告 selected epoch 的 fold/arm/seed 分布、达到 patience 前后的选择差异及各 arm seed 离散度。不根据
该结果追加训练或选择新 epoch；任何优化假设必须在新的 source-only 预注册实验中独立验证。

## 4. 决策树

- **A：correct 与 mismatch 在关系和决策上均无差异。** 先审查视觉—时间—相机—空间绑定和 DINO
  可观测性；禁止直接增加网络。
- **B：correct 仅改善关系 MSE，不改善 choice/regret。** 形成“决策边界加权或结构查询”的候选假设，
  但先与简单 ranking/boundary loss 做匹配小实验。
- **C：correct 有决策增量但 seed 不稳定。** 先做优化/样本效率诊断，不把稳定器包装为主创新。
- **D：removed 恶化而 mismatch 不恶化。** 判定为通道依赖而非场景绑定；检查 positional binding、
  shortcut 与 donor 条件匹配。
- **E：上述均不能解释。** 保持 STOP，并重新评估 `U_ctx` 是否从当前 RGB/DINO 中可辨识。

任何分支只允许提出下一项最小判别实验，不能自动训练复杂 AORTA。

## 5. 产物

- `scripts/diagnose_r005e2b_failure_anatomy.py`；
- `experiments/results/R005E2B_0FDEE4B/failure_anatomy.json`；
- 结果解释写入父实验结果页与 claim–evidence matrix。

