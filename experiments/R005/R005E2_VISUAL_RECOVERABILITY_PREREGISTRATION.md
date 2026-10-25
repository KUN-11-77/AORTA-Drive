# R005E2：上下文决策关系的视觉可恢复性预注册

状态：`PREREGISTERED__OUTCOME_FREE_PREFLIGHT_NEXT__TRAINING_NOT_YET_AUTHORIZED`。

日期：2026-10-25。

## 1. 唯一科学问题

R005E1B 已支持：冻结 source-fit 合法候选 panel 在 native-history teacher-forced
`U_ctx` 下存在非平凡 oracle opportunity。R005E2 只回答下一项必要问题：

> 在已知完整候选轨迹、合法候选 mask 与上一周期 native trajectory 后，正确场景的冻结视觉表示
> 是否仍提供可用于恢复 native-relative `U_ctx` margin 的额外信息？

本轮不检验 Differential Query，不训练视觉 backbone，不读取 development、calibration、test、
NavDream 或 target，不声称完整 pseudo-closed-loop、驾驶收益或安全。

## 2. 数学 estimand

对 recording group `g`、anchor `t`、合法候选域 `V_gt`，native 固定为 index 0：

\[
Y_{gti}=U^{ctx}_{gti}-U^{ctx}_{gt0},\qquad i\in V_{gt}\setminus\{0\}.
\]

令 `C_gt` 为完整候选轨迹、ego/candidate geometry、合法 mask 和候选顺序，`H_gt` 为因果上一周期
native trajectory，`Z_gt` 为冻结 DINO 多相机历史 tokens 与 provenance。比较两个条件信息集：

\[
X_0=(C,H),\qquad X_1=(C,H,Z).
\]

平方损失的 Bayes 风险满足：

\[
\mathcal R^*(X_0)-\mathcal R^*(X_1)
=\mathbb E\left[
\left\|\mathbb E[Y\mid X_1]-\mathbb E[Y\mid X_0]\right\|_2^2
\right]\ge0.
\]

因此，容量与训练预算受控时，correct visual arm 对 trajectory/history-only arm 的稳定风险下降，
是“视觉条件均值信息可被当前函数类利用”的操作化证据。反向不成立：有限模型未改善不能证明
条件互信息为零，可能来自函数类、优化或样本量不足。

决策指标为：

\[
\widehat i_{gt}=\arg\max_{i\in V_{gt}}\widehat Y_{gti},
\quad \widehat Y_{gt0}=0,
\]

\[
r_{gt}=\max_{i\in V_{gt}}U^{ctx}_{gti}-U^{ctx}_{gt\widehat i_{gt}}.
\]

真实独立单位是 recording group。anchor、candidate、pair、appearance 与 seed 都是组内重复或算法重复。

## 3. 数据角色与交叉拟合

- 唯一科学数据：R005E1B 的 42 个 source-fit groups、672 anchors、冻结 ragged membership、
  `U_ctx` components 与因果 previous cache；
- 采用 outcome-free SHA-256 排序形成 6 个 outer folds，每 fold 7 groups；每个 group 仅在一个
  outer fold 中产生 out-of-fold 科学预测；
- 对每个 outer fold，其余 35 groups 内再用独立盐值固定 7-group inner validation；剩余 28 groups
  训练。inner validation 只选择 early-stop epoch，不搜索网络家族、loss 或视觉条件；
- 选择 epoch 后，用相同 recipe 在完整35个 outer-train groups 上重训，再一次性预测7个 outer-test
  groups；
- fold、inner-validation 和 donor mapping 均不得使用 `U_ctx`、margin、hard failure 或模型结果；
- source-fit panel 已参与问题形成，所以结论只能称 `cross-fitted source-panel recoverability`，不能称
  untouched final generalization。

冻结 fold 盐、seed 和实现参数见
`configs/experiment/r005e2_visual_recoverability.yaml`。fold manifest 在 preflight 中生成、哈希并提交。

## 4. 三个训练 arm 与两个干预诊断

### V0：removed / trajectory-history control

完整候选与 previous trajectory 正常输入；视觉 token 替换为只由 outer-train groups 计算的
camera/history-slot 均值 token，并显式设置 `scene_specific_visual=false`。网络骨架、参数、token 数、
FLOPs 与 V1/V2 相同。该 arm 是参数匹配的 trajectory/history-only 对照。

### V1：geometry-matched mismatched visual control

输入来自其他 recording group 的 DINO tokens。donor 只由 outer-train pool 产生，禁止来自 recipient
outer-test fold；禁止同 recording group。匹配只使用 native realized trajectory 的固定时刻位置、
`sin/cos(yaw)`、步长与 yaw 增量分位数、路径长度、终点位移和合法 challenger 数；不使用标签、
challenger outcome、RGB、DINO内容或任何模型结果。描述子逐 fold 只用相应 donor pool 的均值与标准差
标准化，然后求解带跨组禁边的全局线性和分配，而不是逐样本贪心最近邻。

inner epoch selection 使用 `inner-fit -> inner-fit` 的跨组一一置换及
`inner-validation -> inner-fit` 的无放回矩形匹配；最终 refit/test 使用
`outer-train -> outer-train` 的跨组一一置换及 `outer-test -> outer-train` 的无放回矩形匹配。
因此训练错配 arm 保持 donor visual token 的经验边际分布严格不变：若 `pi` 为 donor 置换，

\[
\frac1n\sum_{i=1}^n\delta_{Z_{\pi(i)}}
=\frac1n\sum_{i=1}^n\delta_{Z_i},
\]

同时破坏正确场景对应关系。这样 V1 的差异不能由“视觉样本重复”或“换了视觉边际分布”解释。
测试 donor 永远只来自该 fold 的训练池；相同 arm 在训练与测试中均使用冻结 donor rule，从而成为
公平重训控制，而不是只在推理时制造 OOD。几何匹配只能减小、不能证明消除条件分布偏移，因此仍须
报告匹配距离分布，不能把错配下降单独解释为道路理解。

### V2：correct visual full-context probe

输入 recipient 自己的冻结 DINO tokens、相同候选/history和同一通用 full-context set scorer。
V0/V1/V2 逐层 shape、参数、优化器、更新上限、early-stop 规则、seed 与 batch sampler 完全相同。

### V2-M / V2-R：同一 V2 checkpoint 的输入干预

额外把 V2 的 correct tokens 分别替换成 frozen mismatch 和 removed tokens。它们只解释 V2 是否实际
依赖视觉输入，不冒充公平重训基线，也不能单独证明道路几何理解。

R005E2 不包含 Differential Query、signed support、orbit loss、scale/calibration 或 selective
arbitration。这样即使 V2 通过，也只说明普通 matched scorer 能利用视觉信息；AORTA 的架构创新仍未证明。

## 5. 冻结网络与优化合同

- frozen backbone：与 R004H5/H6 相同的 DINO checkpoint、源码、resize、normalization、
  normalized-last-four-layer mean 和 batch-cardinality replay；
- visual encoder 每个 anchor 只执行一次；candidate 数量不得增加 backbone 调用；
- scorer：2-layer candidate/history encoder、2-layer permutation-equivariant set block、1个
  candidate-to-global-visual cross-attention block、shared native-relative margin head；
- hidden dimension 256、8 heads、dropout 0.1；输出 `tanh` 限定在 `[-1,1]`，native margin固定为0；
- loss：只使用 group/anchor/challenger 等权的 squared native-relative margin error；不加入 ranking、
  contrastive、orbit、factor、uncertainty 或 hard-negative loss；
- AdamW，base LR `3e-4`，weight decay `1e-4`，5% linear warmup 后 cosine；最多40 epochs，patience 5；
- seeds：`20261025, 20261026, 20261027`；确定性算法、混合精度模式、batch size 与梯度累积在 preflight
  后冻结；
- V0/V1/V2 trainable parameter ratio 必须为 `1.000`；measured forward FLOPs ratio 不超过
  `1.01`，视觉 token 数完全一致。

这些数值是 recoverability probe 的冻结实现，不是 AORTA 主架构的最优性主张。

## 6. 外观条件

主要裁决只使用 clean source RGB。主要 checkpoint 冻结后，进行 source-controlled 重复测量：

- clean；
- R004H5 frozen `low_exposure`；
- R004H5 frozen `low_contrast`；
- R004H5 frozen `cool_cast`。

全部变换必须保持候选、previous trajectory、`U_ctx` label、camera/history membership 与投影 exact。
它们只能称 geometry-preserving photometric stress，不能命名为 rain/snow/dusk 或 NavDream OOD。

## 7. 主指标、聚合与统计

每个 arm 先在 seed 内计算 group-level 指标，再对三个 seeds 取均值形成每组主分析值：

1. **proper relation risk**：组内16 anchors等权；每个有challenger anchor内合法 challengers等权的
   squared margin error；zero-challenger anchor贡献0并保留在16-anchor分母；
2. **decision regret**：上述 `r_gt`，16 anchors等权；
3. **sign error / boundary error**：tie tolerance沿用R005E1B；
4. **oracle-normalized visual recovery**：

\[
\rho_{vis}=\frac{R_{V0}-R_{V2}}{0.01665036604823182}.
\]

其中分母是冻结的 group-equal contextual oracle gain，只作机会尺度归一化，不把它当可完全实现的上界。

42个 out-of-fold group paired differences 使用固定 seed `20261025` 的10,000次 group bootstrap，报告
双侧95% CI、完整42组分布、最差组、负收益组数和三个训练 seeds。该CI是在冻结cross-fit训练流程
及其OOF预测条件下对group差异的描述，不冒充包含所有训练集抽样与模型选择随机性的完整学习算法
置信区间；三个seed单独暴露优化波动。anchor/candidate不得作为独立样本。

主要比较按 intersection-union 处理，不因多个必须同时通过的条件放宽阈值：

- V2 minus V0；
- V2 minus V1；
- V2 checkpoint correct minus V2-M；
- V2 checkpoint correct minus V2-R。

context-flip、native hard-failure、late-fork 与 `|filtered margin|<=1/7` near-boundary strata只作预注册
解释，不替代全体组级主指标。

## 8. 冻结裁决

### `GO_VISUAL_INFORMATION_RECOVERABLE`

必须全部满足：

1. provenance、fold isolation、donor isolation、membership、label exactness、finite numerics、primary/replay
   与独立机械复算通过；
2. V2 相对 V0 和 V1 的 group-level proper risk 差异95% CI上界均严格小于0；
3. V2 相对 V0 和 V1 的 group-level decision regret差异95% CI上界均严格小于0；
4. V2-M 与 V2-R 相对 correct V2 的 regret差异95% CI下界均严格大于0；
5. point estimate `rho_vis >= 0.10`，即至少恢复冻结 contextual oracle opportunity 的10%；该10%是
   在训练前冻结的最小实践效应门，不是数学常数；5%与20%只作敏感性报告，不能改变主裁决；
6. 不是由删除zero-challenger、filter-trigger、hard-failure或失败group产生。

通过只解锁 B4 generic full-context scorer 与 B6 Differential Query 的公平架构预注册；不直接证明
Differential Query、外观鲁棒或驾驶收益。

### `MIXED_VISUAL_SIGNAL_NOT_DECISION_SUFFICIENT`

proper risk通过但decision regret、scene-specific controls或10%机会尺度未通过。保留结果，定位
near-boundary与模型容量，不自动增加模块，不进入主架构训练。

### `STOP_CURRENT_VISUAL_MAIN_CLAIM`

V2不能同时优于V0/V1，或correct checkpoint不优于mismatch/removed干预。停止把当前冻结表示写成
新relation视觉证据；先检查输入/标签捷径、backbone信息损失和函数类限制。不得直接训练AORTA复杂架构挽救。

### `INVALID_EXPERIMENT`

任何split/donor泄漏、标签不一致、runtime source hash不符、cache mismatch、非有限值、replay或审计失败，
科学结果不可读，不得通过放宽规则修复。

### controlled-appearance 附加分类

- clean gate通过且三个变换的task/regret与appearance-flip合同通过：
  `RECOVERABLE_AND_CONTROLLED_APPEARANCE_STABLE`；
- clean通过但任一变换失败：`RECOVERABLE_BUT_APPEARANCE_BRITTLE`，允许研究鲁棒性机制但不得宣称已鲁棒；
- 外观结果不能推翻clean主裁决，也不能替代NavDream。

## 9. outcome-free preflight

正式训练前必须由新提交完成：

1. 生成并哈希6-fold outer/inner manifest；每fold严格7个outer-test groups，组间无重叠；
2. 构建mismatch donor map并证明任何outer-test donor都来自outer-train、无同group、无label参与；
3. 记录runtime imported modules、DINO源码/权重/preprocess/cache及所有输入SHA-256；
4. 对synthetic ragged batch完成shape、mask、zero-challenger、candidate permutation、seed replay、
   correct/mismatch/removed和loss-weight metamorphic tests；
5. 在不读取科学endpoint的synthetic数据上证明三arm参数/FLOPs/token完全匹配；
6. 注册预计GPU时间、峰值显存、异常停止与断点恢复；
7. preflight、配置、源码和manifest提交后，才允许单独授权E2正式训练。

## 10. 主张边界

即使GO，本轮最多支持：

> 在当前42-group source panel的严格group cross-fitting中，正确场景冻结视觉特征为`U_ctx`
> native-relative relation提供了轨迹/history控制之外的可用预测信息。

本轮不支持：完全条件互信息证明、untouched test泛化、NavDream、Differential Query优越性、完整
`Q_pi_ref`、actual-history rollout、planner行为收益、clean no-harm、实时部署或安全保证。
