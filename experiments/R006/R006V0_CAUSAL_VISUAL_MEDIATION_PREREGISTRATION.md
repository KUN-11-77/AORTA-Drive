# R006V0：冻结 planner 的低秩外观介导资格实验

状态：`PREREGISTERED_NOT_EXECUTED`。本实验不训练 DINO、adapter、repairer 或 planner。

## 1. 唯一问题

对同一 source 场景、完全相同非图像上下文，是否存在一个低秩、跨场景共享的内部子空间，能够解释受控外观变化诱发的大部分 planner 行为差异，并在组外场景中产生 scene-specific 的双向干预效果？

full clean activation patch 只作为 hook/分解的机械上界，不构成主科学结果。

## 2. 数据、角色与选择隔离

- 仅使用 42 个 source-fit recording groups；group 是统计单位；
- 采用 6-fold outer group cross-fitting，每折 7 个 groups 只用于 confirmation；
- 每个 outer-train 内再由固定 hash 划分 basis-discovery 与 layer/rank-selection；
- outer-confirmation 不参与层、rank、基、阈值或控制设计；
- 每组 anchors、相机历史、车辆状态和 planner 上下文冻结；
- 外观只使用已登记的 `low_exposure`、`low_contrast`、`cool_cast`；
- development、calibration、test、NavDream、target 全部封存；
- 不读取 R005 utility label 设计子空间或选择样本。

## 3. planner、层与 rank 字典

第一阶段只使用已有 hook 与 checkpoint provenance 完整的官方冻结 TransFuser/LTF。

候选层冻结为：early visual、fused BEV、planner memory、trajectory query（secondary）。候选 rank 字典在结果可见前固定为 `r ∈ {4, 8, 16, 32}`，且不得超过该层通道维度。selection split 上选取满足预定义 coverage 规则的最小 rank；若没有 rank 合格，该 fold fail-closed。不得看到 outer-confirmation 后增加 rank 或移动 hook。

## 4. task-weighted drift basis

对 clean/shift pair 定义

\[
\delta_i=z_{c,i}-z_{a,i},\qquad
B_r^*=\arg\min_{B^\top B=I_r}
\mathbb E\|J_i(I-BB^\top)\delta_i\|_W^2.
\]

实现必须使用 JVP/VJP 或等价矩阵自由算子，并用小型显式 Jacobian 合成测试校验。不得以全 feature PCA 冒充该目标。

对空间 feature，basis 仅作用于通道维，投影固定为 `I_spatial ⊗ BB^T`，不得跨 token/BEV 位置任意混合。主输出度量冻结为轨迹 XY 的物理单位 RMS；yaw/heading 只作独立次指标，不参与选层/rank。

selection split 的 rank 合格条件：mean coverage 至少 0.90、最差 recording group 均值至少 0.50；不合格时该层/rank不得进入 outer-confirmation。

### 4.1 冻结求解器

- tensor semantics：`I_spatial ⊗ BB^T` 的共享 channel basis；
- 三次初始化：discovery channel-drift PCA + 两个固定 seed 随机正交基；
- Stiefel Riemannian gradient + QR retraction；
- Armijo `(step0=1, shrink=0.5, c=1e-4)`；
- relative objective decrease `<1e-6` 持续20步或最多500步；
- 只按 discovery objective 选择 restart；
- synthetic dense-Jacobian reference 必须通过 objective、gradient 与 recovered-subspace tolerance。

不得把 PCA 初值本身当成 task-weighted basis。

### 4.2 六折到唯一部署接口

- primary layer 必须在至少5/6 outer folds一致；
- 先确定满足5/6的唯一 layer 众数；只在选择该 layer 的 folds 内统计 rank；
- 同一个 selected rank 必须在全部6折中占至少4折，并且是上述 folds 内的唯一众数；
- 不满足时裁决 `STOP_GLOBAL_SHARED_INTERFACE`；
- 满足时最终 layer 与 rank 分别取上述唯一众数；其他 layer 的 rank 不参与最终 rank 决策；
- 然后只在全部42 source-fit groups上重估一次 \(B\)，不再读取 outer science endpoint 调整任何设置。

OOF结果评价的是选择程序；全42组重估只生成唯一部署方法，不回填新的 source scientific score。

## 5. 两阶段执行

### R006V0S：机械上界 smoke

- clean、shifted、shifted-self identity；
- full same-scene clean patch；
- 只验证 hook shape/order、非图像一致、确定性、finite、tail 分解和 exact replay；
- 1–2 anchors，结果不参与选层/rank或论文主张。

### R006V0：组外低秩干预

对每个冻结 outer fold、选中 layer/rank 和 appearance：

- A0：shifted，不 patch；
- A1：same-scene \(B_r^*\) coefficient patch；
- A2：条件匹配 wrong-scene clean coefficient patch；
- A3：shifted-self identity patch；
- A4：16个固定 seed（`20261026..20261041`）的 Haar channel random-subspace patch；每个实例使用 same-scene drift 投影，随后将 coefficient Frobenius norm 缩放到 A1，先在 anchor 内对16实例求均值；
- A5：16个同样固定 seed 的 Gaussian coefficient-field noise；协方差只由 outer-train 的 A1 coefficient 估计，每个实例再按 anchor 缩放到 A1 Frobenius norm，先在 anchor 内求均值；
- A6：reverse clean-to-shift low-rank patch；
- A7：dose response，强度 `0, .25, .5, .75, 1`。

wrong-scene donor 只由 outer-train metadata 构造，不使用 RGB 内容、activation、trajectory output 或 label。

## 6. 机械门

- clean 与 shifted 除 RGB 外逐字段 exact；
- hook 张量 shape、dtype、device、token/BEV 顺序有 schema；
- A3 逐值复现 A0；
- full patch smoke 满足 tail 分解预期；
- primary/replay 逐 anchor exact；
- donor 不同 group；
- 无非有限值、漏 anchor、角色污染；
- checkpoint、planner source、appearance source、preprocess、roster、fold、rank 字典全部 SHA-256 冻结。

任一机械门失败，不读取科学结论。

## 7. 指标

主恢复量：

\[
\Delta D=D(o_a,o_c)-D(o_{patch},o_c).
\]

连续轨迹报告 ADE、FDE、yaw/heading drift；candidate planner 报告 logit/ranking、clean margin preservation 与 top-choice flip。另报告 held-out basis coverage、rank–coverage 曲线、最差组、干预控制差、reverse/dose response、activation support distance。activation support distance 仅作描述性诊断，不进入 PASS/STOP。source official utility 只有接口合法时才作为独立 endpoint，不把 trajectory restoration 等同于安全。

统计以 42 个 outer-confirmation recording groups 的 OOF 结果为单位，使用配对 group bootstrap 95% CI；anchors/appearances/doses 都是组内重复测量。

## 8. 预注册裁决

`SUPPORTED_LOW_RANK_INTERNAL_MEDIATION` 要求：

1. 6 个 outer folds 都产生合格 layer/rank，不得删 fold；
2. A1 的 \(\Delta D\) group-bootstrap CI 下界大于 0；
3. A1−A2、A1−A4、A1−A5 的 CI 下界均大于 0；
4. group median restoration 的bootstrap 95% CI下界大于0，且至少22/42 outer groups恢复为正；
5. layer在至少5/6 folds、rank在至少4/6 folds一致；
6. 对每个 group，先在组内汇总 reverse damage \(R_g=D(o_{reverse},o_c)-D(o_c,o_c)\)，其42组中位数的配对 bootstrap 95% CI 下界必须大于0；对每个 anchor 的五个 dose–restoration 点做固定截距 OLS，先组内汇总 slope，再要求42组 slope 中位数的 bootstrap 95% CI 下界大于0；
7. 所有机械门通过。

若 full patch smoke 失败，裁决 `STOP_HOOK_OR_REGISTERED_INTERFACE`。若共享低秩 basis coverage 不合格，裁决 `STOP_GLOBAL_LOW_RANK_CPTR`；只允许在独立理论审查后考虑条件化/少量情景 basis。若 low-rank patch 不优于 wrong-scene/random/noise，裁决 `UNSPECIFIC_PATCH_EFFECT__STOP`。

PASS 只解锁 R006V1 的 DINO recoverability 预注册；不得直接训练最终 CPTR。

## 9. 主张边界

PASS 只支持“低秩内部子空间在模型内介导受控外观造成的行为变化”。它不证明语义解释、严格可识别性、DINO 可恢复、NavDream 泛化、真实天气、official utility、闭环安全或跨 planner 通用性。
