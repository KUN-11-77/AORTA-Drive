# R006V0J 结果：真实 LTF 矩阵自由算子资格

状态：`C3_C4_OPERATOR_PASS__C1_C2_MECHANICALLY_EXCLUDED__NO_SCIENCE_CLAIM`。

## 问题

正式 R006V0 需要在不显式物化 Jacobian 的条件下计算 (Jv)，并对 channel basis 求梯度。本实验只验证
真实冻结 LTF 上该算子是否数值自洽、可复算；不估计低秩介导效果。

## 失败链与实现修正

1. strict deterministic CUDA 在 adaptive-average-pool backward 处停止；因此冻结 warn-only，并要求
   same-GPU exact replay；
2. efficient SDPA 不支持二阶导、forward AD；因此冻结 math-SDPA；
3. native-MHA fastpath 不支持 forward AD；因此关闭快路径；
4. 官方 trajectory head 对 heading 原位赋值，破坏高阶图；改写为数学等价的 out-of-place concat，
   与官方输出最大差为 0；
5. 默认 attention 与 math-attention 的 clean 输出最大绝对差为 `1.90735e-6`，低于冻结 `1e-5`；
6. 高维有限差分受分段线性和消差影响，只保留描述性诊断。正式机械门使用独立 JVP/VJP 的伴随恒等式
   \(\langle Ju,w\rangle=\langle u,J^\top w\rangle\)，阈值保持 `1e-4`，未放宽。

## 冻结结果

- C3 planner memory：adjoint relative error `1.86374e-7`，basis gradient norm `0.025672`，PASS；
- C4 trajectory query：adjoint relative error `4.33725e-7`，basis gradient norm `0.023425`，PASS；
- C1 early visual：在多种合法 JVP 实现下稳定约 `7.3e-4`，FAIL；
- C2 fused BEV：稳定约 `2.2e-4`，FAIL。

C1/C2 的张量和梯度均有限且非零，但在当前冻结 CUDA/PyTorch 实现下不能满足数值算子一致性门。因此它们
在任何科学结果可见前被机械淘汰；这不证明这些层不存在因果介导，只说明本研究不能对其给出登记精度的
矩阵自由结论。

C3/C4 primary/replay 逐字节一致，SHA-256 均为
`afe3c1f880adc19fb260debd681cbb2594c0b0ef9f04b3498e19d450f6313de4`。25/25 自动机械审计通过，
封存角色未访问，未产生科学指标。

## 尾部执行资格补充

为避免正式 nested-OOF 运行对每次 JVP 重算图像骨干，项目进一步把 C3/C4 的下游映射写成显式
tail-only 算子：C3 只重放 planner decoder 与 trajectory head，C4 只重放 trajectory head。两条
tail-only 路径与同一冻结模型完整前向的最大绝对差均未超过 `1e-5`；primary/replay 逐字节一致，
SHA-256 均为 `1d1047a2dc89838b65a6be86f04bdb8e64a1ffe3f32c1bb8ddd7c203a9a39cf1`，28/28
机械审计通过。该补充只证明计算图等价、伴随一致和可复算，不产生科学指标。

## 决策

正式 R006V0 的候选层字典在科学运行前收缩为 C3 planner memory 与 C4 trajectory query；rank、fold、
appearance、A0–A7 控制和所有科学阈值不变。变更由
`R006V0_PROTOCOL_AMENDMENT_1.md`登记。

证据位于 `experiments/R006/results/r006v0j_f4feef5/`。
