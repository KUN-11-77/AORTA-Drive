# R004H7-BS3 预注册：前向96×64完整候选域development资格

状态：`PREREGISTERED_NOT_RUN`。本实验是一次性、outcome-blind development几何资格，
不是网络评测。

## 设计依据

H7BS2已否定2.5秒最终路线。fit ROI字典中，前向96×64
`x∈[-32,64], y∈[-32,32], Δ=1m`是第一个同时满足：

1. 保持完整4秒候选时域；
2. 保留原始后向与侧向范围；
3. 支撑3864/3864 fit eligible challenger queries；
4. 节点数6144，小于8192的前向128×64和16384的对称128×128。

因此本轮只检查这一方案，不比较或重新选择ROI。development历史上已被H7B/H7BS1访问，
故只能称当前设计的held-out资格，不能称未触碰测试。

## 硬门

对24 groups×16 anchors中的每个冻结artifact：

\[
F_{40}(\tau_{native})\subseteq\Omega_{96},\qquad
\forall\tau\in\mathcal C_{eligible},\ F_{40}(\tau)\subseteq\Omega_{96}.
\]

所有native与所有full-horizon intent-eligible challengers必须100%满足；一个eligible
candidate越界即失败。另保留原H7B的24/24 groups各至少12/16 valid panels作为供应/评测
可执行性副门。无eligible challenger是candidate supply事实，不伪装成ROI failure，但仍
计入panel分母。

## 输入与禁止项

只读H7BS1已哈希绑定的384个development trajectory artifacts、outcome-free manifest、
H7BS2 fit结果与机械audit。禁止视觉field、DINO、teacher、regret/flip、calibration、test、
NavDream、target以及任何网络更新。阈值、车辆、时域、网格和候选集合冻结后一次运行。

## 解释

PASS只证明该离散1m空间域能承载当前冻结候选。它不证明相机对远端区域可观测、不证明
96×64 decoder准确、不证明1m分辨率充分、不证明时延、驾驶收益、安全或创新性。PASS后
只允许注册视觉可观测性/teacher覆盖与真实推理时延预检；H7-B2和H7-C仍不自动解锁。
