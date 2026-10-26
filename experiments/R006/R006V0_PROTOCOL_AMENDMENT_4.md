# R006V0 协议修正 4：注册site约束与A4/A5等能量控制

状态：`FROZEN_BEFORE_FORMAL_SCIENCE`。

## 原协议缺口

协议已要求C3 status site不可修复、A4为Haar random subspace、A5为outer-train covariance Gaussian
coefficient field，但尚未把site mask写入干预算子，也未唯一规定A5的高维协方差采样。若控制过弱，A1
胜出不能证明效应来自资格化任务子空间。

## 冻结实现

1. 所有A1–A7 correction均先乘注册site mask；C3第65个status site严格为零，C4唯一site可修复；
2. A4对同一clean-minus-shift delta使用seeded Gaussian QR产生Haar channel basis，并投影到相同rank；
3. A5先把35个outer-train groups的same-scene coefficients按anchor×appearance组成
   `K×P×r`，对K维居中；
4. 不物化可能奇异的`(P r)×(P r)`协方差。令居中后的flattened coefficient fields为
   `X∈R^{K×Pr}`，采样`w∼N(0,I/(K−1))`并取`w^T X`。其条件协方差精确等于经验协方差
   `X^T X/(K−1)`，保留跨site与跨rank相关性；
5. A4/A5分别对每个anchor与seed，把correction Frobenius norm精确缩放到该anchor A1 correction能量；
6. A1能量不超过`1e-12`时控制定义为零；若A1非零而control原始能量不超过`1e-12`，机械失败，禁止
   重抽seed；
7. 16个冻结seed先在anchor内平均科学指标，再进入group统计，不把seed当独立样本。

## 未改变

数据、fold、layer/rank选择、A0–A7名称、seed列表、科学阈值与GO/STOP均不变。本修正只让随机/噪声
控制具有与A1相同的site、rank与能量约束，不提供介导证据。
