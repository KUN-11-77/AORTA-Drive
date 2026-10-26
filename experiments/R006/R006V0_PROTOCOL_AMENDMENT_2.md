# R006V0 协议修正 2：shifted-point 算子、接口 site 语义与部署图

状态：`FROZEN_BEFORE_FORMAL_SCIENCE`。日期：2026-10-26。

## 原因

R006V0J 收缩候选层后，真实 LTF 源码表明：C3 为64个8×8 BEV memory tokens加一个status token，
C4为单个trajectory-query token。旧理论把所有 \(P\) 都写成“空间位置”，会导致DINO reader语义和
non-image status 干预错误。同时正式JVP实现实际在shifted activation处求导，旧文档部分段落写成clean点。

## 冻结修正

1. 正式目标统一使用 shifted 点 \(J_a\)，因为修复从 \(z_a\) 出发；
2. \(P\) 统一称为 interface sites：C3只允许修复前64个BEV memory sites，site 64 status永久冻结；
3. C4只有一个trajectory-query site，不宣称物理空间对应；
4. 后续统一 reader 定义为 site-conditioned DINO cross-attention：C3输出64×r系数并为status补零，C4
   输出1×r系数。它不需要伪造像素/BEV一一注册；
5. 机制发现/科学干预使用math-SDPA图；最终快速图必须通过全确认集数值与决策等价门，否则保留科学图
   并报告真实时延。

## 未改变

42 groups、六折nested选择、rank字典、appearance、A0–A7、bootstrap和GO/STOP阈值均不变。该修正
在任何正式科学指标可见前登记，不得用于结果后补救。
