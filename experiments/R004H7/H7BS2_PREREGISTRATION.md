# R004H7-BS2 预注册：候选信息充分性与完整时域支持

状态：`FIT_ONLY_PREREGISTERED`。H7-B2 与 H7-C 均保持 `HOLD`。

## 1. 为什么必须先做本实验

H7BS1 只证明 2.5 秒内的离散 footprint 查询能够落入当前 64×64 场。它没有证明
2.5 秒包含区分完整 4 秒候选所需的信息。若 native 与 challenger 在前缀上产生相同
查询集合，任何视觉网络都不可能通过当前 set-min 接口区分它们；这不是预测精度问题。

本实验只读取 H7BS1 已冻结、已逐 bit 审计的 42×16 fit realized trajectories 与
full-horizon intent eligibility。禁止读取 field、DINO、teacher、regret、ranking flip、
calibration、test、NavDream 或 target。

## 2. 数学对象与可证伪命题

在与原始网格同相位的无限 1m lattice 上，记候选在离散时刻 `0..k` 的车体节点并集为

\[
F_k(\tau)=\bigcup_{t=0}^{k}\left((\mathbb Z^2+(1/2,1/2))\cap B(\tau_t)\right).
\]

对应 set-min 为

\[
M_k(\tau;u)=\min_{q\in F_k(\tau)}u(q).
\]

**不可区分性定理。** 若 \(F_{25}(\tau_i)=F_{25}(\tau_j)\)，则对任意定义在该 lattice
上的场 \(u\)，都有

\[
M_{25}(\tau_i;u)=M_{25}(\tau_j;u).
\]

若同时 \(F_{40}(\tau_i)\neq F_{40}(\tau_j)\)，则存在一个场能在完整时域区分二者：取
差集中的一个节点为负值、其余节点为零即可构造。因此“prefix 集合相同、完整集合不同”
是一个不依赖实际网络、teacher 或 outcome 的精确结构反例。

主反证规则预先固定为：只要任一 full-horizon intent-eligible native/challenger pair 满足
上述条件，`2.5s universal decision sufficiency` 即被否定。零反例只能记为
`NO_EXACT_COUNTEREXAMPLE_OBSERVED_ON_FIT`，不能当作充分性证明。

## 3. 分叉时间

补充定义车辆参考点的首次分叉时间

\[
T_{div}^{\delta}=\inf\{t:\|p_i(t)-p_j(t)\|_2>\delta\}.
\]

主阈值 \(\delta=0.5m\) 来自冻结 provider 的最小 challenger 幅度，而不是结果调参。
`0.25m` 与 `1.0m` 分别是半倍和两倍敏感性分析，不参与主裁决。所有统计以 42 个
recording groups 等权聚合；pairs 与 anchors 只是组内重复测量，不作为独立样本。

## 4. 完整 4 秒 ROI 支持消融

保持候选、4 秒时域、官方车体和 1m 分辨率不变，只改变空间定义域：

| ID | 范围设计 | 节点数 | 目的 |
|---|---|---:|---|
| original 64×64 | `x,y∈[-32,32]` | 4096 | 冻结基线 |
| shifted 64×64 | `x∈[-16,48], y∈[-32,32]` | 4096 | 固定预算，把一半后向余量移到前方 |
| forward 80×64 | 原后/侧范围，前向 +16m | 5120 | 第一档前向扩展 |
| forward 96×64 | 原后/侧范围，前向 +32m | 6144 | 第二档前向扩展 |
| forward 128×64 | 原后/侧范围，前向 +64m | 8192 | 第三档前向扩展 |
| symmetric 128×128 | `x,y∈[-64,64]` | 16384 | 与前向方案对照的各向同性扩展 |

`16/32/64m` 是冻结的二进制粗到细范围梯，不根据结果增删。前向非对称方案由 H7BS0
“全部越界均为 x_max”这一已登记机制观察提出；128×128 作为简单但计算更大的控制。
选择规则是：在 42/42 groups 均达到至少 12/16 有效 anchors 的方案中，先取节点数最少，
再按上表固定顺序打破同成本平局。该选择只产生“fit 上最小支持候选”，不授权网络训练。

节点数只作为本轮 outcome-blind 几何阶段的静态成本代理，不等价于真实推理时延。任何
ROI 被称为最终架构前，必须另行冻结同 GPU、同精度、同 batch、同步计时、固定 warm-up
与重复次数的延迟协议，分别报告 DINO encoder、空间 decoder、候选查询/set-min、仲裁及
端到端的 P50/P95、峰值显存与吞吐。至少比较原 64×64、fit 选择的最小支持 ROI、
128×128 控制及任何多尺度/坐标查询方案。可接受预算必须来自控制周期或事前冻结的
baseline-relative 合同，不能看到结果后设定。本轮不产生 deployability 结论。

## 5. 结果解释与下一步

- 出现精确 prefix-collapse 反例：停止把 2.5 秒 prefix 作为最终方法；H7-B2 不得承担
  完整 4 秒主张。先对 fit 选出的完整时域 ROI 做一次独立 development 支持资格，再设计
  相同信息、容量匹配的网络消融。
- 未出现反例：只说明该 fit panel 未发现精确集合碰撞；仍需 outcome-linked 检验，不能
  宣称 2.5 秒决策充分。
- 没有任何 ROI 方案通过：停止固定均匀网格路线，转向支持域感知的多尺度或坐标查询设计。
- ROI 通过：这只是空间合同，不是算法创新。后续必须重建投影和 teacher，并公平比较
  原 64×64、最小前向 ROI、128×128 与容量匹配表示；禁止把旧场插值成新架构的证据。
- 几何与任务指标通过但时延门失败：该方案不得成为主架构；应比较前向非对称、近密远疏
  多尺度和按候选稀疏查询的 Pareto 前沿，而不是仅继续增大 dense tensor。

本实验的作用是让架构尺寸由任务时域、官方车辆、冻结候选和明确成本规则共同决定，
而不是把 64、2.5 秒或 128 当作没有来源的超参数。
