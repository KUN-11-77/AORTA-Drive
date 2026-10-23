# AORTA-v8.3.2 Geometry Amendment

## Decision

V8.3.1 的 graph-gradient、session calibration、exact min-plus readout 与
relative-intervention risk object 保持不变。唯一的理论改动是将无法可靠生产化的
challenger internal-net 假设替换为可证明的 asymmetric geometry contract。

这是一项 soundness 修正，不作为独立算法创新主张。

## Failure being repaired

旧四角点/离散姿态/raster EDT 路径只能产生评测信号，不能证明：

1. 整个车体而非四个角点位于 nominal road 内；
2. 相邻轨迹姿态之间的连续扫掠体没有越界；
3. 固定图节点形成真实扫掠体的 internal \(h\)-net；
4. raster EDT 与连续边界距离之间的误差为零。

R002B 已分别给出前三类几何/地图捷径的构造性反例。

## Revised mathematical object

对 challenger \(j\)，构造确定性外包络

\[
F_j\subseteq\overline F_j,
\]

并选择固定图节点 \(A_j\)，使每个 \(x\in\overline F_j\) 都存在
\(a\in A_j\) 满足 \(\lVert x-a\rVert_2\le h_j\)。节点不必位于真实
challenger footprint 内。

对 native \(0\)，仍要求

\[
A_0\subseteq F_0.
\]

因此对 1-Lipschitz teacher \(\phi\)：

\[
\inf_{F_j}\phi
\ge \min_{A_j}\phi-h_j,
\qquad
\inf_{F_0}\phi
\le \min_{A_0}\phi.
\]

结合 node-label error 与 discrete robust lower bound，仍得到

\[
\Delta_{j0}^{cont}
\ge L_G(A_j,A_0)
-h_j-\varepsilon_{map,j}-\varepsilon_{map,0}.
\]

外包络多选低势节点只会降低证书；native 不得对称地使用外包络节点，因为这可能
错误降低被减去的 native minimum。

## Frozen interpolation and cover

车体 \(K\) 以 rear axle 为原点。相邻姿态采用线性平移、连续展开 yaw 的线性插值，
并要求 \(|\Delta\theta|\le\pi\)。以中点车体为中心的膨胀半径为

\[
\delta_i=
\frac12\lVert p_{i+1}-p_i\rVert_2
+2R_K\sin\frac{|\Delta\theta_i|}{4}
+\varepsilon_{pose}.
\]

这同时控制中点相对的最大平移偏差、旋转弦长偏差和声明的姿态误差。

固定方格边长为 \(r\)，格心为图节点，\(h=r/\sqrt2\)。选择到中点车体距离不超过
\(\delta_i+h\) 的所有格心，即可覆盖该段外包络。多段取并集。

## Deterministic rejection rules

以下情况不得返回连续证书：

- 任一外包络越出认证 ROI；
- 相邻 yaw 未连续展开或变化超过 \(\pi\)；
- challenger 或 native 节点集为空；
- vehicle/reference/interpolation/grid/transform protocol hash 不一致；
- 不能给出几何浮点运算的外舍入或显式误差预算；
- continuous teacher、node labels 与 \(\varepsilon_{map}\) 尚未闭合。

## Evidence status

- R002B：旧 corner/discrete/raster 捷径已被反例否定；
- R002C：解析外包络与固定格心覆盖已在 clean revision `943e7bb` 上通过
  5 项 synthetic Gate A 测试；
- R002D：exact polygon-SDF interval 与显式 map-error ledger 已在 clean revision
  `497b77b` 上通过 7 项 synthetic 测试；
- R002E：164/164 NAVSIM metric caches、45/45 repository tests 在 clean revision
  `13abb27` 上通过；raw polygon-cover 与 rotation-free disk proof path 获得资格；
- R002F：164/164 caches、49/49 tests 在 clean revision `1ff87a6` 上通过；
  exact-rational trig/affine interval 产生逐节点 \(\varepsilon_{transform}\)；
- full Gate A：保持 BLOCKED，直到 R002G 冻结实际 ROI/resolution 并生成全节点
  raw-cover label/error manifest。

## Falsifiers

任一以下观察直接否定当前 geometry contract：

- 存在连续扫掠点不在解析外包络中；
- 存在外包络点，其所在 grid cell 的中心未被选择；
- 任一 native 选择节点不在真实 native footprint 中；
- ROI clipping 被静默接受；
- 真实 preprocessing 的误差超过声明 budget；
- 修正项使合法 candidate 的 certificate 几乎全部为空。

## Continuous teacher amendment

R002D 的 exact-rational distance 构造在 R002E 后绑定为 raw polygon-cover teacher，
而非 raster EDT 或未经证明的 canonical overlay。成员关系取 raw polygons 的并集，
距离幅值取到全部 raw rings 的最近距离；内部 overlap seam 保守归零。binary64
vertices 与 node coordinates 被解释为精确实数；segment
squared distance 和 point-in-polygon predicate 使用 exact rational arithmetic，
平方根以相邻 binary64 数形成外包区间。

node-label error 不吸收上游语义误差。完整账本仍为

\[
\varepsilon_{map}
=\varepsilon_{transform}
+\varepsilon_{polygon}
+\varepsilon_{dist}
+\varepsilon_{round}.
\]

R002E 已冻结 cache/source hashes、raw polygon validity、vehicle convention 与
rotation-free geometry arithmetic；R002F 已给出不依赖 libm 的确定性 transform
interval 与 \(\varepsilon_{transform}\)。32-ULP scorer-corner comparison 仍只作
兼容诊断。剩余项是实际 ROI/grid 与所有 node labels 的端到端 manifest，而不是新的
经验 tolerance。
