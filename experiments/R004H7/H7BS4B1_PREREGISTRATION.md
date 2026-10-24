# R004H7-BS4B1：Oracle Lipschitz 联合不确定集合非空洞性

状态：`PREREGISTERED_NOT_RUN`。本实验不训练模型，也不读取 learned field、DINO、planner
outcome 或驾驶分数。目标是先判断：在当前视觉支持拓扑下，即使视觉 Oracle 在所有有证据节点
提供正确 teacher 区间，保留图相关性的联合集合是否比匹配的边缘区间更可裁决。

## 1. 集合定义

对 projectively supported 节点 (o\in O)，只揭示 B0A2 的外向 teacher 区间

\[
\ell_o\le d_o\le r_o.
\]

对相邻网格节点 (p,q)，使用 signed-distance 的 1-Lipschitz 结构和两端 teacher 误差账本：

\[
|d_p-d_q|\le 1\mathrm m+\varepsilon_p+\varepsilon_q.
\]

由此定义共享场集合

\[
\mathcal U_{\mathrm{joint}}
=\{d:\ell_o\le d_o\le r_o,\ o\in O;
|d_p-d_q|\le L_{pq},\ (p,q)\in E\}.
\]

hidden teacher 值不得用于集合构造；它们只在集合构造完成后检查包含性和候选 contrast。
实现使用 observed upper intervals 的 McShane 型上延拓作为可行 potential，再把一般 difference
constraints 做 Johnson reweighting；任何所需 binary64 widening 单独记账。

## 2. 三个同覆盖对照

1. `joint_lipschitz`：保留所有 grid edge 和 shared-node 相关性；
2. `shared_node_box`：从 joint set 推出每个节点的精确 marginal interval，但删除节点间 edge
   相关性；native 与 challenger 的公共节点仍是同一变量；
3. `candidate_independent`：使用同一 marginal intervals，但在两个候选间复制变量，连公共节点
   相关性也删除。

三者都必须包含冻结 teacher；改变的只有依赖结构，不改变观测、teacher、候选或 coverage。
因此应有集合包含链

\[
\mathcal U_{\mathrm{joint}}
\subseteq\mathcal U_{\mathrm{box}}
\subseteq\mathcal U_{\mathrm{candidate\text{-}independent}},
\]

以及相应 robust contrast interval 的反向嵌套。任何超出 (10^{-9}\) m 数值账本的违反都使实验
无效，而不是科学失败。

## 3. 精确比较算子

对 directed residual constraints (x_v-x_u\le c_{uv}\)，给定可行 potential (h)，定义

\[
\underline\Delta_{A,B}
=\inf_x\left[\min_{a\in A}(h_a+x_a)-\min_{b\in B}(h_b+x_b)\right].
\]

若 (operatorname{dist}_c(a,b)) 是 constraint graph 的 directed shortest-path distance，则

\[
\boxed{
\underline\Delta_{A,B}
=\min_{a\in A}
\left[h_a-\min_{b\in B}\left(h_b+\operatorname{dist}_c(a,b)\right)\right]
}.
\]

证明要点：约束给出
(x_a-x_b\ge-\operatorname{dist}_c(a,b))；固定 (a) 时取
(x_v=\operatorname{dist}_c(a,v)) 可同时达到全部下界，再对有限 (A) 取最小。实现已通过
随机 directed graph 与独立 LP 的逐例比较。upper bound 由交换 (A,B) 后取负得到。

## 4. 冻结指标与裁决

完整评估 42 groups、672 anchors、3,864 native/challenger pairs。每个 pair 报告 robust interval、
interval width、sign identifiability 与 teacher containment；每个 anchor 报告是否至少一个 pair
可辨识、是否授权正向替换和 teacher regret。

`PASS` 必须同时满足：

- teacher inclusion、contrast containment、两级集合嵌套和 oracle intervention harm 均零失败；
- 42/42 groups 各至少 12/16 anchors 有一个 joint-identifiable pair；
- 以 42 groups 为独立单位，10,000 次 paired bootstrap 后，joint 相对 shared-node-box 的
  identifiable-pair fraction 差异 95% CI 下界严格大于 0。

如果 Oracle joint set 都达不到组级 12/16，裁决为接口信息不足，不允许用更大网络挽救；如果
joint 与 shared box 没有严格差异，则 graph correlation 不具备实验必要性，应简化为共享节点
区间。通过也只说明该结构在 fit teacher Oracle 上非空洞，不授权 learned network、OOD 或驾驶主张。
