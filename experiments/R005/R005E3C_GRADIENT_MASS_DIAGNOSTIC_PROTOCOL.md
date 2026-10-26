# R005E3C：机会稀疏与初始梯度质量只读诊断

日期：2026-10-26
状态：`IMPLEMENTATION_QUALIFIED__READ_ONLY_RUN_AUTHORIZED__NO_WEIGHT_UPDATE`

## 1. 问题

R005E3B删除`tanh`后仍把全部challenger压入低概率尾部。当前仍需区分：

1. 正收益challenger是否过于稀疏；
2. soft-regret在初始化处是否已经给出正确的score级方向；
3. 经共享网络Jacobian映射后，正收益与非正收益challenger的参数梯度是否严重失衡或冲突；
4. 最终近零纠错梯度是否只是score尾部现象，还是初始化处已经没有有效信号。

本轮不更新任何权重，不训练新模型，不改变阈值，也不读取封存角色。

## 2. 冻结对象

- source-fit fold0 outer-train，35 groups×16 anchors；
- seed `20261025`及R005E3B相同初始化；
- identity relative score、正确视觉、`T=0.125`；
- R005E3B primary的最终prediction只作前后score梯度比较；
- development/calibration/test/NavDream/target禁止访问。

## 3. 数学分解

对每个challenger (k)，score梯度为

\[
g_k^s=\frac{p_k}{T}(\mathbb E_p[u]-u_k).
\]

以native-relative utility是否严格为正划分

\[
P=\{k:u_k>u_0\},\qquad N=\{k:u_k\le u_0\},
\]

并在同一初始化、eval-mode Jacobian下定义参数梯度分量

\[
g_P^\theta=\sum J_k^\top g_k^s\mathbf1[k\in P],
\qquad
g_N^\theta=\sum J_k^\top g_k^s\mathbf1[k\in N].
\]

报告`‖g_P‖`、`‖g_N‖`、范数比及余弦。这里的eval-mode分解是确定性局部几何诊断，不等于
含dropout与逐step参数变化的完整训练动力学。

## 4. 固定输出

- 机会anchor比例、正收益challenger比例；
- 正/非正集合的utility、softmax概率、score-gradient有符号及绝对质量；
- 正收益项得到“提高score”方向的比例；
- 非正收益项得到“降低score”方向的比例；
- 初始化处`g_P^theta`与`g_N^theta`的范数、比率和cosine；
- 初始化到E3B最终的score-gradient质量变化；
- 模型state前后SHA完全相同、primary/replay输入SHA与角色封存账本。

## 5. 解释边界

本轮是描述性、只读机制诊断，不设事后性能阈值，也不产生新方法PASS。它只能帮助选择下一个可证伪
假设：

- 若score方向正确但最终概率尾部使正机会梯度巨幅衰减，支持检验不依赖候选当前softmax概率的
  pairwise纠错surrogate；
- 若初始化参数梯度已由非正机会主导或与正机会强冲突，后续surrogate必须显式处理utility gap与pair
  平衡，而不是只调温度；
- 若正机会本身极少，任何平均loss都需要把机会采样/加权作为独立因素，不能与架构改动混合；
- 任一观察都不证明视觉scene binding、OOF泛化、Differential Query或驾驶收益。

诊断完成后，是否运行新训练仍须另行预注册；本协议本身不授权训练。

## 6. 实现资格

- `scripts/diagnose_r005e3b_gradient_mass.py` 已通过定向测试、严格类型检查与全仓库测试；
- primary 与 same-GPU replay 必须除完成时间外逐字段一致；
- `scripts/audit_r005e3c_gradient_mass.py` 必须核验输入、runner、commit、GPU、有限值、零权重更新和
  封存角色；
- 机械通过仍只产生描述性诊断，不自动授权任何训练。

## 7. 结果后闭环（不改写预注册）

正式运行、same-GPU replay与强化版独立复算均完成。560个outer-train anchors中236个存在正收益
challenger，3235个合法challengers中623个为正收益。初始化处正收益项的raise方向比例为94.54%，
并非100%；正/非正机会的聚合参数梯度余弦为-0.99999，非正梯度范数约为正梯度的2.668倍。
训练终点遗漏正收益候选的score-gradient绝对均值从0.1429降至约1.70e-9，衰减约8.42e7倍。

这些结果支持“当前soft-regret训练出现概率尾部纠错衰减和强梯度冲突”的描述，不构成单一根因的因果
证明。预注册中的pairwise建议经理论反例审查后被撤回：逐样本错排界不推出一般多候选总体Bayes排序
一致性。下一处理改由结果读取前冻结的统计原则决定，即identity输出加native-relative proper mean
regression；E3C本身不授权、也不选择该处理。

外部完整性审查裁决为`WARN`：科学数字与最终score-gradient可独立复算，未发现伪造或归一化问题；但
紧凑归档未保存初始化预测和参数梯度张量，故初始化梯度几何只能检查字段与数学自洽性，不能从归档
独立重算。所有结论继续限于source-fit fold-0 outer-train只读优化诊断。

