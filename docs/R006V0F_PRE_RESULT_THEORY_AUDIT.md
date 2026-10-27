# R006V0F 结果前理论审计：估计方差、折间异质与非单调任务秩

状态：`PRE_RESULT_INTERPRETATION_CONTRACT__NO_EMPIRICAL_CLAIM`。

本文在读取 R006V0F 正式结果前冻结，约束后续能够从 repeated-initialization 诊断推出什么、不能推出
什么。父实验 `STOP_GLOBAL_SHARED_INTERFACE` 不可变。

## 1. 科学对象与估计器必须分开

对 fold $f$、rank $r$，令总体任务残差目标为

\[
\mathcal L_{f,r}(P)
=\mathbb E_f\left[\|J_x(I-P)\delta_x\|_W^2\right],
\qquad P^2=P=P^\top,\quad \operatorname{rank}(P)=r.
\]

其总体最优集合记为 $\mathcal P^\star_{f,r}$，数值算法在 seed $s$ 下得到
$\widehat P_{f,r,s}$。观察到的跨折差异至少混合三项：

1. fold 分布改变导致的总体最优集合变化；
2. 有限 discovery/selection groups 的估计误差；
3. 非凸优化和初始化误差。

以 projector Frobenius 距离 $d(P,Q)=\|P-Q\|_F/\sqrt 2$ 为例，对任取
$P_f^\star\in\mathcal P^\star_{f,r}$、$P_g^\star\in\mathcal P^\star_{g,r}$，三角不等式给出

\[
d(\widehat P_{f,r,s},\widehat P_{g,r,t})
\le d(\widehat P_{f,r,s},P_f^\star)
+d(P_f^\star,P_g^\star)
+d(P_g^\star,\widehat P_{g,r,t}).
\]

因此单 seed 的跨折 projector 距离不能识别真实异质性。R006V0F 的 8 个 random-only seeds只用于
估计同一 fold/rank 的算法分散度；seed 不是独立场景样本，不能扩大统计显著性。

## 2. Task coverage 不具备 PCA 式自动单调性

对一个嵌套 projector 序列 $P_{r+1}=P_r+Q$，其中 $Q$ 投影到新增且与 $P_r$ 正交的方向，令

\[
a=J(I-P_r)\delta,\qquad b=JQ\delta.
\]

则增加 rank 后的残差变化精确为

\[
\|a-b\|_W^2-\|a\|_W^2
=\|b\|_W^2-2\langle a,b\rangle_W.
\]

故 rank 增加改善 coverage 的充要条件是

\[
2\langle a,b\rangle_W\ge \|b\|_W^2.
\]

该条件不由 $P_{r+1}\supset P_r$ 自动保证。直观上，新增投影可能删除一个原本在 planner 输出空间
抵消其他残差的分量，使 task residual 反而增大。于是：

- 本问题的 rank 不是普通 PCA explained-variance rank；
- 独立 rank 求解出现非单调 coverage 不必然是代码错误；
- 即使改用共享 ordered flag，也只能消除“不同 rank 学到无关子空间”的混淆，不能自动证明 task
  coverage 单调；
- “最小合格 rank”是冻结阈值下的操作性量，不是无条件内在维数。

后续若提出 ordered flag，必须显式报告每个前缀的 task residual，并将单调性作为可否定经验性质或在
额外的 pullback-orthogonality/commutation 假设下证明，不能沿用 PCA 直觉。
对应的三通道嵌套-projector反例与残差增量恒等式已固化为
`tests/test_r006_rank_geometry.py`，防止后续实现回退到未经证明的单调性假设。

## 3. R006V0F 三条路线的合法解释

### 3.1 `RANK_ESTIMATION_UNSTABLE`

只支持：冻结求解器、有限样本和阈值组成的 estimator 尚不能稳定决定资格。下一步只允许修正 estimator
或改用凸/谱 surrogate 做 matched diagnosis；禁止把 seed mode 当真值，也禁止训练 adaptive-rank reader。

### 3.2 `STABLE_FOLD_HETEROGENEOUS_EFFECTIVE_DIMENSION`

只支持：在冻结 estimator 下，各 fold 的阈值通过模式稳定但不同。它仍不证明：

- 存在由单样本因果输入可预测的“真实 rank”；
- rank 差异来自视觉场景而非有限 group distribution；
- 自适应 rank 会提升 planner 行为；
- 不同 fold 共享相同方向。

只有当低 rank 方向稳定、条件 covariate 在严格 group-held-out 上预测所需预算，并且 matched fixed-budget
基线被可靠击败，才允许把“条件化计算预算”提升为方法候选。

### 3.3 `INCONCLUSIVE_DIRECTION_OR_THRESHOLD_STRUCTURE`

不自动开启更复杂网络。下一步应优先检验 ordered flag、principal-angle 稳定性和 selection-group
敏感性；若跨折方向差异未显著超过 seed 内分散度，停止 global-basis 路线。

## 4. 对 Best Paper 主线的影响

R006V0F 不是论文效果实验，而是决定数学对象是否值得保留的 falsification gate。最强的合法价值是：

> 将“冻结 planner 的外观敏感性”分解为任务 pullback 几何、估计器稳定性和场景条件复杂度，而不是
> 把任意稳定 DINO 表示或固定低秩修复直接当成决策鲁棒性。

只有后续有限干预显示该结构具有 scene-specific causal mediation，DINO 又能组外恢复其系数，并且
容量匹配普通 adapter 不能复现收益，才有资格形成方法贡献。当前文件不提供这些经验结论。

## 5. 预注册 projector bootstrap 的实现缺口

结果前代码核验发现，冻结运行每个 fold/rank/seed 只保存一次完整 discovery 拟合所得 basis，并未保存
recording-group bootstrap 的重复拟合。对由同六折构成的全部 fold-pair overlap 直接 bootstrap 会把高度
相关的 pair 当成独立样本，属于伪重复；对 seed 重采样也只能描述优化分散度，不能替代 group bootstrap。

因此最终审计必须把 projector overlap 标为 `DESCRIPTIVE_ONLY`，把 group-level inference 标为
`UNAVAILABLE_NO_GROUP_LEVEL_REFITS`，且不得据此声称“跨折方向显著稳定/不稳定”或解锁条件化 basis。
如果该方向在 R006V0F 之后仍值得研究，必须另行预注册带真实 group-resample refit 的确认实验；不得把
事后 refit 合并进本轮正式结论。该缺口不改变 frozen rank-mode route，但收缩其后续含义。
