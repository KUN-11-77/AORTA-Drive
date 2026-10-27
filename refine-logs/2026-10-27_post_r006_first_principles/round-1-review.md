# Round 1 严格外审

审稿模型：`gpt-6-astra (xhigh)`；因 research-refine 指定的 GPT-5.4 在当前协作工具不可用，使用可用前沿模型替代。该审查不构成外部同行评议或科学证据。

## 一句话拒稿理由

当前方案从 Stiefel 估计器不稳定跳到标准最小范数逆问题，却尚未定位 DINO 证据在哪里丢失，也未证明在线 VJP 优于一阶等价的 direct-output residual。

## 核心意见

1. R006V0F 只证明当前固定-rank估计资格不稳定，不证明跨样本 row-space 旋转是原因；可以停止 fixed projector，但不能因此宣布 VJP 是答案。
2. clean planner output restoration 与 evaluator-aligned utility 不等价；必须把机制恢复、官方效用和安全主张分开。
3. weighted ridge 解正确，但其一阶输出效果为 Gram-filtered direct-output residual；VJP 必须击败这个等价对照才有必要性。
4. clean/shift/mismatch/removed interchange 很重要，但还需 full-token/no-compression readout，才能区分信息不足与 adapter 未读出。
5. planner-pullback functional invariance 比 VJP 更直接来自 Constant Eye 的缺口，但它很容易退化成普通 output consistency；必须先把 ordinary consistency 当强基线。

## 数学核验

在 \(A=W^{1/2}J\)、\(y=W^{1/2}d\)、\(\lambda>0\) 下，

\[
q^*=(J^\top WJ+\lambda I)^{-1}J^\top Wd
=A^\top(AA^\top+\lambda I)^{-1}y,
\]

且更精确地

\[
q^*\in\operatorname{range}(J^\top W^{1/2})
\subseteq\operatorname{range}(J^\top).
\]

若实际执行步长为 \(\eta\)，局部界应写为

\[
\|f(z+\eta\hat q)-o_c\|_W
\leq
\|\eta J\hat q-d\|_W+
\frac{L_J}{2}\eta^2\|\hat q\|_2^2.
\]

这些是标准正则逆问题与局部光滑性结论，不构成外观鲁棒性或安全定理。欧氏“最小能量”还依赖 activation 坐标尺度和人为度量。

## 建议实验顺序

1. **D1 接口交换审计：**clean / shifted / geometry-matched mismatch / removed DINO tokens；固定历史、其他输入与随机性；同时交换 adapter 后表示与完整因果切口。
2. **D2 信息保留定位：**full tokens、compressed tokens、compressed+full tokens 对同一功能目标的组外 probe；correct/mismatch/removed 全部保留。
3. **D3 单一方法：**若发现接口放大，先比较 functional consistency adapter；只有证明局部 task restriction 有额外收益，才解锁 VJP。

## 七维评分（Round 0）

| 维度 | 分数/10 |
|---|---:|
| Problem clarity | 7 |
| Novelty | 3 |
| Mathematical correctness | 7 |
| Method necessity | 2 |
| Experimental falsifiability | 8 |
| Simplicity | 6 |
| Best-paper story | 3 |

## 必须吸收的修改

- 正式终止 low-rank/Fantope，不再阻塞 DINO 主线；
- 把 DINO interface causal audit 放在任何新架构前；
- 将 VJP 降为条件候选；
- ordinary functional/output consistency、full-token reader、direct-output residual 都作为强基线；
- 只有“广泛机制发现 + 单一不可替代结构 + 行为闭环”同时成立，才恢复 Best-Paper 主张。
