# H7BS4B3 后理论裁决：从 scalar minimum 退回决策对象重定义

状态：`GENERAL_ROAD_MARGIN_RERANKER_STOPPED__NEW_OBJECT_REQUIRES_LITERATURE_AND_ORACLE_REVIEW`。

## 1. 被否定的主张

B3 否定的不是视觉 backbone，而是以下方法链在当前冻结候选面板上的一般适用性：

\[
\text{dense road margin}
\rightarrow \text{global/fixed-suffix/temporal-suffix minimum}
\rightarrow \text{general candidate ranking}.
\]

即便输入是 perfect teacher，temporal fork 也只恢复 170/2,033 个原本被共同 minimum 湮灭的 pair；
因此不能再把 learned network 的失败风险归因于视觉误差，也不能用新 loss 解决算子已删除的信息。

## 2. 为什么 temporal fork 仍会坍缩

设 fork 后两个 suffix 的空间支持为 (B_i,B_j)，再分解

\[
C^+=B_i\cap B_j,\qquad A_i=B_i\setminus C^+,\qquad A_j=B_j\setminus C^+.
\]

则 temporal-suffix contrast 仍为

\[
\min(c^+,a_j)-\min(c^+,a_i).
\]

因此移除早期共同 prefix 不是充分条件；只要分叉后仍有共同/重访区域控制 minimum，零差异定理
再次成立。B3 观察到 1,863/2,033 temporal contrasts 仍为 exact zero，说明该失败在当前 panel 上
不是边缘现象。该计数证明“仍坍缩”，但未单独完成 1,863 个 pair 的新互斥原因分类，原因归因
保持为数学解释而非额外实证主张。

## 3. 为什么 spatial exclusive 也不是现成答案

空间独占 operator

\[
D^{ex}_{ij}(d)=\min_{A_j}d-\min_{A_i}d
\]

在两侧非空时恢复 1,568/1,568 个非零差异；但 465/2,033 pairs 至少一侧为空。这说明它通过
**缩小定义域**换取区分能力。对 nested supports，某条候选没有独占区域，式子本身无定义；
删掉这些 pairs 会改变研究对象并造成 complete-case bias。

因此 B3 的准确结论是：spatial exclusive 可能是 selective diagnostic，却不是一般 panel operator。

## 4. 对原 AORTA “漂亮建筑”的结构修正

现在必须删除两根未经支持的梁：

1. B1 已否定 joint graph uncertainty 的决策必要性；
2. B3 已否定 fork-aware scalar minimum 作为一般 reranker。

仍然保留的地基只有：

- 同一场景外观变化下，held-out visual quotient field 具有非平凡任务信息与 correct-image necessity；
- 完整 4 秒候选域、视觉支持缺失和短前缀不可区分问题均已被机械定位；
- common-offset quotient、Hodge integrability 与 native-preserving relative reasoning 的数学接口成立；
- 但从稳定视觉场到一般驾驶候选排序的**充分统计量尚未找到**。

这比继续堆网络更接近顶会标准：失败的结构退出主方法，不能因故事好讲而保留。

## 5. 下一候选对象（仅研究假设，不授权实现）

一个值得文献查新与 oracle 审查的方向是把轨迹看成归一化时空 occupancy measure
(\mu_\tau)，并用 signed contrast measure

\[
\nu_{ij}=\mu_j-\mu_i,\qquad
S_{ij}(u)=\langle u,\nu_{ij}\rangle.
\]

由于 (\nu_{ij}(\Omega)=0)，有

\[
S_{ij}(u+c)=S_{ij}(u),
\]

而共同 occupancy mass 在 signed measure 中自动抵消；nested support 仍有定义。这给出比 set-min
更自然的 common-mass cancellation 与 gauge invariance。但它只是平均/加权 road evidence，可能
忽略局部灾难性 minimum，也尚未证明与 planner outcome 有关。因此它当前不能成为 AORTA-v9，
更不能立刻训练。

下一步必须先回答：

1. 该 signed-measure operator 与 trajectory cost、risk field、occupancy flow、CVaR/entropic risk、
   direct candidate scoring 的已有工作有何区别；
2. 哪种风险泛函同时满足 common-mass cancellation、nested-support totality、局部灾难敏感性与
   可校准误差界；
3. 在合法 outcome 协议下，它是否比 dense field MAE、direct pair head 和简单 trajectory cost
   更贴近 ranking flip/regret；
4. 单次视觉前向加候选读出的 P50/P95/P99 时延是否满足预算。

在完成这四项理论/查新门之前，只保留为候选，不启动新架构搜索。

## 6. 顶会故事边界

当前最诚实且有 insight 的一句话不是“我们已经有 fork-aware AORTA”，而是：

> 我们发现，外观稳定且任务正确的 dense road field 仍可能在候选读出阶段丢失几乎全部决策差异；
> 这种失败在 perfect teacher 上仍存在，因此视觉鲁棒性必须定义在决策充分的相对算子上，而不是
> feature invariance 或任意 field accuracy 上。

这已经是有价值的问题发现，但要成为 ICCV 方法论文，还缺一个通过 oracle、匹配容量消融、
NavDream/跨 planner 与时延门的新算子。不得把负结果本身包装成已完成方法贡献。
