# Round 1 Refinement：DINO Interface Mechanism First

状态：`REFINED_AFTER_REVIEW__NO_METHOD_EVIDENCE`。

## 修改原则

1. DINOv3-H+ 固定为共同视觉底座，不再重复验证其一般稳定性；
2. low-rank/Fantope/CPTR 正式停止；
3. VJP 从主方法降为只有额外必要性证据才允许解锁的候选；
4. 先定位 DINO-to-planner 接口是放大、压缩、信息不足还是场景先验；
5. 方法只允许由唯一机制裁决自然导出。

## 第一性对象

设冻结 DINO 为 \(D^a=\Phi(g(S,a))\)，DINO 接口为 \(A_\theta\)，冻结 planner tail 为 \(T\)：

\[
O_\theta^a=T(A_\theta(D^a),H).
\]

对合法 appearance orbit \(\mathcal O(S)=\{g(S,a)\}\)，理想条件不是 \(D^a\) 完全相等，而是

\[
O_\theta^a=O_\theta^{a'}
\quad \forall a,a'\in\mathcal A_{legal},
\]

同时保留不同物理场景的任务可分性。由路径积分，

\[
O_\theta^{a'}-O_\theta^a
=\int_0^1J_{T\circ A_\theta}(D_t)(D^{a'}-D^a)\,dt.
\]

因此 DINO feature drift 小不推出 planner drift 小；真正对象是组合接口沿 appearance orbit 的功能增益。

## 一次性机制审计

冻结 DINO、planner、历史和随机性，执行：

- clean / shifted / geometry-matched mismatch / removed DINO token interchange；
- adapter 后 representation interchange；
- full-token、compressed-token、compressed+full-token 的容量匹配组外 readout；
- actual finite output drift、线性预测误差和分层 token dose-response。

唯一裁决：

| 结果 | 解释 | 后续 |
|---|---|---|
| clean full-token 无恢复上界 | DINO-only 信息/接口不足 | STOP DINO-only |
| correct 不优于 mismatch/removed | 场景视觉依赖不成立 | STOP visual mechanism |
| full-token 可恢复、compressed 失败 | generic compression 丢任务证据 | GO full-token task bridge |
| DINO drift 小、接口后/输出 drift 大，interchange 可逆转 | interface amplification | GO functional-orbit constraint |
| ordinary consistency 已闭合 | 无需新复杂算子 | 保留简单方法，删除 VJP |

## 条件主方法：Functional Quotient Bridge

仅在“压缩损失”或“接口放大”至少一项通过时解锁。对冻结 planner 的原生 query/interface states \(Q\) 和完整 DINO tokens \(D\)，使用一个共享 query-conditioned residual bridge：

\[
R_\theta(Q,D,H)=g_\theta(Q,D,H)\odot
\operatorname{CrossAttn}_\theta(Q,D,D),
\]

\[
\widehat Z=Z_{native}+R_\theta(Q,D,H),
\qquad
\widehat O=T(\widehat Z,H).
\]

planner 与 DINO 全冻结。Bridge 不先把 DINO 压成固定通用 grid；原生 planner query 决定读取哪些视觉证据。若同容量 full-token baseline 已存在，则 Proposed 的唯一额外结构是 functional-orbit training contract，不能把 cross-attention 宣称为创新。

训练目标使用 source-only、与 NavDream 风格族不重叠的 appearance interventions：

\[
\mathcal L=
\mathcal L_{task}^{clean}
+\lambda_o\mathbb E_a
\|T(\widehat Z^a,H)-\operatorname{sg}(O_{ref}^{clean})\|_W^2
+\lambda_{id}\|R_\theta(Q,D^{clean},H)\|_2^2
+\lambda_r\|R_\theta(Q,D^a,H)\|_2^2.
\]

- 第一项保持任务正确性；
- 第二项只约束完整冻结 planner 的功能输出，而不是强制全部 feature 相等；
- 第三项直接约束 clean no-harm；
- 第四项限制不必要干预。

ordinary output consistency 是必须击败的强基线。若上述 loss 只是普通 consistency 的改写且没有 matched 增益，论文不声称新算法。

## VJP 的地位

只有当实验进一步证明：相同 reader 的无约束 residual 出现可重复 clean harm，而局部 task-shaped restriction 在相同输出误差和计算预算下改善组外行为，才比较 VJP。否则它留在理论附录，不进入系统。

## 三项证据闭环

1. **发现：**跨 planner 证明稳定 DINO 经接口发生可重复的任务证据放大/压缩；
2. **机制与方法：**Functional Quotient Bridge 在完全相同 DINO 上击败 Constant Eye、full-token普通 reader、matched adapter、ordinary consistency、direct-output residual；
3. **行为：**NavDream zero-shot、多 planner、clean non-inferiority、hard-failure净减少和 batch-1 P95 同时通过。

任一项失败即降低主张或停止对应分支。
