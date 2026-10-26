# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem:** 在不重训原 planner 的条件下，识别并修复外观变化真正破坏的决策相关视觉信息。
- **Must-solve bottleneck:** DINOv3-H+ 已是经 Constant Eye 验证的稳定底座，但稳定不等于 planner 所需信息可识别或可用。
- **Non-goals:** 不恢复完整 RGB；不强制全特征不变；不重新训练 planner；不把 output-to-clean 等同于安全。
- **Constraints:** NAVSIM v1/v2、source-only 开发、DINO/planner 冻结、NavDream 在架构冻结后一次性验证。
- **Success condition:** 低秩内部机制、DINO可恢复性、matched方法优势和行为闭环依次成立。

## Anchor Check

- 原始瓶颈未改变：仍是 frozen planner 的 appearance robustness。
- 修订删除了空泛的一般充分性主张，改为 appearance-effect mediation。
- 未引入 VLM、RL、世界模型或并列贡献。

## Simplicity Check

- 单一贡献：causally qualified planner-tangent repair。
- 单一新训练组件：spatial coefficient reader。
- full patch、Jacobian、DINO、adapter 都是工具/基线，不是贡献。
- uncertainty 继续 HOLD。

## Changes Made

1. 用 task-weighted appearance drift objective 直接定义 \(B_r^*\)，删除 \(C_A/C_T/C_M\) 启发式拼接。
2. 新增 reader error + basis uncovered error + nonlinear remainder 的方法特有界。
3. 对空间特征使用 `I_spatial ⊗ BB^T`，reader 预测 per-location coefficient field，避免巨大任意空间基。
4. R006V0 改为 6-fold nested group-heldout；full patch降为 smoke；加入双向、dose、off-manifold、worst-group 和 rank–coverage。
5. DINO 资格改为条件 Bayes-risk recoverability，不声称 finite experiment 证明严格可识别性。
6. matched baseline 加入 direct-output residual；Constant Eye 分为 faithful 与 frozen-tail 两组公平比较。

## Revised Proposal

### Thesis

外观变化引起的大部分 feature drift 与规划行为无关。AORTA 先在组外数据中定位并验证低秩、planner-sensitive 的内部介导部分，再让冻结 DINO 只恢复该部分，从而在 planner/head 全冻结时实现最小视觉修复。

### Mathematical object

\[
B_r^*=\arg\min_{B^TB=I_r}\mathbb E\|J_i(I-BB^T)(z_{c,i}-z_{a,i})\|_W^2.
\]

空间接口用 \(I_P\otimes BB^T\)。目标系数为 \(A_i^*=\Delta Z_iB_r^*\)，reader 预测 \(\widehat A_i\)，输出 \(\widehat Z_i=Z_{a,i}+\widehat A_iB_r^{*T}\)。

### Method-specific bound

行为误差由 reader coefficient error、basis uncovered planner-sensitive drift 与 Hessian remainder 三项共同控制。若候选分数无穷范数误差小于 clean top margin 的一半，则 clean winner 保持；official utility 与 safety 仍需独立实验。

### Qualification

1. R006V0：nested OOF low-rank internal mediation；
2. R006V1：DINO correct/mismatch/removed/native-only recoverability；
3. R006V2：CPTR vs Constant Eye/plain adapter/full-feature/direct-output；
4. R006V3：NavDream、三 planner、clean、latency、failure boundaries。

### Falsification

- global low-rank coverage 不合格：停止共享 CPTR；
- DINO 无条件增量：停止 DINO lift；
- direct-output/plain adapter 同样有效：删除内部修复创新主张；
- output restoration 无 utility gain：停止行为保持外推。
