# Research Proposal：Causal Planner-Tangent Repair

> 历史初稿：已被 `round-1-refinement.md` 与 `FINAL_PROPOSAL.md` 取代，仅用于保留 refinement 轨迹，不是 canonical 方案。

## Problem Anchor

- **Bottom-line problem:** 在不重训原 planner 的条件下，识别并修复外观变化真正破坏的决策相关视觉信息。
- **Must-solve bottleneck:** Constant Eye 已证明 DINOv3-H+ 是有用的稳定视觉底座，但其 per-planner adapter/head retraining 不解释机制，也不能证明稳定表示为何、何时改善决策。
- **Non-goals:** 不重建完整场景，不强制全特征不变，不把 DINO、adapter、Jacobian 或 activation patching 单独称为创新，不把接近 clean trajectory 等同于安全。
- **Constraints:** NAVSIM v1/v2；source-only 方法开发；原 planner 与 DINO 冻结；最终必须 clean no-harm、NavDream、跨 planner、batch-1 时延闭合。
- **Success condition:** 先以受控干预证明某个低复杂度 planner-sensitive 子空间介导外观失效，再证明 DINO 对其有不可替代的 held-out 信息，最后用最小低秩修复优于 Constant Eye 与容量匹配普通 adapter。

## Technical Gap

Constant Eye 解决“用更稳定的 backbone 接口重新训练 planner”问题，但没有回答：原生 planner 的失败由哪个内部变量介导；DINO 与原生表示是否可识别对齐；为什么修复 feature 能改善效用；是否必须重训规划头。全局 invariance 既可能删除任务信息，也可能在 planner null space 上浪费容量。

## Method Thesis

- **One-sentence thesis:** 外观鲁棒驾驶不需要恢复全部 clean feature，而应先用 causal interchange intervention 定位被外观扰动且会影响冻结 planner 的低秩切空间，再用冻结 DINO 证据只预测该空间中的最小残差。
- **Smallest adequate intervention:** 一个资格化低秩基和一个 coefficient reader；planner/DINO 全冻结。
- **Frontier fit:** DINO 是现成 foundation evidence，不是贡献；贡献是将稳定 evidence 限制到因果定位、planner-sensitive 的修复接口。

## Contribution Focus

- **Dominant contribution:** causally localized planner-tangent minimal repair。
- **Supporting contribution:** sufficient–identifiable–behavior-preserving 三门理论合同。
- **Explicit non-contributions:** DINOv3、SVD/Jacobian、adapter、activation patching、信息瓶颈、Lipschitz bound、NavDream。

## Proposed Method

### Complexity Budget

- Frozen/reused: native planner、DINOv3-H+、candidate generator、evaluator。
- New trainable component: one DINO/native/context-to-coefficient reader。
- Conditional only: uncertainty/abstention head，只有可识别性失败证据授权。
- Excluded: 新 backbone、世界模型、diffusion/RL、全场景重建、多头并列贡献。

### System Overview

1. 在 source same-scene clean/shift pairs 上对 frozen planner 做 layerwise interchange intervention；
2. 由 appearance covariance 与 planner pullback metric 提出候选低秩子空间；
3. 用 wrong-scene/random/noise/reverse 控制资格化该子空间；
4. 用 DINO correct/mismatch/removed probe 验证目标可识别性；
5. 训练 coefficient reader，输出 \(\widehat z_a=z_a+B\alpha\)；
6. 原 planner tail 原样推理。

### Core Mechanism

Planner metric \(G=J^TWJ\) 删除局部行为 null space；appearance covariance 删除未受 shift 影响方向；causal patch 删除仅相关但不介导的方向。最终网络只预测由三者共同资格化的系数。

### Training Signal

主损失为 frozen planner functional restoration；加 task-subspace coefficient、minimum-energy 和 clean-identity 约束。完整 feature L2 只作为 baseline，不承担核心目标。

### Failure Handling

- full clean patch 无恢复：停止 feature repair；
- correct DINO 不优于 mismatch/removed：停止 DINO lift 或引入合法时序证据，不扩网络；
- repair 接近 clean 但 utility 不升：停止“behavior restoration=task improvement”桥；
- clean regression：加强 identity/no-harm 或拒判，但不得事后调 NavDream 阈值。

## Novelty and Elegance Argument

相对 Constant Eye，方法不重训 planner/head，而是先证明需要修复的内部机制，再进行受限低秩修复。相对普通 adapter，新增自由度更少且每个方向有 planner sensitivity、appearance exposure 与 causal mediation 三重依据。相对一般 activation patching，目标不是解释模型，而是将干预发现转化为 frozen-policy repair，并以 DINO 仅作为可识别证据。

## Claim-Driven Validation Sketch

### Claim 1：外观失败由低秩 planner-sensitive 子空间介导

- Minimal experiment: R006V0 same-scene clean patch 与五类控制；
- Metric: group-level raw restoration、control gap、worst-group；
- Falsifier: full patch不恢复或wrong-scene/random同样恢复。

### Claim 2：稳定 DINO 证据可以恢复该子空间，最小修复优于普通 adapter

- Minimal experiment: group-held-out correct/mismatch/removed probe；随后 matched CPTR vs Constant Eye/plain adapter/full-feature regression；
- Metric: target-subspace risk、functional restoration、official utility、clean no-harm、NavDream、latency；
- Falsifier: correct evidence无增量，或matched adapter同样有效。

## Experiment Handoff Inputs

- Must-prove: causal mediation、DINO identifiability、matched architecture advantage、behavior closure。
- Highest-risk assumption: frozen DINO 与 native planner 的 task statistic 可识别对齐。
- First run: zero-training R006V0，禁止先训练最终网络。
