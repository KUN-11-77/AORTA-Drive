# AORTA-Drive Research Brief

> **2026-10-24 PEA-derived method decision:** the audited PEA failure is treated as evidence that unsigned source-moment alignment can move task-critical directions, not as evidence that all adaptation is useless. A task-constrained PEA is specified only as a conditional backup: freeze the planner candidates, project a PEA proposal through decision-margin constraints, and abstain when a lower-bound improvement is unavailable. It is not authorized for implementation before R004H6 and the read-only H7 decision-relevance diagnosis; the single primary thesis remains decision-quotient robustness. Full specification: `refine-logs/2026-10-24_aorta_iccv_positioning/PEA_DERIVED_METHOD_DECISION_CN.md`.

> **2026-10-24 ICCV baseline/novelty audit:** FlowDrive and READ are now mandatory closest-method references, but their published headline scores are not treated as direct wins/losses unless sensor, backbone, data, candidate, and scorer contracts are aligned. PEA is eligible only as a frozen driving-adapted TTA baseline; the current AutoVLA and ReCogDrive MG results remain respectively `test-time core adaptation` and `MG-style adaptation`, not strict official MG-Select reproductions. The provisional method thesis moves from graph quotient alone to a decision quotient defined by frozen candidate contrasts; this is not a qualified contribution until R004H6 passes and R004H7 shows that decision-contrast error predicts and reduces ranking flips beyond field MAE, SDF/BEV, direct edge + projection, direct pair, FlowDrive-style, and READ-style controls. Full audit: `refine-logs/2026-10-24_aorta_iccv_positioning/BASELINE_AND_NOVELTY_AUDIT_CN.md`.

> 状态：AORTA-v8.3.2 V0 的 Gate A 与 R004E-A/B/C 已完成。R004H5 首次给出经过机械复算的正向 fit 视觉机制证据：42 groups/672 anchors 全部完成，learned task MAE 约0.324m、zero 0.550m、teacher cosine约0.771，三种受控外观漂移约0.0087–0.0102m；一次性审查中 learned 在42/42组上优于 leave-one-group-out teacher prior。该结果只支持受控 fit 机制。R004H4P5R 在预注册30分钟上限内未完成首个哨兵，当前严格几何迭代因 computability 停止且无科学几何结果。下一步只允许冻结模型的独立视觉泛化资格实验；决策收益保持 HOLD。development、official calibration、test 与 target 仍封存，正式 coverage、真实天气/OOD、驾驶收益、Gate B 与创新性未成立。

## 1. 问题锚点

AORTA-Drive 研究 NAVSIM v1/v2 上的 source-only appearance robustness。对同一物理场景与驾驶意图，外观变化可能改变冻结端到端 planner 的输出，却不应改变道路可行结构。

核心问题是：

> 能否学习一个与候选无关的道路差分场，在绝对安全零点不可靠时，为任意冻结 planner 的 challenger 相对 native 提供 nominal road-margin non-degradation 证书，并只在证书成立时干预？

## 2. 核心假设

完整道路 signed-distance field 的共同 offset 可能在视觉退化下不可辨识，但局部 clearance differences 可能仍可学习。若 source session 上的 graph-gradient residual 能被同时校准，则一个 field coverage event 可以覆盖任务图内所有自适应 swept-set pairs，并经 challenger 外覆盖/native 内点的非对称几何合同，认证每次授权干预达到冻结的相对改进阈值 \(\gamma_{task}\ge0\)。

该假设可能失败；遮挡、全局 calibration maximum、长路径累积和图错设都可能令证书失去非平凡性。即使证书有效，冻结 candidate provider 也可能没有提供更优候选；candidate supply 与 certificate recall 必须分别报告。

## 3. 冻结范围

- NAVSIM v1/v2；
- frozen DINOv3-H+ 与被审计 planner；
- source-only train/development/calibration；
- 一个具有冻结多相机节点投影/缺失证据协议的 candidate-free node-potential decoder；
- V0 fixed physical scale control 与 v8.4 candidate-independent observation-and-edge scale candidate、single native、exact multi-source solver；
- 预注册 ego-centric ROI、grid、continuous teacher/map error ledger、geometry checker、candidate provider、资格层与 session time set；
- 默认保留 native，仅认证 relative nominal-road non-degradation。

## 4. Non-goals

- 不恢复完整 HD map 或绝对道路安全零点；
- 不联合训练或修改原 planner；
- 不把道路 margin 等同于碰撞、规则、舒适性、真实执行或完整闭环安全；
- 不把 source conformal coverage 写成任意 appearance OOD 保证；
- 不把 DINO、SDF、split conformal、difference constraints、Dijkstra 或 min-plus transform 单独宣称为创新；
- 不在理论/idea 阶段启动 pilot 或长跑实验。

## 5. V0 条件理论链

\[
\text{source-session edge event}
\Rightarrow
\text{all-pair discrete robust bounds}
\Rightarrow
\text{map/geometry/numerical correction}
\Rightarrow
\text{all authorized interventions reach }\gamma_{task}.
\]

既有观察只支持继续研究任务空间接口，不证明 v8.3.2 的局部差分假设、校准锐度、系统收益或创新性。

## 6. 验收门

1. **Gate A:** 数学 oracle、multi-source 实现、geometry checker 和数值误差方向全部通过；
2. **Gate B:** 合法 source-only predictor 产生可接受的 session-tail residual 与非平凡米制证书，并分离 candidate supply、certificate recall 与 selection loss；
3. **Gate C:** 收益存在于具有连续几何意义的 non-inclusion candidates；
4. **Gate D:** 在同一 field event 下迁移到未见 candidate policy/frozen planner；
5. 完成 uncertainty-only 与 end-to-end 两层公平 baseline；
6. 完成独立 novelty-check。

Gate A/B 失败则不进入完整评测；公平 baseline 或查新失败则降级论文定位，不追加模块挽救主张。

## 7. 当前裁定

- V0 条件理论规范：可冻结；
- Gate A primitive/生成器：通过；R004 anchor-aligned teacher instantiation：R004E-A 通过；
- 实际系统：未验证；
- 顶会方法原创性：未授权；
- 下一阶段 A（独立视觉泛化资格）：冻结 R004H5 的 checkpoint、外观算子、metrics 与阈值，只允许在预注册的未见 source recording groups 上一次性检验任务保持；同时冻结错配图像/场景置乱与输入/DINO 扰动强度账本。不得用该阶段调模型，也不得把受控外观命名为真实天气。
- 下一阶段 B（决策收益）：HOLD。R004H4P5R 已在30分钟资源上限内停止且没有完整哨兵结果；当前不恢复细网格、候选修改或128-anchor扩展。只有独立视觉泛化与可计算的决策接口分别闭合后，才注册驾驶收益实验。
