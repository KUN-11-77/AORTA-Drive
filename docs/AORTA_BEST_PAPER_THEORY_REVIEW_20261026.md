# AORTA Visual-First 理论体系 Best-Paper 级审查

日期：2026-10-26  
裁决：`CONDITIONALLY_COHERENT__NO_REDESIGN__EMPIRICAL_CORE_UNPROVEN`

## 1. 总结裁决

现有理论不需要推倒重来。它已经形成一条可证伪、不会把数学工具冒充实验事实的主链：

\[
\text{配对外观效应}
\rightarrow
\text{planner pullback geometry}
\rightarrow
\text{组外内部介导}
\rightarrow
\text{DINO增量可恢复性}
\rightarrow
\text{受限修复}
\rightarrow
\text{OOD行为与时延}.
\]

但它目前只是**有资格被严格检验的条件理论**，不是已经成立的 Best-Paper 理论。Best-Paper 潜力取决于
三个尚未获得的经验事实：跨场景共享低通道秩介导确实存在；正确视觉证据能够组外恢复该机制；受限修复
在相同输入、参数、FLOPs 与训练预算下可靠优于 Constant Eye、普通 adapter 和 direct-output。

本轮没有改变正在运行的 R006V0 数据、层、rank 字典、阈值、求解器或裁决，只修复理论表达中会引起
错误证明或过度主张的地方。

一致性核验：row-vectorized selector/Kronecker 恒等式的数值误差为 `0` 与 `2.22e-16`；
`test_planner_tangent`、`test_r006_basis`、`test_r006_basis_audit`、`test_r006_protocol` 共 16 项测试通过。

## 2. 三轮审查

### Round 1：最强拒稿论证

1. **低秩可能是输出瓶颈的代数假象。** 单样本 Jacobian 的秩受轨迹输出维度限制，不能由此推出跨场景
   共享通道子空间。
2. **site mask 与公式不完全一致。** 原命题 3 使用 \(I_P\otimes B\)，会把 C3 的 non-image status site
   也写成可修复，和代码及协议冲突。
3. **basis vector 不可识别。** \(B\) 与 \(BQ\) 表示同一子空间；若论文解释单列 basis 或直接跨折比较
   coefficient，会产生 gauge-dependent 结论。
4. **“rank-r”容易被误读。** 当前方法是共享通道秩 \(r\)，但仍有 \(P_{repair}r\) 个 site coefficients；
   不能描述为整个修复只有 \(r\) 个自由度。
5. **有限干预界缺少精确范数。** \(W\) 可奇异，必须用 \(W\)-半范数及诱导算子范数陈述 Jacobian
   Lipschitz 条件。
6. **内部 patching 可能离开自然激活流形。** 即使 patch 改变行为，也只说明冻结计算图中的受控介导，
   不自动给出自然语义因果、真实天气因果或安全结论。
7. **恢复 clean 输出不是驾驶正确性。** clean planner 也可能错误；output restoration、official utility、
   clean no-harm 与安全必须分开。
8. **方法必要性尚未成立。** 双编码器、per-planner hook 与 basis 带来部署成本；普通 adapter 或
   direct-output 若同样有效，内部机制只能作为分析贡献。

### Round 2：证据与作者答辩

- 第 1、6、7、8 点已经被项目的 fail-closed gates 正面承认，但仍是决定论文命运的未验证项；不能靠
  新定理消除，只能由 R006V0–V3 和 matched baselines 裁决。
- 第 2–5 点是有效的理论精度缺口，已在
  `docs/AORTA_VISUAL_FIRST_THEORY_V2_CN.md` 与
  `docs/R006_THEOREM_ASSUMPTION_EVIDENCE_MATRIX.md` 修正：
  - 固定 row-vectorization、site selector \(S_\ell\) 与
    \(\Pi_{M,B}=M_\ell\otimes BB^\top\)；
  - 把可识别对象限定为 subspace/projector；
  - 把 rank 限定为 shared channel rank；
  - 用 \(\|\cdot\|_W\) 和 \(\|\cdot\|_{2\to W}\) 写出有限干预条件；
  - 明确共享低秩是经验假设，不是低维输出的推论。
- 冻结预注册中的简写公式保留为历史记录；正式实现语义由 amendment 2/6/7、site mask、逐样本审计及
  当前精化理论共同限定。不得在正式结果运行中改写历史协议。

### Round 3：收敛结论

现有理论的**数学结构已达到可提交给强审稿人检查的条件严谨度**：对象、假设、条件结论、不可识别性与
经验边界可以逐项对应。但论文的创新强度仍处于 `UNTESTED`，不能把“理论闭合”写成“方法成立”。
下一步不应增加新 loss、mixture basis、多层修复或 world model；应让当前最小理论接受最强反例检验。

## 3. 主张—证据矩阵

| 主张 | 类型 | 当前状态 | 决定性证据 | 失败动作 |
|---|---|---|---|---|
| pullback metric 描述局部任务敏感性 | 条件数学事实 | 已闭合 | 可微性、JVP/VJP、冻结 \(W\) | 接口失败则停止该切口 |
| 有限误差可分为 reader/未覆盖/非线性三项 | 条件数学事实 | 已闭合 | 明示 \(L_J\) 假设；真实有限 patch 另测 | 不把 coverage 当修复保证 |
| 存在共享低通道秩介导 | 经验假设 | 未知 | nested group-held-out coverage + finite interventions + controls | 停止 global CPTR |
| DINO 对机制坐标有增量信息 | 函数类受限经验假设 | 未知 | correct/mismatch/removed/native-only | 停止视觉恢复，不扩网掩盖 |
| CPTR 优于普通 adapter/direct-output | 方法创新假设 | 未知 | matched data/params/FLOPs/steps | 降级为机制分析 |
| 对 NavDream 与多 planner 有净收益 | benchmark 主张 | 未知 | official utility、clean、cross-planner、latency | 不投稿完整方法主张 |
| 驾驶安全保证 | 安全定理 | 不主张 | 当前设计无充分证据 | 始终禁止外推 |

## 4. 理论为何值得继续

它的价值不来自 Jacobian、DINO、activation patching 或低秩本身，而来自一个可迁移的问题重写：

> 鲁棒视觉无需恢复所有受扰表示；它只需恢复冻结决策函数对该外观族真正敏感、且外部证据仍可辨识的
> 最小功能子空间。

该重写同时给出三类可行动边界：任务 null-space 不必修、不可恢复信息必须拒判或补充观测、matched
adapter 同样有效时无需内部受限方法。它比“让 feature 更一致”更接近决策问题本身，也允许失败结果
产生明确结论。

## 5. 仍可能导致强拒稿的未解决问题

1. R006V0 若只在 C4 单 query 或高 rank 上通过，低秩机制可能缺乏实质压缩与可迁移 insight。
2. source controlled appearances 与 NavDream/真实天气之间仍有分布鸿沟。
3. DINO + native encoder 的端到端时延与显存可能抵消方法价值。
4. 每个 planner 单独发现 basis 可能证明的是 per-model analysis，而非跨架构方法规律。
5. activation support distance 可能暴露严重 off-manifold patch；它虽不进入当前 PASS/STOP，也必须如实报告。
6. clean behavior restoration 可能在 native 本就错误的样本上维持错误；最终必须由 official utility 与
   hard-failure 分析裁决。

这些问题目前不需要提前增加模块。只有对应的正式失败证据出现，才允许修改理论或架构。

## 6. 最短、最强的后续证据链

1. 完成当前 R006V0 的 inner basis 资格与 outer finite intervention，不读部分科学结果改协议。
2. 若且仅若内部介导通过，冻结唯一 layer/rank/basis，运行 DINO 增量可恢复性。
3. 若且仅若视觉增量通过，实现最小 CPTR，并与 Constant Eye、generic adapter、full-feature、random-basis、
   direct-output 做容量和预算匹配。
4. 最后一次性验证 clean no-harm、NavDream、跨 planner、official utility 与 batch-1 P50/P95。

任何门失败都保留原裁决。Best-Paper 标准在这里意味着证据链没有逻辑捷径，而不是承诺得到正结果。

## 7. 一分钟版本

Constant Eye 说明稳定 DINO 特征可以帮助外观 OOD，但它仍为每个 planner 重建接口。AORTA 追问冻结
planner 真正因外观变化丢失了什么。我们不对齐全部 feature，而在 planner 自身的任务几何中寻找跨场景
共享的低通道秩方向，再用组外、双向、场景特异的内部干预确认这些方向确实介导行为变化。只有随后证明
受扰 RGB 中的 DINO 证据能恢复这些坐标，我们才训练一个被限制在该子空间内的修复器。若普通 adapter
同样有效，方法主张停止；若成立，则得到的是“只修复决策函数真正需要且视觉仍可恢复的信息”，而不是
又一个更大的鲁棒模块。
