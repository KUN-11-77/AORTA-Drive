# R006 Theorem–Assumption–Evidence Matrix

状态：投稿理论边界表；用于区分数学事实、待验证假设与经验主张。

| 编号 | 命题/对象 | 必需假设 | 证据或实验 | 允许结论 | 禁止外推 |
|---|---|---|---|---|---|
| T0 | paired appearance contrast \(\Delta Z=Z_c-Z_a\) | clean/shift 除RGB外的状态、历史与planner输入逐字段一致 | pair ledger、hash、exact non-image audit | 在登记干预内把激活差异归因于RGB外观算子 | 不代表自然天气因果效应或消除所有隐藏混杂 |
| T1 | planner pullback metric \(G(z)=J(z)^\top WJ(z)\) | tail 在指定切口局部可微；\(W\succeq0\) 预冻结 | R006V0J 的 JVP/VJP 伴随一致性、finite、tail equivalence | 指定点附近的一阶任务敏感半度量可计算 | 不代表输出正确、全局线性或安全 |
| T2 | task-weighted shared channel-rank \(r\) subspace | shifted 点 \(J_a\)；site selector固定；登记数值求解器 | synthetic dense oracle；nested discovery/selection；rank–coverage；projector audit；R006V0 fixed-rank STOP | 登记函数类中的数值候选近似最小化未覆盖的一阶任务漂移；当前不具备固定全局最小rank接口资格 | 不声称 Stiefel 全局最优、basis列可识别、rank随coverage单调，或总干预仅有\(r\)个自由度 |
| T3 | 局部到有限干预界 | 星形路径邻域内 \(\|J(z_1)-J(z_2)\|_{2\to W}\le L_J\|z_1-z_2\|_2\) | coverage 与真实 tail patch restoration 同时报；dose/reverse | 条件性地把误差分成 reader、未覆盖、非线性三项 | coverage 单独不证明有限修复；当前实验不估计全局\(L_J\) |
| T4 | 冻结模型内部低秩介导 | 有限source总体中的组外共享性；same-scene patch有效且具scene specificity | R006V0 basis资格后才允许wrong-scene/random/noise/identity/reverse/dose与42组OOF；当前fixed-min-rank资格STOP | 只有完整有限干预通过后，才允许指定planner、层、source组总体和外观族中的内部行为介导结论 | 当前C4层稳定与post-hoc projector overlap不等于介导；不声称唯一自然机制、basis语义、任意分布泛化或真实世界语义因果 |
| T5 | DINO增量可恢复性 | shifted RGB仍含目标信息；site-conditioned evidence binding；匹配函数类可学 | R006V1 correct/mismatch/removed/native-only group-held-out probe | 指定函数类和训练合同下的增量可预测价值 | 不证明理想互信息、严格可识别性或任意OOD恢复 |
| T6 | CPTR功能恢复 | T4、T5成立；reader误差与非线性项受控 | R006V2 coefficient、frozen-tail output、matched adapter/full-feature/random basis | rank-constrained修复优于匹配替代的经验结论 | 不由此推出official utility或安全 |
| T7 | benchmark效用与泛化 | 冻结方法；评估协议合法；独立外观与planner | R006V3 clean/NavDream/cross-planner/latency/official utility | 指定benchmark上的经验收益和边界 | 不宣称任意分布OOD、闭环安全定理 |
| T8 | 科学图到快速部署图可迁移 | 全确认集数值容差与离散决策完全一致 | default/math backend equivalence + batch-1端到端时延 | 可在通过门后使用官方快速图部署 | 单anchor数值接近不能证明全数据决策等价 |
| T9 | planner-pullback residual 的rank增量恒等式 | 嵌套projector满足\(P_{r+1}=P_r+Q\)，且\(Q\)与\(P_r\)正交 | 解析恒等式与`tests/test_r006_rank_geometry.py`三通道反例 | residual变化精确为\(\|JQ\delta\|_W^2-2\langle J(I-P_r)\delta,JQ\delta\rangle_W\)；rank单调改善需要额外条件 | 不得套用PCA explained-variance单调性；ordered flag也不自动修复该问题 |

## 论文中的使用原则

1. 定理只陈述 T1–T3 的条件性数学内容；T4–T7 均写成经验命题。
2. R006V0 basis已因固定最小rank不一致而STOP；若未来新协议PASS，也只解锁对应有限干预或R006V1，不回填DINO、CPTR、NavDream或安全主张。
3. activation patching 证明的是冻结计算图中的受控介导；即使所有对照通过，也不声称找到了唯一自然语义变量。
4. output restoration 与 official utility 分表报告；前者不能替代后者。
5. \(B\) 与 \(BQ\)（\(Q\in O(r)\)）表示同一方法对象；跨运行只比较 projector/principal angles，不比较单列basis语义。
6. “低秩”统一指共享通道秩；site-conditioned coefficient field 仍有 \(P_{repair}r\) 个标量自由度。
7. “最小合格rank”只是冻结threshold与estimator下的操作性诊断；R006V0F只分解其不稳定来源，不能把它升级为内在维数。
