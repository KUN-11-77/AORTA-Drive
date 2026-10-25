# AORTA Best Paper Readiness Audit v1

日期：2026-10-25  
状态：`STRONG_PROBLEM_AND_THEORY_SEED__BEST_PAPER_EVIDENCE_NOT_YET_PRESENT`

## 1. 审查口径

本审查使用高于普通 ICCV 接收的标准。ICCV 官方没有公布一套可机械执行的 Marr Prize 评分表；
CVF 只说明 Marr Prize 由 program chairs 委派的委员会选出。ICCV 2025 reviewer guidelines 对普通
接收要求强调 technical soundness、field contribution、knowledge advance，并明确鼓励 novel、brave
concepts，要求同时权衡 novelty、potential impact 与 performance。因此，本文所称 Best Paper 特征是
根据官方审稿原则和近届获奖工作作出的研究判断，不是官方保证。

官方来源：

- ICCV 2025 reviewer guidelines：<https://iccv.thecvf.com/Conferences/2025/ReviewerGuidelines>
- CVF awards / ICCV Marr Prize：<https://www.thecvf.com/?page_id=413>

## 2. 总裁决

> AORTA 目前具有“可能形成 ICCV 强论文的问题、第一性理论种子和严谨研究协议”，但没有达到
> Best Paper 候选所需的决定性经验与算法证据。

当前优势是问题逐渐收敛为一个可推广的命题：稳定感知不自动推出稳定决策；视觉鲁棒性应定义为
在候选与因果历史条件下，对决策关系的任务充分、外观稳定和可辨识恢复。当前最大缺口是：尚未
证明正确视觉可恢复该关系，尚未证明 differential query 优于 matched generic scorer，更没有证明
该优势能转化为 NavDream、跨 planner 和 sequential behavior 的净收益。

当前状态不是“差一点实验就能拿 Best Paper”，而是“已经有资格用少数决定性实验判断是否存在
Best Paper 级发现”。

## 3. Best Paper 候选通常需要的六项条件

### BP1：问题重述具有领域普适性

论文应让同行改变问题的表述方式，而不只是接受一个新模块。AORTA 需要成立的重述是：

> 自动驾驶视觉鲁棒性的基本对象不是全局特征或场景重建的不变性，而是给定合法行为集合后，
> 影响相对决策效用的视觉证据是否任务充分、外观稳定且可辨识。

该命题必须在不依赖 NAVSIM 专有公式的层面成立，并能解释已有方法的共同失效。

### BP2：存在一个可独立复述的核心数学洞见

理论必须产生新预测、反例或算法约束，而不是为网络加注释。最有潜力的结构是：

1. 条件视觉价值的 Bayes 风险分解；
2. 观测纤维上 margin 符号不一致导致的不可辨识性；
3. 加性任务下 signed support 的精确差分恒等式及其对非加性任务的边界；
4. 同时 margin 下界推出条件性 native improvement 的选择规则。

其中标准定理本身不是原创。论文贡献必须来自它们共同导出的新问题、新网络约束和被验证的新
现象。

### BP3：方法近乎不可避免，而非可替换模块组合

理想状态是审稿人看到失败现象后，会认为：

\[
\text{native-relative target}
\rightarrow
\text{full-horizon signed query}
\rightarrow
\text{global visual evidence}
\rightarrow
\text{distributional relation}
\rightarrow
\text{selective action}
\]

是最小而自然的解决方式。若普通 full-context scorer、D-JEPA-style relation 或 World4Scorer-style
candidate state 同样有效，则 AORTA 的主架构不具备 Best Paper 级必要性。

### BP4：结果不仅正，而且改变理解

Best Paper 级证据不能只是平均分小幅提高。它应至少包含：

- 一个稳定、可重复、此前基线解释不了的现象；
- 一个由理论预言、由消融验证的机制优势；
- near-tie、late-fork、appearance flip 等关键困难层上的显著改善；
- clean no-harm 与失败边界，而不是只报告平均胜利；
- 对至少三类冻结 planner 的可迁移行为结果；
- 可视化能显示模型在什么证据上改变了候选关系。

### BP5：可信度异常完整

包括冻结协议、组级统计、重复 seed、精确 replay、强基线、公平预算、无 target 调参、完整失败报告、
端到端时延和可复现代码。AORTA 当前在实验纪律上较强，但纪律不能替代正科学结果。

### BP6：故事在一分钟内可被准确讲清

理想叙事只有四步：

1. 相同场景改变外观会改变冻结 planner 的候选决策；
2. 稳定全局表示并不保留候选决策差异；
3. AORTA 用行为差异提出视觉问题，并在不可辨识时拒绝替换；
4. 该机制跨外观、跨 planner 改善行为且满足实时预算。

如果主文需要讲完旧 SDF、graph field、几何证书和所有历史 gate 才能成立，论文将失焦。

## 4. 当前课题逐项审核

| 维度 | 当前判断 | Best Paper 缺口 |
|---|---|---|
| 问题重要性 | 强 | 需要证明该问题普遍存在，而非当前 panel 特例 |
| 第一性问题定义 | 强 | 需要压缩成对领域有迁移性的一个核心命题 |
| 数学内部一致性 | 较强 | 多数是条件性或标准结果；方法特有新结论仍弱 |
| 失败驱动设计 | 强 | 需要证明最终机制确实解决失败，而非只解释失败 |
| 网络创新 | 有潜力 | differential query 尚未与最近强基线公平比较 |
| 视觉条件增量 | 未知 | R005E2 尚未完成科学审计 |
| 外观鲁棒性 | 未证明 | NavDream 封存，正确地未参与训练或选择 |
| 决策行为收益 | 未证明 | 只有 oracle opportunity，没有 learned behavior gain |
| 跨 planner 普适性 | 未证明 | 三类 planner 尚未执行 |
| sequential validity | 未证明 | 当前是 native-history teacher forcing |
| 可靠性 | 条件理论存在 | 可能区间过宽、全拒判，non-vacuity 未证明 |
| 时延与部署 | 仅资格演练 | 缺最终 batch-1 P50/P95、显存与 K scaling |
| 实验可信度 | 强 | external scientific review unavailable；最终 test 尚未执行 |
| 论文简洁性 | 可达 | 当前仓库历史过长，主文必须只保留一条主线 |

## 5. 当前最可能导致拒稿的七个问题

### R1：`U_ctx` 不是完整 policy value

正 row-level margin 不保证实际连续选择改善。必须把离线监督、fixed-continuation value 与策略级
行为严格分开，并用 official evaluation 支撑最终主张。

### R2：核心算法可能只是 geometry-biased attention

signed support integral、trajectory query 与 candidate-conditioned attention 均已有强先例。若在相同
预算下不能稳定胜过 generic scorer，创新主张应删除。

### R3：视觉可能只提供弱增量

即使 R005E2 proper risk 改善，若 decision regret、scene-binding control 或机会恢复比例不通过，
仍不足以支撑视觉决策机制。

### R4：source-fit 小规模信号可能不泛化

42 groups 可支持组级实验，但不能单独支撑广泛世界分布主张。必须通过严格 held-out、最终 test、
NavDream 和跨 planner 验证。

### R5：选择性仲裁可能全拒判

coverage 定理在区间过宽时可以形式成立却没有行为价值。必须同时报告 sharpness、intervention rate、
false intervention 与 net gain。

### R6：连续决策会改变历史分布

teacher-forced native history 与实际 AORTA-selected history 不同。若 sequential rollout 退化，离线
scorer 成功不能写成系统成功。

### R7：论文容易被项目历史淹没

旧 field、SDF、几何证书和多轮资格实验应作为动机、附录或失败分析，不应成为最终方法的并列贡献。

## 6. 冲击 Best Paper 的三项决定性发现

不建议增加二十个实验或更多模块。真正需要的是三个高密度发现。

### Discovery A：Decision-sufficiency gap

证明并可视化：全局表示稳定、场误差相近或普通 scorer 正确，并不保证候选关系稳定；错误集中在
外观引发的 near-tie、late-fork 或局部决策证据变化。该现象需跨场景与 planner 重复，并由明确反例
和可辨识性分析解释。

若成立，它提供领域级问题重述。

### Discovery B：Differential evidence is necessary

在完全匹配的 backbone、输入、参数、FLOP、token、数据、optimizer、seed 下，Differential Query
必须稳定优于：

- trajectory/history-only；
- independent candidate scorer；
- generic full-context set scorer；
- D-JEPA-style native-relative relation；
- World4Scorer/DA-WAM/GTRS-style candidate-conditioned scorer。

优势应在 decision regret、appearance flip、关键边界错误和样本效率上同时出现，并由删除 signed、
full-horizon、global context 或 history 的消融精确定位。

若不成立，不能靠其他模块把它包装成核心创新。

### Discovery C：Representation change causes behavior change

在模型、阈值和协议全部冻结后，一次性证明：

- NavDream 未见外观下 relation 和选择更稳定；
- 至少三类 planner 出现正 net gain；
- clean non-inferiority 成立；
- actual-history 与 sequential rollout 不崩溃；
- selective arbitration 非空洞并降低错误覆盖；
- batch-1 P95 满足部署预算。

这一项把视觉表示贡献转化为实际系统贡献。

## 7. 结果—论文级别映射

| 最终结果 | 可支持的论文定位 |
|---|---|
| R005E2 STOP | 当前视觉路线不成立；做可辨识性/表示诊断，不具方法论文主张 |
| R005E2 GO，但 Differential Query 不胜 generic scorer | 有价值的视觉 recoverability/negative architecture 研究；不是当前主创新 |
| Differential Query source-held-out 胜，但 NavDream 失败 | decision-aligned source 方法；不能称外观鲁棒核心贡献 |
| NavDream 胜，但跨 planner/sequence 失败 | 受限 reranking 机制；系统普适性不足 |
| 三项 Discovery 均强、matched baselines 和时延闭合 | ICCV 强论文，并具备 Best Paper 候选资格 |

“候选资格”不等于获奖概率；奖项由委员会在全部接收论文中比较决定，无法由单项目内部保证。

## 8. 最小范围与复杂度纪律

最终论文最多保留：

1. **主贡献：** appearance-interventional differential decision evidence；
2. **从属机制：** identifiability-aware conservative arbitration。

复用冻结视觉 backbone；新增 trainable 机制保持为 differential query/relation head，不主动加入世界
模型、RL、diffusion、LLM、SDF、额外地图 decoder 或多套风险网络。distributional head 应尽可能
作为 relation head 的输出参数化，而不是独立大模型；calibration 为 post-hoc 统计步骤。

复杂度只在已审计失败证明最小机制不足时增加。

## 9. Best Paper 级主图要求

### Figure 1：现象与反例

同一物理场景不同外观下，普通稳定特征/普通 scorer 保持相似，却发生候选关系或行为翻转；标出
决定差异的时空证据。

### Figure 2：一个核心机制图

一次视觉编码；native–challenger full-horizon signed support 提出 query；全局视觉回答；输出 margin
distribution；证据不足保留 native。不得画成十几个并列模块。

### Figure 3：决定性 matched comparison

以 decision regret、appearance flip、near-tie/late-fork 和 sample efficiency 展示 differential
query 相对最强 matched scorer 的优势。

### Figure 4：行为、可靠性与成本闭环

NavDream × 三 planner、clean no-harm、sequential、coverage–intervention–gain 与 batch-1 P95。

## 10. 当前最优研究动作

1. 完成 R005E2，不在中途读取或修改科学协议；
2. 若 GO，只设计一次预算匹配的 generic-vs-differential 核心实验；
3. 若核心机制胜出，再冻结 appearance、calibration、cross-planner、sequential 与 latency 合同；
4. 若任一主门失败，删除对应主张，不用额外模块掩盖；
5. 最终主文只讲第一性问题、一个机制和三个决定性发现，历史路线移入附录。

## 11. 最终判断

- 普通 ICCV 接收潜力：存在，但依赖尚未完成的核心实验；
- ICCV strong accept / oral 潜力：需要 Differential Query 的 matched 优势与 NavDream 行为闭环；
- Best Paper 潜力：只有三项 Discovery 同时形成显著、简单、可推广的发现后才存在；
- 当前裁决：`HOLD_BEST_PAPER_CLAIM__GO_WITH_DECISIVE_FALSIFICATION_SEQUENCE`。

当前不应追求“看起来像 Best Paper”的复杂度，而应追求一个足以改变同行问题表述、并被最少但
决定性证据支持的发现。
