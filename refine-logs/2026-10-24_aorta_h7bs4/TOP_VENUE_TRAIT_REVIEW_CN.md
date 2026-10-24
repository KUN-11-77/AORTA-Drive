# AORTA 顶会论文特质审查（ICCV视角）

日期：2026-10-24  
性质：内部严格审查；未调用外部独立reviewer，不能替代真实同行评审。

## 结论

材料已经具备顶会论文的正确骨架，但当前证据不足以形成一篇可投稿ICCV的方法论文。
最准确的状态是：`STRONG_RESEARCH_THESIS__METHOD_AND_EVIDENCE_INCOMPLETE`。

它的价值不在于“DINOv3+道路场+reranking”的模块组合，而在于一条由反例约束的命题：

> 外观变化下，不必重建完整世界；应学习候选无关、支持域明确、保留完整候选差异的任务
> 表示，并且只有在视觉与误差证据足以区分候选时才允许影响冻结planner。

如果后续实验支持这条命题，它具备ICCV竞争力；如果只得到更低field MAE或少量PDMS提升，
则仍容易被审稿人归类为工程组合。

## 已具备的顶会特质

1. **问题可一句话说明。** 路没有变，天气改变后planner为什么开歪？这比从某个网络模块
   出发更容易形成有记忆点的视觉研究问题。
2. **理论对象有明确边界。** gauge-invariant graph differences、有限支持域、完整4秒候选
   query与可信相对比较都能被形式化；不把道路裕度写成完整驾驶安全。
3. **设计由可证伪反例推出。** 圆盘证书、1m witness、2.5秒prefix和64×64范围均经历
   失败并被保留在证据链中；96×64来自完整候选支持而非随意加大网格。
4. **网络不应成为黑箱补丁。** 视觉投影、missing evidence、相对表示、可靠性和拒答分别
   对应几何绑定、外观不变性、可识别性和风险控制，不是为了堆模块。
5. **系统约束进入方法定义。** 完整时域、真实footprint、候选无关共享计算和P50/P95时延
   被视为方法资格，而不是论文完成后的工程优化。
6. **主张纪律较强。** development不冒充untouched test，projective support不冒充物理
   可见性，source calibration不冒充任意OOD保证，机械audit不冒充独立科学复核。

## 当前仍达不到ICCV接收线的核心缺口

### 1. 核心算法创新仍未冻结

node potential与cut-space projected edge在表达能力上等价，因此potential parameterization
本身不能作为主要创新。96×64也只是经验证的空间合同，不是算法创新。真正可能成为贡献的
对象必须是“support-aware decision quotient”：视觉任务表示、误差/支持结构与候选比较
算子共同构成的、可检验的新接口。

### 2. 决策充分算子未闭合

完整时域解决了prefix删除后段信息的问题，却没有自动解决global set-min被共同早期最小值
支配的问题。最终使用global min、时间裕度曲线、分段worst-case还是其他算子，必须由source
oracle任务反例和决策损失推出，不能看结果后选择。Oracle都不能改善候选排序时，应停止视觉
网络扩展。

### 3. NavDream主张还没有证据

R004H5/H6支持受控外观与historically exposed development上的视觉机制，但不能保证
NavDream雨、雪、夜间等外观。最终需要严格配对、zero-shot、强度账本、clean无害和失败组
报告；稳定性与正确性必须同时成立。

### 4. reliability不是网络输出一个数就成立

预测scale或confidence只有在proper loss、独立校准、coverage/sharpness与OOD边界被审计后
才有统计含义。NavDream上的经验coverage只能写成经验结果，除非另加可验证的分布关系。

### 5. 尚无真实决策与部署收益

当前没有证明方法在多个冻结planner上改善合法候选排序、恢复OOD失败、保持clean且满足时延。
这会是审稿人决定“方法论文”还是“分析论文”的分水岭。

## 推荐冻结的三项贡献

### Contribution 1：任务表示

学习候选无关的相对道路任务场，而不是普通分割/BEV/SDF。其贡献不能写成新SDF，而应写成：
表示被定义在共同offset的商空间，监督与评估直接对应候选决策对比，并保留明确支持域。

### Contribution 2：支持与可信比较

把\(\Omega_{geom}\)、\(\Omega_{visible}(I)\)与\(\Omega_{reliable}(I)\)分开；对完整候选只在
支持和误差条件满足时计算可信decision contrast，否则abstain。贡献是条件化的干预规则与
可审计失效边界，不是泛化的“安全保证”。

### Contribution 3：冻结planner上的可迁移仲裁

视觉编码与任务场对候选共享，复杂度为
\(O(C_{vision}+C_{field}+K C_{query})\)，而不是对每个候选重复视觉网络。用跨planner、
NavDream与真实时延证明该接口有独立价值。

三项贡献必须围绕同一对象，不能扩成三套互不依赖的模块。

## 结果—主张裁决矩阵

| 结果 | 允许主张 | 禁止主张 | 后续 |
|---|---|---|---|
| H7BS4A/B失败 | 当前96×64视觉接口缺少关键证据 | 视觉任务场可用 | 重设可观测接口或停止dense-field路线 |
| 可观测性通过、oracle失败 | 表示可构建但道路裕度不足以改善候选 | 决策算法有效 | 停止大规模网络训练，检查候选/任务定义 |
| oracle通过、learned field失败 | 决策命题成立但视觉学习未成立 | 外观鲁棒方法有效 | 定位表示、监督或不确定性失败 |
| learned决策通过、NavDream失败 | source/development机制有效 | 跨外观鲁棒 | 停止OOD主张，分析外观可辨识性 |
| NavDream、跨planner、clean、时延均通过 | 支持任务表示与选择性仲裁的完整方法贡献 | 完整驾驶安全保证 | 进入最终test与论文主张审计 |

## 最小ICCV证据包

1. H7BS4：候选关键节点可观测性、96×64 teacher误差账本、同步端到端时延；
2. source-only oracle：比较global set-min、预注册的完整时域算子及简单SDF/BEV控制；
3. matched-capacity结构消融：potential/edge-projection/direct pair/普通SDF，固定DINO与预算；
4. NavDream严格配对zero-shot：正确性、漂移、拒答、clean无害和最差组；
5. 至少两个不同冻结planner/provider上的决策收益与失败分析；
6. FlowDrive、READ及合法PEA/MG-style baseline在对齐合同下比较；
7. 参数、FLOPs、P50/P95/P99、峰值显存、吞吐及完整模块分解；
8. submission前执行实验完整性、论文数字和adversarial reviewer审查。

## 论文故事的最简版本

> 天气改变了图像，但没有改变道路。我们发现，鲁棒驾驶的关键不是在所有天气下重建整个
> 世界，而是恢复那些足以区分候选驾驶行为、并且有视觉证据支持的道路关系。AORTA把这种
> 信息表示为支持域明确的相对任务场，完整评价冻结planner的候选，并且只在可信差异足够时
> 仲裁；看不清时不依据该视觉证据强制改写。

这段故事容易理解，背后的技术深度来自四个不可删的条件：跨外观正确性、完整候选信息、
支持/误差可识别性、实时计算。任何一个缺失，故事都会退化成模块组合。

## 当前审稿式裁决

- 问题重要性与叙事：强；
- 理论纪律与失败反例：强；
- 已证明的方法新颖性：不足；
- 视觉到决策的完整证据：不足；
- 当前投稿判断：`WEAK_REJECT / PROMISING BUT INCOMPLETE`；
- 完成上述最小证据包后的潜力：`ICCV-COMPETITIVE`，但不预先保证接收。
