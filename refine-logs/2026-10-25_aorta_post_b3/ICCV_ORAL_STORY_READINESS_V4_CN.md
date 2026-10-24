# AORTA ICCV / Oral 叙事资格合同 v4

状态：`COHERENT_HYPOTHESIS__NOT_YET_ORAL_READY`。

这不是写作包装，而是方法资格门。一个顶会故事只有在“反常现象—新对象—必要机制—行为结果”
四步都由同一组可复算证据连接时才成立。不能快速讲清的公式和模块不进入主线；能讲清但未被实验
支持的内容仍只是假设。

## 1. 唯一问题锚点

同一物理场景的外观变化会改变冻结端到端 planner 的输出。已有视觉表征即使在外观变化下稳定且
保留道路信息，也不保证候选排序稳定，因为驾驶行为取决于候选之间的任务差异，而不是全局特征、
完整道路场或单条轨迹表示本身。

最终问题不是“能否预测一个更好的 BEV/SDF”，而是：

> 能否从同一场景的视觉证据中，直接估计合法 challenger 相对 native 的完整时域任务关系，并在
> 证据不足或统计下界不足时保持 native，从而在外观变化下减少错误替换而不伤害 clean 行为？

## 2. 论文的一条逻辑脊柱

### Surprise

稳定甚至 perfect 的道路场仍可能无法区分候选：B3 中 2,033 个 pair 被共同 scalar minimum 压成
相同证据；BS2 又证明短前缀会删除晚期分叉。瓶颈不只在视觉 backbone，而在被学习和读取的任务
对象。

### Insight

驾驶决策需要的是 native 与 challenger **在哪里不同、这些不同由什么视觉证据支持、其任务后果
是什么**。因此鲁棒性对象应从 feature/field/single-candidate latent 移到 native-relative relation。

### Mechanism hypothesis

用完整时空占据测度差

\[
\nu_{i0}=\mu_i-\mu_0=\nu^+_{i0}-\nu^-_{i0}
\]

生成 challenger-exclusive、native-exclusive 与公共上下文 query；query 可读取全部冻结视觉
token，以免把直接投影缺失误当作全局不可观测。共享有序比较器经反对称化输出

\[
\widehat\delta_{i0}=
\tfrac12\{G(E^+,E^-,E^\cap,Q_{i0})-
G(E^-,E^+,E^\cap,Q_{0i})\}.
\]

外观轨道约束只作用于 relation，任务监督防止常数坍缩；group-simultaneous lower bound 决定是否
替换 native。

### Consequence

证据必须按以下顺序成立：

\[
\text{correct-image dependence}
\rightarrow \text{stable and correct relation}
\rightarrow \text{lower ranking regret}
\rightarrow \text{fewer false interventions}
\rightarrow \text{OOD hard-failure net reduction + clean no-harm}.
\]

任何一箭失败都收窄主张，不能用上游 proxy 替代下游结果。

## 3. 30 秒讲稿：当前假设版

> 现有外观鲁棒驾驶主要稳定特征、BEV 或道路场。但我们发现，即使用 perfect road field，常见的
> 全局读出仍会让大量几何不同的候选得到相同证据，因为共同路径淹没了真正改变决策的分叉。
> AORTA 因此不恢复完整世界，也不独立评分每条轨迹；它用 native 与 challenger 的完整时空差异
> 支持生成问题，再从全部视觉 token 中寻找任务证据，直接预测相对行为 margin。若该关系在观测
> 上不可识别或同时下界不足，系统保留原 planner。我们的核心假设是：可靠驾驶需要稳定“改变
> 决策的证据”，而不只是稳定感知。

在 matched scorer、NavDream、跨 planner、clean no-harm、official-target alignment 与时延通过前，
上段必须明确称为假设，不能改写成完成时。

## 4. 只有两项贡献可以进入摘要

1. **主贡献候选：appearance-interventional differential-support relation learning。** 把外观鲁棒
   对象移到完整时域 native–challenger 差异支持上的任务关系，并用 geometry-biased query、全局
   视觉证据和反对称比较器实现。
2. **从属贡献候选：evidence-qualified native-preserving arbitration。** 将表示不可识别性与统计
   同时误差分开记账，只在二者允许时干预。

DINOv3、Jordan 分解、cross-attention、Huber、conformal quantile、候选生成和普通 uncertainty
head 均是工具，不单列为创新。

## 5. 设计不是任意选择：每一层都必须回答“为什么”

| 设计 | 被哪项失败强制 | 若删除会怎样 | 决定性消融 |
|---|---|---|---|
| 完整 4 s 支持 | BS2 晚期分叉 | 128 pairs 在2.5 s完全不可区分 | full vs 2.5 s |
| signed differential support | B3 common-min | 公共路径继续支配 readout | differential vs independent scorer |
| 全局视觉 key/value | D0只证明直接投影缺失 | 丢失交通灯、遮挡、道路连续性等上下文桥 | global KV vs hard support crop |
| 反对称输出 | 目标是有向 relation | 交换候选不保证符号翻转 | metamorphic swap test |
| task + orbit 双约束 | 单独稳定可坍缩、单独任务可受外观扰动 | 稳定错误或正确但脆弱 | remove task / remove orbit |
| soft provenance + calibration | 直接证据率不是可识别性的充分条件 | 硬门可能错误拒判或近乎全拒判 | no provenance / soft / frozen hard gate |
| native fallback | 观测等价世界与有限样本误差 | 证据不足时仍产生有害干预 | risk–coverage / false intervention |
| previous-action context | 完整官方分数在函数上依赖 two-frame comfort | 省略后不能一般表示 full target | 最小geometry token固定；复杂temporal encoder按matched消融启用 |

新增复杂度只有在表中出现新的、可复算失败证据时才有资格进入系统。

## 6. 三张图讲完整篇论文

### Figure 1：反例与核心对象

同场景不同外观；perfect field + scalar-min 仍折叠候选；右侧显示 \(\nu^+,\nu^-,\mu^\cap\)。这张
图回答“为什么现有抽象不够”。

### Figure 2：一个算子，而不是模块堆

一次冻结视觉编码 → 差异支持 query / 全局视觉 KV → 反对称 relation + scale → calibrated lower
bound → replace or native。每个箭头必须标注其对应定理、反例或消融。

### Figure 3：完整因果证据链

横轴为 appearance severity / planner；纵向依次展示 relation error、ranking regret、false
intervention、hard-failure rescue、clean harm 与 batch-1 P95 latency。不能只展示最终平均分。

## 7. Oral / Poster 资格门

只有同时满足以下条件，才可把当前故事从“连贯假设”升级为“顶会结果”：

1. 合法候选域存在非平凡 official-target oracle；
2. augmented-context full official target 闭合；one-frame只作为oracle gate与`1/7`符号证书子域；
3. differential-support arm 在参数、FLOP、token、数据与 seed 匹配下显著优于 independent scorer、
   DA-WAM/GTRS-style scorer、field readout 与 trajectory-only prior；
4. correct-image 优于 mismatch/removed，排除场景先验和不看图像；
5. NavDream 外观、至少三类 planner 上 relation→ranking→behavior 链成立；
6. 报告 hard-failure 净减少、clean no-harm、abstention coverage 与失败组，而非只报平均 PDMS；
7. batch-1 P50/P95/P99、显存和候选数缩放满足预注册在线预算；
8. 独立审计与强反方 review 无未解决的致命问题。

若第3项失败，差异支持不能作为网络创新；若第2项失败且 temporal context 仍不能修复，当前
decision-relation 主线停止；若第5–6项失败，只能降级为表征论文，不能声称驾驶收益。

## 8. 当前裁决

当前故事已经达到“问题清楚、insight集中、方法对象与失败证据对应”的水平，优于模块拼装式方案；
但仍是 `REVISE / NOT ICCV READY / NOT ORAL READY`。缺口不在措辞，而在三个尚未赢得的事实：

- B4R1 合法 oracle 尚未正式运行；
- 完整 official behavior target 尚未完成对齐；
- differential-support operator 尚未在 matched baselines 与 NavDream/行为/时延上得到支持。

因此下一步不是继续扩展故事，而是让这三个事实逐一通过或推翻故事。

