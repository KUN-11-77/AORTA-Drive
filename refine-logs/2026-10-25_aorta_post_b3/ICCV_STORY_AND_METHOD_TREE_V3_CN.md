# AORTA ICCV/Oral 叙事与方法树 v3

状态：`ONE_STORY_FROZEN_AS_HYPOTHESIS__METHOD_NOT_VALIDATED__B4R1_REQUIRED`。

本文约束最终论文如何讲成一条有吸引力但不牵强的故事。它不是写作包装；任何一句故事都必须对应
一个数学对象、一个网络接口和一个可推翻实验。

## 1. 一句话问题与一句话方法

**问题。** 外观鲁棒方法通常稳定 feature、BEV、道路场或单候选 future latent，但驾驶决策由候选
之间的差异决定；共同路径可以淹没真正的分叉，而不可观测分叉根本不允许可靠判断。

**方法假设。** AORTA 不恢复完整世界，也不把每条候选独立评分；它用 challenger 相对 native 的
完整时空差异支持生成问题、再从全部冻结视觉 token 中寻找答案，学习外观轨道上稳定的任务关系；
只有存在真正的观测等价反例或同时误差界不足时才保留 native。

用户最终应只记住：

> **Robust driving needs invariant evidence on where decisions differ—not merely invariant perception.**

中文：**可靠驾驶需要稳定“决策分歧处”的证据，而不只是稳定的视觉表征。**

## 2. 为什么这个故事不是模块拼装

论文全篇只有一个技术矛盾：

\[
\text{appearance-stable representation}
\centernot\Rightarrow
\text{decision-distinguishable evidence}.
\]

它在四个层次中重复出现：

1. **反常现象：**旧 field 在 held-out development 上稳定、正确且依赖正确图像，但 perfect teacher
   经过 scalar-min readout 仍让 2,033 个 candidate pairs 共用相同 minimum；
2. **数学缺口：**规划需要 \(U(\tau_i)-U(\tau_0)\)，而 global field quality 或单候选表征没有保证
   保留该差异；
3. **网络机制：**直接查询 signed differential support
   \(\nu_{i0}=\mu_i-\mu_0\)，而非把完整 field 压成一个 scalar 或给每个候选独立 pooling；
4. **行为终点：**只在可观测支持与 simultaneously calibrated lower bound 均允许时替换 native。

因此 shared DINO、完整轨迹、support mask、relation loss 与 abstention 不是五项并列创新，而是同一
矛盾的感知、表示、学习和行动实现。

## 3. 与三篇最强近邻的“一刀切”区别

| 近邻 | 它已经解决了什么 | AORTA 不能再宣称什么 | AORTA 必须额外证明什么 |
|---|---|---|---|
| The Constant Eye | appearance-only benchmark、冻结 DINOv3 通用接口、跨 planner 零样本鲁棒 | DINO、NavDream、plug-and-play perception interface | 稳定视觉接口之后，哪个**决策关系**仍会丢失，以及 relation-level intervention 是否减少 ranking/behavior failure |
| DA-WAM | candidate-conditioned future latent、action–representation pairing、factor/utility scorer、hard negatives | candidate cross-attention、trajectory scoring、decision-aligned latent | native–challenger **差异支持**相对普通单候选 scorer 是否在外观 shift 下更稳、更必要 |
| RiskWorld / Expand-and-Prune | selective replacement、候选复用/扩展、不确定性与跨架构 | frozen-planner assistance、candidate pruning、confidence fallback | 拒判是否来自“差异证据不可识别 + simultaneous relation error”，而不是普通 confidence heuristic |

如果 matched experiments 不能支持最后一列，AORTA 就没有独立方法故事；不能靠新名称补救。

## 4. 顶会四拍：Surprise → Insight → Mechanism → Consequence

### 第一拍：Surprise

> 一个道路场可以稳定，甚至可以是 perfect，却仍无法告诉规划器两条候选哪条更好。

主图左侧只放两个反例：

- 两条候选共享早期路径，3 s 后才分叉；2.5 s 表示完全相同；
- 两条候选的 global minimum 相同，但独占支持不同。

这比“雨天分数下降”更有 insight，因为它证明瓶颈不只在 backbone，而在任务读出对象。

### 第二拍：Insight

轨迹 \(\tau_i\) 与 native \(\tau_0\) 的比较只需要它们不同之处的任务证据。用时空占据测度表示，

\[
\nu_{i0}=\mu_i-\mu_0=\nu^+_{i0}-\nu^-_{i0}.
\]

对于加性 task field，公共支持精确抵消；对于非线性 evaluator，公共上下文仍可输入，但不能再次
通过全局 minimum 支配差异。若 \(|\nu|\) 的关键部分不可观测，则存在两个观测相同、关系符号相反
的场景，统一可靠判断不可能，拒判是逻辑后果而非工程保险丝。直接投影缺失本身不等于全局
不可观测；交通灯、道路连续性等区域外上下文仍可能提供证据，必须由全局 token query 与校准实验
判断。

### 第三拍：Mechanism

1. 冻结 DINO 对场景只编码一次；
2. 完整 4 s native/challenger 生成 \(\nu^+,\nu^-,\mu^\cap\)；
3. 三类差异/公共支持生成 geometry-biased query，但 key/value 保留全部视觉 token；
4. 反对称 relation operator 输出合法 \(\widehat\delta_{i0}\)；
5. task loss 保持正确性，appearance-orbit constraint 抑制同场景外观漂移；
6. evidence provenance、可识别性诊断与 group-simultaneous lower bound 决定是否干预。

其中只有 **differential-support decision relation** 是主贡献候选；selective arbitration 是把它接入
frozen planner 的从属机制。DINO、cross-attention、Jordan decomposition、conformal quantile 和
trajectory tokenizer 都是现有工具。

### 第四拍：Consequence

证据顺序必须是：

\[
\text{correct visual evidence}
\rightarrow
\text{stable/correct relation}
\rightarrow
\text{lower ranking regret}
\rightarrow
\text{fewer false interventions}
\rightarrow
\text{OOD hard-failure net reduction with clean no-harm}.
\]

只要其中一箭失败，就收窄主张，不用上游 proxy 替代下游行为。

## 5. 30 秒 Oral / Poster 讲稿合同

> 现有外观鲁棒驾驶通常稳定视觉特征或道路场，但我们发现，即使用 perfect road field，常见读出
> 仍会让大量不同候选得到相同决策证据。原因是驾驶需要的不是完整场是否相似，而是候选在何处
> 分叉、这些分叉是否被正确看见。AORTA 因而把轨迹表示成相对 native 的 signed differential
> support，用分歧生成 query、从完整冻结视觉 token 中查询任务关系；存在观测等价歧义或同时下界不足时保留原
> planner。我们不只是让感知在天气下稳定，而是让“改变决策的证据”在天气下稳定且可拒判。

在 B4R1、matched scorer、NavDream、跨 planner、clean no-harm 与 latency 通过前，讲稿中的
“AORTA 因而……”之后必须标注为 method hypothesis，不得用未来时态伪装成结果。

## 6. 两分钟逻辑链

1. **社区背景：**The Constant Eye 说明外观变化本身足以破坏 planner，DINO 接口能改善鲁棒性；
2. **未解决缺口：**稳定感知不保证下游 readout 保留候选差异；DA-WAM 的普通 candidate-specific
   latent 也没有针对 appearance quotient 或证据不可观测性；
3. **我们的反例：**perfect field + scalar-min 仍大量碰撞；短前缀又删除晚期分叉；
4. **数学对象：**合法候选域上的 \(\delta^*_{i0}\) 与 signed support \(\nu_{i0}\)；
5. **方法：**differential-support relation operator + orbit constraint + provenance-conditioned statistical abstention；
6. **理论：**加性任务差分恒等式、不可观测符号不可识别、\(2\varepsilon\) regret bound、条件性
   native improvement；
7. **决定性实验：**普通 scorer/DA-WAM-style/field readout 匹配消融，NavDream 与三 planner 的
   relation→ranking→behavior 链，clean no-harm 和 batch-1 latency。

## 7. 一张主图应如何讲完整故事

主图不是系统流程图，而是因果图：

```text
same scene, different appearance
            |
            v
 stable field / perfect teacher --scalar min--> indistinguishable candidates  [counterexample]
            |
            v
 native vs challenger full-horizon differential support (+ / - / common)
            |
            v
 correct-image relation + appearance-orbit constraint
            |
 representation ambiguity? calibrated lower bound > 0?
          /                         \
      fail: native fallback     pass: replace native
            |
            v
 relation error -> ranking regret -> hard-failure rescue / clean harm / latency
```

右下角必须同时展示最强竞争解释：ordinary candidate scorer、DA-WAM-style scorer、field readout 与
trajectory-only prior。若图中需要解释五个互不相干的新模块，说明故事已经失焦。

## 8. 贡献只允许两条

### 主贡献候选

**Appearance-interventional differential-support relation learning：**把外观鲁棒性的对象从 feature/
field/单候选 latent 移到合法 native–challenger 差异支持上的 task relation，并以完整时域、正确
图像依赖和 matched scorer 对照验证其决策充分性。

### 从属贡献候选

**Evidence-qualified native-preserving arbitration：**把几何可观测性缺失与统计同时误差分别建模，
只有两者都通过才覆盖 native。

禁止把以下内容列为贡献：DINOv3、Jordan/Hodge、cross-attention、trajectory tokenizer、Huber、
primal–dual、conformal quantile、候选生成、普通 uncertainty head、跨 planner 本身。

## 9. Rebuttal 预死亡问题

1. **“这不就是 DA-WAM scorer 输入换成轨迹差吗？”** 只有 D 相对匹配 C 在 appearance-shift
   regret/false intervention 上有可靠增量，且差异 support 删除实验复现 B3 机制，才能反驳。
2. **“signed occupancy difference 只是 NMP cost difference。”** 承认线性恒等式已有；贡献必须落在
   visual differential-support query、appearance intervention 和 evidence-qualified action 的闭环。
3. **“为什么不训练一个更强 field？”** perfect-teacher B3 反例说明 field 精度不是充分条件；仍需
   READ/FlowDrive-style field 匹配对照。
4. **“拒判只是 conformal prediction。”** 分位数是标准工具；必须证明 support-level
   identifiability gate 与 relation error 分开贡献，且不是全拒判。
5. **“NavDream 上稳定可能只是没看图像。”** correct-vs-mismatch/removed、geometry-matched
   trajectory-only 和最差组分析必须同时通过。
6. **“只是更多参数或更多候选。”** 固定 candidate provider，匹配参数/FLOP/token/seeds，并报告
   candidate-supply 与 relation-estimation gain 分离。
7. **“PDMS 不等于安全。”** 只声称 benchmark-relative decision improvement；报告分项、hard
   failure、NAVSIM v2/伪闭环边界，不写真实道路安全保证。

## 10. 当前真实裁决

当前叙事质量比 v2 更集中，因为网络对象直接由三项失败证据推出，并且与 DA-WAM、The Constant
Eye、RiskWorld 的边界更清楚。但论文状态仍是 `REVISE / NOT ICCV READY`：

- 反例与旧 field 视觉必要性：已有范围内支持；
- v2 differential-support operator：理论候选，未训练；
- 合法 preference oracle：B4 无效，B4R1 只完成 preflight，正式运行未授权；
- matched architecture、NavDream、三 planner、clean no-harm、时延：全部未证明。

顶会故事不是现在写得顺就成立。它必须按相同顺序被实验赢得；若差异支持不优于普通 scorer，主
故事立即退役，而不是再加模块。

