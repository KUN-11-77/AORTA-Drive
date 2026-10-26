# R005E3A：Decision-Aligned Objective Qualification 预注册

日期：2026-10-26

状态：`PREREGISTERED__IMPLEMENTATION_NOT_YET_QUALIFIED`

## 1. 唯一问题

R005E2B与R005E2S共同支持：正确视觉包含可重复的平均relation信号，但该信号没有降低候选选择
regret。本实验只问：

> 在模型、输入、初始化、数据顺序和训练步数完全相同时，将平方margin回归替换成直接对齐候选选择
> 的最小soft-regret surrogate，能否把既有relation信号转化为OOF decision-regret改善？

本轮不是架构实验，不引入Differential Query、adapter、appearance loss、uncertainty或arbitration。

## 2. 第一性理论

平方关系回归在总体无限容量下恢复条件均值，并非普遍“不具决策一致性”。当前问题是有限训练预算下，
平均MSE改善没有集中到argmax相关方向。对真实效用向量\(u\)、合法mask \(V\)及预测\(s\)，定义：

\[
p_i(s;T)=
\frac{\exp(s_i/T)}{\sum_{j\in V}\exp(s_j/T)},
\qquad i\in V,
\]

\[
\mathcal L_{SR}(s,u)
=\max_{i\in V}u_i-
\sum_{i\in V}p_i(s;T)u_i.
\]

它是soft selection下的真实utility regret；exact ties不需要人为指定唯一class。固定\(u\)时，降低该损失
要求将更多概率质量移向高效用候选。它不是proper distributional loss，因此即使通过，也只资格化
decision alignment，不支持校准或条件分布主张。

若部署hard选择为\(\hat i\in\arg\max_i s_i\)，合法候选数为\(n\)，则
\(R_{hard}=u^*-u_{\hat i}\leq n\mathcal L_{SR}\)。本panel有\(n\leq7\)，故该surrogate在逐anchor
意义下确实控制hard regret，但界不等于有限样本泛化保证；完整证明、梯度和有界score失效条件见
`R005E3A_SOFT_REGRET_THEORY_NOTE.md`。

温度不调参。每个outer fold使用outer-train合法候选中严格正的
\(\max_j u_j-u_i\)的中位数；偶数个gap时取两个中央次序统计量的算术平均，ties用`1e-12`排除。
若不存在正gap，实验INVALID。该规则只使用训练角色
target，按效用自然尺度确定平滑宽度，不读取outer-test或封存数据。

## 3. 最小、计算匹配的处理

每个fold×seed复用父R005E2B V2 correct的：

- `FullContextSetScorer`架构和冻结DINO输入；
- refit初始参数hash；
- outer-train roster与seeded batch顺序；
- AdamW、learning rate、weight decay和schedule；
- 父V2通过inner-validation选择的epoch数，即完全相同optimizer-step预算。

唯一变化是训练loss。新arm不重新做early stopping，避免把“loss效果”与“新模型选择策略”混淆。
父V2 OOF预测直接作为并发对照，不重新训练。

## 4. 冻结测试干预

每个新checkpoint在同一outer-test panel上一次性推理：

1. correct visual；
2. S1 primary geometry donor；
3. S2 disjoint-geometry donor；
4. S3 hash-random donor；
5. removed outer-train slot mean。

五个条件的candidate、history、target、mask和model state完全相同。donor与removed仍是输入干预，
不是精确conditional randomization。

## 5. 设计结构

- 独立统计单位：42个recording groups；
- block：6个固定outer folds；
- repeated measures：同group下的16 anchors、3 seeds及5个测试输入条件；
- treatment：训练loss，作用于fold×seed模型；
- 不把672 anchors或2016 anchor-seed观测当独立样本；
- 先在group内聚合anchors，再在group内平均seeds，最后对42个配对group做10,000次bootstrap。

这是受限随机性的配对block设计，不是对真实世界场景的随机抽样因果试验。因而结论限定为冻结
source-fit roster上的OOF算法比较。

## 6. 通过门

设\(R_{SR}\)为新objective correct输入的组级regret，\(R_{MSE}\)为父V2：

1. \(R_{SR}-R_{MSE}\)的95% CI上界严格小于0；
2. \((R_{MSE}-R_{SR})/0.01665036604823182\ge0.10\)；
3. S1、S2、S3、removed各自减correct的regret CI下界均严格大于0；
4. primary/replay、初始状态、父epoch预算、fold/donor、target/mask、finite与sealed-role全部通过。

全部满足才`GO`。若1–2通过而任一输入干预门失败，则`MIXED`；若1或2失败，则`STOP`；机械门失败
则`INVALID`且不读取科学解释。联合门要求所有条件同时成立，不用多次试验中任一显著来制造结论。

## 7. 次要诊断

报告relation MSE、sign/boundary error、prediction/target std ratio、与父V2相比的beneficial/harmful/
neutral choice changes，以及各干预top-choice flips。它们解释主结果，不替代主门，也不得用于事后
选择temperature或子集。

## 8. 结果后的唯一动作

- `GO`：只解锁完整2×3（loss×visual training arm）确认实验的预注册；仍不直接训练DQ；
- `MIXED`：定位scene binding/representation，不扩大方法；
- `STOP`：停止该soft-regret objective，进入表示保留或函数类诊断；
- 任一结果均不解锁NavDream、行为、安全或Best-Paper主张。
