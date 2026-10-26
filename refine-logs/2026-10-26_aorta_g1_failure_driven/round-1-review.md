# Round 1 External Method Review

Reviewer: `gpt-6.1-sol`, reasoning `xhigh`（原要求的 GPT-5.4 在当前环境不可用，已明确降级记录）  
Verdict: `REVISE`  
Score: `7.40 / 10`

## 评分

| 维度 | 分数 |
|---|---:|
| Problem Fidelity | 9 |
| Method Specificity | 8 |
| Contribution Quality | 6 |
| Frontier Leverage | 8 |
| Feasibility | 8 |
| Validation Focus | 7 |
| Venue Readiness | 4 |

## 保留项

1. 问题锚点正确：冻结合法候选和 `U_ctx`，以实际 native-relative gain 而非 loss 改善作为 G1 端点。
2. 严格错排 pair 的 score-margin 梯度下界成立，但只在 pair/score 层成立。
3. 未归一化全 pair loss 对逐样本 hard regret 的上界成立，但不是 population calibration。
4. G1 没有新增可训练模块，适合作为核心视觉方法的资格门而非主贡献。

## 阻塞项

### 1. 多候选 pairwise surrogate 并不 Bayes 一致

固定同一输入 (X)，效用取值为：

| 概率 | (U_0) | (U_1) | (U_2) | (U_3) |
|---:|---:|---:|---:|---:|
| 0.595 | 1 | 1 | 0.59 | 0.40 |
| 0.005 | 0 | 1 | 0.59 | 0.40 |
| 0.400 | 0 | 0 | 0.59 | 0.40 |

条件均值为 ((0.595,0.600,0.590,0.400))，Bayes 应选择候选 1，

\[
G_{Bayes}=0.005,\qquad G_{hindsight}=0.241.
\]

但未归一化全 pair utility-gap logistic 的唯一有限条件风险最优解为

\[
s/T\approx(0,0.04345739,0.17774693,-0.99058401),
\]

它选择候选 2，产生 (G_{model}=-0.005)。因此两候选符号一致性不能推广到多候选 argmax；逐样本
hard-regret 上界也不构成相对 Bayes 的 excess-risk calibration。

### 2. 线性 probe 不能完成信息论归因

令 (Z,C\) 独立均匀取 \(\{-1,+1\}\)，

\[
U_0=0.5,\qquad U_1=0.5+0.25ZC.
\]

完整输入的 Bayes gain 为 0.125，小型非线性网络可实现；但对 ((1,Z,C)) 的线性 ridge 与线性
pairwise probe 都失败。因此 probe 失败只否定该函数类与预算，不能否定视觉信息或输入可辨识性。

### 3. 其他必须修正的边界

- E3C 只能称局部描述性梯度诊断，不能称梯度因果识别。
- 必须冻结 ragged pair reduction、tie、空 panel、温度、loss scale、梯度裁剪和 hard-choice tie-break。
- probe 成功只有在严格函数类包含关系成立时，才可用于收窄神经优化解释。
- 42 groups 已参与前轮方法判断，后续 OOF 是 job 层组外，不是全流程未触碰确认。
- G1 应拆为 optimization qualification 与 visual-information qualification；前者通过不等于视觉成立。
- Differential Query 在 canonical 编号中应保持为 G2，外观鲁棒为 G3。

## 最终意见

保留总体方向，先修正理论与诊断解释，再完成 E3C；pairwise loss只能作为经验资格候选，不能作为严格统计
桥。当前不满足 READY 条件。
