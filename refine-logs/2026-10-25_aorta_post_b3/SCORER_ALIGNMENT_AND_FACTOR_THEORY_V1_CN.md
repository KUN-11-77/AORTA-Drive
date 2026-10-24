# AORTA scorer 对齐与因子关系理论 v1

状态：`SOURCE_AUDITED_THEORY__SURROGATE_ALIGNMENT_UNTESTED__NO_TRAINING_AUTHORIZED`。

本文回答一个容易被忽略的问题：Differential-Support Relation v2 究竟在学习什么标签？B4R1 的
candidate-local one-frame PDM score 是否足以支撑最终 official PDMS / EPDMS 行为主张？结论是：
B4R1 标签适合做 oracle 资格门，但在完成 surrogate-alignment gate 前，不应直接成为最终网络监督。

## 1. 权威源码语义

已在服务器冻结 NAVSIM 源码上读取 `PDMScorer.score_proposals` 与各 factor 实现；权威 source hash
已由 `H7BS4B4R1_SCORER_SEMANTICS_AUDIT.md` 和冻结 config 记录。one-frame score 可写为

\[
F(\tau)=H(\tau)W(\tau),
\]

其中 hard feasibility product 为

\[
H=\mathrm{NC}\cdot\mathrm{DAC}\cdot\mathrm{DDC}\cdot\mathrm{TLC},
\]

weighted utility 为

\[
W=\bar\lambda_{EP}\,\mathrm{EP}
+\bar\lambda_{TTC}\,\mathrm{TTC}
+\bar\lambda_{LK}\,\mathrm{LK}
+\bar\lambda_{HC}\,\mathrm{HC},
\quad \sum_k\bar\lambda_k=1.
\]

所有因子位于 \([0,1]\)。进度在每次内部 `[fixed PDM baseline, evaluated trajectory]` 二元 proposal
set 内归一化；candidate 还各自诱导 reactive traffic。B4R1 标签为两次独立调用之差

\[
\delta^{1f}_{i0}=F_S(\tau_i)-F_S(\tau_0).
\]

它省略跨相邻 anchor 的 two-frame extended comfort，因此不是完整 official PDMS / EPDMS。

## 2. 精确关系分解

令 \(H_i,W_i\) 与 \(H_0,W_0\) 分别为 challenger 和 native 的 hard/weighted aggregate，定义

\[
\Delta H=H_i-H_0,\quad
\Delta W=W_i-W_0,\quad
\bar H=\frac{H_i+H_0}{2},\quad
\bar W=\frac{W_i+W_0}{2}.
\]

则有精确恒等式

\[
\boxed{
\delta^{1f}_{i0}=\bar H\,\Delta W+\bar W\,\Delta H.
}
\]

这说明总 relation 同时来自：

1. hard feasibility 是否改变；
2. 在 hard gate 权重下，进度/TTC/车道/舒适性的软差异。

差异支持不是“道路视觉 margin”的同义词。它是候选比较的几何 query；真正 label 还依赖动态 actor、
交通灯、路线/地图、运动学和 scorer normalization。全局 visual keys 与完整 relative trajectory path
因此是必要输入，而不是对 signed support 理论的例外。

## 3. 因子误差到 score / relation error 的界

对任意 \(a,b\in[0,1]^m\)，

\[
\left|\prod_j a_j-\prod_j b_j\right|
\le\sum_j|a_j-b_j|.
\]

若某候选的 hard factors 与 weighted factors 的预测误差满足

\[
\sum_j|\widehat h_j-h_j|\le\varepsilon_H,
\qquad
\sum_k\bar\lambda_k|\widehat w_k-w_k|\le\varepsilon_W,
\]

则由于 \(H,W,\widehat H,\widehat W\in[0,1]\)，

\[
|\widehat F-F|
\le |\widehat H-H|+|\widehat W-W|
\le\varepsilon_H+\varepsilon_W.
\]

若 challenger 与 native 均满足同一界，则

\[
|\widehat\delta_{i0}-\delta^*_{i0}|
\le2(\varepsilon_H+\varepsilon_W),
\]

再结合已有 selection inequality，可得

\[
\operatorname{Regret}\le4(\varepsilon_H+\varepsilon_W).
\]

该界说明 factor supervision 可以提供诊断性误差账本，但通常比直接 margin 的
`Regret <= 2 epsilon_delta` 更松。因而 factor head 不是数学上必需的主架构，也不能仅凭可解释性
加入；DA-WAM 已使用 factor/utility heads，AORTA 必须把它作为 matched optional arm，而不是新贡献。

## 4. 输入职责而非模块职责

official factors 对输入的依赖不是纯二分，但可形成如下责任账本：

| 因子 | 主要依赖 | 对 v2 的约束 |
|---|---|---|
| NC / TTC | candidate时空footprint、动态actor与reactive traffic | differential support query + 全局动态视觉上下文 |
| DAC | footprint与drivable geometry | 正负差异支持最直接的物理语义 |
| TLC | footprint与红灯token/控制状态 | 不能把视觉keys裁剪到轨迹差异区 |
| DDC / LK | candidate几何、路线/centerline、地图语义 | trajectory path 与场景/路线上下文共同输入 |
| EP | candidate endpoint、centerline及candidate-local normalization | 不能只靠视觉；必须保留完整轨迹与scorer语义 |
| HC | past human states与candidate运动学 | 可以由trajectory/history分支解释，不应强迫其依赖图像 |

因此“correct image 必须让所有 pair 都变好”不是合理主张。若某个 margin 完全由 HC/EP 或显式轨迹
几何决定，mismatch image 不一定改变正确预测。视觉必要性必须在预注册 factor strata 上评价：

- scene-dependent hard/TTC/DAC/TLC/LK disagreement；
- trajectory-dominant EP/HC disagreement；
- mixed-factor disagreement；
- all-tie / score-compressed cases。

全体 pair 的 correct-vs-mismatch 仍报告，但不能掩盖分层机制。

## 5. 主输出与 optional factor arm

v2 主模型保持最小直接输出：

\[
(X,\tau_0,\tau_i)\mapsto\widehat\delta_{i0},\sigma_{i0}.
\]

原因是该输出与 native replacement、同时校准和 regret theorem 完全同型。以下只作为匹配实验 arm：

1. **Direct relation:** 直接预测 \(\delta_{i0}\)；
2. **Factor auxiliary:** 同一 trunk 附加合法 factor targets，但最终仍由 direct relation 决策；
3. **Factor-composed:** 预测候选/native factor vectors，通过冻结 official one-frame aggregation 合成
   margin；
4. **DA-WAM-style factor/utility scorer:** 容量与 token/FLOP 匹配的近邻基线。

若 factor auxiliary 不降低 held-out regret 或只改善 factor MAE，删除它。若 factor-composed 明显优于
direct relation，则主数学对象需要升级为 factor relation vector；不能在结果前预设赢家。

## 6. B4R1 后必须插入的 surrogate-alignment gate

B4R1 只问 \(\delta^{1f}\) 是否在合法 source-fit panel 上存在非平凡 oracle opportunity。它通过后，
在任何 relation 网络训练前，必须另行预注册并验证：

1. one-frame margin 与可合法构造的完整 official target 在 pair sign、top choice 与 regret 上是否一致；
2. 用 one-frame oracle 选择 challenger 是否新增 two-frame extended-comfort failure；
3. score compression/tie 是否集中在特定 factor、planner 或候选幅度；
4. 以 recording group 为独立单位的 sign agreement、false-positive replacement、rank correlation 与
   bootstrap interval；
5. 所有候选仍使用同一冻结 membership，禁止只评估 one-frame 有利的 complete cases。

具体阈值必须在读取 alignment outcome 前，根据最终行为最小有意义差异与统计功效冻结。该 gate 的
逻辑裁决为：

- **PASS：**one-frame surrogate 足以授权后续 matched architecture training；
- **HOLD：**加入 previous selected trajectory / history context，并学习 transition-aware relation；
- **STOP/SCOPE REDUCTION：**若 surrogate 无法预测 official behavior，则不得以其训练 ICCV 主方法，
  只能收窄为 one-frame representation study。

## 7. 对跨 planner 与跨时刻主张的影响

\(F_S(\tau)\) 使用相同 scorer，但候选生成、membership、native 与 score distribution 随 planner
改变。因此跨 planner 的共享对象是**条件 relation mechanism**，不是一个无条件绝对 utility。
calibration 也不能继承。

在连续部署中，若 AORTA 在时刻 \(t\) 替换 native，则下一时刻的历史舒适、观测和 planner state
可能改变。静态 one-frame reranking 无法证明这一闭环效应。最终论文必须把 NAVSIM v1/v2 静态或
伪闭环证据的边界写清楚，并避免把 PDMS 改善写成真实道路安全保证。

## 8. 当前裁决

- signed differential support 仍是合理的**query geometry**，不是完整 utility 的解析表达；
- primary scalar relation 保持最小且与动作同型；
- factor head 降为可删除的 matched arm，避免复制 DA-WAM；
- B4R1 标签继续只承担 oracle qualification；
- B4R1 通过后必须先过 surrogate-alignment gate，再允许任何 relation 网络训练。

这项收紧使理论—标签—网络—行为链更长了一步，但避免在错误 surrogate 上得到漂亮的模型结果。

