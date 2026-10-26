# Round 1 Refinement: 从“新 surrogate”收缩为“可辨识—可实现—可优化”资格链

## 1. 接受审查后的核心修正

原方案把 utility-gap pairwise logistic 放得过近于理论闭环。现在明确撤回这一暗示：

- 它保留逐样本错排梯度与 hard-regret 上界；
- 它在随机多候选条件分布上存在 Bayes 不一致反例；
- 它不能输出条件期望 margin，不能直接接入 utility-unit lower confidence bound；
- 它至多是一个需要冻结实验裁决的优化工具，不是 G1 的默认最终目标。

因此，G1 的第一性统计对象重新固定为条件期望效用：

\[
\mu_i(X)=\mathbb E[U_i\mid X],\qquad
i^*(X)\in\arg\max_i\mu_i(X),
\]

其中 (X=(I,B,h))。任何训练目标若声称完成统计桥，必须证明其 population minimizer 的 argmax 与
上述 Bayes action 一致；否则只能列为经验 surrogate。

## 2. 四层失败分解

后续不再使用排他式三分法，而采用可嵌套的证据层：

\[
G_{hindsight}>0
\not\Rightarrow G_{Bayes}(X)>0
\not\Rightarrow \exists f\in\mathcal F: G(f)>0
\not\Rightarrow \operatorname{Opt}(L,\theta_0)\text{找到它}
\not\Rightarrow \widehat G_{group}>0.
\]

分别对应：

1. **机会层**：冻结 panel 是否存在 hindsight headroom；
2. **信息层**：决策时输入是否改变条件期望 utility；
3. **函数类层**：指定 scorer 类是否能表示有益规则；
4. **优化层**：指定 objective/初始化/预算是否能恢复该规则；
5. **估计层**：有限 groups 上是否泛化并给出正置信下界。

任一有限 probe 的失败只能否定其覆盖的函数类与预算，不能跨层宣布信息缺失。

## 3. Probe 改为有包含关系的 capacity ladder

不再使用“线性 probe 成败决定信息/优化”的二叉树，而使用明确的嵌套资格：

1. **Exact-head probe**：冻结当前 scorer 在初始化时的候选表示，只优化其同构线性 native-relative head。
   该类严格嵌入当前网络；若成功，证明当前网络函数类包含正收益解，但仍不能单独区分 objective 与优化器。
2. **Linear raw-feature probe**：只作为低容量基线；失败不解释为信息缺失。
3. **Small interaction probe**：加入预注册的候选×视觉双线性交互或两层小 MLP，用于检验 XOR 型交互。
   成功只说明该 probe 类可恢复；失败仍只给出 `HOLD / evidence insufficient`。
4. **Correct/removed/mismatch**：只有组外 correct 优于两个控制，才支持视觉信息 qualification。

所有 probe 固定 split、特征、维度、正则化网格与选择规则；不得用科学结果选择容量。

## 4. 优先 objective：proper native-relative mean regression

由于部署决策要求最大化条件期望 utility，统计上最干净的基线仍是 native-relative conditional-mean
regression，而不是 realized-winner 分类或全 pair 排序：

\[
\Delta_i=U_i-U_0,\qquad
L_{mean}=\sum_{i>0}m_i\,(s_i-\Delta_i)^2,qquad s_0=0.
\]

在权重 (m_i=m_i(X)) 只依赖决策时可见输入、且对合法 challenger 严格为正时，条件风险最优解仍为

\[
s_i^*(X)=\mathbb E[\Delta_i\mid X].
\]

它保留 Bayes argmax，一阶梯度为 (2m_i(s_i-\Delta_i))，不含 action-softmax 的 (p_i\) 尾部因子。
R005E2B 已使用平方 margin 回归并得到 relation-only signal，因此下一步不是把它冒充新方法，而是通过 E3C
判断其失败主要来自稀疏机会、参数冲突、表示交互还是有限估计；只有预注册的单因素修改才可继续。

允许的最小修改候选按优先级为：

1. 保持 proper risk 的数值/采样修正（例如输入可见权重、等 group reduction、过采样配逆概率校正）；
2. exact-head 或 small-interaction realizability qualification；
3. pairwise logistic 仅作为经验负/正控制，不用于严格统计主张。

禁止在同一轮同时改变 loss、sampling、network 和 optimization budget。

## 5. E3C 的准确定位

E3C 只报告冻结初始点与 E3B 终点处的：

\[
g_P^\theta=\sum_{i:\Delta_i>0}J_i^\top g_i^s,
\qquad
g_N^\theta=\sum_{i:\Delta_i\le0}J_i^\top g_i^s.
\]

它能描述 score-tail 衰减、局部参数梯度质量与 Euclidean cosine，不能识别训练路径因果、AdamW
预条件冲突或输入信息缺失。结果只用于排除与排序下一假设，不直接触发方法 PASS。

## 6. 冻结实验决策树

1. 完成 E3C primary/replay/audit；
2. 若正机会 score 梯度已在终点衰减，但初始参数梯度并不弱：优先比较 proper mean regression 与历史
   soft-regret，保持 identity scorer 和全部预算不变；
3. 若初始 (g_P^\theta\) 已极弱或与 (g_N^\theta) 强冲突：先做 exact-head / interaction capacity ladder，
   不先换 loss；
4. 若 exact-head 成功而完整网络失败：只能收窄到 objective/optimization/estimation集合，再做一个单因素实验；
5. 任一 source-fit outer-train 成功只解锁 job-level group-held-out qualification；
6. 只有 optimization 与 correct-vs-control information 两门均通过，才关闭 G1 并进入 G2 Differential Query。

## 7. 顶会贡献边界

G1 的价值是建立无 native-fallback 伪阳性的公平训练资格与失败机制证据，而非把 MSE、pairwise loss 或
probe 包装成创新。ICCV 主贡献仍必须来自：candidate differential support 如何在 matched capacity/latency
下更有效地读取 appearance-interventional、decision-sufficient visual evidence，并最终改善可审计行为。
