# NAVSIM 完整目标、one-frame 界与时序上下文合同 v2

状态：`FULL_TARGET_IS_FINAL_ESTIMAND__ONE_FRAME_ORACLE_ONLY__NO_OUTCOMES_READ`。

本文取代 v1 中“alignment 通过后可把 one-frame 当最终训练标签”的宽松表述。修正完全来自官方
源码公式、已通过的 augmented-context roster 预检与表示充分性条件，不使用 B4/B4R1 outcome。

## 1. 完整 official target 是唯一最终行为估计对象

对具有合法 previous row 的当前时刻 \(t\)，官方 one-stage final score 为

\[
F^{full}_{i,t}=\frac78F^{(1)}_{i,t}+\frac18Z_{i,t},
\qquad Z_{i,t}=H_{i,t}C_{2,i,t}\in[0,1].
\]

其中 \(C_2\) 由前一帧与当前帧的模拟轨迹重叠段计算。因此最终 native-relative label 必须是

\[
\delta^{full}_{i0,t}
=F^{full}_{i,t}-F^{full}_{0,t},
\]

其最小决策上下文写成

\[
C_t=(\pi,\tau^{sel}_{t-1},\tau_{0,t},V_t,a_t).
\]

\(a_t=0\) 表示不存在合法 previous context，此时 official target 精确退化为 one-frame；该状态
必须显式编码，不能用零轨迹冒充普通历史。

### 决策 1

B4R1 的 \(F^{(1)}\) 只用于低成本回答“合法候选域是否存在任何任务信号”。即使 B4R1 通过，主
relation 网络的最终监督默认仍为 \(\delta^{full}\)，不能把经验上的高相关改写为标签等价。

## 2. one-frame 能提供的精确最坏情形保证

对任意两个合法候选 \(i,j\)：

\[
\Delta^{full}_{ij}
=\frac78\Delta^{(1)}_{ij}+\frac18(Z_i-Z_j),
\]

所以

\[
\left|\Delta^{full}_{ij}-\frac78\Delta^{(1)}_{ij}\right|\le\frac18.
\]

由此得到无需经验阈值的符号证书：

\[
|\Delta^{(1)}_{ij}|>\frac17
\Longrightarrow
\operatorname{sign}(\Delta^{full}_{ij})
=\operatorname{sign}(\Delta^{(1)}_{ij}).
\]

若 one-frame 最优候选 \(j\) 对所有 \(i\ne j\) 均满足

\[
F^{(1)}_j-F^{(1)}_i>\frac17,
\]

则 \(j\) 也是 full-target 唯一最优候选。若 \(j\) 相对 native 的 one-frame margin严格大于
\(1/7\)，则其 full-target improvement 严格为正。

此外，令 \(j=\arg\max_iF^{(1)}_i\)，\(i^*=\arg\max_iF^{full}_i\)，则

\[
F^{full}_{i^*}-F^{full}_j
\le\frac18.
\]

这是 one-frame top choice 的确定性最坏 regret 上界；\(1/8\) 在 \([0,1]\) score 尺度上并不
足够小，不能据此声称代理可接受。

### 决策 2

\(|\Delta^{(1)}|>1/7\) 只允许被称为“worst-case sign-certified subdomain”，不能用于事后筛除
困难样本或训练后挑阈值。其 coverage、候选供给与 hard-failure 分布必须完整报告。其余 pair
属于 temporal-ambiguous domain，必须显式计算 full target。

## 3. 为什么 previous selected trajectory 是上下文而不是可选装饰

表示充分性要求真实 target 在模型表示的每个纤维上为常数。若模型不读取
\(\tau^{sel}_{t-1}\)，则两个当前图像、当前候选完全相同但 previous overlap dynamics 不同的
上下文会映射到同一表示；只要 \(C_2\) 使 margin 不同，current-only 表示就不具一般充分性。

因此：

1. **估计对象层面：**previous selected trajectory / overlap state 始终属于 full-target context；
2. **最小架构层面：**先用低成本、确定性的 previous-action geometry token 编码，不默认增加第二个
   时序 backbone；
3. **神经网络复杂度层面：**只有简单 context token 相对 current-only 有任务信息，但仍不能恢复
   full margin 时，才允许注册 transition-support attention 或专用 temporal encoder；
4. **部署层面：**输入必须是系统上一周期实际选择并执行/计划的 trajectory，而不是永远使用原
   planner native；否则 interventions 后发生 train–deploy context shift；
5. **首帧/断流：**显式设置 \(a_t=0\)，使用官方 one-frame target 语义并默认保持 native，除非
   one-frame worst-case sign certificate 成立。

这增加的是已知状态输入，不是第二项论文贡献。专用 temporal branch 是否存在由 matched ablation
决定。

## 4. alignment 实验不再承担“挑一个通过阈值”的职责

B4R1 接口有效后，contextual-target 实验固定回答四个事实问题；B4R1 的 one-frame 科学
GO/HOLD/STOP 不替代该实验：

1. one-frame worst-case sign-certified pair / anchor / group coverage；
2. temporal-ambiguous domain 的实际 sign flip、top-choice flip 与 full regret；
3. full oracle 相对 one-frame oracle 的增益、损失和新增 hard harm；
4. previous-context residual \(\Delta^{full}-7\Delta^{(1)}/8\) 是否能由最小 geometry token解释。

任何一个非 tie 的 sign/top-choice counterexample 都足以否定“current-only one-frame 对完整目标普遍
决策充分”的命题。这是反例裁决，不需要人为通过阈值。零 counterexample 也只表示当前 source
样本未反驳，不能证明未来 NavDream 或另一 planner 上充分。

## 5. 条件实验臂

所有 arms 使用同一合法 ragged domain、full labels、视觉输入、参数/FLOP预算、训练数据与 seeds：

| Arm | 输入/目标 | 目的 |
|---|---|---|
| A | current-only，预测 one-frame margin | 低成本 proxy baseline，不是最终方法 |
| B | current/history visual + current candidates，预测 full margin | 检查视觉 history 是否隐式包含所需状态 |
| C | B + previous selected trajectory geometry token | 最小理论一致模型 |
| D | C + transition-support / temporal encoder | 仅在 C 仍不足时注册的复杂模型 |

primary endpoint 是 held-out full-target regret 与 false intervention；full-margin MAE、two-frame factor
MAE 只作诊断。若 C 不优于 B 且没有 source counterexample，保留 B 作为更简单实现，但论文仍不
宣称 previous context 在一般分布上无关。若 D 不优于 C，删除 D。

## 6. full-target materialization 的冻结要求

- 使用 T0 已资格化的672/672真实 previous raw rows；禁止 roster anchor index减一；
- 同一 current anchor 的 native 与全部 legal challenger 共享同一个 previous selected trajectory；
- 保存 \(F^{(1)},H,C_2,Z,F^{full}\) 并逐项复算公式；
- primary/replay/audit 检查 token、gap、overlap index、source hash、ragged membership与完整 ledger；
- 不删除 temporal-ambiguous、zero-challenger、tie 或 hard-failure anchors；
- 不读取 development/calibration/test/NavDream/target；
- B4R1 接口 INVALID 时不执行；接口有效时，无论 one-frame 科学 outcome 为 GO/HOLD/STOP，均按
  独立预注册决定 contextual target 是否有信号。

## 6.1 历史分布一致性

训练标签中的 previous selected trajectory 与部署时真实写回的 previous selection 必须区分：

- `native-history`：始终使用原 planner 上一周期选择，只是 teacher-forced baseline；
- `actual-selected-history`：使用当前策略上一周期真实选择，是部署输入；
- `sequential rollout`：连续更新 history，检验一次干预是否改变后续状态分布。

one-step full-margin 准确不能推出 sequential 改善。若 native-history 与 actual-selected-history 的
regret/false-intervention 明显不同，必须使用 evaluator 支持的 counterfactual-history
materialization 或冻结 rollout/data-aggregation 方案覆盖该分布；复杂 temporal encoder 不能替代
错误输入分布的修复。

## 7. 对论文故事的影响

论文不再声称“one-frame surrogate 与 full target 足够相关，所以可以忽略时序”。更严谨的故事是：

> AORTA 首先在经 evaluator 作用域核验的 contextual decision object 中定义 native-relative
> relation；one-frame 只提供具有
> 显式 \(1/7\) 符号证书的低成本子域。对其余决策，模型读取上一周期已知动作上下文，并在无法
> 建立同时下界时保留 native。

这使 target、输入、网络、仲裁和官方评估使用同一个数学对象，同时把复杂 temporal network 留给
消融裁决。

其中 \(7/8+1/8\) 恒等式必须在冻结 evaluator 中逐项核验其乘性门、过滤、归一化和聚合层级；在
该 scope audit 关闭前，本文的 `full` 只表示 scoped two-frame candidate utility，不得扩写成完整
EPDMS 恒等式。
