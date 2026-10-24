# H7BS4B2 后理论候选：Fork-Aware Decision Quotient

状态：`THEORY_DESIGN_ELIGIBLE__ORACLE_VALIDATION_REQUIRED__NO_NETWORK_TRAINING`。

## 1. 新的中心事实

B2 对 B1 的 2,219 个 exact-zero teacher contrasts 作了运行前冻结的互斥归因：

- 2,033（91.6178%）由共同区域 minimum 同时控制两条候选，覆盖 42/42 groups；
- 186（8.3822%）由 1 m discrete support duplicate 造成；
- distinct-minimum teacher tie 为 0。

因此当前 scalar global set-min 的主要失败不是 teacher plateau，也不是视觉预测误差，而是算子把
**两条候选共同经历的风险**和**分叉后真正区分选择的风险**压缩成同一个最小值。

## 2. 不可区分性定理

令候选 (i,j) 的查询集合分解为共同部分 (C) 与独占分支 (A_i,A_j)。若

\[
c=\min_C d,qquad a_i=\min_{A_i}d,qquad a_j=\min_{A_j}d,
\]

则 global contrast 为

\[
\Delta_{ij}^{\mathrm{global}}
=\min(c,a_i)-\min(c,a_j).
\]

当 (c\le a_i,a_j) 时，

\[
\Delta_{ij}^{\mathrm{global}}=0.
\]

该零差异是算子的确定性性质；更强 backbone、更低 field MAE、更复杂 uncertainty 或更多训练数据
都不能恢复已经被 aggregation 删除的信息。B2 在冻结 fit panel 上观察到该条件解释了绝大多数
exact zeros，因此 task operator 必须先于网络修改。

## 3. Fork-aware factorization

候选方法不删除共同风险，而将它分成两个逻辑角色：

### 3.1 Shared-trunk qualification

\[
G_{ij}(d)=\min_{q\in C_{ij}}d_q-m_0.
\]

共同部分是两种行为都必须经历的结构条件。但这里必须区分两种命题：

1. **相对 non-degradation：**若两候选具有完全相同的时间前缀，则切换分支不会改变此前共同经历；
2. **绝对 road safety：**要求 (\min_C d>0)，但该命题不具有 gauge invariance，不能从 quotient
   potential 单独识别。

因此 B3 的主资格只能使用可辨识的“exact shared temporal prefix + gauge-invariant branch
contrast”。teacher 的绝对正负仅作 oracle 分层。未来若要声明绝对 shared safety，必须增加具有
独立监督和校准责任的 absolute feasibility 接口；在其必要性得到证据前，不把它塞入主方法。

### 3.2 Branch contrast

\[
D_{ij}(d)=\min_{q\in A_j}d_q-\min_{q\in A_i}d_q.
\]

只有 (A_i,A_j) 都非空且 shared-trunk qualification 通过时，才评价 branch contrast。联合不确定
集合下的候选量为

\[
L_{ij}^{\mathrm{fork}}
=\inf_{d\in\mathcal U(X,O)}D_{ij}(d).
\]

干预还需要冻结 candidate legality；(L_{ij}^{\mathrm{fork}}>0) 只表示 model-relative branch
preference，不是现实安全证书。任何绝对合法性/安全性接口必须单列假设、监督与校准，不能由
quotient branch score 暗示。

## 4. 两种“共同部分”不能混同

### Spatial overlap

\[
C_{ij}^{\mathrm{space}}=F_i\cap F_j.
\]

优点是精确、无阈值、直接解释 B2；缺点是不同时间访问同一空间或分叉后重新汇合也会进入交集，
未必等价于共同决策历史。

### Temporal prefix

令离散时刻车体 footprint 为 (F_i^t)。以首次时序 footprint 集合不相等作为无连续阈值的分叉
索引：

\[
t_{ij}^{\mathrm{fork}}=\inf\{t:F_i^t\ne F_j^t\}.
\]

此前为 shared trunk，之后为 branches。该定义符合决策时间顺序，但受 1 m rasterization 影响，
必须与连续位姿 ledger 和更细/连续 query 做敏感性审查。

下一门必须同时实现两种定义并按反例裁决，禁止只报告更有利者。

## 5. 对网络架构的约束

在 oracle fork operator 通过前，不训练新网络。若通过，候选架构应保持最小充分复杂度：

1. 视觉编码器与 candidate-independent quotient potential 只运行一次；
2. deterministic fork-factorization layer 从冻结 candidate panel 构造 shared trunk 与 branches；
3. 先用简单 shared-node uncertainty baseline；B1 未支持 graph-correlation 的决策必要性，不能自动
   恢复复杂 joint head；
4. appearance consistency 约束可辨识 operator，不约束任意绝对 gauge；
5. loss 仍以 proper task relation learning 为主，branch ranking loss 只有在容量匹配消融中单独加入，
   不能先混入后声称 fork structure 必要。

## 6. H7BS4B3 的证明义务

在同一 fit teacher 上，B3 必须报告：

1. spatial 与 temporal factorization 对 2,033 common-min zeros 各恢复多少非零 branch contrasts；
2. 其中多少 shared trunks 本身 unsafe 或 uncertainty-unknown，防止通过忽略共同风险制造可分性；
3. 空 branch、重新汇合、交叉路径、离散 aliasing 与 full-horizon 边界的反例；
4. global min、固定 suffix min、spatial fork、temporal fork 的相同 teacher 对照；
5. 计算量与可共享性，避免每个 candidate 重复神经前向。

若 fork-aware oracle 不能在保持 shared qualification 的条件下显著恢复可分性，停止道路裕度作为
一般 candidate reranker。若通过，只解锁 matched-capacity network design，不直接解锁 NavDream、
H7C 或驾驶收益。

## 7. 顶会故事的当前候选版本

> 外观鲁棒驾驶通常让特征或稠密风险图保持一致，但我们发现 even perfect road margins 也会被
> global minimum 的共同前缀湮灭，无法区分真正分叉的驾驶行为。AORTA 将候选面板因子化为共享
> trunk 与未来 forks：先验证共同部分，再只比较分支特有的任务证据；视觉网络学习跨外观稳定的
> quotient potential，而 deterministic fork operator 保留决策差异。看不清或共同部分不合格时，
> 保留原生决策。

这比“DINO + field + uncertainty + reranking”更连贯，但目前仍是由 fit 机制结果支持的候选故事，
不是已验证方法贡献。
