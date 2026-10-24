# H7BS4B1 后理论修正：从 uncertainty 复杂度退回 task-operator 可辨识性

状态：`B1_UNSUPPORTED__B2_ZERO_CONTRAST_ANATOMY_REQUIRED`。

## 1. B1 排除了什么

B1 在相同 teacher inclusion 与 marginal bounds 下比较 joint Lipschitz field、shared-node box 与
candidate-independent intervals。机械审计通过，但三者的 sign-identifiable pair 数完全相同：
1,301/3,864；joint 相对 shared box 的组级提升为 0，95% CI `[0,0]`，且只有 5/42 groups 达到
最低 anchor coverage。

因此不能继续把“graph-correlated uncertainty 提高候选可识别性”当作当前方法贡献。joint 的平均
interval width 更小，只能证明它保留了部分相关结构，不能证明该结构在冻结决策端点上有用。

## 2. 新暴露的数学瓶颈

令两个候选的完整扫掠节点集合为 (F_i,F_j)，定义

\[
C=F_i\cap F_j,\qquad A=F_i\setminus C,\qquad B=F_j\setminus C.
\]

采用约定 (min\varnothing=+\infty)，并令

\[
c=\min_{q\in C}d_q,\qquad a=\min_{q\in A}d_q,\qquad b=\min_{q\in B}d_q.
\]

原 global set-min contrast 可精确分解为

\[
\Delta_{ij}(d)=\min(c,a)-\min(c,b).
\]

### 命题：共同极小值湮灭候选差异

若 (c\le a) 且 (c\le b)，则

\[
\boxed{\Delta_{ij}(d)=0}.
\]

证明直接由两侧最小值都等于 (c) 得到。该结论与视觉 backbone、网络容量、训练 loss 和
uncertainty sharpness 无关；只要共同区域控制两条候选的 global minimum，任何精确学习该标量
算子的模型都无法区分后续分支。

B1 中 2,219/3,864 teacher contrasts 精确为 0，说明必须先审查这一结构性不可区分性，而不是
继续优化区间或训练新 head。但目前尚未证明 2,219 个零差异主要由共同极小值造成。

## 3. 四种竞争解释

H7BS4B2 必须按互斥优先级将 exact-zero pairs 归因：

1. **Discrete-support duplicate：** (F_i=F_j)。不同连续轨迹在 1 m 节点化后成为同一查询；
2. **Common-min domination：** (F_i\ne F_j)，但共同集合 (C) 含两侧 global minimizer；
3. **Distinct-min teacher tie：** 两侧 minimizer 不在共同集合，却具有相同 teacher 值；
4. **Task equivalence：** 连续轨迹与查询支持确实不同，但道路裕度对该候选差异本就不敏感。

对 (|\Delta|<0.01\) m 但非零的 pairs 另行报告，不得并入 exact-zero 以夸大某一机制。

## 4. H7BS4B2 最小充分诊断

只使用同一 42 个 fit groups 的冻结候选、完整 4 s 轨迹、节点集合与 teacher；不读取 learned
field、DINO、planner outcome、development、NavDream 或 test。逐 pair 输出：

- 连续轨迹最大距离、首次分叉时间与离散节点集合是否相同；
- (C,A,B) 大小及各自 teacher minimum；
- native/challenger argmin 是否落在共同集合；
- exact-zero 的唯一归因类别；
- 非零近 tie 的绝对 contrast 分布。

该实验是 fit-only 机制归因，不提供独立泛化证据。实现和分类规则必须在重新读取逐 pair 数据前
预注册并经合成反例测试。

## 5. 结果驱动的严格分支

- **离散重复占主导：** 优先比较连续坐标或更细查询与 candidate dedup；不发明新的 uncertainty；
- **共同 min 占主导：** 才允许研究“共同前缀资格门 + 分支后缀 contrast”的 fork-aware operator；
- **distinct-min tie 占主导：** 检查 teacher 数值分辨率、SDF plateau 与任务标签；
- **task equivalence 占主导：** 道路裕度只能作为选择性 road-feasibility veto，不能承担一般
  candidate reranking；若需要完整驾驶排序，必须另证多任务算子的必要性。

上述分支先裁决数学对象，再决定网络。不得同时实现四种修复并从结果中挑最好者。

## 6. 条件候选：fork-aware operator（尚未采用）

仅当 common-min domination 被支持时，考虑把 pair 分解为共享资格和分支差异：

\[
G_{ij}(d)=\min_{q\in C}d_q-m_0,
\qquad
D_{ij}(d)=\min_{q\in A}d_q-\min_{q\in B}d_q.
\]

只有共享部分在不确定集合下满足资格时，才允许使用 (D_{ij}) 比较分支；共享部分 unsafe 或
unknown 时 abstain。该结构避免直接删除共同风险，同时防止已证明安全的共同前缀湮灭后续分支
差异。空集、重新汇合、交叉轨迹、时间不同但空间重叠和数值 tie 都是必须处理的反例。

这只是由定理提出的候选接口，不是已验证算法。若 B2 不支持 common-min 解释，应删除。

## 7. 网络与 loss 的当前边界

- 暂停 joint uncertainty head 训练；
- quotient potential 仍是候选表示，但尚未获得容量匹配的结构必要性证据；
- 不新增 branch loss、ranking loss 或多任务 head；
- 只有 B2 固定最终任务算子后，才能重新定义网络输出、proper loss 和结构消融；
- H7C、NavDream 与驾驶收益继续 HOLD。

这一修正使理论更严格：实验失败不是通过更复杂网络掩盖，而是先检查监督对象本身是否携带
候选决策信息。
