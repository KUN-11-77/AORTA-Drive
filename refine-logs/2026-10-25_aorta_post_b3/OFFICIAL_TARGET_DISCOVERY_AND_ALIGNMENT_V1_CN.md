# NAVSIM 完整官方目标发现与 surrogate 对齐设计 v1

状态：`TARGET_SEMANTICS_FOUND__ALIGNMENT_DRAFT__B4R1_PASS_REQUIRED`。

本文在未读取 B4R1 科学结果的条件下审计本地 NAVSIM 官方执行，回答 one-frame 标签与完整官方
行为目标的差别。它不注册或运行新实验，也不授权 relation 网络训练。

## 1. 权威源码与完整目标

服务器源码根：`/home/jbwang/cogatedrive`。本次读取的关键文件与 SHA-256：

- `pdm_scorer.py`：`c9132b8b7019cc05c57573a26a8326cd7762dfdff35592046625a7fa895f6b62`
- `scene_aggregator.py`：`13dec362355a41c4a6e7386301efe05bcb6774a598a79026cf371ebdd3bc8e65`
- `pdm_comfort_metrics.py`：`883b606cc5f98285bd6ff917e9a4129c608b709d58fe6c2435f6b7bd27c2e6ce`
- `run_pdm_score_from_submission.py`：`9deab2ecd274e033b095f5edeaaa2ea6ff91cf9f7721a22187bb9ea44524ee6a`

one-frame scorer 使用 hard product

\[
H=NC\cdot DAC\cdot DDC\cdot TLC
\]

和 EP/TTC/LK/HC 权重 \((5,5,2,2)\)：

\[
F^{(1)}=H\frac{5EP+5TTC+2LK+2HC}{14}.
\]

官方 final-score 注入由相邻帧模拟状态计算的 two-frame extended comfort \(C_2\)，权重为2。令
\(a_t\in\{0,1\}\) 表示当前 evaluation roster 是否含同 log、时间差不超过0.55 s的前一帧，令
\(A_t=(5EP+5TTC+2LK+2HC)/14\)，则 one-stage 官方实现实际为

\[
\boxed{
F^{(official)}_t
=H_t\frac{14A_t+2a_tC_{2,t}}{14+2a_t}.
}
\]

若 \(a_t=0\)，实现把 two-frame metric 与其权重同时置零，故精确退化为 \(F^{(1)}=HA\)；只有
\(a_t=1\) 时分母为16。\(C_2\) 不是当前候选的单帧属性；它读取前一帧与当前帧模拟轨迹的重叠段。
因此完整目标的
最小上下文是

\[
(X_t,\tau^{ctx}_{t-1},\tau_{0,t},\tau_{i,t}),
\]

而不是仅 ((X_t,\tau_{0,t},\tau_{i,t})\)。这解释了为什么 one-frame oracle 只能资格化“当前
候选存在信号”，不能直接授权最终行为 relation 的训练。

## 2. 精确关系与最坏情形符号界

在 \(a_t=1\) 的相邻帧子域上，由权重可直接得到

\[
F^{(full)}=\frac78F^{(1)}+\frac18HC_2.
\]

对 challenger 与 native 作差：

\[
\Delta F^{(full)}_{i0}=\frac78\Delta F^{(1)}_{i0}
+\frac18\Delta(HC_2)_{i0}.
\]

因为 (H,C_2\in[0,1])，有

\[
\left|\Delta F^{(full)}_{i0}-\frac78\Delta F^{(1)}_{i0}\right|\le\frac18.
\]

故 

\[
\Delta F^{(1)}_{i0}>\frac17
\Longrightarrow \Delta F^{(full)}_{i0}>0,
\]

负向同理。这只是相邻帧子域上无需读取实际 \(C_2\) 的最坏情形充分条件；它不说明小 margin
一定翻转，也不应成为事后筛样阈值。实际 alignment 必须显式执行官方 \(C_2\)。

### 2.1 稀疏 roster 的退化反例

对现有42组×16 anchors的冻结 fit roster，只读时间戳审计得到：

- roster 内相邻 current–previous pair：0/672；
- 具有任一 roster 内相邻 pair 的组：0/42；
- 同一 raw log 内相邻抽样 anchor 的最小正间隔约1.999267 s，超过0.55 s官方门。

因此直接对该 roster 调用官方 adjacency mapping 会令所有 \(a_t=0\)，full score 与 one-frame
score 机械相等，不能检验 surrogate alignment。该100%一致若被报告为科学结果将是接口伪证据。

进一步从官方 raw log 的真实顺序恢复前一行，672/672 anchors均存在唯一合法上下文，42/42组均
为16/16；时间差范围为0.498078–0.500686 s。所以正确方案是为每个 anchor 显式加入真实前一帧
context row，而不是把 anchor index减一当作时间或放弃 two-frame 目标。

## 3. 对网络设计的直接约束

1. 主输出仍应与动作同型，预测 full native-relative scalar margin；不把各 metric factor 强行写成
   核心贡献。
2. 若实证显示 one-frame 与 full target 高度对齐，可把 one-frame 用作低成本 source supervision，
   但最终选择和报告仍以 full target 为准。
3. 若翻转主要由 (C_2) 引起，previous selected/native trajectory 必须成为 relation operator 的
   条件输入，并增加 transition-support / overlap-dynamics 编码；这是官方目标强制的必要复杂度，
   不是为了“更复杂”。
4. 若加入 temporal context 仍无法预测 full margin，停止把当前视觉 relation 写成完整行为效用；
   最多收窄为 one-frame proposal-quality 研究。

## 4. 条件性 alignment 实验草案

只有 B4R1 的合法 one-frame oracle 通过后，才把本草案升级为正式预注册。实验必须在任何 full-
target 结果被读取前冻结。

### 4.1 冻结域

- 沿用 B4R1 的合法 ragged candidate membership；
- 每个 anchor 必须使用 raw log 中已资格化的唯一 previous token；native 与所有 legal challenger
  使用同一个冻结 \(\tau^{ctx}_{t-1}\)；禁止用稀疏 roster 的上一 anchor 替代；
- 当前 preflight 显示672/672上下文可用；正式执行仍须 fail-closed 复查。任何缺失不得静默删除，
  应判定接口漂移；
- 不读取 development/calibration/test/NavDream/target；不调整候选、外观或 scorer。

### 4.2 必须复算的量

对每个 legal challenger 记录：

\[
\Delta^{(1)}_{i0},\quad
\Delta^{(full)}_{i0},\quad
C_{2,i},C_{2,0},\quad H_i,H_0.
\]

以 recording group 为独立统计单位，报告：

- 非 tie pair 的 sign agreement 与 one-frame-positive/full-nonpositive false intervention rate；
- one-frame top choice 与 full top choice 的一致率；
- 用 one-frame 选出的候选在 full target 上的 regret；
- two-frame comfort 新增 harm 的数量、分组与 margin 区间；
- 按 \(|\Delta^{(1)}|\)、速度、曲率、appearance、候选数分层的翻转率；
- structural-unavailable、零 challenger 与 tie 的完整分母。

### 4.3 机械审计

- 重新执行官方 `SceneAggregator` 与 `compute_final_scores`，逐候选复算；
- 检查前后帧 token、时间间隔、轨迹采样、overlap index 与 source hashes；
- 检查同一 anchor 的所有候选共享相同 previous context；
- 检查 roster-only aggregation 确实为全 \(a_t=0\)，augmented-context aggregation确实为全
  \(a_t=1\)，防止两条路径被混淆；
- 由保存的 factor 独立复算上式，要求逐项数值闭合；
- primary/replay/audit 三份 materialization 一致后才能读取科学统计。

### 4.4 结论分支

- **ALIGN：** sign、top-choice、regret 与 false-intervention 的预注册门全部通过，才允许 one-frame
  supervision 进入 relation 训练；full target 仍负责最终裁决。
- **TEMPORAL-REDESIGN：** one-frame oracle 存在但 two-frame 造成不可接受翻转；加入 previous-
  action context 后重新注册一次 matched study。
- **STOP/REDUCE-SCOPE：** temporal redesign 后仍不对齐，停止完整行为 relation 主张。

数值门槛尚未在本文件冻结。它们必须由决策成本、官方分数分辨率和可实现的 group-level power
共同推导，而不能在看见 alignment 结果后选择。因此本文件是 target-discovery 结果与实验接口
草案，不是正式 preregistration。

## 5. 当前裁决

完整官方目标已经在源码层面定位，并证明 temporal context 可能是被任务定义强制的必要输入。
现有稀疏 roster 无法直接检验该项，但真实 raw-log previous context 已在 outcome-free 审计中达到
672/672可用；下一步 alignment 必须采用 augmented-context roster。
当前没有理由继续把 one-frame margin 当作最终训练真值；也没有理由在看到实际翻转前立即扩展
网络。正确顺序仍是 B4R1 → frozen alignment → 必要时 temporal redesign → matched architecture。

