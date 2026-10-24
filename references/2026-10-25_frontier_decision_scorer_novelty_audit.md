# 2026-10-25 前沿候选决策与 scorer 查新审计

状态：`TARGETED_FRONTIER_AUDIT__NOVELTY_NARROWED__NOT_SYSTEMATIC_REVIEW`

本轮使用官方 arXiv API 与官方 HTML 正文，审查 2026 年 8--10 月最接近 AORTA 的候选排序、
world-action model、外观/几何表征和选择性校准工作。所有列出的 2026 工作当前均按 arXiv
preprint 对待，不能写成已被顶会同行评审确认。原始响应与可读正文仅保留在本地工作区；仓库提交
官方 API 元数据、URL、抓取时间、SHA-256 与审计笔记，不再分发论文全文。

## 1. 最接近工作的实际覆盖

| 工作 | 已经覆盖的核心机制 | 对 AORTA 的直接约束 |
|---|---|---|
| Liu et al. (2026), **D-JEPA**，arXiv preprint 2609.24749v1 | decision-local prediction gap；完整候选集上的 permutation-equivariant relational operator；bounded correction；outcome supervision；native-relative driving correction；risk penalty、margin gate、native fallback；视觉变化实验 | “候选关系学习”“有界 correction”“native-relative head”“风险门控/保留 native”均不能再作为 AORTA 新意。其 driving 输入已包含 query、native score/rank、auxiliary channels 和完整轨迹几何 |
| Pei et al. (2026), **World4Scorer**，arXiv preprint 2609.36438v1 | trajectory-conditioned JEPA scorer；对所有候选的 simulator outcome supervision；executed-future anchor；scene-matched negative bank；previous-plan inertial reranking | “outcome-grounded scorer”“future target”“hard negatives”“previous plan consistency”均已有直接先例 |
| He et al. (2026), **iDriveVLA / Evaluation Is All You Need**，arXiv preprint 2609.30818v1 | safety-aware criterion scorer、VLM-guided criterion modulation、oracle-aligned progressive ranking | “安全感知 scorer”“oracle gap”“语义调权”不是 AORTA 的独有贡献 |
| Wei et al. (2026), **ManeuverFlow**，arXiv preprint 2610.13409v1 | 候选沿 shared current/future BEV field 查询 path-specific evidence；set scorer；PDM-factor supervision；candidate refinement | “沿候选路径读取未来证据”“共享场上 candidate-specific query”“factor head”均已有很强先例 |
| Xu et al. (2026), **PlanWAM**，arXiv preprint 2610.11382v1 | planning-shaped future representation；privileged future posterior；hindsight-to-foresight distillation；history-only deployment | “任务塑造的未来表征”“训练时未来、部署时历史”不是 AORTA 的新颖性来源 |
| Lu et al. (2026), **GeoWAM**，arXiv preprint 2608.23486v3 | 用 metric point geometry 作为比 pixel 更解耦 appearance/texture/illumination 的状态；future geometry pretraining 与规划 | “几何比像素更适合驾驶/更外观解耦”已被直接提出；AORTA 不能只靠 geometry representation 立论 |
| Schmidt et al. (2026), **Expand-and-Prune**，arXiv preprint 2610.23262v1 | 不更新已训练 planner 参数，在测试时扩展候选支持并按不确定性调节容量；跨 planner/benchmark | “frozen planner plug-in”“test-time candidate adaptation”“不确定性剪枝”“跨 planner”均非新意 |
| Marques et al. (2026), **SLLCP**，arXiv preprint 2610.02765v1 | frozen VLM 上的局部、label-conditional split conformal safety sets | “给冻结视觉模型加 conformal safety monitor”不构成独立新意 |
| Chang & Ahmed (2026), **Barrier Function Conformal Safety Clearance Certification**，arXiv preprint 2608.26533v1 | 固定完整 selection pipeline 后做 post-selection conformal calibration；以 drive session 为 exchangeability unit；组内最大值；有限样本 order statistic；native intervention | AORTA 的 group-level calibration、fixed-pipeline requirement、session maximum 与有限样本拒判属于严谨地基，不是论文主创新 |

官方来源：

- https://arxiv.org/html/2609.24749
- https://arxiv.org/html/2609.36438
- https://arxiv.org/html/2609.30818
- https://arxiv.org/html/2610.13409
- https://arxiv.org/html/2610.11382
- https://arxiv.org/html/2608.23486
- https://arxiv.org/html/2610.23262
- https://arxiv.org/html/2610.02765
- https://arxiv.org/html/2608.26533

## 2. D-JEPA 是当前必须正面超过的最近工作

D-JEPA 对 AORTA 的威胁不是题目相似，而是方法接口高度重叠。其 driving 版本已经：

1. 从 Drive-JEPA 取 32 条共享候选；
2. 对完整候选集做 set-wise relational reasoning；
3. 令 native correction 精确为零，直接回归 candidate benefit；
4. 用 pairwise boundary ordering、risk logits、calibrated margin 和 native fallback 决定是否替换；
5. 保持 predictor、search budget 与 low-level controller 不变。

因此下面这些 AORTA 旧表述必须永久删除：

- “首次直接学习 native-relative driving relation”；
- “首次在冻结 planner 上使用 relation head”；
- “首次用 margin gate 保留 native”；
- “首次证明全局预测质量不保证局部决策正确”。

D-JEPA 尚未从其正文中证明的、也是 AORTA 仍可争取的边界是：

- driving 评估只报告 7 个 focused scenes、3 个 source logs；其 visual-shift 实验发生在 PushT，且
  fitting 与 evaluation 共享已知 appearance families，不是 NavDream 驾驶外观基准；
- driving token 使用 planner query、score/rank、auxiliary channels 与 trajectory coordinates，未把
  **native/challenger 在哪里产生完整时空差异**作为显式视觉查询对象；
- 没有 correct-image vs geometry-matched mismatch/removed 的 driving 视觉必要性合同；
- 没有 appearance-orbit relation constraint、表示不可识别性诊断或 recording-group simultaneous
  relation error 对 false intervention 的闭环；
- 没有在多个冻结 planner 的 NavDream 外观变化下同时报告 clean no-harm、OOD hard-failure net
  reduction 与选择性 coverage。

这些差异只是**待验证空隙**，不是已成立的新颖性。D-JEPA 必须进入 matched strongest baseline，
而不能只在 related work 中文字区分。

## 3. AORTA 现在唯一可防守的主命题

不能再写成“decision-aligned relation learning”。更精确的候选主命题是：

> 在纯外观干预下，可靠的冻结-planner 仲裁需要显式建模 native 与 challenger 的完整时空
> **decision-difference support**，并从全局视觉上下文中只提取足以判断该差异后果的证据；当该
> relation 在观测上不可识别或同时误差下界不足时，系统保持 native。

对应的最小数学对象为完整 official target 的相对 margin：

\[
\Delta^*_{i0}(S,h_{t-1})
=F_{\rm full}(S,\tau_i,h_{t-1})-F_{\rm full}(S,\tau_0,h_{t-1}),
\]

和完整时域 occupancy-measure contrast：

\[
\nu_{i0}=\mu_i-\mu_0=\nu^+_{i0}-\nu^-_{i0}.
\]

`nu+ / nu- / common context` 只负责提出“哪里不同”的查询；所有冻结视觉 token 仍作为 key/value
回答“这些差异意味着什么”。反对称化只保证输出接口正确，不保证预测正确；task supervision 与
paired-appearance constraint 分别承担 correctness 与 invariance；group-simultaneous lower bound
只承担选择资格。

## 4. 决定新颖性生死的单一消融

最强 reviewer 会说：

> 一个容量匹配的 D-JEPA/World4Scorer-style set scorer 已经能从轨迹坐标、native query 和全局视觉
> token 中自己学出差异；signed differential support 只是人工 feature engineering。

因此主创新只由以下 matched study 决定：

| Arm | 输入与预算 | 目的 |
|---|---|---|
| D-JEPA-style relational baseline | native query/score/rank + trajectory geometry + global visual evidence | 当前最强最近工作 |
| World4Scorer-style candidate state | candidate query + shared visual tokens + full-target outcome heads | outcome-grounded scorer 对照 |
| ordinary independent/set scorer | 单候选或完整集合；容量/FLOP/token 匹配 | 检验普通 scorer 是否足够 |
| AORTA differential-evidence operator | signed differential query + common context + 相同 global visual evidence | 只改变归纳偏置 |

判据不是训练 loss，而是未见场景与 NavDream 上的 relation error、pairwise sign、ranking regret、
false intervention、clean harm、跨 planner transfer 和 batch-1 latency。若 AORTA 不稳定优于前两类
matched baseline，`differential-support network innovation` 必须删除；剩余工作最多是严谨评测或
选择性校准系统。

## 5. 可作为论文地基、但不可列入贡献的内容

- Jordan/Hahn decomposition、antisymmetry、graph ranking/Hodge；
- DINO/JEPA/geometry backbone、cross-attention、set Transformer；
- pairwise/listwise ranking、Huber/BCE、factor heads；
- split conformal、session-level exchangeability、order statistic；
- bounded correction、margin gate、native fallback；
- previous-plan token、trajectory candidate bank、hard negatives；
- frozen-planner plug-in 与候选复用。

它们可以让方法严谨，却不能让方法新颖。AORTA 的新意必须来自一个由失败证据强制出的新
**estimand + evidence interface**，并由 matched experiment 证明它不可被普通关系 scorer 替代。

## 6. 审计结论

当前新颖性状态由“中等偏高候选”下调为：

`NARROW_BUT_PLAUSIBLE__D_JEPA_MATCHED_FALSIFIER_REQUIRED`

AORTA 仍有论文机会，但不再有“关系学习本身”的空间。其可投稿价值取决于是否第一次把
appearance intervention、decision-difference support、视觉必要性、选择性 intervention 与
multi-planner driving outcome 连接成可复算闭环。B4R1 尚未运行，网络尚未训练，NavDream、行为
收益与时延均未验证，因此当前不得写成 ICCV-ready 或 oral-ready。
