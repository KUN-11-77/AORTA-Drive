# AORTA-Drive

[![CI](https://github.com/KUN-11-77/AORTA-Drive/actions/workflows/ci.yml/badge.svg)](https://github.com/KUN-11-77/AORTA-Drive/actions/workflows/ci.yml)

**AORTA — Appearance-conditioned Operator for Robust Trajectory Arbitration**

面向 NAVSIM v1/v2 的外观鲁棒候选仲裁研究：在冻结 planner 的合法候选集合中，学习
native-relative contextual relation，并检验完整时空 native–challenger 差异支持能否比通用
scorer 更有效地读取外观稳定的决策证据。

> **研究状态（2026-10-25）：** 旧 graph-gradient field 与 scalar-min road-margin reranker 已按
> 预注册结果停止作为一般候选仲裁主线。R005E1B 已支持 source-fit、native-history `U_ctx` 的合法
> contextual oracle opportunity；R005E2 正在检验给定候选与历史后，正确冻结视觉是否提供可学习的
> 条件增量。Differential Query、NavDream、跨 planner、actual-history rollout、clean no-harm、
> 在线时延和驾驶收益仍未验证。当前 canonical 理论见
> [`docs/AORTA_FIRST_PRINCIPLES_THEORY_V1_CN.md`](docs/AORTA_FIRST_PRINCIPLES_THEORY_V1_CN.md)。

## 当前方法假设

对合法候选 \(\tau_i\) 与 native \(\tau_0\)，当前最小监督对象是：

\[
\Delta_{i0}
=U_{ctx}(\tau_i;h)-U_{ctx}(\tau_0;h).
\]

首先检验视觉表示 \(Z\) 是否在候选与因果历史 \((C,H)\) 之外提供条件增量：

\[
\mathcal R^\star(C,H)-\mathcal R^\star(C,H,Z)
=\mathbb E\!\left[
\left\|
\mathbb E[Y\mid C,H,Z]-\mathbb E[Y\mid C,H]
\right\|^2
\right]\ge0.
\]

若该资格门通过，主架构候选才使用完整时空 signed occupancy difference
\(\nu_{i0}=\mu_i-\mu_0\) 生成 query，同时保留全部视觉 token 作为 key/value，预测
native-relative contextual relation。差异支持只负责提出比较问题，不被假定为完整任务效用的解析式。

## 理论边界

当前主条件链是：

```text
合法 contextual estimand 与 oracle opportunity
  -> 正确视觉具有条件增量
  -> differential query 胜过 matched generic scorer
  -> 外观变化下任务关系稳定且不坍缩
  -> 同时下界支持非真空保守仲裁
  -> actual-history / sequential / cross-planner 行为收益与实时预算
```

native-relative scoring、trajectory query、cross-attention、signed integral、DINO、conformal 与
native fallback 都不是独立创新。唯一主贡献候选是 appearance-interventional differential decision
evidence；其资格取决于与 D-JEPA/World4Scorer/DA-WAM/GTRS 风格 matched scorer 的公平比较及
NavDream、跨 planner、clean no-harm 和时延闭环。

## Repository map

```text
AORTA-Drive/
├── configs/                  # 冻结前的配置合同
├── docs/                     # 问题、方法、claims、协议与决策
├── experiments/              # 实验注册表与可读摘要
├── paper/                    # 稿件和图表规范
├── references/               # 文献索引
├── refine-logs/              # 完整理论细化与独立审稿轨迹
├── scripts/                  # 仓库与可复现性检查
├── src/aorta_drive/          # 研究包；当前仍含旧 affine baseline scaffold
└── tests/                    # 单元与合同测试
```

数据、checkpoints、cache 和运行产物必须位于 `NAVSIM_EXP_ROOT`，不得提交 Git。

## Quick start

不要原地修改共享 `dinov3` 环境。

```powershell
conda env create -f environment.yml
conda activate aorta-drive
python -m pip install -e ".[dev]"
python -m pytest
python scripts/check_repo.py
```

当前 source code 同时保留旧 field/geometry 基线工具与 R005 relation 实验链。R005E2 只负责检验
视觉条件增量，不包含最终 Differential Query、分布校准或 selective arbitration；后续模块严格按
第一性原理规范的证据门解锁。development/calibration/test/NavDream/target 不得用于架构选择。

## 核心文档

- [第一性原理理论与系统规范](docs/AORTA_FIRST_PRINCIPLES_THEORY_V1_CN.md)
- [Best-Paper 候选级实现标准](docs/BEST_PAPER_IMPLEMENTATION_STANDARD_V1_CN.md)
- [Best-Paper-Level First-Principles Blueprint](refine-logs/2026-10-25_aorta_best_paper_blueprint/AORTA_BEST_PAPER_FIRST_PRINCIPLES_BLUEPRINT_V1_CN.md)
- [研究问题](docs/RESEARCH_BRIEF.md)
- [历史 v8.3.2 field/geometry 方法规范](docs/METHOD_SPEC.md)
- [形式化定理与证明](docs/THEORY.md)
- [Claim–Evidence Matrix](docs/CLAIM_EVIDENCE_MATRIX.md)
- [实验协议边界](docs/EXPERIMENT_PROTOCOL.md)
- [架构决策](docs/DECISIONS.md)
- [v8.3.1 冻结提案](refine-logs/2026-10-23_aorta_v8/V8_3_1_PROPOSAL.md)
- [v8.3.1 独立审查与冻结复核](refine-logs/2026-10-23_aorta_v8/V8_3_1_REVIEW.md)
- [v8.3.2 几何修订](refine-logs/2026-10-23_aorta_v8/V8_3_2_GEOMETRY_AMENDMENT.md)
- [Gate A/B 实验计划](refine-logs/2026-10-23_aorta_v8/EXPERIMENT_PLAN.md)
- [五轮审查总结](refine-logs/2026-10-23_aorta_v8/REVIEW_SUMMARY.md)

## Citation and license

作者名单、引用信息与开源许可尚未获明确批准。当前仓库是 pre-release research material；没有许可证不代表授权复用。

