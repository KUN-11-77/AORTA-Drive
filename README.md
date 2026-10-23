# AORTA-Drive

[![CI](https://github.com/KUN-11-77/AORTA-Drive/actions/workflows/ci.yml/badge.svg)](https://github.com/KUN-11-77/AORTA-Drive/actions/workflows/ci.yml)

**AORTA — Appearance-conditioned Operator for Robust Trajectory Arbitration**

面向 NAVSIM v1/v2 的 source-only 外观鲁棒驾驶研究：学习 candidate-free 道路差分场，为冻结 planner 的 challenger 相对 native 提供 nominal road-margin non-degradation 证书。

> **研究状态：** AORTA-v8.3.2 的 Gate A 与 R004E-A/B/C 已完成；R004E-D/E/F 表明冻结 predictor 具有平均视觉信号，却因稀疏 tail 在统一全图最大误差上输给零场。R004E-G 已关闭 heterogeneous edge-envelope 的数学原语，R004H0 已通过候选无关尺度接口、旧势场精确回放与 scale-only 梯度审计。下一步只允许冻结 R004H1 fresh fit-only Stage P；calibration 仍封存，也仍无 learned uncertainty、正式 coverage、Gate B 或顶会原创性结论。

## 核心方法

网络在固定物理图 \(G=(\Omega,E)\) 上输出节点势 \(u_\theta(X,q)\)，只学习 GT road-distance field 的局部差分：

\[
\mathcal L_{edge}
=|E|^{-1}\sum_e
\operatorname{Huber}
\left(\frac{(Du_\theta-D\widetilde d_S)_e}{s_e}\right).
\]

source-session calibration 产生 edge budgets \(w_e=q_\alpha s_e\)。对 native swept set \(B\)，一次标准 multi-source min-plus transform

\[
\psi_B(v)=\min_{b\in B}[u_b+\operatorname{dist}_w(v,b)]
\]

即可为任意 challenger set \(A\) 给出精确最坏相对 margin：

\[
L_G(A,B)=\min_{a\in A}[u_a-\psi_B(a)].
\]

系统只在 proof-carrying nominal geometry gate 成立且 \(L_G\) 扣除 map、geometry 与 numerical budgets 后达到 \(\gamma_{task}\ge0\) 时替换 native。

## 理论边界

当前已建立的主条件链是：

```text
source-session edge event
  -> all-pair discrete robust bounds
  -> map/geometry/numerical correction
  -> every authorized intervention reaches gamma_task relative improvement
```

Graph difference constraints、split conformal、Dijkstra 与 min-plus envelope 是标准工具，不单独构成创新。方法论文地位仍取决于 Gate A/B、两层公平 baseline、meaningful non-inclusion 结果和独立 novelty-check。

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

当前 source code 包含 Gate A exact graph solver、校准/风险合同、raw-cover teacher、rotation-free geometry checker、rational-interval coordinate transform、确定性全节点 label builder，以及已冻结的 R004A–H0 source-session、visual projection、candidate provider、quotient-field probe、失败定位和异质尺度接口协议。下一步是单独注册 fresh fit-only Stage P，不读取 proxy-cal/development/calibration/test/target，也不启动完整性能长跑。

## 核心文档

- [研究问题](docs/RESEARCH_BRIEF.md)
- [v8.3.2 方法规范](docs/METHOD_SPEC.md)
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

