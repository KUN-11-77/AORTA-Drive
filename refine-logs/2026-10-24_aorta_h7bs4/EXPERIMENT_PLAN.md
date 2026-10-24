# Experiment Plan

**Problem**: 前向96×64虽覆盖完整候选，但其视觉可观测性、teacher关键点覆盖和实时性未知。
**Method Thesis**: 完整时域决策场只有同时满足空间定义域、任务关键视觉证据与计算预算，
才有资格进入网络结构必要性和决策相关性实验。
**Date**: 2026-10-24

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| Q1：候选查询获得直接视觉证据 | 排除扩大ROI只增加missing token/坐标先验 | 零全缺失候选；42/42组通过冻结完整性门 | B1 |
| Q2：可能决定set-min的teacher节点可审计 | 区分稳定视觉理解与道路先验外推 | 误差区间导出的critical-set覆盖分布及失败归因 | B2 |
| Q3：完整时域接口计算可接受 | 防止几何正确但无法在线推理 | 同硬件同步P50/P95、显存、吞吐和模块分解 | B3 |
| Q4：联合依赖结构不是装饰 | 决定graph-correlated uncertainty是否值得进入网络 | 同teacher包含下严格增加pair identifiability，且Oracle不过度拒答 | B2.5 |

Anti-claim：结果不能仅来自扩大节点数、空间先验或选择性丢弃远端候选。

## Experiment Blocks

### B1：Fit projective observability
- 数据：42 fit groups×16 anchors；group级统计。
- 系统：冻结96×64、完整4秒、全部eligible candidates。
- 指标：查询节点支持率、前向扩展支持率、零证据候选、组级完整性。
- 成功：无零证据候选且42/42组达到12/16；其余比例不设事后阈值。
- 失败：当前直接视觉接口停止；不自动扩大相机、历史或ROI。
- 优先级：MUST-RUN。

### B2：Interval-critical teacher observability
- 数据：同一fit groups；新96×64 teacher必须保留逐节点外向误差账本。
- 指标：每候选possible-minimizer集合的none/partial/full视觉支持，按距离和group报告。
- 成功：作为架构设计输入，不单独声称视觉正确；零支持必须触发显式不确定性/abstention。
- 失败：固定dense field路线HOLD，评估可观测子图或连续查询。
- 优先级：MUST-RUN。

### B3：Synchronized latency qualification
- 对照：64×64基线、96×64候选、128×128同前向上界控制。
- 固定：GPU、BF16、batch、12 slots、DINO输入、warmup与重复次数。
- 指标：各模块和端到端P50/P95、峰值显存、Hz。
- 成功：满足预先核验的控制周期；若周期未知，只作相对结论。
- 优先级：MUST-RUN。

### B2.5：Oracle joint-uncertainty non-vacuity
- 主张：在完全相同的projective teacher intervals下，保留shared graph constraints能比
  shared-node marginal box严格提高candidate contrast可辨识率。
- 数据：同一42 fit groups、672 anchors、3864 pairs；不读取learned field或outcome。
- 对照：joint Lipschitz set、shared-node marginal box、candidate-independent intervals。
- 主指标：pair sign identifiability；次指标：interval width、每组可辨识anchors、oracle
  intervention coverage与teacher regret。
- 成功：42/42组各至少12/16 anchors有joint-identifiable pair，且joint-box组级配对差95% CI
  下界严格大于0；全部集合包含与no-harm机械检查通过。
- 失败：Oracle近乎全拒答则重设证据接口；joint不优于box则删除graph-correlation主张。
- 优先级：MUST-RUN，先于训练与时延。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 实现与合成自检 | B1 tests | tests/mypy/hash全过 | CPU分钟 | 坐标/身份错配 |
| M1 | fit投影支持 | H7BS4A | PASS才进入B2 | CPU≤120min | 扩展域无直接证据 |
| M2 | teacher关键点 | H7BS4B | 形成架构必要条件 | CPU并行 | teacher生成成本 |
| M2.5 | 联合集合非空洞性 | H7BS4B1 | Oracle覆盖与joint必要性同时通过 | CPU≤120min | 可能几乎全拒答 |
| M3 | 推理代价 | H7BS4C | B1通过后达到冻结周期或报告相对失败 | 单GPU小时级 | DINO主导时延 |
| M4 | development确认 | A/B/C冻结后一次性 | 不调阈值 | CPU/GPU | 历史暴露，仅held-out资格 |

## Compute and Data Budget

- H7BS4A：CPU，120分钟硬上限，≤64MiB结果；
- H7BS4B：先一anchor与八组梯级，再完整fit；
- H7BS4B1：CPU，120分钟硬上限，672 anchors完整机械重放；
- H7BS4C：单GPU，不训练，不读取calibration/test/target；
- 最大瓶颈：96×64 exact teacher生成和12-view DINO端到端计时。

## Final Checklist

- [x] 主张与资格门分离
- [x] recording group为独立单位
- [x] 不用节点伪重复扩大置信度
- [x] 视觉、teacher与时延串行冻结
- [x] development及最终数据继续封存

## Workflow Reference

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
