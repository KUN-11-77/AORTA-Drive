# Experiment Plan

**Problem**: source-only appearance shift 会使冻结 planner 的轨迹违反 nominal road constraint；需要在绝对道路零点不可靠时，只授权 challenger 相对 native 不降低道路 margin 的干预。  
**Method Thesis**: candidate-free graph-gradient field 的 source-session simultaneous event，可以经 exact min-plus readout 与 map/geometry/numerical correction 转化为所有授权 pair 达到 \(\gamma_{task}\ge0\) 的 relative-margin certificate。
**Date**: 2026-10-23

> 本计划用于验证、否证并修正当前条件规范，不预设方法成立。Gate A/B 通过前禁止正式长跑和目标域结论。

**2026-10-23 execution status:** R001–R003 synthetic qualification completed on clean revision `821b060`; R002C asymmetric sweep cover passed on `943e7bb`; R002D exact polygon-teacher ledger passed on `497b77b`. Mathematical core and synthetic geometry/teacher constructions passed. Full Gate A remains blocked by the actual NAVSIM preprocessing, transform/protocol hashes, and geometry-arithmetic ledger. See [`experiments/results/GATE_A_SYNTHETIC_SUMMARY.md`](../../experiments/results/GATE_A_SYNTHETIC_SUMMARY.md).

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1 — candidate-free field certificate 有效且非平凡 | 这是方法区别于候选绑定 scalar calibration 与松弛全域场界的核心 | Gate A 全部 oracle 通过；held-out source-session simultaneous coverage 达到预声明水平；在相同合法 predictor 上，local graph 对 meaningful non-inclusion pairs 比 absolute/global-gauge exact bounds 更紧且非空 | B0, B1, B2 |
| C2 — certificate-guided arbitration 有实际决策价值 | 正确 coverage 若只会 abstain，没有驾驶意义 | 所有授权干预的 relative-margin failure 受控；分别证明 candidate supply、certificate recall、certifiable opportunity 与 final selection 非平凡；收益不以进度、舒适性或其他 NAVSIM 分项显著退化为代价 | B3 |

### Anti-claims to rule out

- 收益只来自 geometry inclusion，而不是视觉道路证据；
- 收益只来自故意削弱 absolute/global/CDF baseline；
- 收益只来自更多参数、更大候选集或更强 backbone；
- source-session coverage 被误写成 arbitrary appearance-OOD guarantee；
- min-plus 的标准求解归约被包装成独立算法创新。

## Paper Storyline

### Main paper must prove

1. field/map/geometry/numerical/selection 数学链在实现中方向正确；
2. candidate-free local field uncertainty 在公平 exact-readout 对照下产生非平凡证书；
3. 该证书能支持有用、保守的 native-preserving arbitration。

### Appendix can support

- 不同图分辨率、路径长度与 calibration alpha 的敏感性；
- appearance corruption 分类诊断；
- learned scale 仅在 fixed-scale V0 成立后作为可删除扩展；
- 更多 planner/candidate generators。

### Experiments intentionally cut

- 新 backbone 搜索、VLM/LLM/diffusion 模块；
- learned shortest-path solver；
- target-domain tuning；
- Gate A/B 之前的完整 NAVSIM 长跑；
- 没有改变 reviewer belief 的大规模模块消融。

## Experiment Blocks

### Block 0: Gate A — theorem, solver, and geometry audit

- **Claim tested:** C1 的数学与实现前提成立。
- **Why this block exists:** 任何乐观符号错误、标签恢复错误或 geometry false acceptance 都会使后续数字失去意义。
- **Dataset / split / task:** 无需 NAVSIM 训练；枚举小图、合成 signed-distance fields、解析 footprints 与固定数值 adversarial cases。
- **Compared systems:** exact closed form；LP/dual oracle；multi-source implementation；geometry analytic oracle。
- **Metrics:** 最大/最小 bound discrepancy；false certificate count；common-shift error；连续 teacher 到 node label 的 \(\varepsilon_{map}\)；geometry false acceptance；最坏乐观浮点误差 \(\varepsilon_{num}\)。
- **Setup details:** chain/cycle/grid；零权边；非零 source labels；\(A=B\)、严格 inclusion、disjoint sets；边界 ROI、姿态插值、连续 teacher、node sampling 与 footprint coverage cases。
- **Success criterion:** 所有 exact cases 在事前容差内一致；零 false-positive certificate；获得分别写入连续证书的保守 \(h_j\)、\(\varepsilon_{map}\) 与 \(\varepsilon_{num}\) 上界；geometry checker 对 nominal protocol 确定性 sound。
- **Failure interpretation:** 修正定理实现或 geometry contract；不得通过增大经验阈值掩盖方向错误。
- **Table / figure target:** Appendix correctness table；主文方法框中的 theorem/implementation statement。
- **Priority:** MUST-RUN。

### Block 1: Gate B — source nontriviality and uncertainty geometry

- **Claim tested:** C1 中的 field certificate 不是空泛界，局部 uncertainty geometry 有独立价值。
- **Why this block exists:** 先决定理论对象在真实 source residual 上是否有用，再支付正式训练成本。
- **Dataset / split / task:** NAVSIM source-only train/development/calibration/held-out-source sessions；route/session 去重，frame 不视为独立样本。
- **Compared systems:**
  1. 合法 absolute-trained field predictor 上的 absolute simultaneous band；
  2. 同一 predictor 上的 global-gauge exact robust readout；
  3. 同一 predictor 上的 local fixed-graph exact readout。
- **Metrics:** session simultaneous coverage；average 与 session-maximum graph residual；\(q_\alpha\) 及 tail concentration；米制 lower-bound gap；\(S_{better}\)、\(R_{cert|better}\)、\(R_{opportunity}\)；按 path length/visibility/ROI 分层的 sharpness；global-q domination；paired appearance \(\rho\)（仅机制诊断）；geometry rejection rate。
- **Setup details:** 先复用已有合法 source predictor；若不存在，允许训练一个最小 source-only absolute field probe。GT 自拟合与直接用 teacher 生成 \(u\) 均不合格。校准前冻结 token-to-node 投影、missing-evidence mask 与 candidate provider；先固定一个 seed 做资格，过门后再补 3 seeds。
- **Success criterion:** coverage 在预声明容差内；\(S_{better}>0\) 证明候选供给存在；\(R_{cert|better}>0\) 证明证书对真实改善机会非空；local graph 在相同 predictor 和 risk object 下，对预注册比例的 meaningful non-inclusion pairs 严格紧于 global gauge。
- **Failure interpretation:**
  - coverage 失败：校准/独立性/实现错误；
  - coverage 正确但全 abstain：理论无部署价值；
  - global gauge 等价或更紧：删除 local graph，回退简单模型；
  - 只在 inclusion pairs 成功：不支持视觉方法主张。
- **Table / figure target:** 主文 Table 1（coverage + sharpness）；Figure 1（path length/visibility 分层）。
- **Priority:** MUST-RUN。

### Block 2: Legal end-to-end method and candidate independence

- **Claim tested:** C1 在各方法合法训练后仍成立，且不绑定 calibration 时的 candidate generator。
- **Why this block exists:** 排除“同一 predictor 控制成立，但 edge training 没有方法收益”以及 baseline handicap。
- **Dataset / split / task:** 同一 source splits；原 candidate policy、held-out perturbation policy、未参与 calibration 的 frozen planner。Target appearance 只作最终经验评测，不提供 coverage 主张或选择信号。
- **Compared systems（最多三族）:**
  1. candidate-specific scalar contrast + session calibration；
  2. structured field family：absolute、global gauge、具体 CDF-style joint field；
  3. AORTA fixed-scale edge-trained field。
- **Metrics:** session coverage；bound sharpness；meaningful non-inclusion certification；candidate-policy/planner transfer degradation；参数量；field inference 与 solver latency。
- **Setup details:** frozen DINOv3-H+；固定多相机/历史帧 token-to-node 投影、可见性与 missing-evidence 协议；相同 field architecture/capacity/source data；冻结且可复现的 source-only candidate provider；3 seeds；相同 geometry gate 与 exact/conservative relative readout。CDF 的 rank、region、score、solver 与优化误差预算事前冻结。
- **Success criterion:** AORTA 在 coverage 合格下，对 held-out candidate policy/planner 保持证书有效性和非平凡率；优势不能由参数量、候选数或不公平 readout 解释。
- **Failure interpretation:**
  - scalar baseline 同样迁移：candidate-free field 必要性不足；
  - CDF/global 等价：降级为结构化应用；
  - edge training 不如 absolute predictor：撤回训练目标主张。
- **Table / figure target:** 主文 Table 2（matched end-to-end comparison）；Figure 2（candidate-policy transfer）。
- **Priority:** MUST-RUN，但仅在 B0/B1 通过后。

### Block 3: Native-preserving arbitration utility

- **Claim tested:** C2。
- **Why this block exists:** 把有效证书转化为驾驶决策，而不是只报告 field metrics。
- **Dataset / split / task:** frozen NAVSIM evaluation protocol；source validation 为风险主对象，预先冻结 target appearance suites 仅报告 empirical robustness。
- **Compared systems:** keep-native/geometry-only；最强 scalar or structured-field baseline；AORTA。
- **Metrics（主指标优先）:** session-level \(R_\gamma\)；\(S_{better}\)、\(R_{cert|better}\)、\(R_{opportunity}\)、\(R_{selected}\)；true relative margin；其次报告 safe-to-unsafe corollary、DAC、NC、EP、comfort、PDMS 分项与 session-clustered 置信区间。
- **Setup details:** 完全相同候选、qualification、progress/deviation budget、选择顺序和 \(\gamma_{task}\)；评估单位先均匀抽 session、再按冻结规则抽时刻；3 seeds；阈值只由 source development/calibration 决定。
- **Success criterion:** source 上 \(R_\gamma\) 不超过预注册风险容差；\(S_{better}\)、\(R_{cert|better}\) 与 \(R_{selected}\) 均非平凡；有意义的非包含干预不是 inclusion 伪收益；其他关键指标无不可接受退化。
- **Failure interpretation:** 正确但无干预表示证书过松；道路收益换取碰撞/进度显著退化表示系统没有部署价值；target-only 收益不支持 source-calibrated theory。
- **Table / figure target:** 主文 Table 3（arbitration）；risk–intervention curve。
- **Priority:** MUST-RUN，但仅在 B2 通过后。

### Block 4: Failure and counterexample atlas

- **Claim tested:** 限定 C1/C2 的适用范围，不制造额外主张。
- **Why this block exists:** 识别理论需要修改的是 observation model、uncertainty geometry 还是候选/geometry protocol。
- **Dataset / split / task:** held-out source 与预冻结 appearance suites；遮挡、极端光照、道路边界缺失、长路径、ROI 边缘。
- **Compared systems:** global gauge vs local graph；成功/拒绝/错误 cases。
- **Metrics:** residual topology；q-dominating edge；path accumulation；candidate supply、certificate recall 与 selection 分解；paired appearance \(\rho\)；geometry rejection；错误类型与证书 gap。
- **Setup details:** 只分析已经运行的主实验，不为定性图另调模型。
- **Success criterion:** 每类失败能映射到一个预先声明的假设或停止条件；不要求所有 OOD 成功。
- **Failure interpretation:** 大量无法归因的错误说明理论对象仍不充分，应返回方法重构，而不是继续调阈值。
- **Table / figure target:** 主文 failure figure + appendix atlas。
- **Priority:** NICE-TO-HAVE；不能延迟 B0–B3。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 冻结 teacher/node projection/candidate/map/geometry/数值协议 | R001–R003 | Gate A 全通过 | CPU，约 1–3 天 | geometry 或 map checker 不能提供确定性上界 |
| M1 | 复现合法 source predictor 与强 baseline | R004–R006 | predictor 与 calibration pipeline 合法 | 约 1–2 GPU-days 特征缓存；probe 10–30 GPU-hours | session split 泄漏或无合法 predictor |
| M2 | 判断 local certificate 是否非平凡 | R007–R009 | Gate B 通过 | 以缓存统计为主，CPU/GPU 数小时至 2 天 | global q/路径累积导致全 abstain |
| M3 | 训练并比较正式最小方法 | R010–R014 | C1 有效且 baseline 公平 | 3 seeds，约 60–180 GPU-hours | CDF/global 等价，原创性不成立 |
| M4 | 仲裁与论文决定 | R015–R018 | C2 成立，否则降级/停止 | 评测成本依 NAVSIM 队列与 planner 数量冻结 | 道路收益损害其他指标 |
| M5 | 失败分析与附录 | R019+ | 不影响主结论 | 利用已有输出，少量追加 | 过度解释 target/OOD 结果 |

## First Three Runs to Launch

1. **R001:** 小图 LP/closed-form/multi-source 等价与 common-shift oracle。
2. **R002:** continuous teacher、node-label map error、footprint coverage 与 deterministic geometry certificate 的反例/property tests。
3. **R003:** nonzero-label recovery、zero-edge、relative \(R_\gamma\) event 与浮点乐观误差上界。

前三项均为 Gate A，不需要服务器 GPU，也不产生方法性能结论。

## Compute and Data Budget

- **CPU correctness stage:** 1–3 天；无需 GPU。
- **Frozen feature extraction:** 初估 1–2 GPU-days，取决于实际 NAVSIM sessions、相机历史与 ROI。
- **Gate B probe:** 10–30 GPU-hours；先 1 seed，过门后再补 seeds。
- **Formal minimal decoder:** 60–180 GPU-hours / 3 seeds 的规划上限；必须在服务器硬件清单与图尺寸确定后重估。
- **Calibration sessions:** 必须与 train/development/test 完全互斥；有限阈值要求 \(n\ge\lceil1/\alpha\rceil-1\)。例如 \(\alpha=10\%,5\%,1\%,0.5\%\) 至少分别需要 9、19、99、199 个真正独立 sessions；实际还需留出独立 development/test。
- **Human evaluation:** 不需要。
- **Biggest bottleneck:** 独立 session 数量、合法 geometry certificate、具体 CDF conservative readout，而不是 Dijkstra 本身。

## Risks and Mitigations

- **全局 q 被远处坏边主导:** 先做分层诊断；若证书空泛，停止而非事后局部调 alpha。
- **路径累积比 global gauge 更松:** exact global-gauge baseline 为强停止条件。
- **geometry checker 只有经验 soundness:** 不授权连续证书；改用可证明上界或保留 native。
- **teacher/node map error 无确定性预算:** 不把离散证书提升为连续道路证书；先建立 \(\varepsilon_{map}\) 上界。
- **baseline 训练不公平:** 强制 Layer 1/Layer 2 双层设计。
- **Gate B 没有合法 predictor:** 训练最小 source-only probe；禁止 teacher-as-prediction。
- **target appearance 结果诱导调参:** target suite 在配置冻结后一次性打开，只作经验报告。
- **服务器资源未知:** 在获得 GPU 型号、数量、显存和存储后更新预算；当前不启动长任务。

## Theory Revision Rules

- Gate A 失败：修正公式、实现或几何合同；不进入模型训练。
- Gate B coverage 失败：优先审计 split/exchangeability/score，再决定校准理论是否需改写。
- Gate B 中 \(S_{better}\approx0\)：候选供给失败，不能归咎于证书；先审计冻结 candidate provider。
- Gate B 中 \(S_{better}>0\) 但 \(R_{cert|better}\approx0\)：证书对真实改善机会过松，否定当前 local field 的部署价值并比较 global gauge；不靠新增模块硬救。
- C1 失败但 CDF/global 成立：降级为基线或结构化应用。
- C2 失败：保留表征研究结论，撤回仲裁系统主张。
- target OOD 失败而 source 合格：不修改 source theorem；把失败用于限定 observation assumptions。

## Final Checklist

- [x] 主张限定为 2 项
- [x] Main paper blocks 覆盖问题、机制与仲裁
- [x] novelty isolation 使用公平 exact-readout controls
- [x] simplicity 通过 fixed-scale/single-native/exact solver 维护
- [x] frontier primitive 仅为 frozen DINO，并有强简单对照；不强行增加 frontier block
- [x] MUST-RUN 与 NICE-TO-HAVE 已分离
- [ ] 服务器硬件与连接信息完成安全配置
- [ ] experiment config 首次冻结
- [ ] Gate A 执行
