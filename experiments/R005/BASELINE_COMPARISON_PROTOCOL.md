# AORTA Baseline–Hypothesis–Experiment–Decision Protocol

状态：`MATRIX_FROZEN__NUMERICAL_MARGINS_LOCKED_BY_M1__NO_TRAINING_AUTHORIZED`

日期：2026-10-25。

本协议把 AORTA 的文献定位、最强竞争方法、机制消融和失败决策绑定到同一证据链。它不修改正在
执行的 B4R1，不授权训练，也不把外部论文在不同协议下的公开数字当作 matched comparison。

## 1. 两项论文主张

### C1：核心算法主张

在相同合法候选域、相同 scoped contextual target、相同视觉与历史信息以及匹配计算预算下，完整
时空 differential-support query 相比最强 full-context set scorer 具有更低的 held-out 决策风险：

\[
\Delta R = R_{\mathrm{AORTA}}-R_{\mathrm{SetScorer}} < 0.
\]

主要风险不是全体 margin MAE，而是 group-level decision regret；pairwise sign/boundary error 与
appearance-induced flip 为解释指标。

### C2：支持性系统主张

当 C1 成立后，选择性 AORTA 在 NavDream 和跨 planner 行为评估中产生正的 OOD 净收益，同时满足
clean no-harm、非真空 intervention、actual-history rollout 与在线 P95 约束。

### 必须排除的竞争解释

1. 收益只来自轨迹形状或数据集先验，而非正确 RGB；
2. 收益只来自更大参数量、更多视觉 token、更多 attention FLOPs 或更宽超参搜索；
3. 收益来自 DINOv3 本身，而非 differential query；
4. 离线 margin 更准但候选选择、完整行为或 sequential rollout 不变；
5. clean 不伤害只来自近乎全部拒判。

## 2. 三套不可混合的证据账本

| 账本 | 作用 | 允许的结论 | 禁止的替代 |
|---|---|---|---|
| L0 理论/接口 | B4R1、contextual oracle、候选域和 evaluator scope | 标签合法、headroom、可辨识性 | 不替代 learned method 或驾驶收益 |
| L1 算法机制 | matched B2–B7、视觉因果对照、结构消融 | differential query 的独立增量 | 不替代完整 EPDMS 或跨 planner 行为 |
| L2 系统行为 | Native、外部系统、NavDream、三 planner、rollout、latency | 实际应用价值与投稿定位 | 不反向证明网络归纳偏置必要 |

## 3. 冻结 Baseline Matrix

| ID | 系统 | 信息与监督 | 科学职责 | 报告位置 |
|---|---|---|---|---|
| B0 | Frozen Native Planner | 原 planner，不干预 | 实际价值基准 | 系统主表 |
| B1 | Full-Context Oracle | 同一合法候选域、官方 contextual target | 可改善空间上限 | 诊断表 |
| B2 | Trajectory-only Scorer | 候选与 history，无 RGB | 排除轨迹先验捷径 | 机制主表 |
| B3 | Absolute Utility Scorer | 逐候选预测绝对效用再相减 | 最简单学习式 scorer | 机制主表 |
| B4 | Matched Full-Context Set Scorer | 全部视觉、候选集合与 history | **核心创新第一裁判** | 机制主表 |
| B5 | AORTA w/o Differential Query | 保留其余 AORTA 组件 | 删除实验 | 消融表 |
| B6 | AORTA Differential Query | signed full-horizon query + global evidence + history | 主方法 | 两张主表 |
| B7 | Frozen DINOv3 Planner / DR | 依原工作协议重训 adapter/head | 视觉鲁棒系统锚点 | 外部系统表 |
| B8 | World4Scorer-style matched implementation | outcome/full-context scorer | 最近 scorer 强对照 | 机制附表 |
| B9 | D-JEPA-style matched implementation | native-relative set relation | 最近 relation 强对照 | 机制附表 |
| B10 | PEA/MG-select adapted implementation | 精确绑定论文、commit、配置后才纳入 | 迁移失败边界 | 附录，不冒充同构主基线 |

B4 与 B6 决定算法创新；B0 与 B6 决定冻结 planner 的实际价值。B7–B10 的原论文数字与 matched
重实现结果必须分表，不能跨候选域、evaluator、历史或训练预算直接排序。

## 4. 公平性硬合同

在 B2–B6 的决定性比较中必须完全相同：

- 训练/验证 recording groups、候选 membership、ragged mask 与候选顺序；
- scoped contextual target、previous-action 来源与 teacher-forcing/rollout 标签；
- 冻结视觉 backbone、相机历史、视觉 token 输入和数据增强；
- optimizer family、最大更新步数、early-stop 规则、seed roster 和搜索次数；
- evaluator 版本、tie tolerance、统计单位和失败处理。

预算匹配规则：

- trainable parameter ratio 不超过 `1.05×`；
- measured forward FLOPs ratio 不超过 `1.10×`；
- 主表使用相同视觉 token 上限；
- 若 richer model 超预算，压缩 richer model，不扩大 AORTA 预算；
- 所有方法另报 batch-1 P50/P95/P99、峰值显存和候选数缩放曲线。

模型宽度、token 数、学习率集合、实用效应阈值和 non-inferiority margin 必须在 M1 contextual target
开封完成后、任何 B2–B6 结果产生前写入独立 preregistration。不得由单方法结果反向选择。

## 5. 数据、统计与指标

- recording group 是唯一独立统计单位；anchor、candidate、pair 和 appearance 都是组内重复；
- stochastic model 默认 3 个相同 seeds；组级 paired difference 为主，seed 作为重复轴完整报告；
- 主比较定义 `Delta R = R_AORTA - R_B4`，其预注册 95% paired group CI 上界严格小于 0 才支持
  C1；CI 包含 0 表示证据不足，不表示等效；
- 同时报告最差组、失败组数、机会组与全部组，禁止 complete-case deletion；
- 正式 calibration/test/NavDream 只在模型、阈值和报告表冻结后一次性开封。

决定性指标依次为：contextual oracle headroom、decision regret、pairwise sign/boundary error、
appearance-induced flip、false intervention/new-hard、official net utility、intervention/abstention、
clean no-harm、actual-history rollout 与在线 P95。

## 6. 五阶段依赖链

| Gate | 运行 | 通过含义 | 失败动作 |
|---|---|---|---|
| E0 | B4R1 + evaluator-scope audit | one-frame 接口有效；contextual target 可合法建立 | 接口无效则修接口；one-frame 无信号不直接否定 contextual target |
| E1 | Full-context oracle | 合法候选存在非平凡 full-target headroom | 无 headroom：停止 scorer，回到候选供给 |
| E2 | B2/B3/B4/B6 matched competition | 视觉必要且 differential query 有独立增量 | 见第 7 节，不堆模块 |
| E3 | full/prefix、signed、RGB、history、orbit loss | 定位增量的最小必要机制 | 删除无证据模块，保留最简单获胜者 |
| E4 | NavDream、三 planner、sequential、clean、P95 | 表征增量转化为系统价值 | 收缩主张或停止投稿主线 |

### 6.1 E1 与 E2 之间的瓶颈分支：视觉信息还是决策接口

B4R1 与 full-context oracle 只回答“合法候选域是否存在可监督、可改善的任务关系”，不能决定最终
方法应修改视觉编码器还是候选关系接口。E1 通过后、训练 B2--B6 前，必须用同一冻结视觉输入、
候选域与 contextual target 做一次预注册的因果诊断：

| 诊断 | 最小对照 | 允许的解释 |
|---|---|---|
| 正确 RGB 是否包含额外任务信息 | correct RGB vs geometry-matched mismatch、removed RGB、trajectory-only | correct 显著更好才支持视觉必要性 |
| 冻结视觉表示是否仍保留该信息 | frozen feature probe vs RGB/feature corruption；同容量读出 | 区分信息在编码阶段丢失，还是读出阶段未利用 |
| 普通完整上下文 scorer 是否能利用信息 | B4 vs B2/B3，预算匹配 | B4 能利用则瓶颈不在视觉编码本身 |
| differential query 是否提供额外归纳偏置 | B6 vs B4，预算匹配 | 只有 B6 稳定降低 held-out regret 才支持核心算法 |

分支规则冻结为：

1. 若 correct RGB 不优于 mismatch/removed，则停止视觉主张，先排查标签、候选与轨迹先验捷径；
2. 若 RGB 有信息但冻结表示的容量匹配 probe 无法恢复，则转向任务约束的视觉鲁棒算子，并与
   DINOv3、PEA-style alignment、普通 Adapter/LoRA 做 matched comparison；
3. 若冻结表示可恢复信息而 B4 仍失败，则研究视觉到决策关系的 evidence interface；
4. 若 B4 已与 B6 相当或更优，则删除 differential-query 核心创新，不把既有投入当作保留理由；
5. 不允许在同一轮同时引入复杂视觉校正器和复杂仲裁器，否则无法归因核心贡献。

该分支把论文的长期目标固定为“从 RGB 获取真正影响驾驶决策的证据”，而不是预先保证某个
具体模块获胜。任何路线都必须最终回到 correct-image necessity、official utility、clean no-harm、
跨 planner 与 batch-1 端到端时延。

## 7. Failure-to-Decision Matrix

| 观察 | 结论 | 下一动作 |
|---|---|---|
| contextual oracle 无改善空间 | scorer 无法创造候选 | 修候选供给，不训练 scorer |
| trajectory-only ≈ AORTA | 正确视觉证据必要性未成立 | 审查输入/标签捷径，HOLD 视觉主张 |
| absolute scorer ≈ AORTA | differential query 尚无增量 | 删除主创新或重新定位失败机制 |
| full-context set scorer ≈/优于 AORTA | 核心架构主张不成立 | 不因理论偏好保留该贡献 |
| correct RGB 不优于 mismatch/removed | 网络未证明使用场景视觉 | 停止 OOD 鲁棒性推广 |
| source 胜、NavDream 败 | 只支持 source 机制 | 不声称未见外观泛化 |
| margin 改善但选择/行为不变 | 只有预测层收益 | 不声称驾驶收益 |
| offline 胜、actual-history rollout 败 | history shift 未解决 | 收缩为离线方法或修正状态建模 |
| clean no-harm 依赖近乎全拒判 | selective rule 真空 | 删除可靠性实证贡献 |
| matched C1 与 E4 均通过 | 主创新与系统价值同时成立 | 才升级为 ICCV/CVPR 核心主张 |

## 8. 文献输入边界

The Constant Eye、World4Scorer、D-JEPA 已有本项目来源复核；新收到的 FROST-Drive、FocusDrive、
DriveTeach-VLA、DA-WAM、TOAD、DriveSuprim、VLA 与 sparse-query 文献只作为待核验候选。每篇必须
先绑定官方正文/仓库版本、输入、候选、监督和可复现资产，再决定属于 matched baseline、外部系统
锚点或仅 related work。PEA 身份未唯一核验前不得进入正式主表。

## 9. 当前状态

- B4R1：已授权并运行；本协议不读取其 partial scientific metrics；
- E1–E4：全部 LOCKED；
- calibration/test/NavDream/target：保持封存；
- 本协议冻结方法身份和公平性原则，但不预填任何实验结果。
