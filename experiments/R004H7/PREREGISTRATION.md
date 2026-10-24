# R004H7 预注册：结构必要性与决策相关性

日期：2026-10-24
状态：`PREREGISTERED_NOT_AUTHORIZED_TO_TRAIN`

## 1. 两条主张

### C13：决策相关性

在冻结候选面板上，精确 set-min decision-contrast error 对 ranking flip 与 teacher regret 的组级解释力高于全局 edge MAE、gauge-centered node MAE 与 DINO feature drift。

### C14：结构与目标必要性

在相同 backbone、数据角色、候选、训练预算和 checkpoint 规则下，decision-quotient supervision 比 edge-only potential、absolute SDF/BEV、direct edge + exact projection 和 candidate-conditioned direct pair 更稳定地保持候选排序。

反主张：优势只来自 DINOv3、更多参数、候选输入、训练预算或更有利的 checkpoint selection。

## 2. 阶段顺序

### H7-A：理论资格

- 审查 `THEORY.md` 中 graph quotient isomorphism、全局非线性 quotient、active-cell 线性化、oscillation bound 与 regret bound；
- 使用无数据的小图 exhaustive oracle 检查 offset invariance、potential/projected-edge 等价、active-set 反例、\(2\eta\) regret 界和 tie policy；
- 任一反例失败则 STOP，不进入 H7-B。

### H7-B：只读 shadow decision diagnosis

- 使用 R004H6 已访问的 24 development recording groups；不得训练或更新网络；
- 复用冻结 R004H6 fields、teacher 与 appearance conditions；
- 只新增冻结候选的离散 shadow query，不进行连续安全认证；
- 每组16 anchors，recording group 是唯一独立统计单位；anchor、candidate 和 appearance 都是组内重复测量。

候选集合由冻结 provider 经官方 execution operator 得到 realized trajectory，再在冻结 1m ROI 上形成“离散时刻 oriented vehicle rectangles 覆盖的节点集合”。它是 shadow task proxy，不是连续 swept-footprint certificate。native 必须有效；无效 challenger 保留并标记，不按模型结果过滤。

H7-B 完整性门：24/24 groups 均存在；每组至少12/16 anchors 具有 native 与至少一个有效 challenger；否则整个 H7-B 记为 `INSUFFICIENT_PANEL_COVERAGE`，不得删除组后继续。

无效 anchor 必须保留在 panel coverage 分母和失败账本中，不得以任何预测场、teacher
margin 或 outcome 为依据过滤。只有上述完整性门整体通过后，组内统计才在有效 panel 上
对四种外观等权求均值；每组仍等权进入组级推断。native footprint 越界、空节点集或
official execution 非确定均使该 anchor 无效；单个 challenger 无效只将该 challenger 留在
失败账本中，只要 native 与至少一个其他 challenger 有效，anchor panel 仍可计算。

每个有效 panel 计算：

- teacher margins 与预测 margins；
- \(\eta=\max_j|\widehat m_j-m_j|\)；
- deterministic ranking flip；
- teacher regret \(m_{j^*}-m_{\widehat j}\)；
- global edge MAE、gauge-centered node MAE、clean-to-appearance DINO drift；
- teacher top-gap、exact-tie rate 与 panel validity。

主统计：先在每组内对 anchors/candidates/appearances 求均值，再对24组等权。使用 seed `20261024` 的10,000次 paired group bootstrap。比较：

1. group-level Spearman association with regret；
2. anchor-level flip AUROC，但置信区间只重采样 recording groups；
3. 决策误差相对每个 control 的关联差异。

H7-B `SUPPORTED` 要求 decision-contrast error 对 regret 的相关为正、其95%组级bootstrap下界大于0，并且相对每个冻结 control 的相关差异下界均大于0；flip AUROC 方向必须一致。否则 `UNSUPPORTED`，不得自动训练新网络。

这里将“flip AUROC 方向一致”在读取 H7-B payload 前冻结为：decision-contrast error 的
cluster-bootstrap 95% 下界严格大于 `0.5`，且它相对 edge MAE、gauge-centered node MAE
和 DINO drift 的 AUROC 差值下界分别严格大于 `0`。bootstrap replicate 若不同时包含
flip 与 non-flip 则不可计算；可计算 replicate 少于总数的95%时整体 fail closed 为
`UNSUPPORTED`。exact teacher ties 不删除，按已冻结的 native-first、随后 challenger-order
tie policy 计算并单独报告。

该诊断因 outcome 与 decision contrast 共享查询结构，只能证明 endpoint relevance，不能单独证明算法创新。

### H7-C：条件公平训练

只有 H7-A 与 H7-B 均通过并完成独立实现审计，才允许训练。数据角色冻结为：

- 42 fit groups：梯度更新；
- 8 selection groups：checkpoint 与预注册超参数字典序选择；
- 16 historically exposed proxy groups：一次性结构审查；
- 24 development groups：H7-B 后不得用于 H7-C 选择或确认；
- official calibration、test、NavDream/target：继续封存。

结构 arms：

1. `QP-EDGE`：node potential，仅 edge supervision；
2. `EDGE-PROJ`：direct edges，经冻结 Euclidean cut-space projection 后查询；
3. `SDF-NODE`：absolute node/SDF supervision，同一 query；
4. `QP-DQ`：potential + edge + DQ + appearance objective，候选 AORTA；
5. `EDGE-PROJ-DQ`：projected edge + 同一 DQ/appearance objective；
6. `DIRECT-PAIR`：candidate-conditioned pair margin head，作为更强但不可复用的对照。

所有 arms：

- 同一冻结 DINO、投影、相机历史和 fit/selection 数据；
- 共享 trunk family、训练 epochs、optimizer family、augmentation 与 wall/GPU cap；
- total trainable parameters 相对 `QP-DQ` 偏差不得超过10%，总 FLOPs 偏差不得超过10%；direct-pair 额外 candidate input 必须明确记账；
- 3个固定 seeds：`20261024, 20261025, 20261026`；seed 是算法重复，不是统计独立数据单位；
- 每个 seed 内随机化 arm 运行顺序，并按 seed/GPU block 记录；
- 每个 arm 只能用8个 selection groups 进行同一字典序 checkpoint selection。

主终点为16个 proxy groups 上的 group-equal teacher regret；共同次终点为 ranking flip rate、decision-contrast MAE、受控 appearance ranking drift、edge MAE、参数/FLOPs和推理时延。每组先跨 anchors 与3 seeds 汇总，再进行 paired group bootstrap。

### H7-C 裁决

- `QP-DQ` 必须以95% paired-group CI 在 teacher regret 上严格优于 `QP-EDGE`、`SDF-NODE` 与 `DIRECT-PAIR`，并在 flip rate 上方向一致，才支持核心结构主张；
- `QP-DQ` 只有优于 `EDGE-PROJ-DQ`，才可主张 potential parameterization 的经验必要性；否则两者应合并解释为等价实现，potential 不作为创新点；
- 若 `DIRECT-PAIR` 持平或更优，主方法应简化为 direct pair，或把 candidate-free transfer 留给后续跨-provider实验证明；
- 若 DQ 只改善 field MAE、不改善 regret/flip，则当前方法主张 STOP；
- 任一容量、数据角色、checkpoint 或运行预算不匹配使比较无效，不得用事后归一化修补。

## 3. 论文边界

R004H7 通过后只允许声称 development/proxy 范围内的结构必要性。驾驶收益、continuous certificate、official test、NavDream/target 与跨 planner/provider 泛化必须由后续独立冻结实验验证。
