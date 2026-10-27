# R007V0：DINOv3-H+ 决策接口因果审计预注册

状态：`PREREGISTRATION_DRAFT__METHOD_TRAINING_HOLD`
日期：2026-10-27

## 1. 研究起点与禁止事项

已知线索是 Constant Eye 与项目既有结果支持：冻结 DINOv3-H+ 能为外观扰动下的驾驶提供正增益。
R007V0 不再检验“DINO 是否一般有用”，而检验更具体的第一性问题：

> 在物理场景与驾驶上下文不变时，DINOv3-H+ 中与决策有关的信息，经过何种真实计算接口进入 planner；
> 当前接口是否放大残余外观漂移、压缩任务证据，或主要依赖场景先验？

本实验禁止：继续搜索固定 rank、Fantope/CPTR 救援、根据结果更换 backbone、提前训练任何新接口、
读取 development/calibration/test/NavDream/target，或把内部恢复指标表述为驾驶安全收益。

## 2. 第一性对象与可证伪假设

对物理场景 (S)、外观 (a)、历史 (H)，冻结 DINO 与原生路径为

\[
D^a=\Phi(X^a),\qquad (Z_n^a,Q^a)=\Psi_{native}(X^a,H).
\]

冻结 DINO 输出为 \(D^a=\Phi(X^a)\)。具体输出必须由源码与 checkpoint 拓扑决定：DINO-only
系统写为

\[
O^a=T_K\circ\cdots\circ T_1(D^a,H),
\]

而只有确有独立 native 与 DINO 路径的 hybrid 系统才写为

\[
O^{a_n,a_d}=T\!\left(F(Z_n^{a_n},Q^{a_n},D^{a_d},H),H\right).
\]

禁止为了统一实验图而给 DINO-only 模型虚构 native path。

只对信息保持外观集合 \(\mathcal A_{pres}\) 检验功能保持；对遮挡、严重低照等信息破坏扰动不要求
强制不变，而要求不确定性和拒判账本。竞争解释允许同时成立：

- **H0 / scene prior：** correct DINO 不优于 geometry-matched mismatch 或 removed；
- **H1 / amplification：** 控制 DINO 后，native/interface 路径的外观改变仍造成显著输出漂移；
- **H2 / compression：** full tokens 存在组外任务增量，而 compressed tokens 丢失该增量；
- **H3 / layer complementarity：**有限预冻结中间层在 last layer / last-four mean 外提供组外任务增量，
  且其 clean-stage interchange 能影响真实 planner 输出或效用；
- **H4 / mixed：** H1、H2、H3 中至少两项独立成立；
- **H5 / current-interface unrecoverable：** 在冻结目标、读出类与预算下，full tokens 也无组外增量。
- **H6 / writer-target mismatch：** DINO/interface evidence 能预测任务，但以 native hidden-state 重建为目标的
  writer 在真实冻结 tail 中产生更差 task output 或 utility；
- **H7 / task-before-compression：** 任务上下文在 raw/full evidence 仍可访问时带来超出主效应的组外交互
  增量，而不是仅由更多 token、参数或训练预算造成。

H5 只否定当前接口/函数类合同，不证明 DINO 在信息论意义上没有相关信息。

## 3. 数据角色与冻结对象

- 统计独立单位：recording group，不以 anchor 或 seed 伪扩充样本量；
- source-fit 42 groups 沿既有角色使用：28 discovery、7 selection、7 outer-confirmation；
- outer-confirmation 只允许在协议、阈值、代码、checkpoint 与裁决函数冻结后读取一次；
- development、calibration、test、NavDream、target 全程封存；
- DINOv3-H+ checkpoint、SHA256、代码版本、288×512 预处理、ImageNet mean/std、patch/token 顺序、
  最后四层聚合、BF16/FP32 边界与缓存 provenance 必须在科学运行前固化；
- planner checkpoint、源码哈希、实际 DINO 接口拓扑、history source、完整切口、随机性、相机帧与
  轨迹时间绑定同时固化。缺少已训练 checkpoint 时只能完成源码/实现资格，不得生成科学机制结论。

任何资产或绑定无法核验，裁决为 `INVALID_ASSET_OR_BINDING`，不得读取科学结果。

## 4. 实验链：每一级只由上一级证据授权

### R007V0S：拓扑与机械资格（2 anchors）

1. 先从源码与 checkpoint state dict 判定 `DINO_ONLY` 或 `HYBRID_NATIVE_DINO`；
2. 原始运行与 identity replacement 必须逐元素一致；
3. `DINO_ONLY`：raw-token、projector、compression、decoder-query 等真实顺序切口逐一验证，full raw-token
   clean replacement 必须复现 clean output；
4. `HYBRID_NATIVE_DINO`：clean/shift 四格组合必须只改变指定路径，joint clean replacement 必须闭合；
5. mismatch/removed 映射、shape、token 顺序、finite、primary/replay 必须闭合；
6. 使用合成反例确认审计器能检出路径串扰、错误 donor、坐标错位、stage 跳连与伪 exactness。

任一失败即 `STOP_MECHANICAL_INTERFACE`，不进入科学解释。

### R007V0D：无训练的拓扑合法有限干预（28 discovery groups）

对 `DINO_ONLY` 系统，按源码拓扑执行 clean-stage-into-shifted 的顺序 interchange，报告每一切口的
有限恢复、相邻阶段恢复损失、最差组和外观分层。对 `HYBRID_NATIVE_DINO` 系统，才对同一
\((S,H)\) 计算完整 2×2 因子：

| native path | DINO path | 目的 |
|---|---|---|
| clean | clean | clean reference |
| shifted | shifted | observed shifted system |
| shifted | clean | DINO 路径恢复上界 |
| clean | shifted | native/interface 路径效应 |

两类拓扑都另加 geometry-matched mismatch、removed、identity 与完整切口。主量为物理单位
归一化的有限输出差，不以 token norm 比值代替机制证据。报告每阶段的增量恢复、交互项、最差组和外观
分层；DINO-only 不报告不存在的 native/DINO 交互，hybrid 不强迫主效应可加，也不强迫唯一归因。

该阶段只用于估计量纲、测量噪声、决定 selection 前冻结的最小有意义效应与等效界；不得据此宣称
组外机制成立。

### R007V0P/F：任务信息增量（1-fold smoke → 6-fold正式）

首先冻结两个互补目标，禁止用其中一个代替另一个：

- \(Y_{task}\)：合法 task supervision / contextual utility，用于判断证据是否与任务正确性有关；
- \(O_{clean}\)：同场景 clean planner output，仅用于判断功能恢复，不作为驾驶正确性的替代品。

随后对相同训练样本、optimizer steps、seed 和参数/FLOPs tolerance 比较：

1. native/history-only；
2. normalized final-layer correct DINO tokens；
3. 冻结 Constant Eye/R004 合同使用的 normalized last-four-layer mean；
4. compressed correct DINO tokens；
5. compressed + full-token skip；
6. 由 discovery groups 上的 outcome-free 规则生成的有限中间层候选；
7. selection 前锁定的最佳单层与等维双层组合；
8. 各视觉臂的 mismatch 与 removed 控制。

层候选不得遍历 outer-confirmation。候选索引由实际 `len(model.blocks)` 计算的固定深度分位点与最后四层
构成，精确索引、归一化与组合规则在任何 selection/outer 结果可见前写入 manifest。双层组合默认采用
等权 normalized mean，从而保持 token 数与通道宽度不变；只有等权组合显示增量后，才允许在后续方法
阶段比较少量 learned scalar mixture。不得把 concat 带来的额外宽度/参数当作层互补证据。

主 endpoint：group-held-out 任务风险与 correct-minus-control 配对差；次 endpoint：选择/轨迹输出误差、
场景难度分层、最差组、训练稳定性和推理成本。probe 结论只限于登记的函数类，不能升级为 Bayes
充分性或不可恢复性定理。任何 layer complementarity 必须同时满足 probe 的组外增量与对应真实 stage
interchange 的 output/utility consequence；只满足前者不得授权多层方法。

### R007V0W：冻结 tail 可实现性与写入目标审计

仅当 R007V0P/F 支持 correct evidence 的组外任务增量时执行。固定同一 reader、writer 切口、冻结 tail、
参数/FLOPs tolerance、样本、优化步数与 seed，比较：

1. native hidden-state MSE；
2. final-feature / strong model-stitching matching；
3. frozen-tail task-output matching；
4. 合法 task supervision。

同时记录 hidden error、tail output error 与 task utility。H6 只有在实际学习模型中出现“feature error 更低，
但 tail output/utility 更差”，且 selection 与 outer-confirmation 方向一致时成立。人工 latent 插值、单样本
异常或仅 discovery 结果不能授权该结论。梯度允许穿过冻结 tail 回到 writer，但 tail 权重不得更新。

### R007V0T：上下文时序 $2\times2$ 审计

仅当 R007V0P/F 支持 full/raw evidence 增量时执行。四臂为：

| evidence | context-free | task-conditioned |
|---|---|---|
| compressed | 普通 adapter | 压缩后读取任务 |
| raw/full DINO | 普通 full-token reader | 任务前置证据读取 |

冻结 DINO/tail、数据、token 输出预算、writer 切口、参数/FLOPs tolerance、训练步数与搜索预算。主估计量为
group-held-out 的 difference-in-differences：

\[
\Delta_{int}=(R_{raw,free}-R_{raw,cond})-(R_{comp,free}-R_{comp,cond}).
\]

风险越低越好，故 $\Delta_{int}>0$ 才与 task-before-compression 方向一致；阈值与等效界必须在 outer
前冻结。若收益可由 token/参数/搜索预算解释，则 H7 不成立。

### R007V0R：应不变与应响应的受控成对实验

分别登记标签保持外观、合法任务指令变化、任务相关场景证据变化与任务无关背景变化。前者检验功能
稳定，后两类中的任务相关变化检验必要敏感性。任何任务/场景反事实必须有受控仿真或经过验证的标签；
否则对应结果只记 `UNRESOLVED_COUNTERFACTUAL_VALIDITY`，不得沿用旧标签。

### R007V0C：一次性 outer-confirmation

冻结以下内容后才可读取 7 outer-confirmation groups：

- 代码提交、资产哈希、全部 arms、训练预算、统计函数；
- discovery 后定义的最小有意义效应、等效界与方向；
- `AMPLIFICATION / COMPRESSION / LAYER_COMPLEMENTARITY / MIXED / SCENE_PRIOR /
  WRITER_TARGET_MISMATCH / TASK_BEFORE_COMPRESSION / UNRESOLVED /
  CURRENT_INTERFACE_UNRECOVERABLE` 的机械裁决函数。

outer 只做一次，不因失败调阈值、改 donor、删组或增加网络模块。

## 5. 从证据到方法的唯一许可映射

| R007V0 结果 | 唯一允许的下一步 | 禁止外推 |
|---|---|---|
| compression only | full-token bridge 与 compressed/skip 对照 | 不引入 VJP、低秩或新 uncertainty head |
| layer complementarity only | 固定等权层组合与同参数 scalar mixture | 不把普通多层融合宣称为贡献 |
| amplification only | 同一结构上的 ordinary functional consistency | 不声称 cross-attention 必要 |
| amplification + compression，且分别不可替代 | 才允许组合一个 task-conditioned DINO evidence interface | 不预设 cross-attention 或旧 FQB 结构 |
| correct ≈ mismatch/removed | 停止当前场景视觉依赖主张 | 不通过增加容量强行恢复 |
| full-token 当前合同无增量 | 停止当前 DINO-interface 合同并复盘 target/alignment | 不宣称 DINO 信息论失败 |
| writer-target mismatch | 最小功能兼容 writer 与强 stitching 对照 | 不把人工反例当作真实机制 |
| task-before-compression | 同预算 task-conditioned reader 与普通 full-token reader | 不把 attention 本身宣称创新 |
| mixed/unresolved | 只允许一个能区分剩余解释的最小实验 | 不进入方法训练 |

VJP 仅在 task-shaped restriction 对同 reader、同预算的 direct-output residual 呈现不可替代 clean-no-harm
或组外收益后才能解锁；低秩/Fantope 永久退出当前主实验序列。

## 6. 论文与主张边界

R007V0 即使通过，也只支持“冻结 DINO 到冻结 planner 的指定接口存在可复现的信息/敏感性机制”。
它不证明新方法有效、NavDream 泛化、official utility 提升、跨 planner 通用、实时或安全。方法级贡献必须
在后续相同 DINO、相同数据、容量和计算预算匹配的最强简单基线上取得组级可靠优势，并通过 clean
非劣、zero-shot NavDream、多 planner 与 P50/P95 时延硬门。

DINOv3-H+ 是视觉基础模型而非 VLM。若论文使用“VLM/VLA 视觉鲁棒性”表述，最终主行为证据必须至少
覆盖一个真实 VLM/VLA 驾驶策略，并把图像派生 caption/query/history 纳入受干预路径；普通 planner 与
Constant Eye 只能分别承担机制控制和 faithful baseline，不能替代这一范围要求。
