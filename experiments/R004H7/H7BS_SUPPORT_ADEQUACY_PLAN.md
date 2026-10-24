# R004H7-BS 预注册：shadow-query 支撑充分性诊断

状态：`PREREGISTERED__H7BS0_ATTRIBUTION_IMPLEMENTATION_GO`。本阶段只回答固定视觉任务图能否完整承载冻结
候选的离散查询，不读取预测场、teacher margin、regret、ranking flip 或任何 H7-B outcome，
也不训练网络。

## 0. 已观察事实与 post-hoc 边界

H7-B 已观察到 265/384 anchors 有效、119 无效；110 个无效原因是 native footprint
越出冻结 ROI，9 个是没有有效 challenger。以下 H7BS-0 归因是在看到该 coverage
结果之后提出，必须标为 post-hoc descriptive diagnosis，不能冒充原 H7-B 的预注册检验。
它只重读已经产生的 realized-trajectory panel artifacts，不产生新的科学 outcome。

### 竞争解释与可区分预测

| 候选解释 | H7BS-0 可观察预测 | 不相容模式 | 状态边界 |
|---|---|---|---|
| H-S：合法轨迹的空间/时间支撑超过固定 ROI | 越界集中在晚时刻；方向与真实行驶进展一致；所需单侧扩张量可由角点包围盒解释 | 初始姿态即大范围越界、方向随机或尺度异常 | 可检验 |
| H-B：坐标、尺度或锚点中心绑定错误 | 初始 realized pose 不是局部原点；早期越界普遍；边界方向与米制 NAVSIM 合同不一致 | 初始原点闭合且越界随时间单调出现 | 可进行 fail-closed 审计，但不能证明所有上游绑定绝对正确 |
| H-V：ROI 内视觉证据不足 | 需要视觉有效性/置信 mask 才能观察 | 当前 panel validity 在读取任何 field 前已决定，且只由几何与 intent 合同产生 | 不能解释本次119个 invalid；未来约束可用性仍未测量 |
| H-Sel：缺失集中于困难驾驶几何 | invalid anchors/groups 具有更大路径长度、横向偏移、总 yaw variation 或曲率 proxy | valid/invalid 的组级几何分布相近 | 仅探索性关联，不作因果或显著性主张 |

appearance 不是本次 coverage 的分层因素：每个 anchor 的 panel 在四个 appearance fields
之前只构建一次，四个条件共享同一 validity。H7BS-0 必须明确报告这一代码与数据事实，
不得把几何缺失误写成某个 appearance 的视觉失败。

### H7BS-0 冻结输出

对全部384 anchors和每个注册候选，确定性计算并保存：

- 四个边界方向的最小 support slack、最大所需扩张量、首次越界时间索引；
- 初始 pose 到局部原点的平移/yaw 残差，检查米/弧度和锚点中心合同；
- native path length、terminal progress、最大横向位移、总 yaw variation、最大离散曲率 proxy；
- valid/invalid 和24个groups的完整计数；
- invalid 与 valid 的 group-equal描述性差异；不计算p值，不把anchors当独立样本；
- 119个 invalid anchors 的只含几何元数据清单，不写入RGB、特征、field或teacher。

H7BS-0 只允许归因并提出后续合同，不允许选择新 ROI、prefix 或阈值。其结果完成机械
复算后，才决定是否继续执行下文的 fixed-prefix support qualification。

## 1. 问题与数学对象

冻结节点图的连续支撑为

\[
\Omega=[-32,32]\times[-32,32]\subset\mathbb R^2.
\]

对 anchor \(i\)、候选 \(j\) 和 official execution 的离散前缀
\(T_k=\{0,\ldots,k\}\)，令车辆矩形为 \(R(p_{ijt})\)，定义

\[
F_{ij}^{(k)}=\bigcup_{t\in T_k}R(p_{ijt}),\qquad
s_{ij}^{(k)}=\min_{t,c}\min\{c_x-x_{\min},x_{\max}-c_x,
c_y-y_{\min},y_{\max}-c_y\},
\]

其中 \(c\) 遍历每个姿态的四个解析矩形角点。于是完整支撑的必要充分条件为

\[
F_{ij}^{(k)}\subseteq\Omega\iff s_{ij}^{(k)}\ge0.
\]

此外仍要求 rasterized node set 非空。anchor 在前缀 \(k\) 上有效，当且仅当 native
有效且至少一个 intent-eligible challenger 有效。该定义只涉及 execution、车辆尺寸和
查询域，不涉及模型误差或选择结果。

## 2. 两种修复不是同一个方法

1. **扩大空间域**会改变神经网络输出格、投影支撑、teacher 标签和计算量；当前64×64
   checkpoint没有域外预测，禁止通过padding、常数外推或裁剪假装已扩大ROI。该路线属于
   新架构/重新训练，不能作为H7-B的实现修补。
2. **缩短固定时间域**保留当前网络与节点图，但把科学对象从完整4秒decision quotient
   改为prefix decision quotient。即使后续通过，也只能支持前缀排序相关性。

因此本诊断只选择一个固定前缀；不在空间大小与时间长度之间做基于outcome的搜索。

## 3. 单位、数据角色与重复测量

- 42 fit recording groups：只用于从冻结字典选择一个前缀；
- 24 historically exposed development groups：只做一次支撑资格复核；
- recording group是唯一独立单位；每组16 anchors、7 candidates和多个时刻都是组内重复；
- calibration、test、NavDream和target继续封存；
- H6 fields、teacher labels和H7-B association结果不得由诊断代码打开。

这不是随机处理实验，而是冻结panel的确定性普查；不进行显著性检验，也不把384 anchors
当作独立样本。

## 4. 冻结选择规则

候选前缀字典按NAVSIM 0.1秒时基固定为

\[
k\in\{10,15,20,25,30,35,40\},
\]

对应1.0至4.0秒。对每个 \(k\)，完整报告：

- 每个group的有效anchor数；
- native越界、无有效challenger、空node set、intent-ineligible的计数；
- \(s_{ij}^{(k)}\) 的group-equal分布；
- 达到每组至少12/16有效anchors的group数；
- 每个候选完整包围盒所需的前/后/左/右扩张量，仅作为未来架构诊断。

在42个fit groups上选择满足42/42 groups均达到12/16的**最大** \(k\)。若不存在这样的
\(k\)，状态为`STOP_CURRENT_FIXED_GRID_SHADOW_ROUTE`，不读取development。

若存在 \(k^*\)，将它连同源码、official NAVSIM hashes、车辆参数、候选顺序和全部失败
规则冻结；随后在24个development groups上只运行一次支撑资格复核。只有24/24 groups
均达到12/16，才可把该固定 \(k^*\) 注册为H7-B2 prefix estimand。否则停止当前fixed-grid
shadow路线，不降低门槛、不按group选择不同前缀。

## 5. 防泄漏与停止规则

- 诊断runner对H6 field路径、teacher label payload和H7-B result payload执行拒绝访问；
- 允许读取source split identity、official metric cache中的native trajectory、车辆参数及
  deterministic execution所需对象；不得读取其评分或teacher派生数组；
- 所有384 anchors保留在coverage分母；candidate失效不删除；
- 任何看到development coverage后的字典、门槛、ROI或prefix修改，都必须成为新的探索，
  不能回写本预注册；
- 通过只授权H7-B2的独立预注册和实现资格，不自动读取预测outcome，更不授权H7-C。

## 6. 解释边界

本诊断能够判定的是当前视觉任务图与冻结候选查询的支撑相容性。它不能证明decision
quotient有效、不能证明神经网络结构必要、不能证明驾驶或安全收益，也不能把prefix
结论外推到完整4秒轨迹。若必须扩大ROI，需重新建立投影、标签、网络输出与计算预算的
架构合同，并重新训练容量匹配baselines。

## 7. 设计依据

本计划遵循group-level replication、组内重复测量和事前停止规则，避免把anchors或
candidates当作独立样本。实验设计流程参考：Kassis, T., Agarwal, V., He, Y., Patel, D.,
& Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for
Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
