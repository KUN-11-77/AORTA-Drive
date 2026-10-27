# AORTA DINO 决策充分性理论 V1

状态：`FORMAL_OBJECTS_CLOSED__EMPIRICAL_PREMISES_UNTESTED`
日期：2026-10-27

## 1. 研究对象

设物理场景为 $S$，外观干预为 $a$，推理时合法任务上下文为 $B$，冻结 DINOv3-H+ 输出为

\[
D^a=\Phi(g(S,a))\in\mathbb R^{N\times d},
\]

驾驶目标为 $Y$。$Y$ 必须在实验前明确绑定为合法任务量，例如轨迹监督、候选 contextual utility
或官方 scorer component；不能在看到结果后切换目标。AORTA 研究的不是 DINO 特征整体是否相似，而是：

\[
\boxed{\text{DINO evidence sufficiency}\;\rightarrow\;
\text{interface preservation}\;\rightarrow\;\text{decision consequence}.}
\]

这里必须区分视觉基础模型与视觉—语言/视觉—动作系统。DINOv3-H+ 只提供冻结视觉证据，不能因为
接入 DINO 就把系统称为 VLM/VLA。普通回归或候选评分 planner 可用于定位机制；论文的 VLM/VLA
行为主张至少需要一个真实多模态驾驶策略作为主实验对象，并核验语言指令、caption、视觉历史缓存等
是否由图像产生。凡是由图像产生的量都属于受外观干预的计算路径，不能被错误地固定在上下文 $B$ 中。

本项目因此研究两个层级：

1. **机制层：**冻结 DINO 证据能否穿过一个已登记的冻结决策接口；
2. **行为层：**同一原理能否在真实 VLM/VLA 驾驶策略中带来任务正确、clean 非劣且成本可接受的收益。

机制层通过不能替代行为层，单一 planner 通过也不能推出跨架构普适性。

## 2. 命题一：关系结构稳定不推出下游功能稳定

定义 patch Gram 结构 $G(D)=DD^\top$。任取正交矩阵 $Q^\top Q=I$，令 $D'=DQ$，则

\[
G(D')=DQQ^\top D^\top=DD^\top=G(D).
\]

对线性接口 $W\in\mathbb R^{d\times k}$，有

\[
D'W-DW=D(QW-W).
\]

因此，对所有可能 $D$ 都保持接口输出不变的充要条件是 $QW=W$。一般正交坐标变化不满足该条件。
这个构造只证明逻辑非蕴含，不证明真实 DINO 在某个外观下发生全局正交旋转。现实中的接口放大必须由
真实 stage interchange 与有限输出变化验证，不能由 Gram/CKA 稳定性直接推断。

## 3. 命题二：压缩造成的条件充分性缺口

设 $Z=C(D)$ 是 DINO 表征的确定性压缩。在平方损失、同一总体分布和不受限可测预测函数类下，

\[
\mathcal R^*(Y\mid B,X)
=\mathbb E\left[\|Y-\mathbb E[Y\mid B,X]\|^2\right].
\]

由条件期望的正交投影性质，

\[
\begin{aligned}
\Delta_{suf}(C)
&=\mathcal R^*(Y\mid B,Z)-\mathcal R^*(Y\mid B,D)\\
&=\mathbb E\left[
\|\mathbb E[Y\mid B,D]-\mathbb E[Y\mid B,Z]\|^2
\right]\ge 0.
\end{aligned}
\]

等号成立当且仅当 $\mathbb E[Y\mid B,D]$ 几乎处处是 $(B,Z)$ 的可测函数。该命题给出理论 estimand，
但有限数据、有限 probe 的测试误差差只能估计指定函数类下的可恢复性，不得直接写成 Bayes risk 或
信息论充分性结论。

### 3.1 三重缺口：信息、冻结解码器兼容性与有限学习

令 $E=C(D,B)$ 是接口输出。对同一数据分布与损失，定义

\[
R_D^*=\inf_f R(f(D,B)),\qquad
R_E^*=\inf_h R(h(E,B)),
\]

以及只允许通过登记的合法 writer $W$ 写入冻结 tail $T_\psi$ 时

\[
R_{T,E}^*=\inf_W R\bigl(T_\psi(W(E,B),B)\bigr).
\]

因为 $(E,B)$ 是 $(D,B)$ 的函数，且冻结-tail writer 类是全部 $(E,B)$ 可测预测函数的子集，对任意
登记模型 $\theta$ 有精确分解

\[
\begin{aligned}
R(\theta)-R_D^*
={}&\underbrace{R_E^*-R_D^*}_{\text{任务信息损失}}
+\underbrace{R_{T,E}^*-R_E^*}_{\text{冻结解码器兼容性缺口}}\\
&+\underbrace{R(\theta)-R_{T,E}^*}_{\text{有限参数化/优化/估计缺口}}.
\end{aligned}
\]

各项非负依赖上述函数类包含关系。若真实系统有未纳入 $D$ 的 native/caption/history 视觉旁路，完整输入
必须改写为 $(D,V,B)$，不能遗漏旁路后作纯 DINO 归因。该分解给出失败路由：full DINO probe 成功而
压缩 probe 失败指向信息损失；压缩 probe 成功而 frozen-tail writer 失败指向兼容性；理论 writer 可实现
而当前网络失败才指向学习缺口。

### 3.2 任务必须在压缩之前进入接口的条件性动机

令 $D=(d_1,\ldots,d_N)$、$d_i\in\{0,1\}$，任务上下文 $B=j$，目标 $Y=d_j$。若先用与任务无关的
$k$ 比特编码 $Z=C(D)$，然后要求对全部 $(D,j)$ 零错误回答，则 $C$ 必须区分全部 $2^N$ 个 $D$，故
$k\ge N$。若编码器先获得任务，$Z=C(D,B)=d_B$，则 $k=1$ 足够。

该反例只证明“先压缩后询问”与“任务条件化压缩”在有限通信预算下可能不等价；它不证明真实 DINO
token 是独立比特，也不证明任务条件化读取必然更好。对应实验必须在相同 token、参数、训练和时延预算
下比较 raw/compressed × context-free/task-conditioned 的 $2\times2$，并以前瞻组外交互效应裁决。

### 3.3 native 特征均值不一定是正确写入目标

令冻结 tail 为 $T(z)=z^2$，原生隐藏状态 $z\in\{-1,1\}$ 均产生正确输出 $1$。若 DINO 保留“正确输出
为 1”却不保留区分符号的 nuisance，则平方特征重建的 Bayes 解为

\[
\mathbb E[Z_{native}\mid D,B]=0,
\]

但 $T(0)=0\ne1=\mathbb E[T(Z_{native})\mid D,B]$。因此非线性 tail 下一般有

\[
T_\psi(\mathbb E[Z\mid D,B],B)
\ne \mathbb E[T_\psi(Z,B)\mid D,B].
\]

对目标输出集合 $\mathcal T_{good}$，定义冻结 tail 的任务等价纤维

\[
\mathcal F_\psi(\mathcal T_{good},B)
=\{z:T_\psi(z,B)\in\mathcal T_{good}\}.
\]

接口只需写入该集合中的可用点，不必恢复唯一 native 坐标。这个反例不证明真实训练发生目标失配；
只有在相同 reader/writer 下出现“native feature error 更低，但 frozen-tail task output/official utility 更差”
并能组外复现，才能把 target mismatch 写成经验机制。

### 3.4 可执行证据是三者的交集，不是单独的信息量

对给定上下文 $B$，令 $\mathcal E_Y(B)$ 表示保留任务目标 $Y$ 所需信息的证据集合，
$\mathcal R_T(B)$ 表示可由合法 writer 写入且冻结 tail 能产生的输出集合，
$\mathcal I_{pres}(B)$ 表示在登记的标签保持外观族上功能稳定的集合。真正可执行的鲁棒证据必须位于

\[
\mathcal E_{exec}(B)=\mathcal E_Y(B)\cap\mathcal R_T(B)\cap\mathcal I_{pres}(B).
\]

该集合表达不是可直接计算的算法，而是研究对象的约束定义。它排除了三种常见逻辑跳跃：DINO probe
可读不表示 frozen tail 可用；clean tail 可用不表示外观鲁棒；外观稳定不表示任务正确。R007 的每个实验
必须只消除其中一个竞争解释。

## 4. 命题三：视觉误差必须相对于决策间隔解释

对固定合法候选集 $\{1,\ldots,m\}$，真实效用为 $u_i$，预测为 $\hat u_i$。若

\[
\max_i|\hat u_i-u_i|\le\epsilon,
\]

令 $i^*=\arg\max_i u_i$、$\hat i=\arg\max_i\hat u_i$，则

\[
u_{i^*}-u_{\hat i}\le 2\epsilon.
\]

证明为

\[
u_{i^*}-u_{\hat i}
\le (u_{i^*}-\hat u_{i^*})
+(\hat u_{\hat i}-u_{\hat i})\le2\epsilon,
\]

其中使用 $\hat u_{\hat i}\ge\hat u_{i^*}$。若真实 top-1 与 runner-up 间隔严格大于 $2\epsilon$，则
$\hat i=i^*$。因此相同视觉预测误差对 near-tie 与大间隔样本的决策意义不同。该命题仅适用于固定
候选集与统一效用；不能直接推广为回归/扩散 planner 的 closed-loop 或 PDMS 保证。

## 5. 命题四：功能漂移的路径积分分解

设完整 DINO-to-output 函数为 $F_B(D)$。对可微路径
$D_t=D^a+t(D^{a'}-D^a)$，

\[
F_B(D^{a'})-F_B(D^a)
=\int_0^1J_{F_B}(D_t)(D^{a'}-D^a)\,dt.
\]

这说明小的无方向 feature norm 不能保证小的任务输出漂移；相关的是残余漂移在组合函数 Jacobian
敏感方向上的作用。但 Jacobian/VJP 不是预设方法，只是解释工具。正式机制证据必须来自真实有限干预，
并报告物理单位输出与任务效用，而不是只报告梯度范数。

### 5.1 功能漂移、clean 改动与平滑任务风险

定义冻结 tail 诱导的功能伪度量

\[
\rho_\psi(z,z';B)=d_\tau(T_\psi(z,B),T_\psi(z',B)).
\]

不同隐藏状态可产生同一输出，因此它通常不是隐藏空间真度量。若任务损失对 $d_\tau$ 是 $L$-Lipschitz，
且 $a$ 属于标签保持的 appearance family，则由三角不等式

\[
R_a(\pi_\theta)-R_0(\pi_0)
\le L\,\mathbb E\left[
d_\tau(\tau_\theta^a,\tau_\theta^0)
+d_\tau(\tau_\theta^0,\tau_0^0)
\right].
\]

两项分别是外观功能漂移与新接口的 clean 行为改变。该界解释为什么一致性和 clean identity 必须同时
评价，但不适用于不连续官方 hard gates，且右侧小不自动证明任务效用优于原模型。

### 5.2 硬事件必须通过安全余量而非平均漂移解释

对一个明确定义的几何事件，设 $c_S(\tau)>0$ 为安全余量，且

\[
|c_S(\tau)-c_S(\tau')|\le L_c d_\tau(\tau,\tau').
\]

则任意 $\varepsilon>0$ 下

\[
\Pr[c_S(\tau_\theta^a)\le0]
\le \Pr[c_S(\tau_\theta^0)\le L_c\varepsilon]
+\Pr[d_\tau(\tau_\theta^a,\tau_\theta^0)>\varepsilon].
\]

因此实验必须单列小 clean margin 场景与功能漂移尾部。该式只覆盖已定义事件、轨迹度量和环境合同，
不是碰撞、交通规则或完整系统安全保证。

### 5.3 单帧稳定不推出闭环稳定

若闭环差异满足

\[
\|\Delta s_{t+1}\|\le\alpha\|\Delta s_t\|+\beta e_t,
\]

且 $\alpha<1$、$e_t\le\varepsilon$，则

\[
\|\Delta s_t\|
\le\alpha^t\|\Delta s_0\|
+\frac{\beta\varepsilon(1-\alpha^t)}{1-\alpha}.
\]

真实驾驶系统未必收缩，所以该条件界不能由单帧实验宣称成立。teacher-forced、actual-history 与
sequential/pseudo-closed-loop 必须分层报告。

## 6. 四级证据链

1. **Evidence availability：**correct DINO 相对 history-only、mismatch 与 removed 提供组外任务增量；
2. **Evidence distribution：**full、compressed、单层与预冻结层组合之间存在可复现增量差异；
3. **Interface causality：**把对应 clean stage evidence 放入 shifted 真实 forward 能恢复输出或效用；
4. **Decision value：**机制授权的最小接口在 matched simple baselines 上改善 official utility，并满足 clean
   非劣与时延门。

任一层失败都停止依赖它的后续主张。尤其是：probe 可读不等于 planner 可用；stage restoration 不等于
驾驶正确；source utility 不等于 NavDream zero-shot 或安全保证。

## 7. 对架构设计的约束

- 不默认使用多层融合、cross-attention、低秩、VJP 或旧 AORTA loss；
- 中间层只有在“相对最后层/last-four mean 的组外增量 + 真实 planner consequence”均成立时才解锁；
- 新接口最多一个主要可训练组件，并必须兼容冻结 planner 的真实输入合同；
- 信息保持外观只约束任务功能，不要求全部 token 相等；信息损失外观要求风险账本与拒判，不强制一致；
- DINO 前向成本与接口/planner 成本分开报告，token 压缩不冒充 backbone 加速。

### 7.1 任务条件化 reader–writer 只能由证据解锁

若 R007 同时支持“压缩次序造成任务信息损失”和“native 特征恢复目标失配”，才允许实现一个最小
reader–writer：

\[
Q_\theta=q_\theta(B,Z_0),\qquad
P_\theta=\operatorname{softmax}\!\left(\frac{Q_\theta\bar D^\top}{\sqrt d}+\Pi\right),
\]

\[
E_\theta=P_\theta D,\qquad
Z_\theta=Z_0+E_\theta W_O,\qquad
\tau_\theta=T_\psi(Z_\theta,B).
\]

$\Pi$ 只能编码真实可得的相机/patch 几何；$W_O$ 零初始化只保证训练开始时的 identity，不保证训练后
clean 无伤害。创新候选不是“使用 attention”，而是经实验支持的两项必要设计：在有限预算下让真实任务
上下文先于证据压缩，以及面向冻结 tail 的任务等价写入。若普通 full-token reader、普通 cross-attention
或最终特征匹配达到相同结果，应删除更复杂的设计和相应贡献主张。

### 7.2 写入目标与上下文时序必须做正交实验

方法训练前冻结两组因子：

- **上下文时序 $2\times2$：**compressed/raw-full evidence × context-free/task-conditioned reader；
- **写入目标：**native feature MSE、final-feature/model-stitching、task-output matching、task supervision。

所有臂保持数据、冻结 DINO/tail、writer 切口、参数量、token 数、优化步数和搜索预算可比。只有前者出现
组外正交交互效应，才支持“task-before-compression”；只有实际学习模型出现更低 feature error 却更差
tail output/utility，才支持“target mismatch”。人工 latent 反例只证明可能性，不能替代该证据。

### 7.3 不变性必须同时包含敏感性

正式成对干预应覆盖：同场景同任务的标签保持外观变化（应保持功能）、合法任务指令变化（应响应）、
任务相关场景证据变化（应响应）与任务无关背景变化（不应无故响应）。任务/场景反事实必须有可靠标签；
无标签图像编辑不能承担“应该改变”的主张。由此避免把“对天气稳定但也看不见新行人”的模型误称为鲁棒。

### 7.4 最终 VLM/VLA 验证合同

主论文至少需要：faithful Constant Eye 作为可训练 planner-head 基线；一个冻结 planner 合同下的方法对照；
至少一个真实 VLM/VLA 驾驶策略的行为验证。两类训练合同分表报告，不能用冻结约束掩盖最终性能差异，
也不能把允许重训 head 的基线当作相同问题。跨 planner 有效与同一接口权重零样本迁移亦是不同主张。

## 8. 当前主张边界

以上命题是给定假设下的数学结论与实验 estimand。尚未证明 DINOv3-H+ 对当前任务条件充分、Constant
Eye 压缩确实丢失证据、中间层具有独立增量、真实接口放大残余漂移，或任何新方法改善驾驶行为。
同样未证明 native feature reconstruction 在真实模型中发生目标失配、task-before-compression 存在交互优势，
或该机制能迁移到 VLM/VLA。论文潜力来自这些命题构成一条可证伪因果链，而不是上述标准数学工具本身。
