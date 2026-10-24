# D0之后的理论结构修订：从二值可见性到相关不确定决策

## 1. 不改变的地基

1. 物理场景在纯外观变换下不变；学习目标应定义在任务等价类而非像素一致性上。
2. 预测对象是相对任务关系\([u]\in\mathbb R^{|V|}/\operatorname{span}(\mathbf1)\)，
   训练与比较应优先依赖\(Bu\)，而非绝对gauge。
3. 网络先产生candidate-free任务证据，再由冻结候选查询；不得把候选标签泄漏进表征学习。
4. 视觉指标、决策指标与驾驶安全必须分层，不相互冒充。

## 2. 被实验否定的接口

- 全候选至少一个直接观测节点：被H7BS4A两个blind native否定，且两者没有比较对象；
- 只要求候选对称差有直接证据：被D0的8个zero-support distinct pairs否定；
- 公共区域自然相消：对set-min一般不成立，公共未知低值可以同时控制两侧。

## 3. 修订后的数学接口

令\(O\)为实际投影证据，网络不直接宣称一个确定场，而输出受gauge、空间关系和校准误差约束的
集合\(\mathcal U_\theta(I,O)\)。对候选pair定义共享变量上的鲁棒差异集合

\[
\mathcal D_{ij}(I,O)=
\left\{R(u,F_i)-R(u,F_j):u\in\mathcal U_\theta(I,O)\right\}.
\]

若\(\inf\mathcal D_{ij}>\delta\)，候选\(j\)才在该证据下可信优于\(i\)；若
\(\sup\mathcal D_{ij}< -\delta\)则方向相反；其余情况abstain。\(\delta\)必须来自独立冻结的
数值/校准预算，不能从decision结果倒推。

相同\(u\)同时进入两个候选函数，因此公共节点的相关性被保留。未直接观测区域不被删除，而是
通过更宽的不确定集合降低可裁决性。该形式也天然保持常数gauge不变性，只要\(R\)对全局常数
平移的候选差异不变。

## 4. 对网络架构的约束

1. **视觉证据层**：DINO关系特征与严格投影support mask分开输出；mask不能被网络悄悄忽略。
2. **部分可观测quotient operator**：在图边/相对势上预测均值与相关不确定性，不把未观测区
   当作确定插值；空间传播必须有可审计的支持距离或图路径条件。
3. **gauge-consistent solver**：从边关系恢复一个规范代表，仅作为计算接口；损失和结论仍定义
   在商空间。
4. **共享鲁棒比较层**：直接求解或保守界定\(\mathcal D_{ij}\)，禁止把两个候选独立置信区间
   粗暴相减后称为精确保证。
5. **拒判与原规划器回退**：证据不足时不改写冻结planner；coverage与conditional utility必须
   同时报告，防止只保留容易样本。

## 5. 下一批证伪顺序

1. fit-only teacher-criticality诊断：8个盲差异节点是否可能改变teacher排序；
2. oracle不确定集合：在不训练新网络前，测试共享鲁棒比较是否保留足够coverage；
3. 容量匹配结构对照：deterministic field、独立interval、相关quotient set三者比较；
4. 通过后才训练，并分别验证appearance stability、calibration、decision utility与latency；
5. 最后才进入NavDream与跨planner，不以表征指标替代行为收益。

该修订是实验驱动的理论候选，不是已证明的创新。其ICCV价值取决于后续是否证明：相关不确定
quotient接口在相同视觉输入、容量和时延下，比普通BEV/SDF、PEA式确定修正及独立uncertainty
基线获得更好的coverage–risk–regret折中。
