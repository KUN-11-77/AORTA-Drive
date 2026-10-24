# R004H7-BS4B0A：前向 96×64 精确 teacher 资格门

状态：`PREREGISTERED_NOT_RUN`。本实验只构建后续关键区域可观测性检验所需的 teacher
工件，不读取预测场、DINO 特征、候选结果或任何 calibration/test/NavDream/target 数据。

## 为什么必须先做这一关

H7BS3R1 已证明前向 96×64 网格可以容纳冻结候选的完整 4 秒几何查询；H7BS4A-D0 又发现
8 个几何上不同的 native/challenger 对在离散对称差区域没有直接 projective evidence。
但是“无证据节点存在”不等于“该节点会控制任务算子”。要检验 teacher-criticality，必须先在
同一 96×64 定义域上得到不经插值、误差方向可审计的 teacher 标签。

对冻结 polygon cover \(S\)，网格节点 \(q\) 的对象是带符号距离

\[
d_S(q)=\begin{cases}
\operatorname{dist}(q,\partial S), &q\in S,\\
-\operatorname{dist}(q,\partial S), &q\notin S.
\end{cases}
\]

实现把二进制浮点坐标解释为精确实数，用有理数谓词判定成员关系，并输出包围真实距离的
binary64 区间。每个节点同时保存坐标变换误差、距离舍入误差及向外舍入的总误差账本。
`polygon_error=0` 只相对于冻结 cache 内的原始 polygon 对象成立，不代表 HD map 或现实世界
没有建图误差。

## 冻结设计

- 网格：\(x\in[-32,64]\)、\(y\in[-32,32]\)、1 m、64 行×96 列；
- 数据：164 个冻结 source NAVSIM metric caches；
- 图层：`ROADBLOCK`、`INTERSECTION`、`DRIVABLE_AREA`、`CARPARK_AREA`；
- 禁止把旧 64×64 标签插值或复制到新增前向区域；
- 独立完整构建两次，每次 164×6144=1,007,616 个节点；
- 两次 manifest 记录、artifact 字节和全部数组必须逐项完全相同；
- 任一 cache 失败、非有限值、误差账本方向错误或 replay 差异均判 FAIL；
- 墙钟上限 120 分钟，超过即停止，不改变参数重跑。

## 结论边界

PASS 仅允许说：前向 96×64 的冻结 source teacher 工件可重复、可进入随后独立预注册的
teacher-critical observability 检验。它不改变 H7BS4A 的 STOP，不证明视觉可观测、决策充分、
模型泛化、实时性或驾驶安全收益。

下一关 B0B 才会在冻结 pair 上用区间 possible-minimizer 集合判断缺失证据是否可能控制
set-min；B0A 的标签值不得用于调整 B0B 的阈值或筛选样本。
