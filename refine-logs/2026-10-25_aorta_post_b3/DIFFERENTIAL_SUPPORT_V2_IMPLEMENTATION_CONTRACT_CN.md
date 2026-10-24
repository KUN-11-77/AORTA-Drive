# Differential-Support Relation v2：实现与反例测试合同

状态：`DESIGN_CONTRACT_ONLY__IMPLEMENTATION_AND_TRAINING_HOLD__B4R1_REQUIRED`。

本文只把 v2 数学对象翻译为张量、mask、前向、loss 与测试合同。它不选择隐藏维度、层数、token
上限或阈值，不授权写训练代码。所有数值超参数必须在 B4R1 通过后的 source-only 预注册中确定。

## 1. 单一系统接口

```text
multi-camera history X
        |
        v
frozen visual encoder (once) -> Z, token geometry/provenance
                                      ^
native + ragged legal challengers     |
        |                             |
        v                             |
full-horizon soft occupancy measures |
        |                             |
        v                             |
positive / negative / common support queries
        |_____________________________________|
                         |
                         v
       shared geometry-biased global attention
                         |
                         v
        antisymmetric native-relative comparator
                  |                 |
            point margin         positive scale
                  |                 |
                  +--------+--------+
                           v
           group-calibrated lower-bound arbitration
```

冻结 planner、candidate provider、DINO、相机/ego 几何、车辆 footprint 与 evaluator。新增可训练对象
仍只有一个 relation decoder；scale 是其输出接口，不是第二个方法。

## 2. 输入与 ragged type contract

对 batch size \(B\)、最大注册候选数 \(K_{max}\)、完整轨迹长度 \(T\)、视觉 token 数 \(P\)：

| 张量 | 形状 | 语义 |
|---|---|---|
| `visual_tokens` | \([B,P,d_v]\) | 冻结 DINO token，只计算一次 |
| `token_geometry` | \([B,P,d_g]\) | camera、patch ray、历史时间、投影 provenance |
| `trajectories` | \([B,K_{max},T,d_s]\) | pose、yaw、velocity/curvature/time；index 0 固定为 native |
| `legal_mask` | \([B,K_{max}]\) bool | 与 B4R1 membership 完全同一对象；native 为 true |
| `soft_support` | ragged \([B,K,T,N]\) 或稀疏等价物 | 每个时刻的 oriented footprint measure |
| `direct_mask` | 与 support 可对齐 | 直接投影证据 provenance，不是 hard attention mask |

`legal_mask=false` 的 candidate 不得进入 support builder、attention、loss、scale、argmax、calibration
或日志聚合。零 challenger anchor 保留，输出空 relation set，最终动作恒为 native。

## 3. 支持构造必须先于网络

对合法候选 \(i\)：

\[
\mu_i=\operatorname{SoftSweep}(\tau_i;\mathcal B,\mathcal Z),
\]

其中 \(\mathcal B\) 是冻结车辆 footprint，\(\mathcal Z=\Omega\times[0,T]\) 使用完整 4 s 物理时间。
禁止用 2.5 s、中心点或四角点替代主对象。然后机械构造

\[
\nu^+_{i0}=[\mu_i-\mu_0]_+,
\quad
\nu^-_{i0}=[\mu_0-\mu_i]_+,
\quad
\mu^\cap_{i0}=\min(\mu_i,\mu_0).
\]

必须逐 pair 记录总变差质量

\[
m^\Delta_{i0}=\|\nu_{i0}\|_{TV}=\int d(\nu^++\nu^-),
\]

直接投影比例 \(\kappa_{i0}\)、首次分叉时间、正负支持质量以及每个时间段的质量分布。它们是
provenance/diagnostic，不自动改变 label 或 sample weight。

## 4. Query 而非裁剪

support encoder 只生成 queries 与 geometry bias：

\[
Q^+=Q_\theta(\nu^+),\quad
Q^-=Q_\theta(\nu^-),\quad
Q^\cap=Q_\theta(\mu^\cap).
\]

三类 query 均读取同一全局视觉 key/value：

\[
E^r=\operatorname{MHA}(Q^r,K(Z),V(Z);b^r_{geo}),
\qquad r\in\{+,-,\cap\}.
\]

禁止把 `direct_mask` 直接用作 key/value 的零一裁剪。若需要稀疏 attention，只能使用在结果前冻结
且对所有对照相同的候选无关 token selection；否则时延提升与方法收益混杂。

空正/负支持使用显式 `empty_support_token` 与零质量标志，不用全零向量冒充普通证据。这样可区分
“轨迹相同”“一侧集合包含”“投影缺失”三种不同情况。

## 5. Comparator 与反对称化

对每个 pair 构造 ordered tuple

\[
H_{i0}=(E^+,E^-,E^\cap,Q_{i0},p_{i0}),
\]

其中 \(p_{i0}\) 包含 \(\kappa\)、正负质量、missing pattern 与 time-of-divergence，但不包含
teacher/outcome。将正向与交换后的 tuple 在同一 batch 中送入共享 comparator：

\[
\hat\delta_{i0}=\tfrac12[G_\theta(H_{i0})-G_\theta(H_{0i})].
\]

交换时必须同时交换 \(E^+,E^-\)、相对轨迹方向和所有有方向的 provenance；公共证据不变。两次
调用通过候选维批处理，不允许重复视觉 backbone。

point hidden state 与 scale hidden state 可以共享 trunk，但输出分别为

\[
\hat\delta_{i0}=W_\delta h_{i0},\qquad
\sigma_{i0}=\sigma_{min}+\operatorname{softplus}(W_\sigma[h_{i0},p_{i0}]).
\]

scale 读取 provenance 不代表 coverage；coverage 仍来自独立 recording-group calibration。

## 6. Loss 合同

### 6.1 主任务损失

\[
\mathcal L_{task}
=\frac{1}{N_{legal}}
\sum_{b,i:\,legal}\rho(\hat\delta_{bi0}-\delta^*_{bi0}).
\]

若 batch 内 `N_legal=0`，该项定义为零并记录 empty batch，不产生 NaN。默认 challenger uniform；
hard-negative weighting 只有独立 source-only 证据才能启用。

### 6.2 外观轨道约束

\[
\mathcal L_{orbit}
=\frac{1}{N_{paired}}
\sum_{b,a,i}\rho(
\hat\delta_{bi0}(T_aX)-\operatorname{sg}[\hat\delta_{bi0}(X)]).
\]

同一 physical scene、candidate panel、membership 与 label 必须保持 bit-identical；否则该 pair 不是
合法 appearance intervention。主优化是 task risk，orbit 项是 source-frozen constraint，而不是
任意加权的第二目标。

### 6.3 Scale proper loss

scale 只能用预注册的 proper residual score 学习相对 sharpness。它与 point loss、最终 conformal
quantile 和 OOD empirical coverage 分开报告。禁止用训练 NLL 直接声称可信区间。

## 7. 推理合同

1. 复核 role、checkpoint、candidate membership 与输入数值；
2. 共享视觉编码一次；
3. 为全部合法 challengers 向量化构造 differential queries；
4. 输出 \(\hat\delta_i,\sigma_i,p_i\)；
5. 用冻结 group-calibration quantile 计算 \(L_i=\hat\delta_i-q\sigma_i\)；
6. 若任何 fail-closed contract 失败、quantile 为无穷或 \(\max_iL_i\le0\)，保留 native；
7. 否则选择最大 \(L_i\) 的 challenger。

`direct_mask`/\(\kappa\) 默认不单独否决候选；hard gate 只有预注册对照证明必要且非真空后才能加入。

## 8. 必过 metamorphic tests

这些测试不证明模型有效，但防止实现破坏数学对象。

| Test | 必须成立的关系 |
|---|---|
| identity | `trajectory_i == native` 时 \(\nu^+=\nu^-=0\)，反对称 point margin 精确为 0 |
| swap | 交换 native/challenger 后 \(\nu^+\leftrightarrow\nu^-\)，\(\hat\delta_{0i}=-\hat\delta_{i0}\) |
| permutation | 合法 challengers 置乱只置乱对应输出，不改变数值 |
| mask noninterference | 插入、删除或修改 masked candidate 不改变任一合法输出 |
| zero-challenger | 输出空集合且动作 native；loss、audit、summary 均有限 |
| late-fork | 仅 2.5–4 s 分叉时 full-horizon \(|\nu|>0\)，prefix arm 为 0 |
| common-prefix | 延长完全相同前缀不改变已有差异支持的正负身份 |
| direct-mask separation | 改变 provenance mask 不改变 support geometry；是否改变 point/scale 由网络输入合同分别测试 |
| global-context access | 即使直接差异区 token 为空，query 仍可读取全局 keys；禁止意外 hard crop |
| appearance pairing | 合法 appearance pair 的 trajectories/membership/labels exact，只有 visual token 改变 |
| backbone once | 候选数变化不增加视觉 backbone 调用次数 |
| finite numerics | 空支持、极小质量、全 missing、ragged batch 不产生 NaN/Inf |

## 9. 反例测试

至少构造以下数据自由或合成反例：

1. **共同 minimum：**两条轨迹共享最低风险节点但差异区不同；scalar-min 相同，signed support 不同；
2. **晚期分叉：**0–2.5 s exact，2.5–4 s 相反；prefix 不可分，full support 可分；
3. **区域外信号：**差异区无直接 token，但区域外 traffic-light token 决定 margin；hard crop 失败，
   global-key query 可表达；
4. **真正二世界：**全部可访问表示相同、隐藏差异风险符号相反；任何 point predictor 至少错一个，
   只有 abstention 能避免统一错误承诺；
5. **trajectory shortcut：**同一 pair geometry 配不同场景与相反 label；trajectory-only head 必败；
6. **appearance collapse：**常数 relation 的 orbit loss 为零但 task loss 非零；
7. **free residual bypass：**v1 中令 difference=0、自由 interaction 拟合 label，证明旧分支职责不可识别。

## 10. 时延与复杂度记账

计时必须包含在线输入解码、DINO、support builder、query attention、comparator、scale 与 arbitration；
同时报告不含 DINO 的增量时延，但不能用它代替端到端时延。同步 batch-1 P50/P95/P99、峰值显存、
吞吐与候选数 \(K\) 缩放均必报。

反对称 comparator 的交换分支必须向量化；若两次 comparator 使时延失败，可以比较结构等价的 odd
readout，但不得取消 swap test 后仍声称反对称保证。

## 11. 解锁顺序

1. B4R1 正式 oracle 通过；
2. 仅实现 support builder、tensor contract 与上述 metamorphic tests；
3. source-only 小规模 implementation qualification，不读取 NavDream/development outcome；
4. 预注册参数/FLOP/token/seed 匹配的普通 scorer 与 DA-WAM/GTRS-style 对照；
5. 结构 study 通过后才进入外观、选择性仲裁、跨 planner 与行为实验。

任何阶段失败都删除对应复杂度。本文的目的不是保证 v2 被采用，而是保证它若被检验，检验的确实
是同一个数学对象。

