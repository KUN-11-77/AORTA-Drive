# AORTA-v0.5 候选理论与系统建筑

状态：`DRAFT_REVISE__EXPERIMENT_GATED`。这是D0后的系统候选，不替代当前`METHOD_SPEC v0.4`，
也不授权训练或性能主张。

## 1. 建筑原则：最小充分复杂度

复杂度只有满足以下至少一项才可进入主系统：

1. 有明确反例证明删除它会破坏可识别性、可信性或完整时域；
2. 它是某个定理假设或方向正确数值界不可缺少的实现；
3. 容量匹配消融证明它带来不能由更简单基线解释的收益；
4. 它降低整体计算或使多planner共享成为可能。

“模块少”不是目标，“没有无证据模块”才是目标。允许复杂的确定性solver、误差账本和审计层；
trainable novelty仍尽量集中为一个quotient field decoder与一个相关scale/uncertainty head。

## 2. 五层承重结构

### L1 外观干预与信息目标

结构因果图为\(S\to X\leftarrow A\)，\(S\to Y\)，其中\(A\)是天气/光照，\(Y\)是
任务关系。paired appearance只干预\(A\)，不能改变\(S\)。目标不是恢复完整\(S\)，而是找到
对候选决策充分且对\(A\)尽量不敏感的统计量。

### L2 Hodge-consistent quotient representation

网络预测node potential representative\(u_\theta\)，任务监督与机制指标使用\(B_0u\)。
potential参数化强制cycle consistency，并通过mean-zero或固定参考节点选择计算gauge；候选差值
始终在商空间定义。

### L3 Support-conditioned correlated uncertainty

投影模块同时产生视觉evidence和support topology。uncertainty head输出edge budgets，但必须满足：

- 正值、有界、数值可审计；
- missing/support-distance不能使预算虚假变窄；
- map、projection、model、calibration与rounding误差分账；
- 相同edge变量在所有候选之间共享。

输出是\(\mathcal U_\theta(X,O)\)，不是一张带confidence颜色的确定热图。

### L4 Full-horizon robust decision functional

冻结tracker把所有candidate变成完整4秒query sets。每个native只运行一次multi-source solver，
所有challenger复用envelope；禁止短前缀和逐candidate重复神经推理。精确robust contrast保留
公共节点和路径相关性。

### L5 Selective runtime supervisor

候选供给、视觉可识别性、certificate coverage和最终选择分别记账。只有严格lower bound通过
才干预；其余保留native。最终指标是OOD rescue、new hard failure、clean non-inferiority、
coverage–risk–regret以及端到端P50/P95。

## 3. 系统图

```text
multi-camera RGB history
    -> frozen DINOv3 relation tokens
    -> audited projection + support topology O
    -> Hodge-consistent quotient decoder u_theta
       + bounded correlated edge-scale head w_phi
    -> graph uncertainty set U_theta(X,O)

frozen planner(s) -> native + eligible challengers -> official tracker -> full-horizon F_i

U_theta + {F_i}
    -> shared-variable exact robust contrasts
    -> eligibility / deviation / numerical contracts
    -> choose challenger only if identifiable; otherwise native fallback
```

## 4. 模块必要性账本

| 模块 | 目的 | 直接依据 | 删除实验 |
|---|---|---|---|
| frozen DINO关系tokens | 提供外观较稳定视觉先验 | H5/H6 | weaker/backbone-matched control |
| audited projection+mask | 区分空间包含与视觉证据 | H7BS4A | ignore-mask / learned missing token |
| potential/Hodge head | gauge不变、edge可积 | 理论 + H7C待验 | direct edge / cycle penalty |
| uncertainty head | 表达局部异方差与missing | D0 | fixed global / independent node band |
| exact joint solver | 保留共享相关性 | set-min反例 | independent intervals / nominal score |
| abstention | 防止证据不足时强改 | D0 + selective risk | always-intervene / confidence threshold |
| full horizon | 保留晚期分叉 | H7BS2 | 2.5s prefix |
| asymmetric 96×64 | 完整空间定义域 | H7BS3 | 64×64 / 128×128 cost control |

## 5. 不默认加入的复杂模块

- **Sheaf fusion**：只在跨camera/history冲突而非缺失成为主要误差时启用；
- **continuous neural field/operator**：只在1m aliasing或固定网格时延被证明为瓶颈时启用；
- **PEA/task-constrained adaptation**：只作强基线或条件备选，不与主贡献并行；
- **diffusion/RL candidate generator**：candidate supply成为主要瓶颈后另立研究问题；
- **depth/occlusion网络**：projective support与物理visibility差异被关键节点实验证明后再考虑。

## 6. 三个目标命题与证明义务

### 命题A：关系表示不变性与非坍缩

若appearance nuisance只诱导常数gauge变化，则edge relation与候选差精确不变。真实网络必须
额外以task error、场能量、mismatch/removal和未见外观验证其不是常数塌缩。

### 命题B：共享不确定集合上的排序界

若\(d^*\in\mathcal U_\theta\)，则exact infimum\(L_{ij}\)是实际candidate contrast的下界。
证明必须覆盖集合包含、相等集合、公共minimizer、missing路径、disconnected support、有限精度
和完整车辆footprint反例。

### 命题C：选择性仲裁

若field coverage和非field contracts均成立，授权pair满足冻结margin。概率主张只在明确的session
exchangeability/calibration单位下成立；NavDream作为经验OOD，单独报告coverage与失败，不继承
source保证。

## 7. 最近实验门

### Gate B0：teacher-critical observability

构建带数值误差账本的96×64 fit teacher，定义候选possible-minimizer集合

\[
C(F)=\{q\in F:\ell_q\le\min_{p\in F}r_p\}.
\]

检查D0的8个blind difference regions是否与\(C(F_i)\cup C(F_j)\)相交，并在全部pair报告
none/partial/full visual support。该实验只回答未观测节点是否可能控制teacher set-min，不训练网络。

### Gate B1：oracle uncertainty non-vacuity

使用teacher残差或冻结合成误差构造方向正确的联合集合，比较global、independent-node、graph-
correlated三种uncertainty在相同coverage下的selective coverage与ranking regret。若graph route
仍近乎全abstain，停止该主创新，不用增加模块挽救。

只有B0/B1通过，才进入H7C容量匹配网络结构消融与正式时延实验。

## 8. 当前未闭合项

- 96×64 teacher及其map/error ledger尚未生成和审计；
- support-conditioned uncertainty的确定性floor尚未定义；
- graph set在实际误差下是否非平凡未知；
- 1m差异节点是否是teacher-critical未知；
- 时延、NavDream、三planner和clean no-harm均未验证；
- 外部独立method reviewer不可用，本文不是独立审稿结论。
