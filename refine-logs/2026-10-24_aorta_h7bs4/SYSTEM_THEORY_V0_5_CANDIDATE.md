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

### 3.1 单一论文主张与30秒故事

现有外观鲁棒方法通常优化特征不变或最终轨迹一致，但两者都没有回答：在视觉只部分覆盖未来
任务区域时，哪些稳定信息足以可靠地区分完整驾驶候选？AORTA的候选主张是：**学习可积的
相对任务势及其相关不确定集合，在共享变量上直接判断候选排序是否可识别；可识别才干预，
否则回退。** 因而外观稳定、任务信息、完整时域与拒判不是四个并列模块，而是同一
“decision identifiability under partial visual evidence”问题的四个必要条件。

该故事只有在B0/B1、容量匹配结构消融、未见外观和实际决策收益都成立时才可写入论文主张；
任何一关失败都必须缩小或替换主张。

### 3.2 候选网络参数化

令冻结视觉编码器产生多相机历史token \(Z\)，投影与支撑审计产生局部证据\(O\)。共享图解码器
输出未定gauge的node logits \(a_\theta\)和edge scale logits \(s_\phi\)：

\[
u_\theta=P_\perp a_\theta,
\qquad P_\perp=I-\frac{1}{|V|}\mathbf1\mathbf1^\top,
\qquad z_\theta=B_0u_\theta .
\]

因此\(u_\theta\)选取mean-zero代表元，真正进入监督与决策的是exact 1-cochain
\(z_\theta\in\operatorname{im}B_0\)，并由架构恒有\(B_1z_\theta=0\)。这比给direct-edge
head事后增加cycle penalty更强：可积性不是软偏好，而是可行域。

edge预算采用硬方向参数化

\[
w_{\phi,e}=w_{\mathrm{ledger},e}
              +w_{\mathrm{support},e}(O)
              +\operatorname{softplus}(s_{\phi,e}),
\]

其中ledger项不可学习，support floor在缺失或离支撑边界更远时不得下降。具体floor形式只有在
B0/B1测出关键盲区分布后才能冻结；当前禁止凭直觉指定常数。

### 3.3 候选核心loss：proper relation learning而非装饰性多项加权

teacher node potential为\(d^*\)，edge target为\(z^*=B_0d^*\)，残差
\(r_e=z_{\theta,e}-z^*_e\)。中心与scale联合训练的基本项采用Laplace proper score：

\[
\mathcal L_{\mathrm{proper}}
=\frac{1}{\sum_e\omega_e}
 \sum_e\omega_e
 \left(\frac{|r_e|}{b_e}+\log b_e\right),
\qquad
b_e=b_{\min,e}(O)+\operatorname{softplus}(s_{\phi,e}).
\]

\(\omega_e\)只编码预先冻结的合法teacher/support mask；不能由预测误差事后重加权。
\(\log b_e\)阻止网络仅靠扩大不确定性降低残差代价，硬floor保证missing区域不能虚假变窄。

对同一场景的clean/appearance pair，定义

\[
\mathcal L_{\mathrm{app}}
=\frac{1}{|E_O|}\sum_{e\in E_O}
\rho\!\left(z^{(a)}_{\theta,e}-z^{(0)}_{\theta,e}\right).
\]

不把多个lambda任意拼成“创新loss”，而把稳定性写成约束优化：

\[
\min_{\theta,\phi}\ \mathcal L_{\mathrm{proper}}
\quad\text{s.t.}\quad
\mathbb E[\mathcal L_{\mathrm{app}}]\le\delta_{\mathrm{app}},
\]

再用非负对偶变量进行primal–dual优化。\(\delta_{\mathrm{app}}\)必须在fit协议中预注册；常数场
虽然满足约束，却会被proper task score惩罚。Hodge consistency由参数化保证，不再添加重复的
cycle loss。候选排序loss仅作后续消融，因为把它放入主loss会引入planner/candidate-provider
依赖，削弱跨planner主张。

### 3.4 核心推理算法

一次视觉前向产生\((u_\theta,w_\phi,O)\)，一次multi-source图solver构造共享不确定包络；所有
完整4秒candidate复用该包络。对challenger \(j\) 与native \(0\)，计算

\[
L_{j0}=\inf_{d\in\mathcal U_\theta(X,O)}
\left[\min_{q\in F_j}d_q-\min_{q\in F_0}d_q\right].
\]

只有\(L_{j0}\ge\gamma\)、几何/资格合同通过且candidate存在时才替换native。没有合法候选、
证据不足、集合过宽或数值审计失败都映射到同一个明确动作：abstain并保留native。该算法的
创新候选不是Dijkstra本身，而是“结构化任务表示—相关集合—共享比较—选择性干预”的闭环。

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

先以B0A在164个资格cache上审计96×64生成器，再以B0A2对H7BS4A的672个精确fit token
一一重建带数值误差账本的teacher；两者禁止跨token替代。随后定义候选possible-minimizer集合

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

- 96×64 teacher生成器已在164个资格cache上双构建PASS，但672个精确fit-token工件尚待B0A2；
- support-conditioned uncertainty的确定性floor尚未定义；
- graph set在实际误差下是否非平凡未知；
- 1m差异节点是否是teacher-critical未知；
- 时延、NavDream、三planner和clean no-harm均未验证；
- 外部独立method reviewer不可用，本文不是独立审稿结论。
