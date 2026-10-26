# R005E3A 决策目标资格实验正式结果

日期：2026-10-26  
冻结实现：`77f37e43862bc352a753d6ee9141c907c9ac926b`  
机械审计：`PASS`  
冻结科学分类：`MIXED_DECISION_IMPROVEMENT_WITHOUT_COMPLETE_SCENE_BINDING`  
机制裁决：`ALL_NATIVE_COLLAPSE__ZERO_ABSOLUTE_ORACLE_CAPTURE`  

## 1. 本轮问题

在模型、DINO输入、初始化、数据顺序、optimizer、schedule、epoch与step预算逐job匹配时，仅将父
R005E2B的margin MSE替换为soft selection-regret，是否能把已有平均relation信号转化为候选选择收益，
并且该收益是否依赖正确场景视觉？

本轮不是Differential Query、外观鲁棒、NavDream、完整pseudo-closed-loop或驾驶安全实验。

## 2. 执行完整性

- 6 folds × 3 seeds = 18 primary + 18 same-job same-GPU replay；
- 4,749,313个参数，与父V2完全一致；
- primary/replay prediction bit-exact，final model state exact；
- 父初始化、selected epoch、optimizer-step预算与训练顺序合同通过；
- OOF fold/target/mask覆盖、有限值、source-only与outer-train-only temperature重算通过；
- development/calibration/test/NavDream/target均未访问；
- 完整远端目录：`/data/khsong/aorta_runs/r005e3a_77f37e4`；
- 本地紧凑证据：`experiments/results/R005E3A_77F37E4/`。

## 3. 冻结结果

| 条件 | Relation MSE ↓ | Decision regret ↓ | Sign error ↓ | Boundary error ↓ |
|---|---:|---:|---:|---:|
| Parent V2 | **0.112362** | 0.019301 | 0.193758 | **0.410218** |
| Soft-regret correct | 0.732367 | **0.016650** | **0.192262** | 0.410714 |
| S1 mismatch | 0.732367 | 0.016650 | 0.192262 | 0.410714 |
| S2 mismatch | 0.732367 | 0.016650 | 0.192262 | 0.410714 |
| S3 mismatch | 0.732367 | 0.016650 | 0.192262 | 0.410714 |
| Removed | 0.732367 | 0.016650 | 0.192262 | 0.410714 |

组级配对bootstrap：

- treatment − parent regret：`-0.0026503`，95% CI
  `[-0.0054378,-0.0008577]`；
- 登记的相对父模型“opportunity recovery”：`0.159175`；
- S1/S2/S3/removed − correct regret：全部精确为`0`，CI均为`[0,0]`；
- 输入干预top-choice flips：全部为`0`。

因此按冻结规则只能判`MIXED`：相对父模型regret下降，但四项scene-binding门全部失败。该冻结分类
不可被事后修改。

## 4. 关键机制解剖

审计后只读复算发现，2016个OOF anchor-seed观测中：

- correct、S1、S2、S3、removed predictions逐元素完全相同；
- 所有合法challenger得分精确为`-1`，native得分为`0`；
- native选择率为100%，非native选择数为0；
- 模型相对native的实际收益`G_model=0`；
- oracle opportunity为`G_oracle=0.0166504`；
- absolute oracle capture为`G_model/G_oracle=0`。

随后使用冻结primary checkpoints在各自outer-train split上复算全部18个job：18/18均为全部challenger
精确`-1`、100%选择native、absolute oracle capture为0。故退化在训练域已经存在，不是单纯OOF泛化
失败。可复算产物见`posthoc/collapse_diagnosis.json`。

父模型regret高于native fallback，因此本轮regret改善只表示消除了父模型的有害干预：

\[
G_{oracle}=G_{model}+R_{model},\qquad
G_{model}=0.
\]

原登记比率

\[
\frac{R_{parent}-R_{model}}{G_{oracle}}=0.159175
\]

是相对一个有害parent的regret修复，不是绝对oracle机会捕获。该协议缺陷不改变冻结`MIXED`，但禁止
在未来继续使用此量作为单独正向门。

## 5. 第一性失败解释

soft selection-regret的梯度为

\[
\frac{\partial L}{\partial s_k}
=\frac{p_k}{T}\left(\mathbb E_p[u]-u_k\right).
\]

当前challenger经过`tanh`限制到`[-1,1]`，native固定为0，所有job的冻结温度均为`T=0.125`。当
challenger进入`s=-1`时，softmax因子约含`e^{-8}`，同时`tanh`导数接近0；两者共同造成近吸收的
低梯度区域。多数负收益/近tie challenger又使“全部拒绝”成为容易达到的经验解。

这支持`OPTIMIZATION_AND_PARAMETERIZATION_COLLAPSE`，不支持“RGB没有信息”“DINO失败”或
“Differential Query失败”。总体无界logit一致性定理没有保证当前有界函数类、温度与优化预算可实现
非退化解。

## 6. 下一步边界

- `STOP`：不得进入2×3确认、Differential Query、NavDream或行为实验；
- `GO`：只允许L2优化/可实现性资格，先验证简单generic scorer能否在outer-train捕获严格正的
  native-relative收益；
- 新协议必须同时要求：`G_model`组级CI下界大于0、absolute oracle capture达到预注册阈值、存在有益
  nonnative选择、correct相对S1/S2/S3/removed具有决策敏感性；
- 候选修正（无界中心化logit、非饱和margin、具有非消失梯度的ranking surrogate、opportunity-aware
  weighting）目前都只是待比较假设，不得直接升格为方法。

## 7. 主张边界

本结果只支持：当前有界`tanh` scorer + `T=0.125` + soft-regret组合在source-fit OOF与outer-train均
退化为native-only。它不支持或否定外观鲁棒性、最终AORTA架构、NavDream、跨planner、策略级收益、
驾驶安全或Best Paper资格。

