# R004H7-BS4B4R1：合法候选域 preference oracle 预注册

状态：`PREREGISTERED_NOT_RUN`。本轮只修正 B4 的候选域类型错误，不训练网络，不读取新的科学
结果，不自动启动正式运行。

## 1. 为什么允许 R1，但不能覆盖 B4

B4 的 primary/replay 与机械审计可重复，却把 168 个冻结 intent-ineligible challengers 纳入
oracle，同时把 `pose_exact AND all_eligible` 命名为 reconstruction。其科学数值永久保持无效。

R1 的唯一设计变化不是根据 B4 outcome 选择的：候选 membership 直接采用在 B4 之前已经生成并
冻结 SHA 的 H7BS1 `challenger_eligible` mask；B4 的数值、winner、margin 和 hard outcome 被列入
禁止输入。原 B4 的 42-group 信号阈值、tie tolerance、bootstrap seed/resamples 与全部诊断 margin
原样保留。因此 R1 是确定性的 type repair，不是结果驱动的 threshold search；但 fit source 已有
历史暴露，仍不能替代最终独立确认。

## 2. 数学对象先于代码

对 group (g)、anchor (a)，六个注册 challenger 的冻结 membership 为
(e_{gai}\in\{0,1\})。合法顶点集定义为

\[
V_{ga}=\{0\}\cup\{i\in\{1,\ldots,6\}:e_{gai}=1\}.
\]

只有 (i\in V_{ga}) 才允许调用 candidate scorer、形成 pair edge、进入 oracle、hard rescue、
Laplacian 或后续 loss。当前冻结域包含 3,864 个 eligible challengers；15 anchors 只有 native，
657 anchors 至少有一个 challenger。对应总计 4,536 个 scored candidates 与 13,402 个 complete-
graph pairs。这些计数只来自预 B4 membership artifact，不来自 B4 scores。

零 challenger anchor 不删除。定义

\[
G_{ga}=\max_{i\in V_{ga}}(y_{gai}-y_{ga0}),
\]

故 (V_{ga}=\{0\}) 时 (G_{ga}=0)、pair rate 为 0。每组仍严格平均 16 anchors：

\[
G_g=\frac1{16}\sum_{a=1}^{16}G_{ga}.
\]

禁止 complete-case deletion，避免通过删除 candidate-supply 困难 anchor 抬高 oracle ceiling。

## 3. 合法标签与评分合同

每个合法 candidate 独立调用当前 official one-frame NAVSIM PDM path；内部固定 PDM baseline、
candidate-conditioned reactive traffic 与 human penalty filter 保持不变。一次把所有候选交给 scorer
或对 ineligible candidate 评分均为协议失败。

### 3.1 正式运行前的 scorer 语义勘误（未读取 outcome）

源码检查确认 `pdm_score_from_interpolated_trajectory` 对每条被评分轨迹 \(\tau\) 构造的是

\[
[\tau_{PDM}(S),\tau],
\]

随后返回 `pred_idx=1` 的分数；progress 在该内部二元 proposal set 内归一化。runner 对 native 与
每个合法 challenger 分别调用该函数，并未把 `[native, challenger]` 联合送入 scorer。因此本轮
estimand 精确定义为

\[
F_S(\tau)=\operatorname{PDM}_{1f}([\tau_{PDM}(S),\tau])_{pred},\qquad
\delta_{i0}^*=F_S(\tau_i)-F_S(\tau_0).
\]

native 仍然是候选生成、membership 和最终差分的 reference，但不是 challenger scorer 调用中的
progress-normalization baseline。旧 B4 文档中的 `[native, candidate]` 表述作废，不能用于论文。
该勘误不改变任何 candidate、threshold、seed、统计分母或 scorer 调用；它在正式 R1 outcome 前
由官方源码与现有 runner 静态核对得到。

由于 `PDMScorer` 会读取 `PROGRESS_MODE`，正式 runner 必须在任何 scoring 前 fail-closed 要求
`PROGRESS_MODE=eval`（环境变量未设置时官方默认同样为 `eval`），并把实际模式写入 primary/replay
与机械审计。`gen_gt` 或其他模式一律使接口无效。

该标签不含跨 anchor 的 two-frame extended comfort，因此不是完整 official PDMS。它只用于判断
source-fit 上是否存在值得学习的 preference 信号。

## 4. 分离的实现资格门

必须分别记录，禁止再次合并命名：

1. 672 个 metric cache、H7BS1 panel artifact、role identity 与 SHA 全部匹配；
2. 全部注册轨迹（native+6 offsets）的 realized poses 与冻结 panel bit-exact；
3. membership mask shape、artifact SHA 与冻结计数 exact；
4. scored registered indices 精确等于 (V_{ga})，ineligible score count 必须为 0；
5. 全部合法 outcomes 有限且位于合法范围；
6. 42 个 anchor-0 sentinel 对合法 candidates 的 outcome/state bit-exact；
7. primary 后再做一次完整 materialization，两份 672-anchor ledger exact；
8. `PROGRESS_MODE` 与冻结 `eval` 合同 exact；
9. 独立 audit 从 ledger 重算 summary 和裁决。

任一失败即 `INVALID_PREFERENCE_LABEL_INTERFACE`，科学值不可解释。

## 5. 统计设计与裁决

recording group 是唯一独立单位；anchor、candidate、pair 均为组内重复。原 B4 判据不变：

1. 42/42 groups 至少有一个合法 nonzero pair；
2. 至少 21/42 groups 至少有一个合法 challenger 严格优于 native；
3. 42 个 (G_g) 的 group bootstrap 95% CI 下界严格大于 0；
4. 全部实现门通过。

通过只得到
`GO_TO_MATCHED_STRUCTURE_PREREGISTRATION__NO_TRAINING_YET`；不是训练授权。若零组有改善则
`STOP_CURRENT_LEGAL_PANEL_NO_ORACLE_IMPROVEMENT`，其他为
`HOLD_CURRENT_LEGAL_PANEL_PREFERENCE_SIGNAL_INSUFFICIENT`。

### 5.1 开封前新增的科学解释边界

上述 GO/HOLD/STOP 的机械定义、阈值、候选、seed、统计分母和输出字符串保持不变；但其科学作用域
严格限定为 **candidate-local one-frame supervision**：

1. `INVALID_PREFERENCE_LABEL_INTERFACE` 仍压倒一切科学解释，必须先修复接口；
2. 接口有效时，GO/HOLD/STOP 均不得直接裁决带 previous-action context 的两帧目标，因为
   temporal term 可能产生 one-frame 中不存在的排序信号；
3. 因此原 `STOP_CURRENT_LEGAL_PANEL_NO_ORACLE_IMPROVEMENT` 在论文中只能解释为
   `STOP_ONE_FRAME_SUPERVISION_ON_CURRENT_PANEL`，不能写成“当前 panel 没有任何 full-context
   oracle opportunity”；
4. 任何接口有效 outcome 之后，都必须另行预注册 scoped contextual target 的 materialization；
   只有该 contextual oracle 也缺乏非平凡机会时，才停止当前 candidate panel；
5. 本解释修正发生在 formal outcome 前，不读取任何 B4R1 科学数值，也不授权运行或训练。

`0/.001/.005/.01/.02/.05` 绝对差异率只作冻结 sensitivity；不得据此选择后续 margin。hard rescue
仍是描述量，不事后升级为 gate。

## 6. 反例与必须覆盖的合成测试

- membership 为 `[T,F,T,F,F,F]` 时，scored indices 必须恰为 `[0,1,3]`；插入 index 2 必须失败；
- 零 challenger anchor 必须保留且所有 pair diagnostics 为 0，不得 NaN；
- membership 总数漂移即使 oracle signal 很强也必须判 INVALID；
- pose reconstruction、membership artifact、ineligible-score count、sentinel replay 任一失败均须
  压倒正向信号；
- 可变 (K) 下 candidate/pair counts 必须由合法域机械重算，不允许 `672×7` 或 `672×21` 硬编码。

这些测试通过只证明实现按定义工作，不证明 source oracle 存在。

## 7. 论文与理论边界

R1 若通过，只支持“合法候选域上存在 source-fit candidate-local score-difference supervision 与
oracle opportunity”。
它不支持 exact coboundary head 必要性、视觉泛化、NavDream、跨 planner、完整 PDMS、实时性、
clean no-harm、驾驶收益、安全或 ICCV readiness。

对未来理论，(K_{ga}=|V_{ga}|) 是 context-dependent：当 (K_{ga}\ge2) 时，可在连通图上定义
Laplacian 与谱隙；当 (K_{ga}=1) 时，商空间维数为 0，没有 preference supervision 或
\(\lambda_2\)。网络必须用 ragged mask 排除该 candidate，而不是给它伪造 score/edge。

无论 R1 的 one-frame outcome 为 GO/HOLD/STOP，只要接口有效，它都不构成完整 contextual target、
sequential policy 或最终 EPDMS 的裁决。
