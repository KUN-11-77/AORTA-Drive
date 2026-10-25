# R005E0：NAVSIM 目标作用域与反事实价值审计

状态：`SOURCE_SCOPE_AUDIT_PASS__DATA_TARGET_NOT_RUN__TRAINING_LOCKED`。

日期：2026-10-25。

## 1. 审计结论

用户提出的三层区分必须进一步细化为四个不可互换的数学对象。AORTA 后续不得再用
“full target”同时指代单阶段分数、两帧行效用和完整两阶段 benchmark。

### 1.1 B4R1 原始单帧对象

\[
F^{1f}_{\mathrm{raw},i}
=\texttt{PDMScorer.\_aggregate\_pdm\_scores}(\tau_i).
\]

它是 B4R1 已验证有 headroom 的唯一对象。它先于候选级 human filter 的 component 修改形成。

### 1.2 过滤后单阶段对象

\[
\widetilde F^{1f}_i
=M_i\frac{5EP_i+5TTC_i+2LK_i+2HC_i}{14},
\quad M_i=NC_iDAC_iDDC_iTLC_i.
\]

这里所有 factors 必须来自 filter 后返回 components。R004H7 的 posthoc 机械审计已证明
`F_raw` 与 `F_tilde` 在 506/4,536 candidates 上不相等，因此二者不得代换。

### 1.3 带真实上一周期轨迹的 contextual row utility

对当前候选 \(\tau_i\) 与因果上可用的 previous selected trajectory \(h\)，定义

\[
U^{ctx}_i(h)
=M_i\frac{5EP_i+5TTC_i+2LK_i+2HC_i+2EC(h,\tau_i)}{16}
=\frac78\widetilde F^{1f}_i+\frac18 M_iEC(h,\tau_i).
\]

这是一行候选在 two-frame comfort 注入后的 scoped contextual utility。等式要求：相同的
filter 后 components、相同乘性门、权重 `[5,5,2,2,2]`，且没有混入阶段聚合。它不是完整
pseudo-closed-loop benchmark。

### 1.4 固定后续策略下的两阶段 action value

官方 two-stage aggregator 对一个 mapping 的两个 frame streams 分别计算 stage-one 分数与
按 endpoint/start-point 高斯核加权的 stage-two 分数，再先相乘、后对两条 stream 平均。令
\(z_k^+\) 为 current stream 的 second-stage scenes，\(\pi_{ref}\) 是冻结的后续 planner，

\[
w_k(i)=
\frac{\exp[-\|e(\tau_i)-s(z_k^+)\|^2/(2\sigma^2)]}
{\sum_l\exp[-\|e(\tau_i)-s(z_l^+)\|^2/(2\sigma^2)]},
\quad \sigma^2=0.1,
\]

则当前动作在固定 continuation 下的代码对齐价值可写为

\[
Q_{\pi_{ref}}(i\mid x,h)
=\frac12\left[
U^{ctx}_i(h)\sum_k w_k(i)U^{ctx}(z_k^+;\pi_{ref})+C^-(h,\pi_{ref})
\right].
\]

\(C^-\) 是 previous stream 项；在 previous trajectory 与 continuation 固定时，对当前候选为常数。
若权重和数值下溢，必须严格复现官方均匀回退，不能只实现理想化公式。

`Q_pi_ref` 是**固定后续策略反事实 action value**，不是任意自适应未来策略下的真实闭环价值；
它只有在候选替换、stage-two mapping、交通反应和 metric cache 的反事实语义都通过 candidate-validity
审计后才可 materialize。

最终 benchmark 是策略级量

\[
J(\pi)=\operatorname{Mean}_{mappings}
\frac12\left(S_{0}^{\pi}S_{0,+}^{\pi}+S_{-}^{\pi}S_{-,+}^{\pi}\right),
\]

而不是一条候选的 `U_ctx`。离线候选监督最多估计某个明确 continuation 下的 \(Q\)，不能直接称
完整 official target，更不能称驾驶安全。

## 2. 精确源码绑定

本审计绑定官方 NAVSIM commit `0a380a9063d7162ec93d0f51e9990ebac585f720`：

- `docs/metrics.md`：NAVSIM v2 EPDMS 与 two-stage pseudo-closed-loop 定义；
- `run_pdm_score.py`：filter 后 components 注入 two-frame comfort 并重算 `score`，以及 stage1×stage2；
- `scene_aggregator.py`：真实 previous/current overlap、Gaussian endpoint weights 与均匀回退；
- `pdm_scorer.py`：raw candidate-local `pdm_score` 的形成位置。

B4R1 实际服务器 fork 与官方 main 的 `scene_aggregator.py`、comfort 和 `run_pdm_score.py` 内容一致；
`pdm_scorer.py`/`pdm_score.py` 含项目 fork 修改。R005 正式运行只能绑定一个完整来源根和源码哈希，
禁止把官方 main 与 fork 的局部文件拼接成未实际执行的 evaluator。

| Source artifact | SHA-256 |
|---|---|
| official `docs/metrics.md` | `f28dd00f3c85469705d4492e8d86996c8122440cfb70d1816f4f545233d0f978` |
| official `run_pdm_score.py` | `c0a3a0648b72178203a8d6559a41e8993e81860d9e0367eac91dd94c26a06ecf` |
| official `scene_aggregator.py` | `13dec362355a41c4a6e7386301efe05bcb6774a598a79026cf371ebdd3bc8e65` |
| official `pdm_comfort_metrics.py` | `883b606cc5f98285bd6ff917e9a4129c608b709d58fe6c2435f6b7bd27c2e6ce` |
| official `pdm_scorer.py` | `9ed5527bace1a272c869a26ed714dcddceced16219d8e2859f45503d3cf032eb` |
| official `pdm_score.py` | `066ff166f8cd9de69499f522d78b9d82207c1ce0daf36bd4d0870ad9439e4b1a` |
| B4R1 fork `pdm_scorer.py` | `c9132b8b7019cc05c57573a26a8326cd7762dfdff35592046625a7fa895f6b62` |
| B4R1 fork `pdm_score.py` | `7d0b82b8420c52dfedfda2a56b4aa44bd371cd1adbac58f001c19ccda60cf99c` |

固定来源链接：

- [NAVSIM v2 metrics](https://github.com/autonomousvision/navsim/blob/0a380a9063d7162ec93d0f51e9990ebac585f720/docs/metrics.md)
- [official run and final-score aggregation](https://github.com/autonomousvision/navsim/blob/0a380a9063d7162ec93d0f51e9990ebac585f720/navsim/planning/script/run_pdm_score.py)
- [official two-stage scene aggregation](https://github.com/autonomousvision/navsim/blob/0a380a9063d7162ec93d0f51e9990ebac585f720/navsim/planning/simulation/planner/pdm_planner/scoring/scene_aggregator.py)

## 3. R005 的可证伪问题

1. `U_ctx` 在合法 ragged panel 上是否仍有组级正 headroom？
2. `argmax F_raw`、`argmax F_tilde` 与 `argmax U_ctx` 多大程度不同？
3. 忽略 history 的 contextual regret 是否非平凡？
4. 现有数据是否足以定义 candidate-specific `Q_pi_ref`；若可以，它是否相对 `U_ctx` 再产生排序翻转？
5. 上述对象能否由独立 replay/audit 精确复算，且 previous history 无未来泄漏？

## 4. 当前裁决

- `TARGET_FIDELITY_SOURCE_AUDIT = PASS`：公式、代码阶段和边界已区分；
- `B4R1_TO_FILTERED_EQUIVALENCE = FAIL`：不能直接复用 raw 标签；
- `U_CTX_DATA_MATERIALIZATION = NOT_RUN`；
- `Q_PCL_COUNTERFACTUAL_VALIDITY = UNRESOLVED`；
- `NETWORK_TRAINING = LOCKED`；
- development/calibration/test/NavDream/target 保持封存。

