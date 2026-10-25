# R004H7-BS4B4R1：过滤后单阶段分数语义审计

状态：`POSTHOC_SEMANTIC_DIAGNOSTIC__REGISTERED_B4R1_DECISION_UNCHANGED`。

## 审计问题

B4R1 登记的 `one_frame_pdm_score` 来自 `PDMScorer._aggregate_pdm_scores()`。该值在候选级
human false-positive filter 之前形成；filter 随后可能修改返回的 hard/weighted metric components，
而官方完整评估在注入 two-frame comfort 后会从这些已修改 components 重新计算 `score`。

因此必须区分：

\[
F^{1f}_{\mathrm{raw},i}=\texttt{one\_frame\_pdm\_score}_i,
\]

以及按返回 components 复算的过滤后单阶段分数

\[
\widetilde F^{1f}_i
=M_i\frac{5EP_i+5TTC_i+2LK_i+2HC_i}{14},
\quad
M_i=NC_i\,DAC_i\,DDC_i\,TLC_i.
\]

二者相同是一个需要核验的经验事实，不是接口保证。

## 机械结果

对 B4R1 primary ledger 的全部 672 anchors、4,536 candidates 与 13,402 无序 pairs 逐项复算：

| 检查 | 结果 |
|---|---:|
| `F_raw != F_tilde` 的 candidates | 506 / 4,536 |
| 至少一项不一致的 anchors | 111 / 672 |
| 至少一项不一致的 groups | 41 / 42 |
| 最大绝对分数差 | 1.0 |
| pair sign 不一致 | 473 / 13,402 |
| top-set 不一致 anchors | 55 / 672 |
| raw 正 oracle-gain anchors | 292 / 672 |
| filtered 正 oracle-gain anchors | 291 / 672 |
| oracle-gain 符号不一致 anchors | 9 / 672 |

脚本仅读取已冻结结果并使用固定容差 `1e-12`；不重跑 scorer、不改变候选、阈值、统计或
B4R1 登记裁决。

## 裁决边界

1. B4R1 的原始结论仍成立于其预注册对象：合法 source-fit panel 上的
   `F_raw^{1f}` 存在正 oracle headroom。
2. 该结论不能自动转写为 `F_tilde^{1f}`、带历史的 contextual row utility 或官方两阶段
   pseudo-closed-loop action value 的 headroom。
3. 旧的 `7/8 + 1/8` 分解只适用于 `F_tilde^{1f}` 与同一乘性门下的 two-frame comfort；不得把
   B4R1 的 `F_raw^{1f}` 直接代入。
4. R005 必须先绑定精确 evaluator 版本、过滤顺序和阶段聚合，再生成监督；网络训练继续锁定。

机器可读结果：`H7BS4B4R1_FILTERED_SCORE_SEMANTICS_AUDIT.json`。

