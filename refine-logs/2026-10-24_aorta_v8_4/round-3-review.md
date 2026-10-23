# Round 3 Independent Method Review

> Same independent reviewer as Rounds 1–2. No files were modified and no literature
> search was performed.

<details>
<summary>Full raw review</summary>

**总体 Verdict：REVISE；R004E-G：READY，可以注册并执行。**

本轮已关闭上一轮主要理论与协议缺口。没有发现需要阻止 CPU synthetic gate 的机制
矛盾。后续真实 R004H 仍须等待该 gate 通过并完成注册；顶会方法原创性仍未建立，不能
把 gate readiness 等同于论文 READY。

| 维度 | 分数 | 判断 |
|---|---:|---|
| Problem Fidelity | 9 | 原始 candidate-independent relative-margin certification 锚点保持。 |
| Method Specificity | 9 | 训练、数据角色、完整 residual 算术和评价对象已足够具体。 |
| Contribution Quality | 6 | 贡献候选集中且可证伪；超出标准方法组合的增量仍待证明。 |
| Frontier Leverage | 9 | 冻结视觉表征与单个 scale head 合适，无需添加其他大型模块。 |
| Feasibility | 9 | scale-only 训练与无参数 pooled control 成本合理，可复用现有认证实现。 |
| Validation Focus | 9 | 同一 potential、共同 opportunity 分母与分层解释已基本闭合。 |
| Venue Readiness | 5 | 尚无真实机制收益、最强直接基线或独立查新证据。 |

\[
0.15(9)+0.25(9)+0.25(6)+0.15(9)+0.10(9)+0.05(9)+0.05(5)
=\boxed{8.05/10}.
\]

**本轮核心修复的核验**

1. **proxy-cal 降级正确解决历史依赖问题。** 42/8/16 现在只描述本轮拟合、选择和流程
   排练角色，历史曝光仍在账本；实际 proxy threshold 不再承载 \(16/17\) coverage。
2. **四项区间 residual 方向正确。** 两个 gradient intervals 给出
   \(r_e\in[g_u^- -g_d^+,g_u^+-g_d^-]\)，所定义 \(a_e^+\) 覆盖 \(|r_e|\)。
3. **invalid groups 对当前 maximum-rank proxy 已闭合。** 16 组不删除；任何无效 score
   为 \(+\infty\)，因 \(k=n=16\) 导致拒绝。
4. **pooled-scale 对照补齐关键归因。** learned 对 uniform 支持灵活尺度；同时优于
   pooled 才支持 edge allocation。
5. **共同 opportunity 分母消除循环定义。** \(Q_B\) 独立于被测方法，invalid/fallback
   不通过过滤提高 recall。

**CRITICAL：NONE，针对 R004E-G 的注册与执行。**

**IMPORTANT**

1. 注册中将 exchangeability 写成 \(Z_1,\ldots,Z_n,Z_{new}\) 在冻结外部训练/选择信息
   条件下联合可交换，且 score construction 不依赖 calibration sample。
2. 统一一般 official calibration invalid 规则。建议任一非法 calibration group 直接使
   整个 variant calibration 失败；这比仅放入一个 \(+\infty\) 后取 \(k<n\) 阶更明确。
3. R004H 再冻结 teacher opportunity、\(\delta_M,\gamma_{task}\)、non-inclusion、实际
   最小收益、零机会处理和不等长 session 权重。
4. Development 面板已参与方法形成，只能作为 qualification，不能重写成未见确认集。

**MINOR**

- pooled scale 可把确定性 represented output 当作 predictor 定义，只需校准/推理一致；
- 离线 group invalid 与未来在线即时 fallback 语义需分开；
- 不把有限 head 输出解释成真实条件误差或 epistemic uncertainty。

**Drift Warning：NONE。**

**Simplification Opportunities**

1. 保持 Stage P 冻结后只训练 Stage S；
2. R004E-G 只用 synthetic、注册 ID 元数据和 exact oracle；
3. CPU primitive、真实机制和论文原创性三层结论分别记录。

**Modernization Opportunities：NONE。**

最终建议：**现在可以注册并执行 R004E-G。** R004E-G 通过不等于 Gate B、正式 coverage
或算法创新成立。

</details>
