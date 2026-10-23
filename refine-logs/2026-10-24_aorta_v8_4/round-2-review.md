# Round 2 Independent Method Review

> Same independent reviewer as Round 1. No files were modified by the reviewer and no
> literature search was performed.

<details>
<summary>Full raw review</summary>

Verdict：**REVISE**。本轮明显改善了方法的可实现性和归因设计：原问题保持，scale-only
训练更清楚，同一 \(u\) 的公平对照与连续几何修正后的终点都已补入。仍有两项需要在
宣称理论合同闭合前修正：历史数据使用与内部 coverage 的关系，以及完整 residual 的
向外舍入定义。没有发现必须放弃该方向的致命机制矛盾。

| 维度 | 分数 | 判断 |
|---|---:|---|
| Problem Fidelity | 9 | 原始 candidate-independent relative-margin 问题完整保留。 |
| Method Specificity | 8 | 三路数据角色、两阶段训练、尺度参数化与推理路径已具体；残差算术及无效样本处理仍有歧义。 |
| Contribution Quality | 6 | 机制更集中，删除不成立的贡献表述；原创性仍是待验证假设。 |
| Frontier Leverage | 9 | 冻结视觉表征与单个 scale head 合适，没有不必要的新模块。 |
| Feasibility | 9 | 删除 scale 的 tail gradient 后，训练与调试成本显著下降。 |
| Validation Focus | 8 | 已对齐最终证书，主要归因成立；空间定位与场景整体尺度仍需区分。 |
| Venue Readiness | 5 | 理论资格方案更成熟，但尚无证据证明超出标准方法组合的算法贡献。 |

\[
\text{Weighted overall}
=0.15(9)+0.25(8)+0.25(6)+0.15(9)
+0.10(9)+0.05(8)+0.05(5)
=\boxed{7.75/10}.
\]

**对本轮六项重点的判断**

1. **Anchor：保持。** 冻结 planner、candidate-free 输入、道路相对 margin、
   native-preserving arbitration 和 source-only 边界均未改变。“最小充分机制”降为
   “最小候选机制”是正确收窄。

2. **42/8/16 与 \(n=16,\alpha=.1\)：rank 算术正确，但实际数据资格仍需说明。**

   \[
   k=\lceil17\times0.9\rceil=16,\qquad
   k/(n+1)=16/17\approx0.9412.
   \]

   对合法交换的 score，取 16 个 calibration scores 的最大值确实给出至少 \(16/17\)
   的边际覆盖。\(n=16\) 不能提供 \(\alpha=.05\) 的有限阈值，这一点也已写对。
   关键是：这项保证需要最终 score construction 对 calibration 数据没有不对称依赖，
   不能仅由重新划分和重置权重推得。

3. **bounded Laplace scale training：主要表述正确。** 固定 \(u\)，在理想的逐特征可
   自由选择尺度的函数空间中，

   \[
   \mathbb E[a/s+\log s\mid H]
   \]

   的最优尺度为 \(\mathbb E[a\mid H]\)，加范围约束后取 clipped optimum。删除 scale
   上的 tail 项后，上一轮的目标性膨胀反例已排除。它仍不是尾部分位数学习，也不保证
   高风险边可预测；修订已承认这一点。

4. **outward arithmetic：方向正确，但尚未完全闭合。** 除法、乘法、分位数和 graph
   readout 的要求合理；“两个 directed subtractions”必须明确作用于完整四项 residual，
   不能作用于已经普通舍入的 \(Du,D\widetilde d\)。

5. **同一 \(u\) 与 corrected-certificate endpoint：足以识别 scale mechanism 的总体
   增量。** 这能排除 potential 质量变化和仅改善 edge summary 的假收益。但仍不能单独
   证明收益来自“边级空间定位”，而非输入相关的整体尺度变化；终点定义也还需要固定
   分母与无效样本处理。

6. **是否存在致命漏洞：目前没有发现不可修复的理论矛盾。** 上述问题可以通过收窄
   内部保证、明确区间算术和冻结指标定义解决。当前不宜写“全部理论/数据合同已闭合”。

**CRITICAL**

1. **不要把 fresh fit 自动等同于恢复了历史独立性。**

   这些 16 个 internal-cal groups 来自旧模型已经训练过的 66 组；v8.4 的设计又受到旧
   模型实验与诊断的影响。重新训练可以消除旧权重直接携带这些样本的信息，却不能自动
   消除历史方法选择对这些样本的依赖。

   因此，“internal-cal 不参与任何拟合/超参/选择”需要明确是完整历史，还是仅本轮新
   训练。若仅指新训练，不足以直接授权实际这 16 组上的精确有限样本保证。

   最小修改是：保留 42/8/16 作为严格记录的新 probe 数据角色；将其阈值称为
   **internal development proxy**，不声称历史暴露数据上的精确 coverage；将真正的
   split-conformal guarantee 留给最终模型冻结后首次读取的官方 calibration。

2. **把 residual upper bound 写成完整的区间运算。**

   当前措辞允许一种不安全实现：先用普通浮点计算两个梯度，再对它们做双向上舍入
   减法。例如：

   \[
   u_j=1,\quad u_i=-2^{-53},\quad
   \widetilde d_j=1,\quad \widetilde d_i=0.
   \]

   精确 residual 为 \(2^{-53}\)，但普通 binary64 计算的两个梯度都可能是 \(1\)。此后
   再做上舍入减法无法恢复已丢失的误差。

   建议直接冻结：

   \[
   g_u^-=\operatorname{down}(u_j-u_i),\quad
   g_u^+=\operatorname{up}(u_j-u_i),
   \]

   对 \(g_d^\pm\) 同样处理，并定义

   \[
   a_e^+=\max\left\{
   \operatorname{up}(g_u^+-g_d^-),
   \operatorname{up}(g_d^+-g_u^-),0
   \right\}.
   \]

   或使用等价的、经过证明的四项有向计算。随后再执行上舍入除法和乘法。R004E-G 应
   包含上述 cancellation 反例。

**IMPORTANT**

3. **补充无效 calibration groups 的处理，禁止静默过滤。** 固定 16 组不删除；任何
   calibration group 无法计算合法 score，则令其 score 为 \(+\infty\)，或直接判定该
   calibration attempt 不可部署。测试与 development 中因尺度无效而 fallback 的案例，
   也应保留在固定评价分母中。

4. **把两个 co-primary endpoints 写成可执行定义。** opportunity 必须复用原
   `METHOD_SPEC` 中独立于被测证书的 frozen candidate、几何资格、meaningful
   non-inclusion 与 teacher improvement 门槛；同时冻结聚合、invalid/fallback 和数值
   容差。

5. **区分“尺度适应有效”和“空间定位有效”。** learned \(s_e(X)\) 对 uniform 的优势
   可能来自每个 observation 的整体幅度。若要支持空间机制，需要增加无需新网络的
   pooled-scale 对照；否则收窄主张。

6. **原创性状态保持未授权。** 当前是成熟的 falsification protocol，尚不是顶会 READY。

**MINOR**

- conditional mean absolute residual 的结论只适用于理想函数空间最优；
- sigmoid 对有限 logits 位于开区间，边界 optimum 只是受约束函数问题的最优/极限；
- 删除 group-tail surrogate 只指 Stage S，Stage P 仍沿用旧 potential objective；
- rank proof 应先覆盖 outward upper score，再推出真实 residual coverage。

**Drift Warning：NONE。**

**Simplification Opportunities**

1. 保留两阶段 scale-only 实现；
2. 历史暴露 internal-cal 降为 proxy，最终官方 calibration 承担正式保证；
3. 若不主张空间定位，可暂不增加 pooled-scale 对照。

**Modernization Opportunities：NONE。**

</details>
