# AORTA-v8 Grounding Notes

检索日期：2026-10-23。仅记录本轮直接核对过的原始公开页面；这些论文的结论用于限定创新边界，不作为本项目实验结果。

## DriveReferee

- 原文：[DriveReferee: Geometric Safety Verdicts Need Not Be Learned for Driving World-Action Models](https://arxiv.org/html/2609.22762v1)
- 版本：arXiv:2609.22762v1，2026-09-19。
- 本轮核对：方法直接预测可行驶区域与车辆占据 BEV，再执行零参数解析几何规则；部署时在同一预测地图上比较候选。§III-D 明确指出，相对 margin 排序对共同局部地图偏差较不敏感，但误差并未消失。
- 对 AORTA-v8 的约束：不能把“同图候选相对 margin”“解析几何规则”“默认保留原轨迹”单独作为创新。必须在无绝对锚点时的可校准相对证据、跨冻结 planner 接口和明确的不退化决策合同上形成区别。
- 抓取 SHA256：`be5a3dabdaceb2a33895d96de580ad63a90d6db9ea86f45f39ebf3f95b1f0dc7`。

## Conformalized Distance Fields

- 原文：[Conformalized Distance Fields](https://arxiv.org/html/2607.00776v1)
- 版本：arXiv:2607.00776v1，2026-07。
- 本轮核对：工作对残差距离场做 FPCA，在低维系数空间构造 conformal 区域，并以 sup-norm conformal quantile 单独覆盖投影残差；最终将场下界用于几何规划约束，并显式处理离散化分辨率。
- 对 AORTA-v8 的约束：不能把“低维联合误差＋conformal＋投影残差”作为新颖性。v8 应避开完整场集合，转而研究绝对锚点不可辨识时的商空间对比算子及其基线相对决策合同。
- 抓取 SHA256：`49605b211db476707dc24b1543cfc1a401c1e532ca3a6f0c46ba266aedcb2dfc`。

## SPIBB

- 原文：[Safe Policy Improvement with Baseline Bootstrapping](https://proceedings.mlr.press/v97/laroche19a.html)
- 发表：ICML 2019，PMLR 97。
- 本轮核对：该工作研究相对数据收集 baseline 的安全策略改进，在高不确定性处回退 baseline。
- 对 AORTA-v8 的约束：“相对 baseline 不退化”不是一般意义上的新思想。可能的新贡献必须来自视觉道路约束下的可辨识对象、轨迹算子、同时校准和冻结 planner 仲裁机制。
- 抓取 SHA256：`1e793c49768ceae29e5ae7d4b6513aa323b8bc44ef75f333b8ab6a68a4a0a566`。

## 尚未完成的查新边界

- 2026-10-23 追加使用 arXiv 官方 API 做术语发现：`conformal prediction AND graph` 的首批结果主要是图节点预测与图结构反馈；`conformal AND distance field` 再次定位到 *Conformalized Distance Fields*；`difference constraints AND uncertainty set`、`graph Lipschitz AND uncertainty` 的首批结果未发现直接等价标题。
- 上述查询只是发现性检索，不能用“没有搜到”证明新颖。v8.3 因而把 graph difference constraints、Lipschitz extension 与 min-plus shortest-path readout 全部列为标准工具。
- 本轮不是完整 novelty search。
- 尚需系统核对 trajectory-conditioned implicit fields、structured conformal prediction、set-valued ranking、graph uncertainty sets、runtime assurance 和 cross-planner frozen interfaces。
- 因此文档中的创新性只能写为“候选机制主张”，不能写成“首次”。
