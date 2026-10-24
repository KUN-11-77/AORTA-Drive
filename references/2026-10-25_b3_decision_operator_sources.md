# B3 后 decision-operator 文献来源账本

检索时间：2026-10-25（Asia/Shanghai）。来源均为 arXiv 官方摘要或官方 HTML；下面只记录可由
摘要/正文直接支持的内容，不将 preprint 写成 peer-reviewed 论文。原始抓取及 SHA 元数据保存在
`F:\AORTA-Drive-artifacts\literature\2026-10-25_b3_decision_operator`。

| 文献 | 版本/状态 | 官方来源 | raw SHA-256 | 本次使用 |
|---|---|---|---|---|
| Zeng et al., *End-to-end Interpretable Neural Motion Planner* | CVPR 2019 Oral；arXiv:2101.06679v1 | https://arxiv.org/html/2101.06679 | `39ceba511831b6bf6e1231698ef08594351370e48295b76ae8b149c77908c8af` | 时空 cost volume 沿轨迹求和、采样选择 |
| Casas et al., *MP3: A Unified Model to Map, Perceive, Predict and Plan* | arXiv:2101.06806v1 | https://arxiv.org/html/2101.06806 | `1905b2d2e33b5897b27a9441600ecd34d53cfb9d34ff2bfc790c5c515423106e` | online map/prediction 与 neural planner 闭环 |
| Li et al., *Generalized Trajectory Scoring for End-to-end Multimodal Planning* | 2025 challenge report/preprint；arXiv:2506.06664v1 | https://arxiv.org/html/2506.06664 | `72acfcd865ff80d4297bd94cdd0cade05a63b260f8b159274929ae57ff98f3e1` | dense vocabulary、dropout、sensor augmentation、refinement、NAVSIM-v2 |
| Zhong et al., *DA-WAM: Decision-Aligned Future Latents for Driving World Models* | 2026 preprint；arXiv:2608.19085v3 | https://arxiv.org/html/2608.19085 | `a57a7b9f57e76d558fde1a21f8f3110812581c7f996aab064562a1e338231ebc` | candidate-conditioned future、factor/utility head、hard negatives |
| Xu et al., *Test-Time Trajectory Optimization for Autonomous Driving* | 2026 preprint；arXiv:2606.07170v2 | https://arxiv.org/html/2606.07170 | `477cc11d0b58350c531a414cb6625742468e3f5e11f17716be15ef5ea38f02f3` | scorer-as-reward、off-proposal generalization、trust region、runtime |
| Mandi et al., *Decision-Focused Learning to Predict Action Costs for Planning* | 2024 preprint；arXiv:2408.06876v2 | https://arxiv.org/abs/2408.06876 | `36c697dd2516cff769420e10636c9b72bcfb7ed322067a0362e4a3dbe1a4860c` | decision-focused cost learning 与 solver cost |
| Dixit et al., *Risk-Sensitive Motion Planning using EVaR* | ECC 2021；arXiv:2011.11211v2 | https://arxiv.org/abs/2011.11211 | `0550435624738596aebc422a8641e7add0c2c20c62dfdbe68cce42f777d4287c` | EVaR/CVaR motion-planning baseline |
| Dixit et al., *Risk-Averse Receding Horizon Motion Planning using Coherent Risk Measures* | Artificial Intelligence 325 (2023)；arXiv:2204.09596v2 | https://arxiv.org/abs/2204.09596 | `4bdfa880dd49cab5066d807e421f83acb9079633f7bcae304ce60525331a2da4` | CVaR/EVaR/g-entropic risk 与 recursive feasibility |
| Hakobyan & Yang, *Distributionally Robust Risk Map for Learning-Based Motion Planning and Control* | arXiv:2105.00657v1 | https://arxiv.org/abs/2105.00657 | `fbea05778f3da4bae8216873c7944735b534e003de8c7c263864f6180cc27e10` | ambiguity-set worst-case CVaR risk map |

检索式还覆盖 `trajectory scoring + autonomous driving`、`decision-focused learning + planning`、
`occupancy measure + trajectory`、`CVaR + motion planning` 与 `risk field/cost map + trajectory
planning`。本轮是 B3 后定向约束审查，不是最终 systematic review；正式 novelty claim 前仍需
补充同行评审 venue、引用链、代码合同和 2025–2026 最新工作。
