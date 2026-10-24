# AORTA relation/story 定向来源账本

检索时间：2026-10-25（Asia/Shanghai）  
状态：`TARGETED_STORY_AND_ARCHITECTURE_AUDIT__NOT_SYSTEMATIC_REVIEW`

> 后续更新：D-JEPA、World4Scorer、iDriveVLA、ManeuverFlow、PlanWAM、GeoWAM 与近期 conformal
> driving work 的正文审查进一步收窄了边界。最新结论见
> `2026-10-25_frontier_decision_scorer_novelty_audit.md`；若有冲突，以后者为准。

本轮只审查两件事：一是顶会方法论文如何把问题、机制与证据收成一条主线；二是
`candidate-conditioned scorer`、`selective replacement` 与 `frozen DINO interface` 是否已经被近邻
工作占据。摘要只能支持问题与高层方法判断；架构细节仅采用已经保存的官方 arXiv HTML 正文。

## 1. 直接来源

| 工作 | 状态 | 官方来源 | 对 AORTA 的约束 |
|---|---|---|---|
| Zeng et al., *End-to-end Interpretable Neural Motion Planner* | CVPR 2019 Oral | https://arxiv.org/html/2101.06679 | 从“分模块指标不等于规划质量”的矛盾推出时空 cost volume 与可微规划；沿轨迹累加 cost 不是新意 |
| Li et al., *Generalized Trajectory Scoring for End-to-end Multimodal Planning* | 2025 challenge report / preprint | https://arxiv.org/html/2506.06664 | 从静态词表与动态候选各自的泛化缺陷推出 generator、vocabulary generalization 与 refinement；普通 trajectory scorer、dropout、sensor augmentation 不是新意 |
| Zhong et al., *DA-WAM* | 2026 preprint | https://arxiv.org/html/2608.19085 | 已明确提出“预测表征未必区分不同决策后果”，并用 candidate-conditioned future latent、factor/utility scorer 与 hard negatives 解决；AORTA 不能再以 candidate-conditioned cross-attention 作为核心创新 |
| Xu et al., *TOAD* | 2026 preprint | https://arxiv.org/html/2606.07170 | 已把冻结 scorer 当 reward 做 test-time trajectory optimization；reranking/search、off-proposal scoring 与 trust region 不是 AORTA 独有 |
| Li et al., *Expand-and-Prune* | 2026 preprint | https://arxiv.org/html/2610.23262 | 已覆盖冻结 planner、候选空间适配、不确定性剪枝、跨架构和时延；plug-and-play assistance 不是核心新意 |
| Zheng et al., *Gen-Drive* | 2024 preprint | https://arxiv.org/html/2410.05582 | 已使用 pairwise preference 训练 driving reward model；“偏好学习 + scalar reward”不是新意 |
| *DriveReward* | 2026 preprint | https://arxiv.org/html/2606.08525 | 已覆盖多模态轨迹 reward/scoring；学习一个通用 trajectory evaluator 不是新意 |
| *The Constant Eye* | 2026 preprint, arXiv:2602.12563v1 | https://arxiv.org/abs/2602.12563 | 已定义 appearance-only shift、NavDream、冻结 DINOv3 通用接口与跨规划器零样本外观鲁棒；AORTA 不能以这些元素本身为贡献 |
| Zeng et al., *RiskWorld* | 2026 preprint, arXiv:2609.18442v1 | https://arxiv.org/abs/2609.18442 | 已覆盖共享 occupancy forecast、候选复用、planning anchor 与 selective trajectory replacement；AORTA 的拒判必须与 appearance-relation coverage 和 native-relative evidence 绑定 |

## 2. 新增原始抓取

原始文件目录：
`F:\AORTA-Drive-artifacts\literature\2026-10-25_relation_story_audit`

| 文件 | URL | SHA-256 |
|---|---|---|
| `constant_eye_abs.html` | https://arxiv.org/abs/2602.12563 | `b38557a8c6e8f924d3806156f8a2d987af3b60afa76b481248f2283ee92ff70d` |
| `riskworld_abs.html` | https://arxiv.org/abs/2609.18442 | `866de845ebc205818fb7fe6255c553ef494c05ef19fb2deb41d3eb9c18543924` |
| `arxiv_trajectory_scoring.xml` | arXiv API 定向检索 | `8f70871ebb6638df16776badd595819b25a86717cef3dfcdde3921b987068019` |
| `arxiv_constant_eye.xml` | arXiv API 题名核验 | `06f1fc26913d909f4951daaf35a3d5e94081f162dad781888eef2a5fdde11420` |
| `s2_signed_occupancy.json` | Semantic Scholar API 定向检索 | `74af1b4e8ecbf6db80133be0695c71f24338cf3e095b851d6b5813ef1dce2e92` |

两次额外 Semantic Scholar 查询返回 HTTP 429，已保留失败元数据；不能把本轮写成系统性查新。

## 3. 从这些论文抽出的叙事规律

强方法论文通常不是“模块清单”，而是同一技术矛盾的四次重述：

1. **问题层：**现有对象或接口在什么具体情况下失败；
2. **机制层：**为什么该失败由表示/优化接口本身造成，而不只是模型不够大；
3. **方法层：**最小的新对象或算子如何精确消除该失败；
4. **证据层：**一项匹配消融和一个行为终点排除最强竞争解释。

NMP 的主线是“独立子任务指标无法保证规划”到“task-coupled cost volume”；DA-WAM 的主线是
“共享/弱绑定的 future representation 无法区分 action consequence”到“candidate-conditioned
future”；GTRS 的主线是“静态/动态候选各自泛化不足”到“统一 coarse-to-fine scoring”。它们的方法
公式、网络图与主实验都在重复同一矛盾，而不是并列展示若干技术。

对 AORTA 而言，唯一可以继续争取的矛盾是：

> 外观稳定或甚至正确的场，不保证候选关系仍可区分；共同路径聚合会掩盖分叉，而分叉区域若不可
> 观测，任何视觉 scorer 都不能可靠判断。

因此下一架构若成立，必须直接处理“差异支持域”和“不可观测则拒判”。普通 candidate-conditioned
latent、普通 scalar scorer、普通 uncertainty head 或普通 plug-and-play replacement 都已不足以构成
独立故事。

