# AORTA 宏观价值审计：为什么研究内部受限修复，而不是更常见的替代路线？

状态：研究路线边界；不是实验结果。

## 1. 领域问题，而非局部工具问题

目标不是证明activation patching、Jacobian或低秩基“有趣”，而是回答：

> 冻结驾驶策略在外观变化下失效时，是否存在一种比重训视觉入口或任意adapter更小、更可解释、且不损害
> clean行为的修复接口？

R006V0只是检验这一宏观命题的第一道机制门。若内部介导效应弱、非scene-specific或不稳定，研究应停止
内部修复，而不是继续优化patching。

## 2. 主要替代路线与AORTA的必要性门

| 路线 | 解决方式 | 已知优势 | AORTA只有在什么条件下更有价值 |
|---|---|---|---|
| 输入增强/鲁棒训练 | 扩展训练分布、重训planner | 简单、部署成本低 | 冻结planner无法重训，且AORTA在未见外观与clean no-harm上有净收益 |
| Stable backbone / Constant Eye | 用DINO等稳定表示重建planner接口 | 已有强实证、故事直接 | AORTA保持原planner/head冻结，并在matched预算下优于faithful与frozen-tail版本 |
| BEV/SDF/world reconstruction | 先恢复完整场景再规划 | 语义清晰、可复用 | 完整重建不是必要条件，受限task coordinate在数据/时延/效用上更优 |
| 直接output residual/distillation | 直接修正轨迹或分数 | 最短路径、易训练 | 内部rank约束在OOD、clean no-harm和样本效率上显著优于direct-output |
| 普通adapter | 任意改写内部特征 | 容量强、工程成熟 | CPTR在同输入、参数、FLOPs、步数下可靠胜出，且效应来自资格化子空间 |
| scorer/reranker | 不改planner内部，只重排候选 | 适合多候选系统 | 当前LTF回归planner的内部失效确实可修且跨范式复现；否则应转向决策层方法 |

## 3. 第一性选择逻辑

1. 若稳定backbone已经足够，使用Constant Eye，不需要AORTA。
2. 若行为差异不存在共享、非平凡的低秩介导，停止CPTR。
3. 若DINO不含该目标的组外增量信息，停止视觉恢复。
4. 若direct-output或matched adapter同样有效，内部因果定位只保留为分析，不作为方法创新。
5. 只有“介导非平凡 + 视觉可恢复 + 受限修复优于替代 + 行为/时延闭合”同时成立，AORTA才是有价值的方法。

## 4. 不钻牛角尖的实验纪律

- 每个技术模块必须对应一个会改变路线选择的竞争解释；
- 不因R006V0失败而加入mixture basis、多层修复或更大rank；
- 不因R006V1失败而微调DINO、增加VLM/world model；
- 不因R006V2落后而削弱matched baseline或隐藏双编码器成本；
- 负结果优先更新问题认识，再决定转向输入层、输出层或决策层，而不是默认修补当前方法。

## 5. 一分钟宏观定位

Constant Eye说明稳定基础视觉能够提升外观OOD，但仍需为不同planner重建接口。AORTA追问更根本的问题：
冻结策略真正因外观变化丢失了什么？我们先在planner自身任务几何中定位并用组外有限干预验证这些内部
坐标，再检查DINO是否仍保留恢复它们的信息。只有答案都为是，才做低秩受限修复。这个路线的价值不在
patching本身，而在于给出一个可证伪的判据：何时应修复冻结planner，何时应改用更宏观的替代方案。
