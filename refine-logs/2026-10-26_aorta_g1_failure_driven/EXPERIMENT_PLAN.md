# G1 claim-driven experiment plan

| 顺序 | 实验 | 唯一问题 | 解锁条件 |
|---:|---|---|---|
| 1 | R005E3D | proper mean能否避免冻结训练域native-only塌缩 | 四项资格门全过 |
| 2 | E3D-OOG | 同一方法在job-level group-heldout是否仍正收益 | 组级CI与预注册最小效应通过 |
| 3 | Visual increment | correct是否优于mismatch和removed | relation与decision端点联合通过 |
| 4 | Differential Query | 差异归纳偏置是否优于matched generic scorer | regret、near-tie、样本效率、时延共同占优 |
| 5 | NavDream/behavior | 外观与planner变化下是否转化为净收益 | clean no-harm、跨planner、时延和行为门通过 |

R005E3D失败时停止自动架构搜索，按端点区分无正收益、capture不足、无有益选择或无非native选择；只有
新的可判别诊断才能授权下一变更。
