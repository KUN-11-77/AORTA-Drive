# Round 4 Review：真实接口语义与有限干预复核

**Overall:** 7.5/10  
**Verdict:** REVISE

## 最强拒稿理由

当前方法可能被视为 activation patching + Jacobian subspace + adapter。更具体地，R006V0J把候选层收缩到
C3 planner memory和C4 trajectory query，但旧架构仍把两者统一写成空间位置；若DINO证据绑定与真实token
语义不一致，方法从理论到实现并未闭环。另一个阻塞点是单点Jacobian并不自动解释有限非线性干预。

## 七维评分

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9 |
| Method Specificity | 7 |
| Contribution Quality | 7 |
| Frontier Leverage | 8 |
| Feasibility | 7 |
| Validation Focus | 9 |
| Venue Readiness | 6 |

## 必须修改

1. 把低秩规律从“观察”降为待检验假设；
2. 用路径积分和非线性余项连接局部Jacobian与有限patch；
3. 按真实C3/C4 token语义定义reader和repair mask；
4. 区分理论 \(B_r^*\) 与数值近似 \(\widehat B_r\)；
5. 区分math-SDPA科学图与官方快速部署图，并加入全确认集等价门和端到端时延。

## 明确拒绝的复杂化

不加入world model、VLM、diffusion/RL repair、mixture basis、多层同时修复、DINO微调、planner联合训练或
新risk head。当前最小充分方法仍是资格化基 + 一个site-conditioned coefficient reader。
