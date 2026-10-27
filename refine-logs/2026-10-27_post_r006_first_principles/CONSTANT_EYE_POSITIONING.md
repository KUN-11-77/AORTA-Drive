# Constant Eye 之后：AORTA 的研究边界

状态：`POSITIONING_NOTE__NOT_NOVELTY_PROOF`。

## Constant Eye 已经解决的内容

根据论文正文，Constant Eye 已完成：

1. 用 navdream 把 appearance shift 与几何/场景变化分离；
2. 冻结 DINOv3-H+ 作为跨 regression、diffusion、scoring planner 的统一视觉底座；
3. 通过 MLP + CNN adapter 将 DINO feature grid 压缩并聚合，再为不同范式训练各自的 planning head；
4. 用 PCA、单场景 token t-SNE 与 attention 可视化说明 DINO 特征和规划焦点更稳定；
5. 在 navdream、navtest、navhard 上报告多范式性能。

因此 AORTA 永久不把 DINOv3-H+、冻结 foundation backbone、轻量 adapter、跨 planner 接入或 NavDream 本身写成创新。

## 仍未解决、且可直接形成 AORTA 问题的缺口

1. **planner 并未完全冻结。** Constant Eye 冻结 DINO，但训练 paradigm-specific adapter 与 planning head；它没有解决如何为已有冻结 planner 做最小、受约束的视觉修复。
2. **接口是通用压缩器，不是任务机制。** 4-layer MLP + CNN 的作用是维度投影与空间聚合，adapter 深度通过 navtest 性能消融选择；论文没有定义“哪些 DINO 信息对当前 planner 决策有效”。
3. **稳定性分析主要是相关性。** PCA、t-SNE、attention 可视化不能证明稳定 token 是必要决策证据，也不能区分正确视觉、错配视觉和数据先验。
4. **没有干预最小性。** 通用 adapter 可修改所有下游表示，没有机制保证只改变当前 planner-sensitive 自由度，也没有干预能量、trust region 或 clean no-harm 的结构性约束。
5. **行为闭环仍有限。** 论文明确以 open-loop/pseudo-closed-loop benchmark 为主，未来 closed-loop interaction 仍是限制；没有给出“何时应拒绝修复”的证据边界。

## AORTA 的单一升级命题

在完全相同且冻结的 DINOv3-H+ 下，不再学习一个任意 DINO-to-planner adapter，而学习**输出侧控制**，并通过冻结 planner 当前样本的 VJP 形成任务有效、最小能量的内部修复：

\[
\hat q_i=J_i^\top W^{1/2}\hat\alpha_i(D_i,z_{a,i},H_i).
\]

Constant Eye 回答“稳定视觉底座是否有用”；AORTA 必须回答“稳定视觉证据应通过什么不可替代的任务接口进入冻结 planner”。

## 生死对照

在相同 DINOv3-H+、数据、planner、参数/FLOPs 与训练预算下：

- Constant Eye 式 adapter/head；
- matched residual adapter；
- direct-output residual；
- AORTA sample-conditioned VJP repair。

若 AORTA 不能在未见组上同时改善单位干预能量的行为恢复、clean no-harm 与最终 NavDream utility，核心创新即不成立。不能用额外模块或不匹配预算挽救。
