# Claim–Evidence Matrix

任何数字或比较在没有注册实验与不可变结果引用前都保持 `proposed`。

| ID | Intended claim | Required evidence | Falsifier | Status |
|---|---|---|---|---|
| C1 | source-session edge event 可同时覆盖声明时刻的全图差分 residual | 独立 session calibration/validation；simultaneous coverage、sample-size eligibility 与 sharpness | undercoverage、split reuse 或阈值空泛 | proposed |
| C2 | exact min-plus 实现与 robust set-min infimum 一致 | 小图枚举、LP/dual oracle、shift/zero-edge/inclusion cases | 任一方向错误或数值乐观 | Gate A synthetic pass |
| C3 | map/geometry/numerical ledger 给出有效 continuous correction | continuous teacher、node-label error、解析 geometry bound、protocol hash 与数值预算 | 任一通过 gate 的乐观反例 | Gate A PASS：legacy route falsified；R002E raw-cover/rotation-free geometry、R002F exact transform、R002G 164 caches × 4096 nodes 完整账本通过 |
| C4 | local graph 在相同合法 predictor/coverage 下比 absolute/global gauge 更紧 | Layer-1 uncertainty-only control，exact readout | 优势消失或仅来自 baseline handicap | proposed |
| C5 | 完整 edge-trained system 优于合法 absolute/global/CDF 最近邻 | Layer-2 matched-capacity/source comparison | 等价或更差 | proposed |
| C6 | 所有授权干预达到 \(\gamma_{task}\) relative improvement，并产生 meaningful non-inclusion utility | session \(R_\gamma\)、\(S_{better}\)、\(R_{cert|better}\)、\(R_{opportunity}\)、\(R_{selected}\) | relative failure、供给近零、证书近空或只靠 inclusion | proposed |
| C7 | candidate-free event 迁移到未见 candidate policy/frozen planner | 同一校准事件下 held-out policy/planner test | coverage/utility 随候选机制崩溃 | proposed |
| C8 | 运行代价可接受 | 冻结图规模上的 field 与 multi-source profiler | 不满足预注册资源门 | proposed |
| C9 | graph potential differences 是 appearance transfer 的相关机制 | paired same-scene \(\rho\)、session-tail residual 与 empirical target coverage | \(\rho\) 与证书退化无关或目标域严重失效 | partial mechanism evidence only：R004E-F development mean signal + R004H1 historically exposed selection mean signal；paired/OOD/coverage 未验证 |
| C10 | observation-conditioned bounded edge scale 可在冻结 potential 上被稳定学习 | proper-loss fit、cap diagnostics、pooled/uniform controls 与后续 common-denominator task endpoint | learned proper loss 不优于 uniform、边界塌缩或 learned 不优于 pooled | first instance falsified：R004H2 learned `2.566971` vs uniform `0.380000`; conditioning diagnosis required |

## Claim language policy

- `guarantees` 只用于完整陈述假设的定理；
- `shows` 只用于已注册、可追溯结果；
- `safe` 必须附带 nominal road constraint 范围；
- source coverage 不写成 arbitrary-OOD coverage；
- V0 规范冻结不写成系统已验证或论文原创性成立。
