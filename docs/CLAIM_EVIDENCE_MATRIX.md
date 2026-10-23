# Claim–Evidence Matrix

任何数字或比较在没有注册实验与不可变结果引用前都保持 `proposed`。

| ID | Intended claim | Required evidence | Falsifier | Status |
|---|---|---|---|---|
| C1 | source-session edge event 可同时覆盖声明时刻的全图差分 residual | 独立 session calibration/validation；simultaneous coverage 与 sharpness | undercoverage 或阈值空泛 | proposed |
| C2 | exact min-plus 实现与 robust set-min infimum 一致 | 小图枚举、LP/dual oracle、shift/zero-edge/inclusion cases | 任一方向错误或数值乐观 | proposed |
| C3 | proof-carrying geometry gate 给出有效 continuous correction | 解析上界、property tests、协议 hash 与误差预算 | 存在通过 gate 的反例 | proposed |
| C4 | local graph 在相同合法 predictor/coverage 下比 absolute/global gauge 更紧 | Layer-1 uncertainty-only control，exact readout | 优势消失或仅来自 baseline handicap | proposed |
| C5 | 完整 edge-trained system 优于合法 absolute/global/CDF 最近邻 | Layer-2 matched-capacity/source comparison | 等价或更差 | proposed |
| C6 | 仲裁产生 meaningful non-inclusion 干预且不新增 nominal road failure | session H、intervention、连续几何差异、GT strict improvement | 干预为空、只靠 inclusion 或新增失败 | proposed |
| C7 | candidate-free event 迁移到未见 candidate policy/frozen planner | 同一校准事件下 held-out policy/planner test | coverage/utility 随候选机制崩溃 | proposed |
| C8 | 运行代价可接受 | 冻结图规模上的 field 与 multi-source profiler | 不满足预注册资源门 | proposed |

## Claim language policy

- `guarantees` 只用于完整陈述假设的定理；
- `shows` 只用于已注册、可追溯结果；
- `safe` 必须附带 nominal road constraint 范围；
- source coverage 不写成 arbitrary-OOD coverage；
- V0 规范冻结不写成系统已验证或论文原创性成立。
