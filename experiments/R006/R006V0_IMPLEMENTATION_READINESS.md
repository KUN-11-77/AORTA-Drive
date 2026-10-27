# R006V0 实现资格清单

状态：`BASIS_CHAIN_MECHANICAL_PASS__STOP_GLOBAL_SHARED_INTERFACE__OUTER_RUNNER_NOT_AUTHORIZED`。

2026-10-27：六折basis discovery/selection完成，主运行与replay逐文件一致，聚合审计93/93 PASS。
C4层为6/6，但最小合格rank计数`{4:2,8:1,32:3}`未达到至少4/6共识，故按冻结协议停止。
outer-confirmation runner不再是待实现阻塞项，而是**未获实现授权**；R006V1/CPTR/NavDream继续HOLD。

2026-10-26 更新：synthetic dense-Jacobian oracle 与 2-anchor R006V0S primary/replay 已通过；55 项
机械审计为 PASS。旧 C1 hook 因位于多轮 fusion 之后而被反例否定，现已冻结为第一次视觉编码输出、
任何跨模态 fusion 之前的完整因果切口。详见 `R006V0S_RESULTS.md`。这不是低秩介导的正向证据。

同日 R006V0J 对真实 LTF 矩阵自由算子完成资格：C3/C4 primary/replay 与25项审计PASS；C1/C2在冻结
`1e-4`伴随门下失败并于科学运行前排除。详见 `R006V0J_RESULTS.md`与协议修正1。

随后 C3/C4 tail-only 执行路径完成资格：其输出与完整冻结模型一致，primary/replay逐字节一致，
28/28机械审计PASS。正式运行可缓存因果切口及固定下游上下文，再仅对资格化尾部计算JVP；该变化不
改变估计对象或科学阈值。

正式 discovery 的流式 Stiefel solver 已实现为固定顺序、按样本数加权的完整目标值与梯度精确归约；
19个非整除合成样本上与稠密参考目标一致并恢复同一真实投影子空间。primary trajectory二次型同时
冻结为 `W=I_(2T)/T`，即每时间步二维物理位移平方均值。无科学endpoint资源benchmark随后按冻结
规则选择batch=128，38/38审计PASS；详见协议修正6与`R006V0R_STREAMING_RESOURCE_RESULTS.md`。

同日一组×16 anchors 的真实 activation-cache smoke 完成，4种冻结条件、C3/C4 schema、repair mask、
sealed-role与no-science-metrics共14/14审计PASS。它只资格化缓存管线，不替代42组正式缓存或科学实验。
详见 `R006V0C_CACHE_SMOKE_RESULTS.md`。

42组×672 anchors正式缓存随后完成provenance修复重放：V2审计265/265 PASS，实际执行`navsim/`子树
clean且Git tree object固定；与第一次完整运行的42/42 shard hashes逐组一致，replay审计65/65 PASS。
详见`R006V0C_FORMAL_CACHE_V2_RESULTS.md`。该结果仍然只是机械资格。

每折basis discovery/inner selection运行器与独立审计器现已实现：运行器只读取28个discovery组和7个
selection组，审计器从逐样本task-drift重新计算coverage、验证basis正交性/哈希/角色隔离，并机械复现
唯一layer/rank选择；primary/replay必须同GPU逐字节一致。详见协议修正7。outer-confirmation仍封存。

本文件只管理实现来源与机械资格，不改变 R006V0 科学问题、数据、层、rank、控制或裁决阈值。

## 已冻结

- 理论对象：task-weighted channel basis 与逐位置 coefficient field；
- 数据角色：42 source-fit groups；其余角色全部封存；
- 6-fold nested group-held-out 选择程序；
- 每折7 outer-confirmation、28 discovery、7 selection及固定hash salts；
- C3/C4唯一选择顺序：最小合格rank→最差组coverage→组等权mean→固定层顺序；
- 全部干预强制registered site mask，A4 Haar与A5完整经验协方差等能量控制；
- 672个A2 wrong-scene donors outcome-free唯一映射与独立审计；
- primary trajectory metric固定为`W=I_(2T)/T`，流式目标按样本数精确归约；
- candidate layer、rank 字典、外观算子与科学 arms；
- 16 个固定 random/noise seeds；
- group-level bootstrap 统计和 GO/STOP 结构；
- canonical 协议：`experiments/R006/R006V0_CAUSAL_VISUAL_MEDIATION_PREREGISTRATION.md`；
- canonical config：`configs/experiment/r006v0_causal_visual_mediation.yaml`。

## 正式 smoke 前必须写入 manifest 的不可缺项

1. 单一 planner checkpoint 的绝对路径与 SHA-256；
2. planner 源码提交 SHA 与 dirty-state；
3. hook 名称、shape、token/BEV 顺序和空间坐标 schema 的 SHA-256；
4. 图像预处理、受控 appearance 实现和 source manifest SHA-256；
5. `coverage_epsilon`：在归一化 W 度量中冻结，不得由 science outcome 调整；
6. synthetic dense-Jacobian oracle 的 objective、gradient 和 projector tolerance；
7. bootstrap seed、CI 类型与有效重采样比例；
8. dose OLS 截距的具体语义；
9. GPU roster、软件环境、最大并发、预计 wall time、异常停止与恢复规则。

## 当前代码资产边界

已有外部 instrumentation 原型提供 `early visual / fused BEV / planner memory / trajectory query` hooks，以及 same-scene / wrong-scene full patch。它只能作为移植参考：

- 不直接在原 GTRS checkout 中修改；
- 必须移植到隔离 worktree；
- 不沿用其 NavDream manifest；
- R006V0S 仅使用 source-controlled appearance；
- full patch 仅是机械 smoke，不进入科学主张。

## 解除阻塞的唯一顺序

1. ~~移植 hook 与 tail replay，生成 schema；~~ `PASS`
2. ~~冻结 checkpoint/source/hook provenance；~~ `PASS`
3. ~~运行显式 Jacobian synthetic oracle；~~ `PASS`
4. ~~运行 1–2 anchors 的 identity/full-patch smoke；~~ `PASS`
5. ~~运行一组真实 activation-cache smoke 并审计 site semantics；~~ `PASS`
6. ~~用 smoke 的非科学耗时与真实 tensor shape 冻结正式预算；~~ `PASS: streaming batch=128`
7. ~~冻结每折basis运行器、独立审计器与config SHA；~~ `PASS: this frozen change`
8. ~~一次性执行六折basis discovery/selection primary+replay，期间不读取部分科学指标；~~
   `MECHANICAL PASS / SCIENTIFIC STOP`
9. 只有六折产生唯一稳定接口后，才实现并冻结A0–A7 outer-confirmation运行器；当前条件未满足，禁止实现。

当前nested fold、真实JVP basis runner及其独立审计器均已完成；完整缓存、资源预算和batch=128均已
资格化。六折结果否定了预注册fixed-global-rank接口，因此全部A0–A7 outer-confirmation运行器保持未实现，
状态为`SCIENTIFICALLY_STOPPED_BEFORE_OUTER_INTERVENTION`。后续只允许独立登记的inner failure
decomposition，不得把实现完整度当作继续科学链的理由。
