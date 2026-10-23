# Architecture Decision Log

## ADR-0001 — Use a `src/` Python package layout

- **Status:** accepted
- **Reason:** 防止 repository-root accidental imports，并支持可测试的研究包。

## ADR-0002 — Retire affine panel factorization as the main method

- **Status:** superseded
- **Reason:** 该分解是代数恒等式；固定候选 panel、anchor set 与 scalar contrast 未建立不可替代的方法贡献。
- **Preservation:** 现有 reference code 保留为历史 baseline，不能代表当前 v8.3.1 方法已实现。

## ADR-0003 — Use a candidate-free graph-quotient field

- **Status:** accepted for V0 qualification
- **Reason:** 共享物理 field 允许一次 source-session event 同时覆盖任意域内 swept-set pairs，并显式消除共同 offset。
- **Reversal:** Gate B 证书空泛，或 exact global-gauge/CDF baseline 等价。

## ADR-0004 — Fixed physical edge scale in V0

- **Status:** accepted
- **Reason:** learned scale 不是理论闭合所必需，且会混入额外的效率与校准问题。

## ADR-0005 — Exact multi-source min-plus readout

- **Status:** accepted
- **Reason:** 它是 graph difference polytope 的标准精确读出，复杂度明确且保持证书方向。
- **Non-claim:** Dijkstra/min-plus 不是项目原创贡献。

## ADR-0006 — Native-preserving arbitration

- **Status:** accepted
- **Reason:** 只在 sound geometry gate 与 relative non-degradation bound 同时成立时干预。

## ADR-0007 — Separate V0 correctness from paper novelty

- **Status:** accepted
- **Reason:** 条件理论闭合不证明系统有效、实时、OOD-calibrated 或顶会原创。

## ADR-0008 — No license until the owner selects one

- **Status:** pending user decision
- **Reason:** 许可改变法律权限，不应推断。

## ADR-0009 — Freeze v8.3.1 as the Gate A/B specification

- **Status:** accepted
- **Reason:** v8.3.1 strengthens the primary theorem to every authorized relative intervention, adds map/geometry/numerical error accounting, and freezes visual-node and candidate-provider interfaces without adding trainable modules.
- **Boundary:** this is not a paper-novelty or empirical-success decision.

## ADR-0010 — Separate candidate supply, certification, and selection

- **Status:** accepted
- **Reason:** zero intervention can arise from no better candidate, a vacuous certificate, or the selection policy. Report \(S_{better}\)、\(R_{cert|better}\)、\(R_{opportunity}\) and \(R_{selected}\) on one predeclared session-time sampling unit.

## ADR-0011 — Asymmetric continuous geometry lifting

- **Status:** accepted for synthetic Gate A; NAVSIM ledger pending
- **Reason:** challenger 外包络 + fixed-grid \(h\)-cover 与 native internal nodes 足以推出连续 relative-margin lower bound，且避免不可生产化的 challenger internal-net 假设。
- **Boundary:** R002C 只验证解析构造与拒绝逻辑；continuous polygon teacher、coordinate transform 和 geometry arithmetic 的确定性误差预算仍阻塞 full Gate A。

## ADR-0012 — Use a raw-cover teacher and rotation-free proof path

- **Status:** accepted for R002E qualification; coordinate-transform ledger pending
- **Reason:** NAVSIM cache polygons materially overlap，canonical overlay 会引入额外数值证明义务。teacher 改为 union membership 加到全部 raw rings 的距离，在重叠内部 seam 处保守归零。正式 swept-footprint 证书使用 rear-axle disk outer cover 与 native inner disk，避免 scorer 四角点及三角函数舍入进入 soundness 链。
- **Boundary:** 32-ULP corner check 只检查接口约定，不是 certificate error budget；ego-grid 到 global raw polygons 的坐标变换仍须单独闭合。

## ADR-0013 — Certify coordinate transforms with rational trig intervals

- **Status:** accepted for R002F qualification
- **Reason:** 40 阶 rational Taylor polynomial 加 41 阶 Lagrange remainder 可在冻结的 \([-4,4]\) rad domain 内包住精确 sin/cos；其后的二维 affine interval 与 outward Euclidean radius 直接产生逐节点 \(\varepsilon_{transform}\)，无需假设平台 libm 正确舍入。
- **Boundary:** 该预算只覆盖 binary64 表示值到 paper transform 的数值误差，不覆盖物理定位、地图配准或 pose convention 错误；最终 ROI、resolution 与全部 node-label manifest 仍须冻结。

## ADR-0014 — Freeze the V0 graph at 64×64 one-meter cells

- **Status:** accepted for R002G and subsequent Gate B
- **Reason:** \([-32,32]^2\) m 是被审计 NAVSIM 模型族的共同 source-side BEV support；1 m 对应 0.25 m 输入 BEV 的 factor-4 decoder stride，并将 graph 固定为 4096 nodes/8064 edges。该选择在读取 target score 和 calibration residual 前冻结。
- **Tradeoff:** \(h=1/\sqrt2\) m 可能降低 certificate sharpness；若 Gate B 近乎全空，触发预注册 stop/sensitivity rule，不能事后用 target outcome 改写主配置。

## ADR-0015 — Define zero polygon error only against the frozen cache object

- **Status:** accepted
- **Reason:** raw cache binary64 rings 本身定义 V0 paper teacher；相对于这一计算对象，polygon semantic approximation 为零，transform/distance/serialization 误差仍逐节点记账。
- **Boundary:** 不主张相对于真实道路、另一 map release 或完整 HD-map API 的 polygon error 为零。

## ADR-0016 — Split NAVSIM source data by complete physical recording

- **Status:** accepted for Gate B protocol
- **Reason:** 官方 navtrain 的 1192 个导出 log fragments 来自 162 个较长录制；把 fragment 或 frame 当独立样本会让同一录制跨 split，造成伪重复和过于乐观的 session calibration。R004A 因此以去掉文件名末尾区间后的 recording group 为唯一独立单位，并固定 66/24/40/32 个 train/development/calibration/test groups。
- **Allocation:** 只用首次读取任何 predictor residual 前可见的 scene count、log count、month 与 vehicle ID 做精确组容量的元数据均衡；MILP 达到时限时保存的可行 incumbent 被哈希冻结，不宣称全局最优，也不得根据后续结果重新抽样。
- **Time set:** 每个 group 按时间顺序固定 16 个 midpoint-quantile anchors；session 风险与效用结论仅覆盖该有限 (T(Z))，640 个 calibration anchors 仍只构成 40 个独立 calibration sessions。
- **Boundary:** 该决定只关闭 data-independence 与有限时刻集，不证明 source exchangeability、coverage、predictor validity 或 Gate B 成功。

## ADR-0017 — Distortion-aware projective support with explicit missing evidence

- **Status:** accepted for Gate B protocol
- **Reason:** NAVSIM raw camera frames carry nonzero five-coefficient lens distortion. R004B therefore projects ideal ego-ground rays with the OpenCV Brown--Conrady forward model, then maps raw pixel centers through a no-crop 512×288 half-pixel resize into the 18×32 DINOv3-H+ patch grid. Silently dropping distortion changes the per-view support mask and moves supported nodes by a material fraction of a patch.
- **Calibrated domain:** the ideal normalized ray radius is capped at 1.0. Across the 52 train/development calibration fingerprints, the maximum inverse-projected image-boundary radius is 0.93342 and the minimum radial-only first turning radius is 1.14705. This rejects nonphysical polynomial fold-back outside the calibrated field of view.
- **Missing evidence:** a node-slot pair is supported only with positive depth, membership in the calibrated ray domain, raw-image membership, and membership in the patch-center convex hull. Unsupported slots may be stored as zero only together with an explicit mask; downstream pooling must be masked, all-missing nodes use a dedicated learned token, and hard best-view selection, padding, clamping, and extrapolation are prohibited.
- **Boundary:** the mask certifies projective token support, not physical visibility or occlusion. R004B freezes the interface only; it does not qualify a learned field predictor, establish calibration coverage, or pass Gate B.
