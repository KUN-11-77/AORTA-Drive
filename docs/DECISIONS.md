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

## ADR-0018 — Use a planner-independent deterministic kinematic candidate library

- **Status:** accepted for Gate B protocol
- **Reason:** the primary candidate provider must not use a map, GT, paired clean observation, learned field, certificate, or planner score. R004C therefore generates at most 35 challengers from the current scalar speed using a frozen Cartesian product of constant longitudinal accelerations and path curvatures. Native remains a separate mandatory candidate. This keeps candidate supply independent of the field event and permits later transfer to a different frozen planner or candidate policy.
- **Dynamics:** each 0.5 s interval is integrated analytically in traveled arc length under constant acceleration and curvature; braking truncates at zero speed and never reverses. Controls are rejected above 20 m/s, 3 m/s² absolute longitudinal acceleration, or 4 m/s² lateral acceleration. Ordering, duplicate removal, horizon, provenance, and the full 35-control budget are deterministic.
- **Source-side repair before freeze:** the first train/development audit showed that (-2\,\mathrm{m/s^2}) left 12/1440 high-speed anchors without even one challenger whose rear-axle samples remained inside the frozen ROI. Adding the predeclared (-3\,\mathrm{m/s^2}) emergency-deceleration row removed those empty cases without consulting maps, labels, planner outputs, calibration residuals, or target data.
- **Boundary:** rear-axle samples inside the ROI are not a full-vehicle geometry certificate. Every used challenger still passes the frozen asymmetric swept-footprint checker, and Gate C must separately establish meaningful non-inclusion and GT usefulness relative to native.

## ADR-0019 — Freeze a masked quotient-field probe before reading calibration residuals

- **Status:** accepted after R004D structural qualification
- **Reason:** the predictor must respect the actual R004B evidence interface and the common-offset nullspace of the downstream task. R004D therefore uses shared per-slot projection, learned camera/history context, masked soft attention, an explicit all-missing token and evidence-count feature, followed by a small multiscale spatial decoder. The output spatial mean is removed exactly, so optimization occurs in a fixed representative of the quotient space rather than spending capacity on an unidentifiable common offset.
- **Tail alignment:** mean edge Huber alone is not aligned with the session maximum used by calibration. The frozen objective adds a centered log-sum-exp surrogate over edge residuals per anchor, while checkpoint selection uses the exact maximum over all 16 anchors and 8064 edges in each of the 24 development recording groups. The primary development criterion is the mean of the five worst group maxima, followed by the all-group mean and earliest epoch.
- **Leakage boundary:** candidates, maps beyond the frozen training labels, certificates, planner scores, calibration/test samples and target-domain samples cannot enter training or checkpoint selection. This is a legality and feasibility probe, not a performance claim; calibration residuals remain unread until its code, configuration and structural audit are committed.
- **Replay environment:** deterministic evaluation enables PyTorch deterministic algorithms and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before the first CUDA linear-algebra operation. A seed without this CUDA workspace contract is insufficient for the frozen replay claim.

## ADR-0020 — Instantiate labels on the exact frozen R004 anchors

- **Status:** accepted after R004E-A qualification
- **Finding:** the R002G formal manifest contains 164 valid cache-label artifacts, but none of their scene tokens occurs among the 1440 R004 train/development anchors. R002G qualifies the teacher implementation and those exact cache objects; it cannot supervise R004 by token substitution or nearest-scene reuse.
- **Decision:** run the official NAVSIM metric-cache processor only on the frozen R004 train/development tokens, then apply the already-qualified R002G label builder. The join key is exact `(log_name, scene_token)`, with one cache and one label per anchor. Existing R002G labels remain proof artifacts and are prohibited as R004 training labels unless their exact token is present.
- **Execution:** one anchor, then eight complete recording groups, then all 90 groups. Calibration/test/target remain unread. The independent unit for any later validation statement remains the recording group; 16 anchors are nested observations.
- **Replay correction:** an independent one-anchor regeneration showed that official compressed metric-cache pickle bytes are not canonical even when the resulting 4096-node label artifact is byte-identical. Raw pickle SHA remains provenance only. Replay is therefore judged by a canonical SHA-256 over the exact teacher inputs: rear-axle binary64 pose, selected layer names and every selected polygon outer/hole ring, invariant to component order, multipolygon order, cyclic ring start and ring direction. Label artifacts must still replay byte-for-byte.
- **Qualification evidence:** the ladder passed at one anchor, eight complete groups (128 anchors) and all 90 train/development groups (1440 anchors). The final independent audit reopened 1440 label artifacts, found zero duplicate IDs, hash mismatches or array failures, and confirmed that calibration/test/target data were not accessed. Metric and label manifest SHA-256 values are recorded in `experiments/results/R004EA.md`.
- **Boundary:** this repairs training-data alignment only. It does not train the predictor or establish residual sharpness, calibration coverage, certificate recall or driving utility.

## ADR-0021 — Cache dense frozen features and projection coordinates, not node tensors

- **Status:** accepted after R004E-B qualification
- **Problem:** directly persisting a `[4096,12,1280]` node-slot tensor for 1440 anchors would require roughly 169 GiB and would duplicate each dense camera feature across thousands of projected nodes.
- **Decision:** persist each content-addressed `[1280,18,32]` frozen DINO map as exact BF16 bit patterns, plus float64 patch coordinates and a Boolean support mask per anchor. Materialize node-slot evidence with the frozen bilinear operator at load time. PCA, lossy quantization, learned offline projection, padding and clamping are prohibited for the primary probe.
- **Replay contract:** DINO checkpoint and canonical Python-source-tree hashes are frozen. Independent one-anchor builds must have byte-identical feature/projection artifacts and identical logical manifests. NumPy materialization must agree with an independent torch `grid_sample` path within `1e-5`, and invalid slots must be exact zero after masking.
- **Qualification evidence:** the ladder passed at one anchor, eight complete groups and all 90 train/development groups. The full audit reopened 17,133 dense features and 2880 projection arrays, matched all 1440 frozen labels, and used 24.65 GiB. Exact hashes and resource figures are recorded in `experiments/results/R004EB2.md`.
- **Boundary:** this closes the training-data interface only. It neither trains the quotient-field predictor nor establishes calibration, certificate sharpness, driving utility or OOD robustness.

## ADR-0022 — Materialize frozen dense features at runtime before the predictor

- **Status:** accepted after R004E-C0 qualification
- **Decision:** decode the 12 persisted BF16 maps exactly to float32, apply one vectorized `grid_sample` with `align_corners=True`, and set every unsupported node-slot value to exact zero after sampling. The teacher target is the right/down incidence image of raw `label_m`; its common offset cancels exactly, while the numerical teacher ledger remains provenance for later continuous lifting rather than an optimization weight.
- **Qualification evidence:** the real-anchor torch path agreed with the independent NumPy oracle to `9.7751617431640625e-6` under the frozen `1e-5` gate, preserved the 8064-edge order and zero gauge, was exactly invariant to invalid storage, replayed identically, and produced finite backward gradients. The numerical sampler gate passed narrowly and must not be characterized as a wide margin.
- **Preserved failures:** two pre-model executions exposed status-literal and nested-artifact-path mismatches. Both logs remain in the evidence bundle and both repairs have regression tests; neither failed execution performed an optimizer update.
- **Boundary:** C0 qualifies one real input/backward path only. It does not show optimization success, held-out group generalization, calibration coverage, certificate nontriviality, utility or OOD robustness.

## ADR-0023 — Freeze the epoch-1 checkpoint but withhold calibration pending visual-necessity controls

- **Status:** accepted after R004E-C2 completion
- **Execution:** the sole registered seed completed 20 epochs over 66 train groups and evaluated all 24 development groups after every epoch. The frozen lexicographic group-tail rule selected epoch 1 with worst-5 mean `1.475400 m` and all-24 mean `1.394614 m`; the checkpoint and complete ledger are hash-frozen.
- **Negative evidence:** from epoch 1 to 20, train mean objective fell to `0.7527×`, while development worst-5 and all-24 group-tail statistics rose to `1.5402×` and `1.4281×`. Thus the current anchor-level mean plus centered log-sum-exp surrogate does not empirically track the intended cross-group maximum endpoint under continued optimization.
- **Decision:** do not read calibration merely because a checkpoint exists. First compare the frozen visual checkpoint, without any further fitting, against a zero-difference field and a train-label mean spatial prior on the already-used development endpoint. Calibration is permitted only if the visual checkpoint strictly improves both worst-5 and all-24 statistics over both controls.
- **Interpretation boundary:** this pattern does not yet identify whether the cause is representation, missing evidence, or objective/statistical-level mismatch. It is neither a Gate B pass nor proof that all quotient-field approaches fail.

## ADR-0024 — Reject calibration entry when the learned field loses to the zero-difference control

- **Status:** accepted after R004E-D failure
- **Control theorem:** the continuous signed-distance teacher is 1-Lipschitz, so on a 1 m edge the true difference magnitude is at most approximately 1 m after the separately bounded label error. The zero-difference predictor therefore provides a deterministic residual control near 1 m without visual input.
- **Evidence:** on the same 24 development groups, zero-field worst-5/all-24 statistics were `1.000000954/1.000000725 m`; the frozen visual checkpoint gave `1.475400233/1.394614086 m`. The visual checkpoint beat the train-label mean spatial prior but failed both preregistered strict comparisons against zero.
- **Decision:** do not open calibration and do not add a new module speculatively. First attribute the learned model's excess residual to predicted-gradient overshoot, edges incident to all-missing evidence, or errors on fully observed edges. Those cases imply different theory and architecture repairs.
- **Boundary:** zero field is a diagnostic minimax control, not a useful certificate: a roughly 1 m per-edge uncertainty can still accumulate into vacuous path bounds. R004E-D rejects the current learned predictor/objective, not the broader task-equivalence research question.

## ADR-0025 — Reject projection-only and binary-observability repairs

- **Status:** accepted after R004E-E mixed diagnosis
- **Evidence:** none of `3,096,576` development edges exceeded the registered predicted-gradient limit `1 + 1e-5 m`; group-global maxima were attributed to fully observed non-overshooting edges in 15/24 groups and missing-incident edges in 9/24. Neither reached the preregistered 20/24 single-mechanism threshold. The teacher Lipschitz ledger had zero violations.
- **Decision:** do not add graph-Lipschitz projection as a claimed repair: projection onto the already-satisfied registered constraint cannot remove the measured errors. Do not equate projective support with reliable geometry or claim that removing all-missing nodes solves the problem: the both-observed subset still has `1.453571/1.370929 m` worst-5/all-24 group maxima, worse than the zero-field control.
- **Next test:** before designing a new network, compare visual and zero-field group-mean edge error while retaining the exact group maximum, and include a frozen sign-flipped field only to detect a convention/inversion failure. This determines whether useful average visual information exists but the statistical target fails, or whether the representation itself is uninformative.
- **Boundary:** R004E-E is development-only localization, not causal proof. It does not authorize calibration, a learned confidence mask, a new architecture or Gate B.
