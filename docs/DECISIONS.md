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

## ADR-0026 — Preserve the visual field but replace the uniform tail interface

- **Status:** accepted after R004E-F diagnosis
- **Evidence:** the frozen visual field reduced mean edge MAE from `0.554261` to `0.494434 m` and had positive predicted/teacher cosine in every development recording group, but its exact group maximum was worse than zero in every group. The preregistered tail/objective-mismatch event held in 24/24 groups. Exact sign flipping worsened mean MAE to `0.638136 m` and worst-5 maximum to `1.847585 m`, ruling out a global convention inversion.
- **Decision:** retain node-potential prediction as a source of useful central information, but do not retrain the same uniform-scale objective or enlarge the decoder speculatively. Develop an observation-conditioned positive edge uncertainty shape and calibrate the session maximum of standardized residuals. Heterogeneous edge budgets must enter the same exact min-plus theorem, remain candidate-independent and pay locally for unreliable edges.
- **Required safeguards (superseded in part by ADR-0027):** freeze the scale parameterization, calibration score and same-potential controls before training. The proposed scale-tail surrogate was rejected by the v8.4 review because adding it moves the proper-loss stationary scale from `a` to `(1+lambda)a`; it must not enter Stage S. A learned scale is not accepted merely because it lowers normalized residual.
- **Boundary:** the evidence is development-only and does not show that uncertainty shape is learnable, calibrated, nontrivial or novel. R004E-G must close the mathematics and synthetic oracles before any new source training.

## ADR-0027 — Qualify heterogeneous envelope primitives but reserve all empirical claims

- **Status:** accepted after R004E-G PASS
- **Data qualification:** the former 66 train groups are deterministically assigned 42 fit, 8 selection and 16 proxy-cal roles. All carry an explicit historical-exposure label; proxy-cal cannot support a formal coverage claim. Formal split conformal remains conditional on calibration and new sessions being jointly exchangeable after the complete predictor and score are frozen.
- **Numerical decision:** certification recomputes each residual interval from the four represented node values. It encloses both `u_j-u_i` and `d_j-d_i`, then outward-bounds their absolute difference before division. This catches a registered cancellation case where naive rounded gradients yield zero for an exact represented-value residual of `2^-53`. Division, threshold-times-scale and graph operations remain outward-safe; any invalid calibration group fails the entire variant.
- **Training decision:** Stage P must be freshly trained on fit only and selected on selection only. Stage S freezes Stage P and trains one bounded right/down scale head using only Laplace proper loss. No group-tail term enters scale fitting. Learned scale, a no-parameter per-observation pooled scale, uniform scale and evidence-only scale use the same potential and separate thresholds.
- **Qualification evidence:** 19 focused and 131 full tests passed; 1000 randomized residual/score and 1000 budget cases were non-optimistic; scale gauge and heterogeneous graph oracles passed. Exact hashes are in `experiments/results/R004EG.md`.
- **Boundary:** R004E-G qualifies primitives and the protocol only. R004H needs a separate preregistration and one-read development rule. Official calibration remains sealed; Gate B, learned uncertainty, OOD robustness and algorithmic novelty remain unestablished.

## ADR-0028 — Qualify the scale interface before any heterogeneous training

- **Status:** accepted after R004H0 PASS
- **Reason:** the v8.4 repair needs to reuse the potential trunk without silently changing
  its represented field, while exposing exactly one bounded right/down scale per graph edge.
  R004H0 therefore requires strict old-checkpoint loading, exact potential replay, a
  candidate/planner-free forward signature, exactly 194 trainable scale parameters and
  gradient isolation before any optimizer step is authorized.
- **Evidence:** on one deterministic fit-role anchor the refactor replay error was `0.0 m`,
  all 194 scale-gradient elements were nonzero and finite, potential-gradient count was
  zero, all 8064 scales were initialized to 1 m within `5.97e-8 m`, and peak CUDA allocation
  was `0.929 GiB`. Focused tests passed 11/11 and the repository passed 142/142 tests.
- **Decision:** R004H1 may now be registered, but not inferred from the old epoch-1
  checkpoint. It must initialize a fresh Stage P model, optimize only on 42 fit groups and
  select only on 8 selection groups. H0 does not authorize Stage S, proxy-cal, development
  or official calibration access.
- **Boundary:** structural gradient flow is not evidence that heteroscedastic uncertainty is
  learnable or calibrated. Coverage, utility, Gate B, OOD robustness and novelty remain
  unestablished.

## ADR-0029 — Require fresh selection-side central signal before scale fitting

- **Status:** accepted after R004H1 PASS
- **Reason:** the old 66-group checkpoint had already used every future proxy role and was
  selected by a development tail that later failed the zero-field control. Stage P therefore
  restarts from the registered random seed, optimizes only 42 fit groups and chooses exactly
  one checkpoint on 8 selection groups. The primary key is macro group-mean edge MAE because
  R004E-F specifically established transferable central signal; the two-worst group maximum
  is secondary and cannot silently replace the primary question.
- **Evidence:** 20 complete epochs and 13,440 updates selected epoch 20. Its selection macro
  mean was `0.3800000122 m`, strictly below the same-label zero field `0.5506349756 m`
  (`30.99%` relative reduction). Mechanical replay reproduced the selector and strict-loaded
  both checkpoints. The two-worst group maximum remained `2.2441165447 m`.
- **Decision:** authorize registration of H2 scale-only proper-loss training. Stage P must be
  loaded from checkpoint SHA-256 `8b5110b312341f1516067c55d6bb5a9cdec9b230de3837f451f993d5dbc0906b`
  and frozen permanently.
- **Boundary:** all selection groups are historically exposed and only one seed was run.
  This supports a narrow mechanism-formation claim, not confirmation, tail control,
  uncertainty calibration, task utility, Gate B, OOD robustness or novelty. External
  cross-model integrity review was unavailable; the mechanical audit is not a substitute.

## ADR-0030 — Stop the first scale fit and diagnose conditioning before redesign

- **Status:** accepted after R004H2 STOP
- **Evidence:** the legal 10-epoch run updated only 194 scale parameters and selected epoch
  9, but its selection proper loss was `2.5669707581` versus uniform 1 m
  `0.3800000343`. The selected geometric-mean scale was `10.4165 m`; 44.1% of scales were
  within the registered upper-cap neighborhood. Pre-clip gradient norms reached
  `714,505.625` and stayed at `10^5` scale despite unit clipping.
- **Decision:** H3 proxy rehearsal is prohibited. Do not reinterpret cap-gate PASS as model
  success and do not tune learning rate post hoc. First register a read-only diagnostic that
  measures frozen trunk activation scale and compares bias-only, raw full-head and
  normalization-only initial gradients on fixed fit/selection anchors.
- **Rival explanations:** (A) the unnormalized Stage-P spatial representation makes the
  1x1 weight Jacobian ill-conditioned; (B) the linear bounded-sigmoid scale map is itself
  ill-conditioned; (C) the registered optimizer step is too large despite gradient clipping.
  A revision must target the supported explanation rather than add arbitrary complexity.
- **Boundary:** this falsifies the first architecture/optimization instance, not the
  heterogeneous-envelope theorem. No proxy-cal, development or official calibration data
  were opened.

## ADR-0031 — Freeze an analytic first-step diagnosis before any H2 repair

- **Status:** accepted before R004H2D execution
- **Reason:** global gradient clipping does not by itself bound AdamW's first effective
  coordinate update because the first-moment scale and square-root second-moment scale
  cancel a common gradient multiplier. A large frozen activation can therefore turn a
  nominal `3e-4` weight update into an arbitrarily large scale-logit change.
- **Decision:** inspect four deterministic anchors from one fit group and four from one
  selection group. Compare raw and uncentered-RMS-normalized feature Jacobians, verify the
  analytic gradient against autograd, and replay the first AdamW update without constructing
  an optimizer or mutating parameters. Require at least 6/8 cause-specific votes.
- **Precedence:** derivative amplification first, activation scale second, learning rate
  third, otherwise unresolved. Activation scale wins over learning rate only when the same
  registered step becomes stable under magnitude-only normalization.
- **Boundary:** this is a cause diagnosis, not a scale-fit result. It may authorize one
  targeted H2 revision but never H3, coverage, task utility, Gate B, OOD or novelty claims.

## ADR-0032 — Repair only the diagnosed activation gauge

- **Status:** accepted after R004H2D
- **Evidence:** all 8/8 fixed anchors voted for activation scale. Raw feature RMS was
  `1776`–`2195`; raw first-step logit changes were `82`–`133` with `75%`–`93%` cap entry.
  Uncentered global-RMS removal reduced changes to `0.041`–`0.065` with zero cap entry.
  Median full-gradient reduction was `2026.85×`. Mechanical replay passed.
- **Decision:** authorize registration of exactly one R004H2 revision that divides the
  detached scale-branch input by its per-observation uncentered global RMS. Keep the
  potential path, 194-parameter head, objective, optimizer, learning rate, schedule,
  fit/selection roles, seed, selector and stop gates unchanged.
- **Rejected changes:** no LayerNorm or centering, no learned affine normalization, no
  learning-rate tuning, no alternate scale parameterization, no extra head capacity and no
  proxy/development read.
- **Boundary:** H3 remains blocked until the revised H2 independently passes both its
  proper-loss and cap gates.

## ADR-0033 — Accept the gauge repair and advance only to proxy rehearsal

- **Status:** accepted after R004H2R1 PASS
- **Evidence:** with all H2 contracts unchanged except detached scale-input RMS removal,
  selected proper loss became `0.0015641139` versus uniform `0.3800000343`; both cap
  fractions were zero and maximum pre-clip gradient norm was `4.6456`. Mechanical replay
  verified epoch 9, both gates and both checkpoint hashes.
- **Decision:** the scale interface is qualified for R004H3 registration. Use the selected
  checkpoint SHA-256 `aacb6bf43ce909860d6e459c9115eb2ca471c84f2426ea560545e0e471cd2b48`.
- **Boundary:** H3 uses historically exposed proxy groups only to rehearse the frozen
  calibration machinery and compare learned, pooled, uniform and evidence-only controls.
  It cannot produce a coverage claim or authorize official calibration by itself.

## ADR-0034 — Calibrate every proxy control independently at the session level

- **Status:** accepted before R004H3 execution
- **Unit:** one recording group is one calibration observation; its score is the maximum
  over all 16 registered times and 8064 canonical edges. The 256 anchors are not treated as
  independent samples.
- **Threshold:** with `n=16` and `alpha=0.10`, every control uses rank 16, its maximum
  complete-session score. Learned, pooled, uniform and evidence-only thresholds are
  separate; threshold sharing is prohibited.
- **Invalidity:** one invalid group makes the complete affected control variant infinite.
  No group may be filtered, replaced or winsorized. H4 registration requires all four
  variants finite and replay-exact.
- **Boundary:** proxy roles were historically exposed. H3 qualifies execution plumbing
  only; it is not calibration coverage, utility, Gate B or official-calibration evidence.

## ADR-0035 — Preserve the task endpoint despite proxy tail inflation

- **Status:** accepted after R004H3 PASS
- **Evidence:** all four proxy variants were finite and mechanically replayed. Learned
  threshold was `14.8486` and its median calibrated edge budget `5.7849 m`, versus uniform
  `2.6315 m`; learned nevertheless had a lower minimum budget `1.0733 m` because it remains
  spatially heterogeneous.
- **Decision:** do not redesign the scale objective, clip the threshold or select pooled
  scale from proxy diagnostics. Register H4 exactly as predeclared and let candidate-level
  common-denominator certificate recall decide whether localized low budgets offset the
  inflated joint tail.
- **Boundary:** H3 groups were historically exposed, and budget summaries are not task
  utility. Development remains sealed until every H4 denominator and decision rule is frozen.

## ADR-0036 — Prove continuous non-inclusion before opening development

- **Status:** accepted before R004H4P Stage 1
- **Problem:** challenger outer nodes and native inner points are sufficient for the relative
  margin theorem but cannot prove that both continuous footprint differences have positive
  area. A finite node-set difference would be a category error.
- **Decision:** use cached PDM-Closed as the explicitly named native for this mechanism
  probe, sample it at the same `0:0.5:4.0 s` horizon, and carry the global-to-ego transform
  radius. For each direction, count only complete 1 m grid cells proved inside one
  trajectory's rear-axle inner disks and disjoint from every disk in the other trajectory's
  continuous outer cover. Require at least `1.0 m^2` in both directions and a conservative
  continuous teacher improvement lower bound of at least `0.5 m`.
- **Intent contract:** progress ratio `[0.8,1.2]`, terminal cross-track `<=3.5 m`, terminal
  heading difference `<=0.35 rad`, terminal speed difference `<=4.0 m/s`, with native
  terminal displacement `>=2.0 m`. This filter cannot read the teacher or prediction.
- **Boundary:** the registered single fit anchor is an interface/runtime preflight only. It
  cannot estimate opportunity frequency, register H4, open development, establish coverage
  or support method claims.

## ADR-0037 — Reject an unexercised geometry branch as H4P PASS

- **Status:** accepted after the first H4P execution integrity check
- **Evidence:** all 35 frozen challengers failed the intent/progress filter, so no challenger
  entered continuous geometry. The initial implementation nevertheless emitted `PASS` and
  marked the non-inclusion interface closed.
- **Decision:** retain that raw artifact as invalid, change no scientific threshold, and
  replay the same fit anchor after making PASS require at least one intent-eligible and one
  continuous-geometry-valid challenger. Record all four intent metrics for every candidate.
- **Boundary:** the zero-support observation may motivate a separately registered expanded
  fit-only supply diagnosis; it cannot justify relaxing `[0.8,1.2]`, adding acceleration
  values, or reading development.

## ADR-0038 — Diagnose candidate supply before touching geometry or thresholds

- **Status:** accepted before R004H4P2
- **Decision:** read all 16 anchors from the first eight fit rows in frozen manifest order.
  Keep the original 35-control provider and every intent threshold unchanged. Report the
  anchor and group fractions with at least one fully eligible candidate, all independent
  gate pass fractions, sequential first failures and nearest progress gap.
- **Adequacy gate:** at least `0.90` of anchors and at least `0.75` within every group must
  have a fully eligible candidate. This tests whether the provider can represent the frozen
  native intent before road geometry or learned uncertainty is allowed to influence the
  analysis.
- **Boundary:** PASS permits fit-only geometry preflight; failure permits registration of
  one physically structured provider revision. Neither outcome opens development or
  authorizes post-hoc threshold relaxation.

## ADR-0039 — Require candidate-level diagnosis totals to close exactly

- **Status:** accepted after the first R004H4P2 accounting audit
- **Evidence:** the anchor-level supply result was internally consistent, but the
  first-failure histogram added zero for eligible candidates and therefore summed to 4157
  instead of all 4256 candidate evaluations.
- **Decision:** invalidate that accounting artifact, change the increment to count every
  candidate's terminal category exactly once, add a payload-free audit requiring histogram
  sum = raw evaluations and eligible category = ledger eligible count, then replay the same
  128 fit anchors.
- **Boundary:** this repair cannot change provider controls, thresholds, selected anchors or
  scientific decision; it only makes the diagnostic evidence exhaustive.

## ADR-0040 — Audit terminal-speed semantics before revising candidate supply

- **Status:** accepted before R004H4P2S
- **Evidence:** all 128 H4P2 cached PDM-native terminal dynamic-state speeds were reported as
  zero despite nonzero trajectory displacement; one inspected native travels `54.188953 m`
  over four seconds and `5.970444 m` in its final half-second while its exported dynamic
  velocity remains exactly zero. NAVSIM's PDM generator initializes the exported state array
  to zero and writes its SE2 fields, while the longitudinal speed is held separately in the
  IDM state array.
- **Decision:** do not interpret H4P2's speed gate as a valid physical comparison yet. Audit
  the same 128 fit anchors by comparing the cached terminal dynamic field with rear-axle
  displacement over `[3.5,4.0] s`. Reject the cached field if any moving anchor has more than
  `1.0 m/s` disagreement. Hash the two NAVSIM source files that define this conversion.
- **Boundary:** the audit reads no labels, images, predictions, geometry, development,
  calibration, test or target data. Rejection permits only a same-scope replay where both
  native and challenger use the same position-derived final-segment average-speed quantity;
  it does not permit a threshold or provider change.

## ADR-0041 — Compare native and challenger through the same positional speed observable

- **Status:** accepted after R004H4P2S and before R004H4P2R1
- **Evidence:** R004H4P2S found zero cached dynamic terminal speed on all 128 anchors, while
  119 had at least `0.5 m/s` final-segment positional average speed and 113 materially
  disagreed with the cached field. The raw ledger and payload-free mechanical audit agree.
- **Decision:** replay the exact R004H4P2 scope and decision rule. For both native and
  challenger, define the speed observable as
  `||p(4.0)-p(3.5)|| / 0.5`. Keep the `4.0 m/s` difference budget, all other intent gates,
  current-speed input and the 35-control provider unchanged.
- **Boundary:** this is a measurement-interface correction, not threshold tuning. It makes
  no instantaneous-speed claim and cannot open geometry or development unless the original
  supply adequacy rule passes.

## ADR-0042 — Treat the remaining supply gap as route-conditioning failure

- **Status:** accepted after R004H4P2R1
- **Evidence:** the corrected replay supplies 82/128 anchors (`0.640625`) with a minimum
  group rate of `0.5`, failing the unchanged `0.90`/`0.75` gate. Among 46 unsupplied anchors,
  23 nearest-progress candidates already lie inside the progress interval but fail
  cross-track or heading; another 23 fail progress or the deliberately undefined low-native-
  displacement case.
- **Decision:** a denser acceleration grid is not an adequate repair. The one authorized
  provider revision must condition on the frozen native route and introduce smooth,
  endpoint-invariant mid-horizon perturbations. It must preserve the final `[3.5,4.0] s`
  segment, derive heading from the perturbed tangent, and pass physical-envelope and
  adversarial implementation tests before any fit-data supply run.
- **Boundary:** no provider revision is registered yet. The four native trajectories below
  `2 m` remain outside the intent frame; thresholds are not relaxed. Geometry and development
  remain closed.

## ADR-0043 — Qualify a fixed-normal C2 bump before any fit-data provider run

- **Status:** accepted before R004H4P3S
- **Design:** cross six analytic native families with six signed amplitudes. These are
  deterministic paired construction cases, not statistical replicates; no randomization,
  significance or prevalence estimate is applicable. Freeze excessive-amplitude and
  insufficient-displacement negative controls before execution.
- **Implementation contract:** reconstruct a C2 piecewise-quintic native position curve;
  perturb it along the normal to native terminal displacement using
  `64 u^3(1-u)^3` until `3.5 s`; derive active-horizon heading from the perturbed tangent;
  copy `[3.5,4.0] s` native samples bit-for-bit. Screen complete active polynomial segments
  with recursive Bernstein bounds for speed and acceleration envelopes.
- **Decision:** PASS requires both amplitude signs in every analytic case, exact endpoint
  intent invariants for every accepted candidate, nontrivial middle displacement, bit-exact
  deterministic replay, both negative controls, and at least 209 repository tests passing.
- **Boundary:** the Bernstein implementation is a conservative floating arithmetic screen,
  not a formal real-arithmetic proof. This stage reads no fit data, geometry, teacher,
  prediction or development. PASS can authorize only R004H4P3 registration.

## ADR-0044 — Evaluate the single provider revision on the unchanged fit denominator

- **Status:** accepted before R004H4P3
- **Precondition:** R004H4P3S passed all registered construction gates with 209 tests and
  mechanical audit PASS; independent cross-model integrity review remains unavailable and
  is carried as a provisional qualifier.
- **Design:** replay exactly the first eight fit groups × sixteen anchors used by
  R004H4P2R1. Replace only the candidate provider. Keep all intent thresholds and the
  overall `0.90` / each-group `0.75` adequacy rule. Treat recording group, not candidate,
  as the independent unit. Report both-sign supply and all continuous-envelope rejections.
- **Decision:** PASS requires the unchanged adequacy rates and zero endpoint-invariant
  failures. Failure ends this provider-revision branch; it does not authorize another
  provider, more amplitudes or relaxed thresholds.
- **Boundary:** no geometry, teacher, learned prediction, image, development, calibration,
  test or target payload may be read. PASS permits only a separately registered fit-only
  continuous-geometry preflight.

## ADR-0045 — Audit endpoint identity structurally, not by literal derived equality

- **Status:** accepted after the first R004H4P3 execution audit
- **Evidence:** an accepted candidate and native had bit-identical origin and final two pose
  rows, zero cross-track/heading/speed difference and an eligible decision, but the computed
  progress ratio was `0.99999999999999989`. The first implementation incorrectly required
  this derived dot/norm result to equal the literal float `1.0`, producing false endpoint-
  invariant failures.
- **Repair:** retain bit-exact origin/final-segment equality as the structural invariant.
  Audit the four derived eligibility residuals with the frozen bound `64 * eps_binary64 =
  1.4210854715202004e-14`. Add direct unit tests for one-ulp progress roundoff and a changed
  final segment. Record every maximum residual in the raw ledger.
- **Boundary:** no data, provider amplitude, dynamic envelope, intent threshold, anchor or
  adequacy gate changes. The first raw artifact and its expected mechanical-audit failure
  remain evidence of the implementation defect. The corrected replay may still fail supply.

## ADR-0046 — Insert the official NAVSIM execution operator before task geometry

- **Status:** accepted after corrected R004H4P3 and official-source inspection
- **Evidence:** corrected R004H4P3 mechanically passes but its self-constructed polynomial
  screen accepts only 154/768 desired candidates. Official `pdm_score` does not score that
  polynomial: `transform_trajectory` creates desired `EgoState`s with zero dynamic fields,
  `get_trajectory_as_array` resamples native and prediction at 0.1 s, and `PDMSimulator`
  applies `BatchLQRTracker` plus `BatchKinematicBicycleModel` before scoring.
- **Theory correction:** distinguish desired reference \(\hat\tau\) from realized trajectory
  \(\tau=\mathcal S_{x_0}(\hat\tau)\), where \(\mathcal S\) is the frozen official execution
  operator. Intent eligibility, swept footprint, teacher opportunity and certification must
  use paired realized native/challenger trajectories. The field-coverage theorem remains
  candidate-independent once these realized sets are supplied.
- **Decision:** do not revise the provider again or relax its gates. Register a read-only
  execution-interface audit on fit data to close sampling, coordinate, simulator and state-
  array semantics. The polynomial screen result is retained as a failed formulation, not
  evidence about official realized feasibility.
- **Boundary:** no geometry teacher, prediction, image or development read is authorized.

## ADR-0047 — Make implementation validity the zero gate for realized-trajectory evidence

- **Status:** accepted before R004H4P4E execution
- **Problem:** the first draft of the interface runner passed global native coordinates to a
  provider whose contract requires the current-ego local graph. This was detected before
  registration and execution. Any result from that draft would have been invalid regardless
  of its metric values.
- **Repair:** obtain the provider input only through the already audited
  `sample_cached_native_trajectory` local sampler. Pin the new bridge and provider source
  hashes, require 214 repository tests plus static type checks, then exercise the bridge
  against the frozen official NAVSIM modules. On every one of 128 fit anchors, require the
  expected `7 x 41` batch, finite states, one bit-identical realized initial state, bit-exact
  repeated simulation, and desired-pose coordinate roundtrip within `1e-8 m` and `1e-10 rad`.
- **Decision:** only these interface gates decide R004H4P4E. Realized intent supply,
  desired-to-realized error and endpoint-identity survival are recorded as secondary
  diagnostics and cannot turn an interface failure into PASS or establish method utility.
- **Boundary:** PASS authorizes registration of a separate realized-trajectory intent and
  geometry preflight only. No geometry, teacher, prediction, development, calibration, test
  or target-domain payload is read here; no provider or threshold change is authorized.

## ADR-0048 — Parse registered experiment configuration before payload access

- **Status:** accepted after the first R004H4P4E launch stopped before payload access
- **Evidence:** the registered YAML placed `decision_use` at mapping indentation beneath a
  block sequence. `yaml.safe_load` rejected the file before anchors or cache payloads were
  read. The official one-anchor smoke test had exercised the implementation directly and
  therefore did not expose this orchestration defect.
- **Repair:** represent secondary diagnostics as a mapping with explicit `metrics` and
  `decision_use` fields, and add a repository test that parses the exact registered file and
  checks its key gate values. Preserve the failed launch log; rerun under a new implementation
  revision. No threshold, data, provider or decision endpoint changes.
- **Boundary:** this is an implementation-integrity repair. The failed launch carries no
  scientific result and cannot support or reject the theory.

## ADR-0049 — Retire desired endpoint identity as a realized-trajectory invariant

- **Status:** accepted after R004H4P4E
- **Evidence:** all 128 official execution-interface gates passed with maximum coordinate
  roundtrip errors `4.6635e-10 m` and `1.9429e-15 rad`; the payload-free mechanical audit
  passed. Nevertheless, bit-exact desired final-segment identity survived the stateful
  LQR+bicycle execution for all candidates on `0/128` anchors. Maximum desired-to-realized
  differences reached `2.6965 m` for native terminal position and `4.3483 m` over challenger
  trajectories.
- **Decision:** provider endpoint copying remains a desired-reference construction property,
  not a physical or mathematical invariant of the realized candidate. All H4 intent,
  non-inclusion, swept footprints, map-error budgets and opportunity labels must be evaluated
  from paired official realized trajectories. The diagnostic realized supply `122/128` and
  minimum group `14/16` may motivate a preregistered preflight but cannot be promoted to a
  passed supply endpoint after the fact.
- **Boundary:** only a fit-only realized-trajectory preflight may be registered next. No
  provider/threshold revision and no development, calibration, test or target access is
  authorized. External cross-model integrity review remains unavailable.

## ADR-0050 — Make the payload-free result audit fail closed on non-finite fields

- **Status:** accepted during R004H4P4E evidence sealing
- **Problem:** the first mechanical checker compared roundtrip fields only with upper
  thresholds. In IEEE arithmetic, `NaN > threshold` is false, so a corrupted result could
  evade that isolated check even though the actual raw result contained only finite values.
- **Repair:** require exact implementation revision and config hash, completed status,
  finite nonnegative interface/diagnostic values, valid candidate-count ranges, agreement
  between count and supply booleans, feasible both-sign counts and positive finite elapsed
  time. Re-audit the unchanged raw result and preserve both audit versions by hash.
- **Boundary:** this does not rerun or alter the experiment and cannot change scientific
  endpoints. It strengthens only evidence validation; external independent review remains
  unavailable.

The first hardened-audit replay also failed because the manually pinned config digest had
one extra hexadecimal `d`. That failed `audit_v2` is retained. The only permitted correction
was to match the digest to both the raw result field and an independent hash of the frozen
config; the unchanged raw result was then audited as `audit_v3`.

## ADR-0051 — Qualify realized geometry with one sentinel per recording-group block

- **Status:** accepted before R004H4P5I
- **Design:** use anchor index zero from each of the first eight frozen fit recording groups.
  Recording group is the experimental unit; these deterministic sentinels are an interface
  exercise and make no prevalence, significance or performance claim.
- **Implementation gate:** require exact metric-cache/label-cache hash identity, frozen
  source hashes, official Pacifica dimensions, rear-axle analytic corners agreeing with
  NAVSIM preprocessing within 32 global-coordinate ULPs, deterministic realized execution,
  and a complete candidate/failure ledger.
- **Geometry gate:** use all 41 official realized states, an arithmetic-safe disk sweep with
  certified pose error, inner-cell non-inclusion witnesses and the exact-cache polygon
  teacher error ledger. At least one sentinel must reach an actual teacher-bound computation;
  otherwise diagnose the geometry architecture before any 128-anchor expansion.
- **Boundary:** PASS permits only registration of the expanded fit-only preflight. It cannot
  estimate supply/opportunity prevalence, change the provider, thresholds, grid or ROI, or
  open development. The exact polygon teacher is relative to the frozen cache object and is
  not asserted equivalent to Shapely boundary semantics or real-world map truth.

## ADR-0052 — Diagnose the disk-certificate resolution barrier before scale-up

- **Status:** accepted after R004H4P5I, before any 128-anchor geometry expansion
- **Evidence:** all eight interface sentinels passed and six reached geometry plus teacher
  computation, but every one of the 36 geometry-valid candidate ledgers returned zero in
  both certified set-difference directions. All teacher improvement lower bounds were
  negative. Two additional sentinels were correctly rejected when the native disk sweep
  left the frozen ROI.
- **Mathematical concern:** the Pacifica rear-axle outer radius is
  `R=4.208735350434855 m`, its yaw-invariant inner radius is `r=1.127 m`, and the 1 m grid
  cell radius is `h=0.7071067811865476 m`. For paired samples with certified pose error
  `epsilon`, the disk construction cannot produce an inner-cell witness whenever the
  matched rear-axle separation is at most `R-r+2h+2epsilon`, approximately
  `4.49594891280795 m` here. This follows by the triangle inequality against the
  corresponding outer segment disk; it is a property of the certificate, not the data.
- **Decision:** do not interpret zero witnesses as zero physical non-inclusion and do not
  spend the 128-anchor budget until a registered fit-only diagnosis checks the analytic
  barrier against every realized sentinel candidate. If the barrier explains all zero
  ledgers, retire the rotation-free disk non-inclusion architecture and qualify a tighter
  direction-aware rectangular cover. The official provider and intent thresholds remain
  frozen.
- **Boundary:** this is a post-result mechanism diagnosis. It cannot support prevalence,
  performance, coverage, Gate B or novelty claims and cannot open development,
  calibration, test or target-domain data.

### R004H4P5D outcome

The registered diagnosis passed its implementation and mechanical-audit gates. All `48/48`
realized candidates were below the sufficient barrier. The `36/36` candidates with valid
predecessor geometry and bidirectional zero lower bounds were therefore fully explained by
the certificate-resolution theorem. The maximum paired separation upper bound was
`1.4476203847423528 m`; the minimum no-witness threshold lower bound was
`4.49594891280795 m`. The disk non-inclusion construction is retired for this provider.
This does not assert that a tighter certificate will find a positive opportunity.

## ADR-0053 — Require a numerically certified orientation-aware rectangle interface

- **Status:** accepted for one terminal qualification after R004H4P5D; data not yet read
- **Outer set:** for each adjacent realized pose pair, cover the continuous body sweep by
  the body rectangle at the midpoint pose dilated by half endpoint translation, certified
  pose error and the rational upper bound `R |Delta yaw| / 2` on the exact chord term
  `2 R sin(|Delta yaw|/4)`. Stored-midpoint translation/yaw rounding is also added. Linear
  unwrapped yaw is part of this interpolation contract; it is not described as arbitrary yaw.
- **Inner set:** accept a grid cell only when interval-certified transforms of all four cell
  corners lie inside a pose-local body rectangle shrunk by pose error. Convexity then places
  the whole cell inside the physical footprint.
- **Disjointness:** accept a witness only when an outward-safe lower bound on distance from
  the complete cell to every opposite segment cover is positive. Libm-only trigonometry or
  nearest-rounded threshold comparisons cannot carry the proof claim.
- **Resolution separation:** the frozen 1 m teacher graph and continuous-area witness grid
  are distinct mathematical objects. Synthetic qualification showed that 1 m cells can be
  vacuous for a 1.5 m lateral shift purely from grid phase, while a separately frozen
  0.25 m geometry grid certifies both directions at the unchanged `1.0 m^2` area threshold.
  The finer grid reads no labels and must be fixed before any sentinel execution.
- **Registration gate:** adversarial synthetic tests must include identical sweeps, lateral
  shifts around the resolvable boundary, high yaw change, last-sample witnesses, ROI exits,
  non-finite inputs and monotonic loss of witnesses as pose error grows. Only after these
  pass may the same eight fit sentinels be read.
- **Terminal design:** R004H4P5R freezes the same eight sentinels and all existing candidates,
  compares a dyadically refined polygon diagnostic, the retired disk control and the oriented
  certificate at `1.0/0.5/0.25/0.125 m`. Four fixed grid phases are diagnostic; only the
  canonical phase enters the primary decision. Polygon output is not called exact ground truth
  and is not used as an upper-bound soundness oracle: a union of sampled rectangles does not
  upper-bound the continuous set difference when both sweeps are refined.
- **Stop rules:** if polygon differences persist but the finest grid remains vacuous, stop grid
  refinement and evaluate a conservative polygon certificate. If geometry becomes nontrivial
  while all teacher bounds remain below `0.5 m`, stop geometry work and review the provider or
  task margin. No 128-anchor expansion occurs before both soundness and nontriviality pass.
- **Boundary:** no provider, intent threshold, teacher graph, area threshold, ROI or data-role
  change is authorized. Development remains sealed.

The first terminal launch failed closed before payload access because the legacy
`geometry.py` hash no longer matched R004H4P5I. The exact revision diff contained only the
additive `DiskNonInclusionResolutionBarrier` diagnostic introduced by R004H4P5D; none of the
disk cover, official execution, teacher or candidate semantics changed. R004H4P5R therefore
pins the new complete file hash explicitly while retaining exact predecessor hashes for every
other local and official source. This is a provenance repair, not permission to accept an
arbitrary source mismatch.

The second legal launch was terminated at the registered 30-minute wall-time cap before
one sentinel completed. It therefore produced no scientific geometry result: polygon area,
formal non-inclusion and teacher improvement are all unobserved, not zero. This is a
computability failure of the current exact-rational per-cell implementation. In accordance
with the terminal scope, geometry work is now frozen: no finer grid, threshold adjustment,
candidate reselection, 128-anchor expansion or open-ended certificate refactor is authorized
in this stage. The visual mechanism decision takes priority.

## ADR-0054 — Start fit-only visual mechanism validation in parallel with the terminal geometry gate

- **Status:** accepted and registered before R004H5 data execution
- **Reason:** R004H1 shows fit/selection mean edge signal but does not show that the frozen field
  is stable under appearance changes or that any observed stability retains road-task content.
  Completing every geometry detail first would leave the central visual hypothesis untested.
- **Design:** freeze the R004H1 checkpoint, DINO, projection, teacher and all 42 fit groups.
  Apply three deterministic RGB photometric operators to the exact same scene/camera/history
  images and rerun the frozen feature pipeline. Measure field change only in graph-edge space,
  task error against the unchanged teacher, field energy/correlation, and predefined zero,
  leave-one-group-out mean-field, pooled-token and uniform-token controls.
- **Statistical unit:** recording group. Sixteen anchors and all appearance variants are paired
  repeated measurements within each group; treating 672 anchors as independent is prohibited.
- **Interpretation:** this is a training-role mechanism diagnostic. It cannot establish held-out
  source generalization, real weather, target OOD, conformal coverage, driving benefit or safety.
  The transforms are not named rain, snow or dusk. Feature noise is not an acceptable substitute
  if raw-image replay fails.
- **Decision:** low appearance distance supports the mechanism only when every registered
  transform also beats zero-field task error, retains positive teacher-edge correlation and
  avoids constant-field collapse. A failure redirects work to the visual objective/architecture,
  not to further geometry refinement.

The first full R004H5 launch stopped at anchor 340 on its first clean DINO bit-replay mismatch.
The preceding 339 anchors and the one-anchor integration smoke were exact. Diagnosis identified
one image from the original cache builder's final five-image batch: replaying it in a batch of
eight changed the BF16 output materially, while replaying the exact original batch cardinality
of five restored bit equality for all five images. The repair freezes, for every image hash, the
original cache-build batch cardinality and pads incomplete condition batches with duplicate
inputs whose outputs are discarded. No appearance transform, model, checkpoint, data role,
metric or decision threshold changes. The failed partial fields cannot enter any summary.

The repaired run completed all 672 anchors with zero clean replay mismatches. Mechanical audit
reopened and recomputed all 672 field artifacts with zero failures. Every registered non-clean
condition passed the fit mechanism gate: macro task MAE stayed near `0.324 m` versus zero-field
`0.550 m`, appearance edge MAE was `0.0087--0.0102 m`, and teacher-edge cosine stayed near
`0.771`. A frozen one-shot follow-up found learned task MAE below the leave-one-group-out teacher
prior, pooled probe and uniform probe in all 42 recording groups under every condition. These are
fit mechanism results only. Independent visual generalization is GO; decision benefit is HOLD;
the current open-ended geometry certificate iteration remains STOP.

## ADR-0055 — Fail H7-B closed on shadow-panel support coverage

- **Status:** accepted after the frozen R004H7-B execution and mechanical recomputation.
- **Observed gate:** all 24 groups and 384 anchors remained in the denominator, but only 265
  anchors had a valid native plus at least one valid challenger. Only 11/24 groups reached the
  preregistered minimum of 12/16 valid anchors, so the result is
  `INSUFFICIENT_PANEL_COVERAGE`; association and flip-AUROC endpoints are intentionally absent.
- **Mechanism evidence:** 110 anchors had an invalid native, nine had no valid challenger, and
  770 candidate failures were caused by an oriented footprint leaving the frozen 64x64, 1 m
  ROI. The provider emitted all six registered offsets without its own rejection on every
  anchor, and 352/384 anchors had all six challengers pass intent eligibility. The dominant
  defect is therefore the support contract of the discrete query panel, not evidence that the
  decision quotient is useful or useless.
- **Decision:** H7-C remains prohibited. Do not delete groups, lower the 12/16 gate, select short
  trajectories, clip rectangles, or enlarge the ROI after reading outcome metrics. Register one
  label- and outcome-blind support-adequacy diagnosis that measures the complete support needed
  by native and challengers and makes the estimand change from ROI expansion versus horizon
  truncation explicit. Any H7-B2 requires a new frozen common support contract and separate
  authorization.
- **Boundary:** the mechanical audit passed 384/384 artifacts and reproduced the coverage
  verdict, but the required external reviewer backend was unavailable. No decision relevance,
  driving, safety, NavDream/target, final-test or novelty claim follows.

## ADR-0056 — Treat task-support containment as a decision-quotient domain condition

- **Status:** accepted after the post-hoc, outcome-blind R004H7-BS0 attribution.
- **Observed mechanism:** every one of the 110 native ROI failures and 770 candidate ROI
  failures first crossed the forward `x_max` boundary. Native first crossings occurred only at
  indices 21--40, with median 30.5; initial translation and yaw residuals were exactly zero.
  Within 23 paired groups, native-invalid anchors had 24.300 m greater mean path length than
  valid anchors, while yaw variation and curvature proxies were not larger. This is consistent
  with a full-horizon forward-support mismatch and a short-reach complete-case bias, not with a
  demonstrated coordinate-origin defect or difficult-turn concentration.
- **Theory:** a field observed only on `Omega` cannot identify a set-min over `F(tau)` when
  `F(tau)` leaves `Omega` unless an extension class and its error are specified. Clipping to
  `F(tau) intersect Omega` changes the estimand and has unbounded discrepancy without such an
  assumption. Fail-closed support checking is therefore part of the mathematical interface.
- **Decision:** preserve the original H7-B result. Do not pad, extrapolate, clip, remove groups,
  or tune ROI/horizon on decision outcomes. A fixed prefix is a different near-horizon claim;
  a larger ROI is an architectural retraining decision. Both require separate qualification.
- **Boundary:** H7BS-0 is post-hoc descriptive evidence. It cannot establish visual evidence
  quality, decision relevance, driving benefit, safety or novelty. Prefix selection, H7-B2 and
  H7-C remain on hold pending review of the estimand and architecture trade-off.

## ADR-0057 — Qualify a 2.5-second prefix as a queryable diagnostic domain, not a sufficient decision horizon

- **Status:** accepted after outcome-blind fit selection, one-shot development qualification and
  two mechanical artifact replays.
- **Selection:** the frozen prefix dictionary was evaluated on 42 fit groups without fields,
  teacher margins or decision outcomes. Prefixes through 2.5 seconds passed 42/42 groups at the
  registered 12/16 anchor threshold; 3.0 seconds passed only 37/42. The maximum-feasible rule
  therefore selected 2.5 seconds.
- **Qualification:** the selected prefix produced 360/384 valid anchors and 24/24 qualifying
  development groups. All 384 panel hashes and prefix statuses were independently reproduced.
- **Implementation event:** the original fit execution wrote all 672 artifacts but failed strict
  JSON serialization on an auxiliary NaN. Recovery regenerated no scientific panel: all 672
  official executions and intent decisions had to replay bit exactly before the result was
  materialized and audited.
- **Decision:** retain 2.5 seconds only as a mechanically qualified near-horizon diagnostic domain.
  H7-B2 remains on hold until an outcome-blind H7BS2 audit measures late candidate divergence,
  exact prefix node-set equivalence and full-horizon ROI requirements. H7-C remains prohibited.
- **Boundary:** this changes the estimand from the complete 4-second query to a near-horizon
  query. It establishes support containment only, not that 2.5 seconds preserves the information
  needed to rank 4-second candidates. In particular, equal prefix swept sets imply equal set-min
  scores for every possible field even when candidates diverge later. It does not establish visual
  correctness, decision relevance, driving benefit, safety, target generalization or novelty.
  External review remains unavailable.

## ADR-0058 — Retire the 2.5-second final route and qualify full-horizon forward support

- **Status:** accepted after H7BS2 fit-only diagnosis and exact mechanical replay.
- **Evidence:** among 3864 eligible native/challenger pairs, 128 pairs in 36/42 groups have
  identical 2.5-second lattice footprint sets but distinct 4-second sets. Their prefix set-min
  scores are equal for every possible field. At the registered 0.5m threshold, 151 pairs in
  40/42 groups first diverge after the prefix; 34.21% of complete distinguishing nodes are absent
  from the prefix on a group-equal basis.
- **Decision:** stop treating 2.5 seconds as a candidate final method. It may remain a diagnostic
  endpoint only. H7-B2 and H7-C remain on hold.
- **ROI correction:** the preregistered panel-level rule selected shifted 64×64, but that domain
  retains only 3574/3864 eligible challengers. Because the candidate-independent operator requires
  every compared query set to lie in its domain, that rule is insufficient for the final system.
  The first frozen fit option with 3864/3864 support is forward 96×64 (`x=[-32,64]`).
- **Next gate:** preregister one outcome-blind development qualification for exactly forward
  96×64 with all-eligible-candidate preservation. Passing does not authorize network training;
  visual observability and synchronized P50/P95 latency/memory gates remain mandatory.
- **Boundary:** H7BS2 is discrete-time fit geometry. It does not validate field accuracy, teacher
  relevance, decisions, latency, driving safety, target generalization or novelty.

## ADR-0059 — Freeze forward 96×64 as the qualified discrete query domain

- **Status:** accepted after H7BS3R1 one-shot development execution and mechanical replay.
- **Evidence:** all 384 native queries and all 2206 intent-eligible challenger queries on 24
  historically exposed development groups lie inside `x=[-32,64]`, `y=[-32,32]` at 1m for the
  complete four-second horizon. Zero anchors have a support failure; 24/24 groups satisfy the
  12/16 panel gate. Nine anchors lack an eligible challenger and remain candidate-supply cases.
- **Decision:** forward 96×64 becomes the only qualified discrete query-domain candidate for the
  next architecture preflight. Do not train it yet. First preregister distance-stratified visual
  observability/teacher coverage and synchronized end-to-end P50/P95 latency, memory and throughput.
- **Boundary:** domain containment is necessary but not sufficient for task sufficiency. It does
  not establish visual information at long range, decoder accuracy, 1m resolution adequacy,
  decision relevance, driving benefit, safety, final-test/target generalization or novelty.

## ADR-0060 — Fail the all-candidate direct-evidence precondition without blaming forward range

- **Status:** accepted after H7BS4A fit execution and exact mechanical replay.
- **Evidence:** 2/4536 query candidates have zero directly supported query nodes, so the frozen
  strict gate fails even though all 42 groups meet 12/16. Both are native queries with 10 or 12
  nodes and no node in the added `x>=32m` region. In contrast, all extension nodes queried by
  1104 candidates have direct projective support.
- **Decision:** preserve the STOP. Do not delete the two cases, relax the zero-failure rule or
  call mean support a pass. Hold teacher generation and network training. Permit one outcome- and
  teacher-free post-hoc attribution of common versus candidate-distinguishing query regions.
- **Theory implication:** a task-sufficient visual interface should be defined on information
  needed to contrast candidates, not assumed to require direct evidence at every candidate's
  invariant footprint. This is a hypothesis to test, not yet an algorithmic contribution.
- **Boundary:** projective support is not occlusion-aware physical visibility or task correctness;
  no NavDream, latency, decision, driving, safety or novelty claim follows.

## ADR-0061 — Reject binary difference-only support; model correlated partial observability

- **Status:** accepted after preregistered post-hoc H7BS4A-D0 and exact mechanical replay.
- **Evidence:** both H7BS4A blind natives have no eligible challenger, so they diagnose candidate
  supply rather than an actual pairwise arbitration. Separately, 8/3678 geometrically distinct
  native–challenger pairs have zero directly supported symmetric-difference nodes. Their discrete
  differences contain only 1–2 nodes while their common regions contain 5–25 supported nodes.
- **Decision:** keep the parent STOP and separate no-challenger supply accounting from decision
  support. Do not adopt a difference-only binary visibility gate. The next admissible theory target
  is a partially observed quotient-field uncertainty set with shared-variable robust comparison and
  abstention; training remains HOLD until teacher-criticality and oracle coverage are diagnosed.
- **Reasoning:** deleting unobserved nodes hallucinates sufficiency, while independent candidate
  intervals discard the correlation induced by shared field variables. A joint uncertainty set can
  preserve this dependence and force abstention when the decision sign is not identifiable.
- **Boundary:** D0 did not read teacher, fields or outcomes. It neither validates the new target nor
  supports prediction, decision, driving, safety, latency, NavDream, novelty or publication claims.

## ADR-0062 — Govern complexity by falsified simpler alternatives, not module count

- **Status:** historical post-D0 policy; its Hodge/core-chain content is superseded by B3, B4 and
  ADR-0067, while the general complexity-governance principle remains accepted.
- **Decision:** retain complex deterministic structure when it is required by a counterexample,
  theorem assumption, auditable error direction or shared-compute requirement. Limit trainable
  novelty to the quotient decoder and correlated uncertainty mechanism unless a later registered
  failure proves another component necessary.
- **Historical theory candidates:** discrete-exterior/Hodge consistency, additive-gauge quotient,
  support-conditioned graph uncertainty, exact shared-variable robust readout and selective
  Simplex-style fallback. These are not current adopted contributions; B3 retired the scalar
  road-margin readout and ADR-0067 removed native-star Hodge exactness from the novelty claim.
- **Conditional only:** sheaf fusion, continuous neural fields, PEA-style adaptation, learned depth,
  diffusion/RL candidate generation and target-shift weighting. Each needs its own failure evidence
  and capacity/latency-matched deletion test.
- **Reasoning:** parsimony means no unsupported mechanisms, not artificially few mechanisms. A
  system can be mathematically layered and operationally complex while retaining one dominant
  contribution: identifiable decision arbitration from partially observed appearance-robust task
  relations.
- **Boundary:** the literature mapping and candidate architecture do not validate novelty or any
  theorem assumptions on NAVSIM/NavDream; B0/B1 remain required before training.
## ADR-0063：teacher生成器资格与机制实验token必须分离（2026-10-25）

- **Status:** accepted
- **Evidence:** R004H7BS4B0A在164个R002G资格cache上完成两次forward96构建并机械PASS；
  同时角色审计确认这些token不等于H7BS4A-D0的672个fit anchors。
- **Decision:** B0A只资格化生成器实现。teacher-critical observability必须先运行B0A2，用
  H7BS4A冻结的`(log_name, scene_token)`精确连接R004E cache；禁止近邻、跨token或旧标签插值。
- **Reason:** 实现可重复性不能替代实验对象对齐；否则会用错误场景的道路几何解释视觉盲区。
- **Boundary:** B0A/B0A2都不证明可观测性、决策充分性、模型收益或OOD泛化。

## ADR-0064 — Retire scalar road-margin minima as the general reranking object

- **Status:** accepted after H7BS4B1–B3 and exact mechanical replay.
- **Evidence:** graph-level correlated uncertainty added no sign-identifiable pairs in B1；B2
  attributed 2,033 exact-zero contrasts to common-minimum domination；B3 temporal first-fork
  recovered only 170/2,033 pairs with a group-bootstrap interval of 6.34–10.29%. Spatial-exclusive
  minima recovered all 1,568 pairs on which both exclusive branches existed, but applied to only
  77.13% and therefore failed the frozen 95% applicability gate.
- **Decision:** stop global, fixed-prefix and temporal-fork scalar road-margin minima as a general
  AORTA reranker. Keep spatial-exclusive clearance only as a selective diagnostic. Do not train a
  replacement network until a new decision estimand passes source-only label legality and oracle
  ceiling gates.
- **Boundary:** this rejects one readout family on one frozen source-fit panel. It does not reject
  visual task representations generally and provides no learned, held-out, driving, safety,
  NavDream or novelty evidence.

## ADR-0065 — Make candidate utility explicitly native-conditioned

- **Status:** accepted before H7BS4B4 scientific execution, after official scorer source audit.
- **Evidence:** the frozen NAVSIM scorer normalizes progress by the maximum progress in the current
  score panel. Adding a higher-progress candidate can therefore change another candidate's score;
  an absolute panel-independent candidate utility is not the evaluator's actual estimand.
- **Decision:** define each source label by independently evaluating one `[native, candidate]` pair
  with the official execution, candidate-conditioned reactive traffic and one-frame PDM scorer.
  Write the future model as `s(X, native, candidate)` and construct pair preferences only by
  subtracting node scores under the same native context. Batch-scoring all seven candidates to
  manufacture labels is prohibited.
- **Consequence:** antisymmetry, cycle consistency and common-gauge invariance remain exact within
  one native context, while cross-planner transfer becomes conditional generalization to a new
  native trajectory rather than unconditional absolute-score transfer.
- **Boundary:** the B4 label excludes two-frame extended comfort and is not full official PDMS.
  It does not yet establish label signal, learned prediction, architecture necessity, planner
  transfer, OOD robustness, driving benefit or safety.

## ADR-0066 — Treat candidate membership as part of the preference estimand

- **Status:** accepted after R004H7BS4B4 completed with
  `INVALID_PREFERENCE_LABEL_INTERFACE`.
- **Evidence:** primary and replay were mechanically exact, but 168 challengers across 48 anchors
  were marked ineligible by the frozen intent-match interface and still entered oracle aggregation.
  All 48 flagged anchors reconstructed their realized poses bit exactly; the apparent reconstruction
  failure came from conjoining pose equality with `all(challenger_eligible)`.
- **Decision:** define the vertex set before labels: native is always present and only candidates
  admitted by the frozen membership rule may define graph vertices, pair edges, Laplacian losses,
  oracle gains and regret statements. Pose reconstruction and candidate eligibility are independent
  audit fields. Variable-size panels and zero-challenger anchors require explicit denominators and
  fail-closed handling.
- **Consequence:** B4 positive diagnostics are invalid and cannot authorize training. A corrected
  run requires a new preregistration; deleting the eligibility conjunct or filtering results after
  observing them is prohibited.
- **Boundary:** this is an interface/type-safety correction, not a validated architecture innovation
  or evidence of preference signal, generalization, NavDream robustness, driving benefit or safety.

## ADR-0067 — Do not claim Hodge/coboundary novelty on a native-star interface

- **Status:** accepted before any preference-network training.
- **Proof:** for the star incidence operator \(D_\star s=(s_i-s_0)_{i>0}\), every
  \(q\in\mathbb R^{K-1}\) is representable by \(s_0=0,s_i=q_i\). Hence
  `image(D_star)=R^(K-1)` and the star has no cycle; a node-score difference head and a direct
  native-relative vector head have the same output hypothesis space.
- **Decision:** Hodge/cycle consistency is not a core claim. Complete-graph structure remains only
  a falsifiable hypothesis and must beat capacity/FLOP/data/seed-matched native-vector, star-score
  and free complete-edge controls on held-out regret/selection, not merely training cycle residual.
- **Consequence:** if that matched study is null, delete the Hodge narrative and retain at most the
  appearance-quotient relation target plus selective simultaneous-uncertainty arbitration.
- **Boundary:** the proof does not show that a shared scorer is ineffective; it shows only that its
  exactness is not structurally restrictive on the deployed star. No learned or driving result
  follows.

## ADR-0068 — Define the minimal estimand before choosing a relation parameterization

- **Status:** accepted as a theory correction before preference-network training.
- **Decision:** the deployed estimand is the ragged vector of legal native-relative task margins
  `delta_i0 = y_i - y_0`. It does not require an absolute candidate utility or a complete candidate
  graph. A node-score difference head and complete-edge losses are optional parameterizations to be
  decided only by matched held-out regret experiments.
- **Reasoning:** this ordering keeps label, loss, inference and arbitration in the same unit and
  prevents a convenient architecture from being mistaken for the mathematical problem. It also
  yields the direct bound `regret <= 2 * max_i |delta_hat_i0 - delta_i0|` without a graph spectral
  assumption.
- **Learning contract:** optimize native-relative task correctness subject to a source-frozen
  paired-appearance relation-drift constraint. Task supervision prevents constant collapse;
  correct/mismatched/removed-image controls test scene-conditional visual necessity.
- **Boundary:** quotient factorization, conditional mutual information and constrained optimization
  are standard tools, not novelty claims. No generalization, calibration, NavDream, driving benefit
  or real-time result follows until their registered experiments pass.

## ADR-0069 — Treat one-frame NAVSIM labels as differences of candidate-local scores

- **Status:** accepted after pre-formal-run official-source inspection; no scientific outcomes read.
- **Evidence:** `pdm_score_from_interpolated_trajectory` constructs `[metric-cache PDM trajectory,
  evaluated trajectory]`, scores both, and returns `pred_idx=1`. The runner invokes this separately
  for native and every legal challenger. It does not jointly score `[native, challenger]`.
- **Decision:** define `F_S(tau)` as the evaluated-trajectory score in its internal fixed-PDM-baseline
  proposal set and define the label as `delta_i0 = F_S(tau_i) - F_S(tau_0)`. The native remains the
  candidate-generation and difference reference, not the progress-normalization baseline.
- **Implementation consequence:** require `PROGRESS_MODE=eval` before any scorer call, record it in
  primary/replay, and audit it against the frozen config. `gen_gt` fails closed.
- **Boundary:** these labels are one-frame, independently normalized surrogate scores and omit
  two-frame extended comfort. A positive oracle would not establish complete official PDMS gain,
  an absolute utility, safety, or cross-planner calibration.

## ADR-0070 — Calibrate simultaneous intervention error at the recording-group level

- **Status:** accepted as the only current route from predicted scale to a finite-sample coverage
  statement; empirical validation remains pending.
- **Decision:** freeze the point/scale model, reduce every calibration recording group to the maximum
  normalized residual across its registered appearances, anchors and legal challengers, then use
  split-calibration order statistic `k = ceil((n+1)(1-alpha))`. If `k > n`, set the radius to infinity
  and abstain.
- **Reasoning:** anchors, appearances and candidates inside one recording group are correlated and
  cannot be counted as independent calibration samples. The group maximum makes the event match the
  simultaneous arbitration theorem.
- **Finite-sample consequence:** if a future untouched calibration split has 42 independent groups,
  95% coverage has finite `k=41`; 99% coverage requires `k=43` and is therefore vacuous without
  extra assumptions. Existing fit/development groups cannot be relabeled as calibration, and the
  paper may not inflate the sample size to 672 anchors.
- **Boundary:** exchangeability applies only to the calibration distribution with fixed model and
  candidate mechanism. NavDream, another planner or another candidate policy receives empirical
  coverage and abstention analysis, not the source guarantee.

## ADR-0071：以差异支持关系替代不可识别的 shared-difference / free-interaction 分支解释

- **Decision:** `ADOPT_AS_CONDITIONAL_ARCHITECTURE_HYPOTHESIS__NO_TRAINING_AUTHORIZED`。
- **Reason:** v1 的 `Psi(Z,A_i)-Psi(Z,A_0)` 只有在非线性查询输出层已满足候选共同加性干扰时才
  精确消偏；共享参数本身不保证该条件。自由 `Omega` 又可绕过 difference 分支，使分支职责不可
  识别。B3 的 common-min、BS2 的晚期分叉与 BS4A-D0 的无证据差异共同指向更直接的对象：完整
  时空占据测度差 `nu_i0=mu_i-mu_0` 的正/负/公共支持。
- **Consequence:** 新候选架构用 challenger-exclusive、native-exclusive 与 common support 生成
  geometry-biased query，同时保留全部视觉 token 作为 key/value，再由反对称 relation operator
  输出合法 margin。直接投影证据率只作为 provenance 输入 scale/calibration；它本身不证明全局
  不可观测，也不自动触发硬拒判。真正的不可识别条件是全部可访问表示相同而真实 margin 符号
  相反。该结构仍需 B4R1、普通 scorer/DA-WAM-style/field matched study、NavDream、跨
  planner、clean no-harm 与 latency 证明；Jordan decomposition、antisymmetry、cross-attention 与
  conformal quantile均不作为单独创新。
- **Artifacts:**
  `refine-logs/2026-10-25_aorta_post_b3/DIFFERENTIAL_SUPPORT_RELATION_ARCHITECTURE_V2_CN.md`；
  `refine-logs/2026-10-25_aorta_post_b3/DIFFERENTIAL_SUPPORT_V2_IMPLEMENTATION_CONTRACT_CN.md`；
  `refine-logs/2026-10-25_aorta_post_b3/ICCV_STORY_AND_METHOD_TREE_V3_CN.md`；
  `references/2026-10-25_relation_story_sources.md`。

## ADR-0072：B4R1 one-frame margin 只作 oracle 资格门；训练前增加 surrogate-alignment gate

- **Decision:** `ONE_FRAME_LABEL_NOT_YET_AUTHORIZED_AS_FINAL_TRAINING_TARGET`。
- **Reason:** 官方源码显示 one-frame score 是 NC/DAC/DDC/TLC 的乘积与 EP/TTC/LK/HC 加权平均的
  乘积，且 progress 在每条 `[fixed PDM baseline, trajectory]` 内独立归一化；它省略跨 anchor 的
  two-frame extended comfort。signed differential support 只能作为 comparison query geometry，不能
  解析等同于完整 utility。
- **Consequence:** v2 主输出仍为与动作同型的 scalar native-relative margin；factor head 只作
  DA-WAM-matched optional arm。B4R1 即使通过，也必须先预注册 one-frame 对完整 official target 的
  sign/top-choice/regret/two-frame harm alignment，才能授权网络训练。若不对齐，加入 previous-action
  context 或收窄研究范围，不得用 surrogate 结果冒充 official behavior gain。
- **Artifact:**
  `refine-logs/2026-10-25_aorta_post_b3/SCORER_ALIGNMENT_AND_FACTOR_THEORY_V1_CN.md`。

## ADR-0073：以单一决策差异故事约束方法，并把two-frame context设为条件复杂度

- **Decision:** `ONE_STORY__TEMPORAL_CONTEXT_ONLY_IF_ALIGNMENT_REQUIRES`。
- **Reason:** B3与BS2已经给出“稳定/perfect field仍不保留候选差异”的反例，足以把论文收紧为
  differential-support relation，而不是继续并列field、Hodge、geometry certificate与adaptation故事。
  进一步源码审计得到`F_full=(7/8)F_one+(1/8)H*C2`，且`C2`依赖相邻帧重叠轨迹；完整行为目标
  可能需要previous-action context。
- **Consequence:** Oral/poster只讲“decision-difference evidence”这一条脊柱。当前网络保持最小
  current-frame v2；只有冻结surrogate-alignment显示two-frame引起不可接受翻转时，才加入previous-
  action/transition-support条件分支。任何新模块都必须对应新的反例、定理假设或官方目标依赖，并有
  删除消融。
- **Boundary:** 故事连贯不等于创新成立。B4R1、full-target alignment、matched scorer、NavDream、
  跨planner、clean no-harm与latency未通过前，状态仍为`NOT_ICCV_READY / NOT_ORAL_READY`。
- **Artifacts:**
  `refine-logs/2026-10-25_aorta_post_b3/ICCV_ORAL_STORY_READINESS_V4_CN.md`；
  `refine-logs/2026-10-25_aorta_post_b3/OFFICIAL_TARGET_DISCOVERY_AND_ALIGNMENT_V1_CN.md`。

## ADR-0074：full-target alignment必须扩充真实previous context，禁止在稀疏anchor roster上伪闭合

- **Decision:** AUGMENT_TRUE_PREVIOUS_ROW__ANCHOR_INDEX_IS_NOT_TIME。
- **Evidence:** 现有672-anchor roster中0个current token在同一roster内具有≤0.55 s predecessor，
  42/42组均为0 pair；直接官方聚合会对全部样本关闭two-frame权重。逐官方raw log恢复后，
  672/672 anchors存在唯一立即前一行，42/42组16/16，gap为0.498078–0.500686 s。冻结提交
  06d3ab0的primary/replay完整ledger逐字节一致，SHA-256为2ae90f5c…512c。
- **Consequence:** future alignment为每个current anchor增加真实previous context row；native与全部legal
  challenger共享相同previous trajectory。不得用anchor index减一、最近抽样anchor或roster-only
  aggregation。部署时若上一帧已被AORTA替换，下一帧应使用实际selected trajectory；是否需要
  stateful relation由alignment与sequential评估决定。
- **Boundary:** 当前是outcome-free接口资格；不证明one-frame/full alignment、preference signal、
  网络收益或驾驶收益。B4R1通过前不运行full-target科学实验。
- **Artifacts:** experiments/R004H7/H7BS4B4T0_TEMPORAL_CONTEXT_PREFLIGHT.md；
  src/aorta_drive/evaluation/temporal_context_roster.py；
  scripts/audit_r004h7bs4b4t0_temporal_context_roster.py。

## ADR-0075：用理论—网络—行为闭环和单一故事资格门约束后续设计

- **Decision:** `ONE_DOMINANT_MECHANISM__EVERY_COMPONENT_NEEDS_FAILURE_THEOREM_AND_ABLATION`。
- **Reason:** 数学、网络、loss、拒判、时延和叙事已分别建立，但若不共用同一估计对象，仍可能在
  实现中重新变成模块堆叠。当前唯一部署对象冻结为合法 ragged 域上的 native-relative official-task
  margin；完整时域 signed differential support、global visual evidence、反对称 relation、task+orbit
  constraint、同时校准和 native fallback 必须各自对应已观察失败、条件命题与删除实验。
- **Consequence:** 摘要最多保留一项主贡献（appearance-interventional differential-support relation
  learning）和一项从属机制（evidence-qualified native-preserving arbitration）。公式复杂度、DINO、
  cross-attention、Jordan/Hodge、conformal 和 uncertainty head 不单独列创新。任何新增模块都需满足
  “新反例或官方依赖—数学职责—可区分预测—matched deletion”四联证据，否则不进入主方法。
- **Boundary:** 该闭环是可证伪设计合同，不是效果证据。B4R1、full-target alignment、correct-image
  necessity、matched scorer、NavDream、跨 planner、clean no-harm 与 batch-1 latency 未通过前，
  仍为 `NOT_ICCV_READY / NOT_ORAL_READY`。
- **Artifact:**
  `refine-logs/2026-10-25_aorta_post_b3/THEORY_ARCHITECTURE_CLOSURE_V1_CN.md`。

## ADR-0076：旧绝对安全区间仲裁与新native-relative relation仲裁严格分离

- **Decision:** `SEPARATE_ESTIMANDS__NO_SEMANTIC_REUSE`。
- **Reason:** 历史 `arbitration.native_preserving.arbitrate` 接收每条候选的绝对上下界，要求先证明
  native 上界低于零，再寻找安全 challenger；新方法则直接估计
  `delta_i0 = F(tau_i) - F(tau_0)`，其条件定理只要求 challenger 的同时 lower bound 严格大于零。
  两种规则的零点、输入和保证不同，复用旧函数会把新理论错误翻译为“必须先证明 native 不安全”。
- **Consequence:** 保留旧函数供历史实验复现；新增
  `theory.relation_guarantees.arbitrate_native_relative` 作为数据无关参考。它允许零 challenger、对
  NaN/Inf/负半径 fail closed、按最大严格正 lower bound 选择，并使用冻结向量顺序打破平局。
  42-group split quantile 在95%返回第41次序统计量，在99%返回无穷并触发全拒判。
- **Evidence:** 新增穷举性质测试验证：在同时误差事件上每次 intervention 的真实 relative margin
  都严格为正；还验证 task correctness 推出 `2 epsilon` appearance drift、反向不成立，以及
  group maximum 不把组内比较冒充独立样本。
- **Boundary:** 这些是数学/软件合同测试，不证明模型 residual 满足半径、不证明 source 或
  NavDream coverage，也不授权 B4R1 或网络训练。

## ADR-0077：B4R1正式开封前强制primary/replay来源闭合并使用120分钟上限

- **Decision:** `PROVENANCE_FIELDS_ARE_GATES__WALL_CAP_120_MINUTES`。
- **Reason:** 原 auditor 能重算 ledger 与 summary，却未把 status、source commit、config hash、
  frozen-input hash 和 implementation hash 纳入主重放 exact gate；同时配置仍保留180分钟上限，
  与用户已要求的120分钟不一致。两项都能在任何 scorer outcome 产生前修正，不构成结果驱动调参。
- **Consequence:** runner把冻结输入与实现哈希写入结果；auditor要求 primary/replay 的 provenance、
  sealed-role、full-PDMS boundary、estimand、progress mode 和完整 ledger 全部一致。wall cap固定为120
  分钟，超时即失败，不自动续跑。服务器来源提交`53f16c1`的12项合同测试和compile通过。
- **Boundary:** provenance PASS不提供 preference signal。formal B4R1仍需用户明确授权；运行期间
  不读取partial scientific metrics、不修改candidate、membership、threshold或统计协议。
- **Artifact:** `experiments/R004H7/H7BS4B4R1_IMPLEMENTATION_PREFLIGHT.md`；本地镜像
  `F:/AORTA-Drive-artifacts/gate-b/R004H7/r004h7bs4b4r1_preflight_53f16c1.json`。
