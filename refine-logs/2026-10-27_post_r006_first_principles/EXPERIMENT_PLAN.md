# Experiment Plan：R007 DINO Interface Audit → Conditional Method

**Problem**：冻结 DINOv3-H+ 的稳定视觉证据在何处被 planner 接口放大或丢失？
**Method thesis**：只有在干预审计证明具体接口机制后，才训练与该机制一一对应的最小 bridge。
**Date**：2026-10-27

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| C1：DINO 驾驶证据的分布、压缩损失与接口后果可被组外复现 | 这是超越 Constant Eye 的发现性基础 | correct DINO相对非视觉/错配有任务增量；full/compressed/层级差异与真实planner consequence闭合；outer-confirmation前瞻复现 | B1 |
| C1b：任务前置压缩与功能兼容写入是被真实机制要求的，而非架构偏好 | 决定reader–writer是否具有必要性 | 等预算context timing 2×2交互成立；native feature error与frozen-tail task utility出现可复现目标失配 | B1-W/T |
| C2：机制导出的单一接口优于相同 DINO 的简单方法 | 决定算法是否有独立价值 | 相同输入、参数/FLOPs、训练预算下优于 ordinary full-token reader/consistency/direct residual，且 clean 非劣 | B2 |
| C3：收益转化为跨外观行为价值 | 决定论文是否成立 | NavDream zero-shot × 多 planner 的净收益、hard-failure减少、clean非劣与实时门同时通过 | B3 |

Anti-claims：增益只来自 DINOv3-H+、更多参数、额外数据、普通 output consistency、重训 planner head，或事后选择最有利外观/组。

## Paper Storyline

- 主文必须证明：B1 机制发现、B2 方法必要性、B3 行为闭环。
- 附录支持：线性化误差、token dose-response、VJP 条件基线、更多外观分层。
- 明确 CUT：Fantope、升 rank、CPTR 重启、无机制授权的多模块组合。

## Experiment Blocks

### B1：R007V0 DINO 接口因果审计（MUST-RUN）

- **Claim tested**：C1。
- **Dataset/split**：既有 source-fit 42 recording groups 沿冻结 28/7/7 角色使用；28 discovery、7 selection，最后 7 outer-confirmation 只允许一次前瞻裁决。development/calibration/test/NavDream/target 不访问。
- **Backbone**：唯一 DINOv3-H+ 权重、预处理、分辨率、token 顺序和精度；先锁 SHA 与缓存 provenance。
- **接口**：先用源码与checkpoint判定真实拓扑。Constant Eye TransFuser-DINO候选为DINO-only，
  依次审计raw tokens、projector、CNN compression、decoder query与trajectory output；只有真实hybrid
  planner才使用native×DINO因子设计。第二planner只在同样拓扑资格闭合后加入。
- **hybrid因子化条件（不适用于DINO-only）**：
  1. native-shift + DINO-shift；
  2. native-shift + DINO-clean；
  3. native-clean + DINO-shift；
  4. native-clean + DINO-clean；
  5. native-shift + DINO-mismatch；
  6. native-shift + DINO-removed；
  7. adapter/interface 后 clean/shift swap；
  8. joint full-cut clean/shift 上界。
- **双目标**：`Y_task`裁决任务正确性，`O_clean`只裁决功能恢复；禁止用复制clean planner输出替代驾驶正确。
- **信息 probe**：相同容量/训练预算下比较 history-only、normalized final layer、冻结last-four mean、
  compressed tokens、compressed+full skip与有限中间层候选；每臂含correct/mismatch/removed。
- **层级候选**：实际模型深度的固定分位点加最后四层只在discovery生成；selection前锁定。双层只用等权
  normalized mean，保持token/width不变；outer不搜层，不用concat增加容量。
- **主 endpoint**：group-held-out `Y_task` risk/correct-minus-control，以及真实stage interchange产生的
  physical output/official utility consequence；两者缺一不能称可用驾驶证据。
- **次 endpoint**：`O_clean` functional restoration、DINO/hidden drift ledger、最差组、跨外观分层。
- **裁决标签**：`AMPLIFICATION`、`COMPRESSION`、`LAYER_COMPLEMENTARITY`、`MIXED`、`UNRESOLVED`、
  `CURRENT_INTERFACE_UNRECOVERABLE`；标签可多选，不能强迫唯一。
- **Success criterion**：selection 上预冻结的最小效应与组级 CI 通过，并在 outer-confirmation 同方向复现；具体阈值在任何 outer 结果可见前由 discovery 的量纲和测量噪声冻结。
- **Failure interpretation**：只停止对应接口/读出合同，不外推为 DINO 信息论不可能。
- **Writer target audit**：同一reader/writer/cut下比较native MSE、final-feature stitching、tail-output matching与task supervision，联合报告feature/tail/task三层结果。
- **Context timing audit**：compressed/raw-full × context-free/task-conditioned严格等预算，主量为group-held-out risk difference-in-differences。
- **VLM/VLA scope**：DINO不是VLM；普通planner用于机制控制，最终行为链至少需要一个真实VLM/VLA驾驶策略，并干预所有图像派生caption/query/history。
- **Figure target**：主文 Fig.2 因子化交换图；Fig.3 full/compressed risk；Table 1 机制裁决。

### B2：R007V1 单一方法 matched study（CONDITIONAL MUST-RUN）

- **Unlock**：仅 B1 支持 amplification/compression 之一。
- **Compared systems**：
  1. frozen native planner；
  2. faithful Constant Eye（单列，允许原文式重训 head，不冒充 matched）；
  3. frozen planner + full-token ordinary reader；
  4. frozen planner + matched residual adapter；
  5. same structure + ordinary output consistency；
  6. direct-output residual；
  7. mechanism-derived single method（只 compression→full-token bridge；只 amplification→functional constraint；
     both→组合一个最小 task-conditioned DINO evidence interface，不继承旧 FQB 结构）。
- **Control**：所有 frozen-interface arms 共享 DINO tokens、训练样本、optimizer steps、seed、参数/FLOPs tolerance；3 seeds，统计单位仍是 group。
- **Metrics**：group-held-out functional restoration、official source utility、clean non-inferiority、correct-minus-mismatch、干预能量、参数/FLOPs、batch-1 P50/P95。
- **Success criterion**：Proposed 对 strongest matched simple arm 的组级 paired CI 下界超过预冻结最小有意义效应；clean utility 满足非劣界；不是全拒判。
- **Failure interpretation**：simple arm 等效则删除复杂结构；ordinary consistency 等效则只保留简单 consistency；direct residual 等效则停止内部 repair 主张。
- **Table target**：主文 Table 2；关键 deletion study 为 Table 3。

### B3：R007V2 zero-shot behavior closure（CONDITIONAL MUST-RUN）

- **Unlock**：B2 通过并冻结最终 pipeline。
- **Data**：NavDream 只作最终 zero-shot；另报告 clean/navtest/navhard 与至少三类 planner。不得用 NavDream 选超参数、阈值或架构。
- **Systems**：frozen base、faithful Constant Eye、strongest matched simple arm、final Proposed。
- **Metrics**：official utility/EPDMS、hard-failure净减少、appearance-induced flip、clean group-level non-inferiority、coverage/abstention、batch-1端到端P50/P95、显存。
- **Success criterion**：预注册主指标的 group-level CI 与最小效应同时通过；clean非劣；P95在预算内。
- **Failure interpretation**：restoration 不能挽救行为失败；停止 ICCV/Best-Paper 方法主张。
- **Table target**：主表；Fig.1 discovery→method→behavior 总图。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision gate | Rough cost | Risk |
|---|---|---|---|---|---|
| M0 | 资产/语义资格 | DINO SHA、Constant Eye资产、pair binding、C3/C4切口、2-anchor replay | 机械 exact/finite；否则 STOP | <1 GPU-h | checkpoint/接口不全 |
| M1 | B1 无训练交换 | 2 anchors smoke → 28 discovery groups | full-cut 与因子化路径可复算 | 4–12 GPU-h，依缓存而定 | 路径耦合 |
| M2 | B1 组外证据分布 probes | 1 fold smoke → 6-fold/3 seeds → 7 outer一次确认 | task增量、压缩/层级差异与planner consequence同时闭合 | 20–60 GPU-h | probe容量混淆/层搜索泄漏 |
| M3 | B2 方法 | strongest simple arms → conditional method | 必须超过 simple arm | 每planner约30–80 GPU-h | 方法不必要 |
| M4 | B3 闭环 | zero-shot NavDream、多planner、latency | 行为/clean/runtime同时通过 | 依官方评测约60–150 GPU-h | 行为不转化 |

以上仅为资源量级估计；M0 后用实测吞吐更新，不改变科学合同。

## First Four Runs

1. `R007V0S`：源码/checkpoint拓扑审计 + 2-anchor topology-legal swaps + full-cut exactness，不读取封存角色；
2. `R007V0D`：28 discovery groups 的无训练顺序stage或hybrid因子交换与 stage ledger；
3. `R007V0P`：单折 final/last-four/intermediate/full/compressed/skip 的
   correct-mismatch-removed probe；先闭合双目标、容量和层候选规则，再冻结正式6-fold。
4. `R007V0W/T`：仅在任务信息增量通过后执行 writer 目标与上下文时序正交审计。

## Risks and Mitigations

- **Constant Eye checkpoint不可用**：先核验服务器资产；若只有 DINO 权重，faithful reproduction 单列为 M0 baseline task，不用未对齐权重冒充复现。
- **mismatch/removed OOD**：只与正确图像差、任务误差和 full-token probe联合解释，不单独做信息论结论。
- **原生与DINO路径混淆**：使用完整2×2因子化交换与joint swap。
- **反复使用42组**：它们只作 discovery/selection；outer 7一次确认，之后需要新的独立确认单位。
- **简单方法已经足够**：视为成功的科学收缩，删除复杂接口/VJP而非继续堆模块。

## Final Checklist

- [x] 主张不超过2个机制/方法主张
- [x] 所有正式实验围绕冻结 DINOv3-H+
- [x] 低秩/Fantope 已切除
- [x] 最强简单基线在方法前
- [x] clean、行为、时延硬门独立
- [ ] DINO/Constant Eye 服务器资产已核验
- [ ] R007V0S 代码与合成反例已通过
- [ ] 预注册最小效应/等效界/非劣界已冻结
