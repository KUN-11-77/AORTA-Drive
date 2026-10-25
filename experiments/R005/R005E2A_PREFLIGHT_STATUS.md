# R005E2A outcome-free preflight status

日期：2026-10-25。  
状态：`COMPLETE_OUTCOME_FREE_PREFLIGHT__GO_TO_SCIENTIFIC_RUNNER_FREEZE__NO_RESULT_YET`。

## 已完成

- R005E2科学命题、三arm、统计单位、交叉拟合、主指标与GO/MIXED/STOP/INVALID已预注册；
- 只读取R005E1B anchor ledger中的recording-group ID，生成42-group、6-fold outer cross-fit：
  每fold严格7个outer-test groups，每个group恰好一次作为outer test；
- 每个outer fold的35个outer-train groups由独立盐值固定为28个inner-fit与7个inner-validation；
- fold算法对输入顺序不变、对盐值变化敏感，并对overlap、missing、错误inner计数、outcome flag和
  duplicate group fail-closed；
- manifest：`experiments/R005/inputs/r005e2_crossfit_manifest.json`；
- Git-blob/LF canonical manifest SHA-256：
  `f0be65e54517a80aee5a7d09d5027e73f22f1204e28a5c7568ee33c46836d878`；
- R005E1B source result SHA-256在manifest中绑定为
  `7673f6fe216db7b91e90aaabc2eba48311c6432c96bbb4769691f06dda8dfd5f`；
- 定向协议测试8项通过；连同R005E1B contextual tests共16项通过；
- 已从冻结context roster、R005E1B identity和B4R1 panel hash提取outcome-free panel identity roster；
  该文件不含utility、candidate score、RGB、DINO或模型输出；
- panel identity roster：`experiments/R005/inputs/r005e2_panel_identity_roster.json`；
- panel identity roster SHA-256：
  `b0c5b14a7bb935f381ca246c568b616d24c58b5c8e88a737824b4e498731d62f`；
- donor算法已冻结为native几何描述子上的全局线性和分配：训练映射是跨组一一置换，validation/test
  映射从相应训练池无放回取donor；因此训练visual token经验边际严格不变，且禁止same-group与
  outer-test donor；
- donor core新增11项测试；与cross-fit协议合计19项通过；strict mypy通过；
- 服务器冻结提交`998d85a`的28/28定向测试通过；两个frozen panel roots按SHA精确解析为
  base `671` + replacement `1`；
- donor primary/replay byte-exact，SHA-256均为
  `9a5f4878e9856a6cb020c1d814f034233aa7b6585cc2507af16ab29cb2373093`；
- 6 folds、24 stages、7,392 assignments全部通过独立机械复算：recipient roster、donor pool、
  same-group禁边、无放回约束和训练边际精确置换均闭合；
- audit SHA-256：`5c05ccbd320b00e0e4388300b4f23e0411f384a58e635c664abc8c902f0d519d`；
- 可读manifest：`experiments/R005/inputs/r005e2_mismatch_donor_manifest.json`；完整压缩bundle：
  `experiments/results/R005E2A_998D85A/r005e2a_outputs.tar.gz`。
- 通用`FullContextSetScorer`已按预注册实现：共享2-layer时序encoder、显式candidate-vs-previous
  context fusion、无candidate位置编码的2-layer set encoder、单层candidate-to-visual cross-attention、
  shared score head与` tanh(s_i-s_native)`输出；native margin恒为精确0；
- 网络本地13项synthetic合同通过：ragged mask、zero challenger、challenger permutation equivariance、
  invalid candidate/visual storage non-interference、finite backward、seed replay和loss weighting；strict mypy通过。
- 服务器真实shape scorer preflight通过：V0/V1/V2逐层同构，均为4,749,313参数、6,912 visual
  tokens和7,623,183,799 profiler FLOPs，parameter/FLOPs ratio均为`1.0`；
- RTX 3090、BF16、batch-1的scorer-only P95为V0 `16.97ms`、V1 `17.16ms`、V2
  `16.12ms`，三臂peak allocated memory均约245MB；同输入推理bit-exact、native margin精确0；
- 独立审计绑定scorer/runner source、可解析配置、真实shape、参数、FLOPs、确定性和sealed-role边界并PASS；
  result SHA-256 `cf0f158b...620c1f`，audit SHA-256 `ad80ff97...532141`；
- 完整记录：`experiments/results/R005E2A_32F179F/`。scorer-only延迟不含DINO、I/O、训练backward或
  完整系统仲裁，不得升级为实时系统主张。
- 冻结的672-anchor视觉缓存已用同一DINO合同完成primary与完整replay：42 groups、每anchor 12 slots、
  8,064个唯一图像特征；DINO checkpoint SHA-256为
  `7c1da9a5...f41488`，151个Python源码文件的tree SHA-256为`dde43545...cf1bf8`；
- 8,064份BF16 dense features与1,344份投影产物均被独立审计器重新打开、检查有限值并逐文件核验；
  primary/replay的逻辑manifest、特征和投影全部精确一致，sealed roles与科学标签均未访问；
- cache audit SHA-256为`94c14166...bf4194`，完整紧凑证据包见
  `experiments/results/R005E2A_C957AA0/r005e2a_visual_cache_compact.tar.gz`；
- 已在任何科学输出可见前物化S2 disjoint-geometry与S3 hash-random donor敏感性映射；两次构建
  byte-exact，14,784 assignments全部满足fold隔离、跨组、无放回和无标签/视觉内容参与，S2与主S1
  recipient-donor边完全不重合。它们是non-gating解释性分析，不能挽救主门失败；
- 单fold真实I/O资源演练使用28 inner-fit groups、448 anchors、零科学target与native-shape history替身，
  完成448/448 backward steps；loss/gradient有限，峰值显存365,554,176 bytes，总耗时37.999 s，
  训练step P95为80.900 ms。原result中的`0.570 GPU-hours`仅覆盖每个arm/seed/fold一次448-step
  selection epoch，漏计refit和最多40 epochs，不能作为完整预算；按同一实测重算，训练-only的
  一次selection+一次refit下界为1.282 GPU-hours，40+40 epochs最坏上界为51.299 GPU-hours，均不含
  validation/test/I/O/audit。提交后的runner与auditor已把epoch和448/560 step比例写成硬合同；
- checkpoint恢复后的模型tensor、AdamW状态与同batch前向均精确一致。首次审计因错误假定
  `P95 >= mean`而fail-closed；冷启动离群值使data-wait mean高于P95，这是合法统计形态。提交
  `81a2cb5`修正为只要求maximum不低于mean和P95，新增反例测试，443项全量测试通过；对原始同一
  result/checkpoint重新审计PASS，未重跑训练，失败日志完整保留；
- 训练资源audit SHA-256与原始产物哈希见
  `experiments/results/R005E2A_C957AA0/training_rehearsal_SHA256SUMS.txt`。外部cross-model reviewer当前
  不可用，因此独立科学复核状态为`REVIEW_UNAVAILABLE`；本阶段没有科学endpoint可供解释。
- 完整runner冻结前的预算复核发现上述epoch漏乘问题；它不改变已完成448-step运行与checkpoint证据，
  但原`0.570`字段被明确退役。纠正账本见
  `experiments/results/R005E2A_C957AA0/resource_projection_correction.json`。

## 资格裁决与下一步边界

R005E2A outcome-free资格门机械`PASS`。结合用户已明确授权“通过DINO来源审计和单折演练后启动冻结
视觉增量实验”，裁决为：

- `GO`：实现、测试并冻结R005E2B科学runner、独立auditor和异常停止合同；
- `HOLD`：在runner源码、输入哈希、完整replay和审计路径冻结前，不得读取科学结果；
- `STOP`：仍不解锁Differential Query、NavDream、development/calibration/test/target或驾驶安全主张。

资格通过只证明实验可以忠实执行，不证明correct visual提供增量；该科学命题仍为`UNTESTED`。

首次物化尝试在任何assignment或科学指标产生前fail-closed：672个anchor中有1个经R005E1A修复的
replacement panel，不在原H7BS1单一artifact root。该失败不改变fold或donor规则。解析器现要求在
冻结base与replacement roots中按预登记panel SHA-256精确查找；同token但hash不符、缺失或多份同hash
但bytes不一致都会失败。该修复后的正式manifest已由提交`998d85a`完成primary、完整replay与独立审计。

首次真实shape GPU演练在输出前被deterministic-algorithm gate拒绝：CuBLAS workspace未在进程启动前
冻结。未产生resource结果或科学指标。协议保持deterministic模式，新增硬门
`CUBLAS_WORKSPACE_CONFIG=:4096:8`；禁止通过关闭确定性算法绕过。修复后提交`32f179f`已完成上述
服务器preflight；独立审计随后发现并修复了原配置`primary_endpoints`的YAML结构错误，审计以可解析
配置SHA `79e8a1f8...a04b`重新执行并PASS。
