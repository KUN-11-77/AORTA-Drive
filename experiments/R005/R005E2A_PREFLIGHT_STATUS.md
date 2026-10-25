# R005E2A outcome-free preflight status

日期：2026-10-25。  
状态：`IN_PROGRESS__CROSSFIT_AND_DONOR_MANIFEST_PASS__NO_SCIENTIFIC_TRAINING`。

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

## 尚未完成，继续阻止训练

1. runtime imported module、DINO源码/权重/preprocess/cache provenance；
2. 在服务器真实`[1,12,18,32,1280]`视觉shape复核网络合同，并验证三arm逐层参数完全相同、
   token相同、forward FLOPs比不超过1.01；
3. 单fold资源 rehearsal 与GPU/wall/memory合同；
4. 剩余outcome-free资格门的统一机械审计与独立复核。

在上述四项全部关闭、配置与源码再次冻结并获得单独运行授权前，不得启动R005E2B科学训练，不得
读取任何correct/mismatch/removed科学比较。

首次物化尝试在任何assignment或科学指标产生前fail-closed：672个anchor中有1个经R005E1A修复的
replacement panel，不在原H7BS1单一artifact root。该失败不改变fold或donor规则。解析器现要求在
冻结base与replacement roots中按预登记panel SHA-256精确查找；同token但hash不符、缺失或多份同hash
但bytes不一致都会失败。该修复后的正式manifest已由提交`998d85a`完成primary、完整replay与独立审计。
