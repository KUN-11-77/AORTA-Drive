# R005E2A outcome-free preflight status

日期：2026-10-25。  
状态：`IN_PROGRESS__CROSSFIT_MANIFEST_PASS__NO_SCIENTIFIC_TRAINING`。

## 已完成

- R005E2科学命题、三arm、统计单位、交叉拟合、主指标与GO/MIXED/STOP/INVALID已预注册；
- 只读取R005E1B anchor ledger中的recording-group ID，生成42-group、6-fold outer cross-fit：
  每fold严格7个outer-test groups，每个group恰好一次作为outer test；
- 每个outer fold的35个outer-train groups由独立盐值固定为28个inner-fit与7个inner-validation；
- fold算法对输入顺序不变、对盐值变化敏感，并对overlap、missing、错误inner计数、outcome flag和
  duplicate group fail-closed；
- manifest：`experiments/R005/inputs/r005e2_crossfit_manifest.json`；
- manifest SHA-256：`e0d62eff0314ea8c02e83632515a4a601644419d41d3aea3f1e6860c0c110983`；
- R005E1B source result SHA-256在manifest中绑定为
  `7673f6fe216db7b91e90aaabc2eba48311c6432c96bbb4769691f06dda8dfd5f`；
- 定向协议测试8项通过；连同R005E1B contextual tests共16项通过。

## 尚未完成，继续阻止训练

1. geometry-matched mismatch donor manifest与无label、无outer-test donor证明；
2. runtime imported module、DINO源码/权重/preprocess/cache provenance；
3. V0/V1/V2 scorer的ragged mask、zero-challenger、permutation、finite、seed replay测试；
4. 三arm逐层参数完全相同、token相同、forward FLOPs比不超过1.01；
5. 单fold资源 rehearsal 与GPU/wall/memory合同；
6. outcome-free mechanical audit与独立复核。

在上述六项全部关闭、配置与源码再次冻结并获得单独运行授权前，不得启动R005E2B科学训练，不得
读取任何correct/mismatch/removed科学比较。
