# R005E3A 实现与正式运行状态

日期：2026-10-26

状态：`FORMAL_COMPLETED__MECHANICAL_PASS__MIXED__ALL_NATIVE_COLLAPSE`

## 冻结实现

- 预注册提交：`cdb496f`；温度中位数澄清提交：`483f697`；
- 实现提交：`77f37e43862bc352a753d6ee9141c907c9ac926b`；
- 唯一训练处理：`equal_anchor_challenger_margin_MSE`替换为
  `soft_selection_regret`；
- 架构、DINO cache、初始化、outer-train顺序、AdamW、schedule、父V2 selected epoch及
  optimizer-step预算保持逐job匹配；
- 温度只由outer-train严格正utility regret gaps的中位数确定，偶数时取两个中央次序统计量的
  算术平均；不搜索、不读取outer-test target来定温度。

## 实现资格证据

- 本地全仓库：`485 passed`；
- 服务器正式解释器全仓库：`485 passed`；
- 新增定向测试：30项通过；
- 本地strict mypy：6个新增/修改源文件通过；Black check通过；
- 服务器`dinov3`环境未安装ruff/mypy，未修改该环境；服务器动态测试使用与R005E2B相同的
  `/data/jbwang/anaconda3/envs/dinov3/bin/python`；
- GPU 0单fold×seed演练：primary/replay均退出0，14 epochs、7,840 optimizer steps；预测数组、
  final model state、temperature ledger、training-order hash及父初始化/epoch预算exact；sealed role与
  partial-scientific-metric标志均为false。

## 正式链与结果

- 服务器目录：`/data/khsong/aorta_runs/r005e3a_77f37e4`；
- GPU roster：`[0,1,2,3]`；每卡最多4个独立job；
- roster：18 primary + 18 same-job same-GPU replay + 一次性机械audit；
- 18 primary + 18 replay全部完成，正式auditor退出0；
- predictions bit-exact、model state exact、父初始化/epoch/step预算、fold/target/mask、finite、same-GPU、
  temperature来源与sealed-role门全部通过；
- 冻结科学分类为`MIXED_DECISION_IMPROVEMENT_WITHOUT_COMPLETE_SCENE_BINDING`；
- post-hoc只读复算确认18/18 outer-train checkpoints全部将challenger推至`-1`并100%选择native，
  absolute oracle capture为0；
- development、calibration、test、NavDream与target domain保持封存。

完整结果见`R005E3A_RESULTS.md`。冻结`MIXED`不追溯修改，但soft-regret当前实现不具视觉决策机制
资格；下一步只允许优化/可实现性诊断，不解锁Differential Query或NavDream。
