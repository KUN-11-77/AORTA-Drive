# R007V0S 资产与真实拓扑审计

日期：2026-10-27  
状态：`SOURCE_AND_DINO_WEIGHT_VERIFIED__TRAINED_CONSTANT_EYE_CHECKPOINT_NOT_LOCATED`

## 已核验事实

1. 冻结 DINOv3-H+ 权重存在，SHA-256 为
   `7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488`，与 R004/R005 合同一致；
2. Constant Eye 归档源码包含 TransFuser-DINO、DiffusionDrive-DINO 与 GTRS-DINO 三条实现；
3. TransFuser-DINO 关键源码哈希：
   - model：`6bb7788a1df7f7156ebfe6c6d5488ba6f2a14e7bf7ccf1b9c06e2755d5e5826d`；
   - backbone：`ef351f8afd1d220ca2306114e4909276c4807da1017c1940384177ece430877d`；
   - config：`98794e4a6b4f140940e19a27646c2ce13f5ac2c2897a0e2664b7e806f5db75d7`；
4. TransFuser-DINO 是 `DINO_ONLY` 拓扑：完整 DINO token 经 `_dino_proj`、CNN 上/下采样、
   positional embedding、Transformer decoder 与 trajectory head 形成输出；不存在可独立固定的 native
   image path。因此 R007V0 对该模型必须使用顺序 stage interchange，不能使用 native×DINO 2×2；
5. 已检查的归档包中没有 TransFuser-DINO、DiffusionDrive-DINO 或 GTRS-DINO 的训练 checkpoint；
6. 一个已存在的 GTRS/inspection checkpoint 经 state-dict 审计只包含普通 TransFuser 路径，没有 DINO
   adapter/projector 参数，不能冒充 Constant Eye checkpoint。

## 当前资格边界

- DINO 权重与 Constant Eye 源码：`PASS`；
- topology-conditioned 审计设计：`PASS`；
- 本地审计内核、真实拓扑 runtime 与反例：19 项相关测试通过；四个目标文件 strict mypy PASS；
- 真实 2-anchor Constant Eye forward/interchange：`HOLD_MISSING_TRAINED_CHECKPOINT`；
- 任意 amplification/compression、任务增益或方法结论：`UNTESTED`。

## 允许的下一步

1. 已完成真实源码拓扑对应的只读 stage capture/replacement runtime；下一步在真实外部源码构造模型
   做随机权重 shape/identity smoke，但不得解释科学效果；
2. 继续优先定位学长训练 checkpoint；若确认无法取得，则将 faithful Constant Eye reproduction 登记为独立
   baseline reproduction，不用随机或普通 TransFuser 权重替代；
3. reproduction 必须冻结原文架构、训练数据、损失、步数和评价合同，先复现其 clean/NavDream 量级，
   再允许 R007V0D 机制审计；
4. checkpoint 缺失期间不启动任何新 DINO 接口、VJP、Differential Query 或方法训练。

## 不能推出

源码中的多层 projector/CNN 不等于已经证明压缩丢失；DINO-only 拓扑也不等于 native 路径无价值。
当前审计只修正实验对象并关闭伪基线风险。

