# R005E2S Donor Sensitivity Implementation Status

状态：`IMPLEMENTATION_FROZEN__READY_TO_RUN__SECONDARY_ONLY`

## 科学职责

R005E2B 已机械通过但科学主门为
`UNSUPPORTED / STOP_CURRENT_VISUAL_MAIN_CLAIM`。R005E2S 不重新训练、不选择模型，也不能改变该
裁决；它只检验此前 scene-binding 解释是否依赖单一 geometry-matched donor 构造。

## 冻结干预

- `S2_disjoint_geometry`：禁止主 S1 donor edge 后重新执行全局几何匹配；
- `S3_hash_random`：在同一合法二分图上使用预注册 salt 的确定性无放回随机分配；
- 复用 R005E2B 的 18 个 V2 correct checkpoints；每个 fold 同时推理 3 seeds 与 2 schemes；
- 候选、history、target、mask、模型参数及计算图均不改变，只有 donor visual identity 改变。

## 实现与完整性门

- 实现提交：`73b5e46d5e06e064d1da60007d3af6e8878bf746`；
- 正式链：6 fold primary → 同 fold 同 GPU replay → 独立机械 audit；
- 冻结 donor manifest SHA256：
  `8c1f4e096dfadb683b7d1a6ddf92c256d32c33cf2ad1b6ecb29760ffbd993ce2`；
- 审计覆盖配置、runner、bundle、cross-fit、donor manifest、parent audit、checkpoint file/state、
  target/mask、fold/donor isolation、finite values、exact replay 与封存角色；
- 统计单位为 42 个 recording groups；先组内汇总 anchors 与 seeds，再进行 10,000 次组级 bootstrap；
- 全仓库 `469 passed`，新增脚本通过 strict mypy、ruff 与 `py_compile`。

## 主张边界

结果只允许归类为：一致 decision binding、relation-only binding、donor-sensitive/inconsistent 或无一致
scene binding。它不是条件互信息检验，不证明 Differential Query、外观鲁棒性、NavDream、驾驶收益
或安全性。任何结果都不得挽救或改写父实验 STOP。
