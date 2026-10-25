# R005E2 资源调度修订：每卡多独立作业

日期：2026-10-25（任何正式 job 完成、科学指标产生或读取之前）。

状态：`RESOURCE_ONLY_AMENDMENT__SCIENTIFIC_CONTRACT_UNCHANGED`。

## 原因

首次正式队列使用 GPU `[1,2]`、每卡一个独立 job。运行健康检查显示每个进程只使用约
`0.7 GiB` 显存与 `14--18%` GPU，CPU 约一个核心；此观察只涉及资源，不涉及 loss、prediction、
epoch ledger 或任何科学 endpoint。用户随后明确要求提高每卡利用率。

## 修订

- 正式 GPU roster：`[0,1,2,3]`；
- 每张 GPU 最多并行 4 个相互独立的 fold/arm/seed job，总并发上限 16；
- 每个 job 的 primary 与 replay 仍绑定同一物理 GPU；
- 每个进程仍使用原冻结 `batch_size=1`、4 个 DataLoader workers、相同 deterministic/CUBLAS 合同；
- GPU 在没有本队列作业时仍必须低于冻结 idle-memory 门，防止占用他人卡；
- 完整 replay 和 bit-exact audit 仍是发布科学结论的必要条件。

## 未改变

模型、DINO/cache、bundle、candidate/history、target、fold、donor、arm、seed、初始化、样本顺序、
loss、optimizer、LR、schedule、early stopping、阈值、统计、GO/MIXED/STOP 均未改变。

原低并发目录在无任何 completed job、无 scientific metrics emitted 的状态下终止并永久保留，不能
与新正式运行合并。该修订只减少 wall-clock，不改变单个 job 的数学计算；若多进程资源竞争导致
任一失败、非有限值或 primary/replay 不精确，实验按原合同判 `INVALID`，不得放宽审计。
