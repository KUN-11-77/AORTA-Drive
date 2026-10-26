# R006V0C 正式 activation cache V2 结果

状态：`MECHANICAL_PASS__42_SHARDS_EXACT_REPLAY__SCIENCE_NOT_STARTED`。

## 完整结果

- 运行目录：`/data/khsong/aorta_runs/r006v0c_f4b1183_formal_v2`；
- 冻结实现提交：`f4b1183956f1f7ee34a10fe4dc17aa67adc63b36`；
- source archive SHA-256：`ff96e382cead030a07ef7f6d419f12f9fd2f0a8c83d97fbd6625e567926ad943`；
- runner exit code：0；V2 audit exit code：0；
- 42/42 source-fit groups、672/672 anchors、42/42 shards完成；
- cache audit：265/265 PASS；
- 与第一次完整运行比较：42/42 shard SHA-256逐组完全一致，replay audit 65/65 PASS；
- 封存角色未访问，未产生coverage、restoration或其他科学指标。

## planner provenance

- checkpoint SHA-256：`d08e97ba6beb937412697d619171553041622859e026a06935a2e0e118e6309d`；
- planner commit：`b0798122597ec8d9f3d09c0a5d019d6d06e2414f`；
- 整个planner仓库dirty：true，仅来自执行子树之外的研究文档；
- 实际执行子树：`navsim/`；
- `navsim/` dirty：false；
- `navsim/` Git tree：`615a051a2d6bef1ef2ca49a7e439b0020bd1aaf4`。

## 哈希

- V2 manifest：`f3658ea5b0a85547129587ed39691bda6ba1853e83e0c83cf569854bfc0f01bb`；
- V2 cache audit：`1a4da6800daa71b4574fd8d3864731c7a9339288c1ef002e8ec5f5e639c23284`。

## 裁决

正式activation cache的完整性、执行源码子树provenance与重复确定性闭合。下一步仅解锁无科学endpoint
的streaming JVP资源benchmark及正式runner/auditor冻结；R006V0低秩介导科学实验仍未开始。

本结果不支持低秩介导、DINO可恢复性、CPTR、NavDream、行为、时延或安全主张。

