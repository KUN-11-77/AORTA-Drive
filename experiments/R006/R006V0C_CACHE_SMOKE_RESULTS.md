# R006V0C activation-cache smoke 结果

状态：`MECHANICAL_PASS__FORMAL_CACHE_NOT_COMPLETE__NO_SCIENCE_CLAIM`。

## 1. 本轮只回答什么

本轮只检查已资格化 C3/C4 切口能否在真实冻结 LTF 上，按 source-fit recording group 稳定地产生
可审计的 activation cache。它不估计低秩介导、不选择 layer/rank、不训练 DINO 或 repairer，也不读取
development、calibration、test、NavDream 或 target。

## 2. 范围

- 1 个 source-fit recording group，16 anchors；
- 4 个冻结条件：clean、low exposure、low contrast、cool cast；
- C3 planner memory：`[1,65,256]`，只允许前 64 个 BEV memory sites 修复，status site 64 冻结；
- C4 trajectory query：`[1,1,256]`，唯一 query site 可修复；
- 冻结 LTF checkpoint、planner 源码版本、输入角色、appearance 实现与随机种子写入运行 manifest。

## 3. 机械结果

- cache runner exit code：0；
- cache manifest：PASS；
- 独立 audit：14/14 PASS；
- shard 数量与 anchor 数量一致；
- shard SHA-256 与 manifest 一致；
- C3/C4 shape、repairable/frozen site semantics 一致；
- sealed roles 未访问；
- `scientific_metrics_emitted=false`；
- manifest SHA-256：`cda3b113a73c8e868e68e427e282074ad0fe5628343ee7d9b7abaf8ccbfe41c8`；
- shard SHA-256：`2a921f29077460eac78858c41e45e81e526f4d70b4de680827d76535a1a2cb00`。

## 4. 失败记录

首次 smoke 已完成数值缓存，但 provenance 写入因跨用户 checkout 的 Git ownership 检查而 fail-closed。
随后只修复只读 HEAD 解析，不改变 checkpoint、数据、appearance、tensor schema 或科学协议；本记录对应
修复后的全新 smoke。旧失败不作为科学运行，也不被覆盖成 PASS。

## 5. 结论边界

本轮只资格化“小规模缓存管线可执行、角色隔离和接口语义可审计”。它没有证明：

- 42 组完整缓存一定能完成；
- 存在共享低秩介导；
- DINO 能恢复该介导统计量；
- CPTR 优于 Constant Eye、普通 adapter、direct-output repair 或 scorer；
- 外观 OOD、驾驶收益、安全性或实时性。

下一步仍是冻结正式缓存预算、nested-OOF 求解器、A0–A7 控制与最终 manifest；全部资格完成后才可
一次性运行 R006V0。
