# R004H7-B 实现资格审查

状态：`PASS_IMPLEMENTATION_QUALIFICATION__PAYLOAD_HOLD`

本资格只说明 shadow-decision runner、统计汇总与机械审计接口已经实现并通过无数据测试；
24 个 development groups 尚未读取，因而没有 H7-B 科学结果，也没有解锁 H7-C。

## 冻结对象

- 实现 revision：`432d0473b6828bceca910cc1763a44ed8180e0f1`；
- config SHA-256：`26716748bf8329d448fa6024896af9cb8d0897b37c9ef98bbf28e3f6c97be660`；
- shadow evaluator SHA-256：`9753d13a4f172a8a6155ade09d448657fae127a1e8d4c707f2cf10e38ac5d1b5`；
- runner SHA-256：`dac827f3daa84222886f88b8d5b59f47018f072ac852d765c56216610a708c4a`；
- auditor SHA-256：`5ed8cd59adfcbfd61149b1c141f5868459d980b31daa8502f90966c01f8bbecd`；
- H6 result、exact-teacher audit、source/feature/label/metric manifests 均已在 config 中绑定；
- official NAVSIM execution 与 candidate/intent 源码在 runner 启动时逐项验 hash。

## 已闭合的实现合同

1. 候选由冻结 normal-offset provider 生成，并统一通过 official NAVSIM `0.1s` execution；
2. discrete query 使用真实长宽与 yaw 的 oriented rectangles，不使用旧圆盘证书；
3. ROI clipping、空 native、无有效 challenger 或非 bit-exact execution 均 fail closed；
4. 单个无效 challenger 留在账本中，不删除 anchor 上其他合法 challenger；
5. 每个条件重算 exact set-min contrasts、最大 contrast error、flip、teacher regret、top gap、tie；
6. 每次查询同时检查 oscillation bound 与 `2η` regret bound；
7. 统计以 recording group 为唯一独立单位；anchor/candidate/appearance 均为组内重复；
8. invalid anchors 保留在覆盖率分母；任一组少于12/16合法 anchors，整体直接
   `INSUFFICIENT_PANEL_COVERAGE`；
9. 10,000次 paired cluster bootstrap 同时比较 decision error 与 edge MAE、centered-node
   MAE、DINO drift；AUROC 不可计算比例超过5%时 fail closed；
10. 独立 auditor 从保存的 realized trajectories 重新生成 provider、eligibility、node panel、
    contrasts、outcomes 与全部 bootstrap，不信任 runner 汇总。

## 无数据验证

- H7B focused tests：`7 passed`；
- H7A+H7B focused tests：`13 passed`；
- 完整仓库：`281 passed`；
- strict mypy：新增 evaluator、runner、auditor、tests 全部通过；
- line-length 与 `git diff --check` 通过；
- ruff：当前环境未安装，记为 `TOOLING_UNAVAILABLE`，不以其他检查冒充。

## 科学边界与下一权限

- `GO`：在另一次明确授权下执行一次冻结 H7-B development shadow run；
- `HOLD`：当前回合读取 development payload；
- `HOLD`：H7-C 六臂训练、official calibration/test、NavDream/target；
- 离散 shadow panel 不是 continuous footprint certificate，regret/flip 也不是驾驶安全收益；
- 即使 H7-B `SUPPORTED`，也只说明 endpoint relevance，结构创新仍必须由 H7-C 公平消融证明。

