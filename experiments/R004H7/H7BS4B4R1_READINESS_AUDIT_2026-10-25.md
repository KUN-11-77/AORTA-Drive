# R004H7-BS4B4R1 Formal Readiness Audit

审计日期：2026-10-25。

裁决：**AUTHORIZED__UNBOUNDED_RESOURCE_AMENDMENT_LOCAL_PASS__SERVER_REPREFLIGHT_PENDING**

## 1. 本地来源状态

- branch：experiment/gate-b-protocol；
- 原 readiness 审计提交：34d7084；
- 2026-10-25 投稿版理论修订只更新科学解释边界，不改 runner、config、candidate、threshold、seed、
  wall cap 或冻结输入；
- 工作树：clean；
- preregistration 状态：PREREGISTERED_NOT_RUN；
- implementation preflight 状态：SERVER_PROVENANCE_REPREFLIGHT_PASS__FORMAL_RUN_NOT_AUTHORIZED；
- formal wall contract：`maximum_wall_minutes: null`，无固定 wall cutoff；
- calibration/test/NavDream/target：保持封存。

### 固定文档哈希

| Artifact | SHA-256 |
|---|---|
| H7BS4B4R1_PREREGISTRATION.md | 0B6454E47D8B43A9E88F549D4D555B03AC842E556A6E596ECD870BA9114C328F |
| H7BS4B4R1_IMPLEMENTATION_PREFLIGHT.md | CACACFEFC70314DF1766992EA30BB9960E02D1F1CEEC33F031A00ABACF41A204 |

旧 preregistration hash
`9DB3AFE2069936CB87D8C6A074EE205FE347C952CCCBD801DD42290B59CE3BAB` 作为历史值保留；新 hash
只来自 formal outcome 前加入的 one-frame/contextual scope addendum。机械 gate 内容未改。

## 2. 服务器只读状态

在服务器 /data/khsong/aorta_runs 下只发现以下预检目录：

- r004h7bs4b4r1_preflight_08081cc；
- r004h7bs4b4r1_preflight_53f16c1；
- r004h7bs4b4r1_preflight_9664e4b。

没有发现名称匹配 r004h7bs4b4r1 或正式 runner 的活动进程，也没有发现正式结果目录。该检查只证明审计时未运行，不授权创建新任务。

## 3. 已闭合的开封前条件

- candidate-local scorer 语义已核对；
- eligible ragged candidate domain 与冻结 membership 已定义；
- zero-challenger anchors 保留；
- PROGRESS_MODE=eval fail-closed；
- primary/replay/audit provenance 与 hashes 纳入 gate；
- 服务器合同测试和 smoke 已通过；
- 无固定 wall cutoff 的资源合同已在任何 scorer outcome 前固定；
- 最终 claim-driven 实验 tracker 中 B4R1 为 HOLD，后续项均 LOCKED。

## 4. 当前启动条件

用户已于 2026-10-25 明确授权正式运行 B4R1，并在 scorer outcome 产生前要求取消固定时间上限。
启动前仍必须使无上限修订后的 runner/config 在服务器完成 hash、合同测试和环境复检。

授权后只允许：

1. 使用现有冻结 commit/config/input/hashes；
2. 启动一次 B4R1 primary/replay/audit；
3. 运行期间不读取 partial scientific metrics 调整任何协议；
4. 超时或失败后不自动第三次运行；
5. 完成后先机械审计，再解释 GO/HOLD/STOP。

GO/HOLD/STOP 的科学解释只适用于 one-frame supervision。接口有效后，contextual target 由独立
预注册裁决；该解释修正不改变 formal B4R1 的运行方式。

## 5. 当前边界

本 readiness audit 不提供 oracle、preference、网络、NavDream、驾驶收益、clean no-harm、实时性或 ICCV readiness 证据。

本地权威工作树现迁移至 `D:/AORTA-Drive-worktrees/gate-b-protocol`；仓库文件与提交历史由同一
Git 远端恢复，机器绝对路径不进入 formal provenance。
