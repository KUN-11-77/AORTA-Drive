# R004H7-BS4B4R1 Formal Readiness Audit

审计日期：2026-10-25。

裁决：**READY_BUT_NOT_AUTHORIZED__NO_FORMAL_PROCESS_RUNNING**

## 1. 本地来源状态

- branch：experiment/gate-b-protocol；
- 审计时 HEAD：ff2212f；
- 工作树：clean；
- preregistration 状态：PREREGISTERED_NOT_RUN；
- implementation preflight 状态：SERVER_PROVENANCE_REPREFLIGHT_PASS__FORMAL_RUN_NOT_AUTHORIZED；
- formal wall cap：120 分钟；
- calibration/test/NavDream/target：保持封存。

### 固定文档哈希

| Artifact | SHA-256 |
|---|---|
| H7BS4B4R1_PREREGISTRATION.md | 9DB3AFE2069936CB87D8C6A074EE205FE347C952CCCBD801DD42290B59CE3BAB |
| H7BS4B4R1_IMPLEMENTATION_PREFLIGHT.md | A406F6E3A7403E13FF66EDB505409FDFA80FE064055AA54436FAC92E160E66C5 |

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
- 120 分钟异常停止规则已固定；
- 最终 claim-driven 实验 tracker 中 B4R1 为 HOLD，后续 16 项均 LOCKED。

## 4. 尚缺的唯一授权条件

需要用户明确表达授权正式运行 B4R1。普通“继续”、理论讨论或自动 goal continuation 不解除该门。

授权后只允许：

1. 使用现有冻结 commit/config/input/hashes；
2. 启动一次 B4R1 primary/replay/audit；
3. 运行期间不读取 partial scientific metrics 调整任何协议；
4. 超时或失败后不自动第三次运行；
5. 完成后先机械审计，再解释 GO/HOLD/STOP。

## 5. 当前边界

本 readiness audit 不提供 oracle、preference、网络、NavDream、驾驶收益、clean no-harm、实时性或 ICCV readiness 证据。
