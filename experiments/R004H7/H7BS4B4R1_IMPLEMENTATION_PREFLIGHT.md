# R004H7-BS4B4R1 implementation preflight

状态：`SERVER_PROVENANCE_REPREFLIGHT_PASS__FORMAL_RUN_NOT_AUTHORIZED`。

> 2026-10-25 正式运行授权后、任何 scorer outcome 产生前，用户明确取消固定时间上限。原
> `53f16c1` preflight 仍证明旧 runner/config 的来源闭合，但新的无上限 runner/config 必须重新完成
> hash、合同测试和服务器 preflight，才能启动正式 primary。

> 无上限修订提交 `7050c6e` 已完成服务器重检：source archive SHA-256 为
> `be1829f0...db6456b`，13/13 合同测试与 compile 通过，config/runner/audit/preregistration hash
> exact。正式 primary 已启动；该运行状态不提供任何科学 outcome。

> 2026-10-25 provenance hardening：冻结提交 `53f16c1` 在正式 outcome 前补齐 primary/replay 的
> `status`、`code_commit`、config hash、frozen-input hash、implementation hash、sealed-role 与
> full-PDMS boundary 审计，并按用户要求把 wall cap 从180分钟收紧为120分钟。服务器同一 NAVSIM
> 环境12/12合同测试通过，未调用 scorer、未访问任何科学 outcome。

> 2026-10-25 pre-run source audit 发现旧文档把 official scorer 误写为 `[native, challenger]`
> 联合评分。实际实现对 native 和每个 challenger 分别在固定 `[PDM baseline, trajectory]` proposal
> set 中评分。该勘误发生在 formal R1 前且未读取 outcome；runner/audit 已增加
> `PROGRESS_MODE=eval` fail-closed 合同。下述 `9664e4b` server smoke 仍是合法的候选域证据，但其
> source bundle 已被新 runner/audit 取代；当前提交 `08081cc` 已完成下述服务器语义复检。

## 已闭合

- 新 core summary 支持每 anchor 可变合法 candidate domain；不再硬编码 7 candidates / 21 pairs；
- 零 challenger anchor 保留为 native-only，gain/pair rate 为 0，不产生 NaN；
- scored indices 必须精确等于冻结 membership mask 导出的顶点集；任何 ineligible score 直接失败；
- pose reconstruction、membership artifact、domain counts、label、sentinel replay 分别作为独立门；
- implementation failure 优先于任何正向 oracle signal；
- recording group 仍为唯一统计独立单位，所有 group mean 固定包含 16 anchors；
- R1 thresholds、bootstrap 与 sensitivity margins 通过配置测试确认沿用 B4，未根据 B4 outcome 调整；
- 15 个 B4/R1 core tests 通过，其中 R1 新增 11 项，覆盖 variable panel、zero panel、mask injection、
  domain drift、四类 fail-closed gate 与独立 summary/audit 复算；
- 全库 `354 passed`，未破坏既有实验合同；
- runner/audit 通过 Python compile，配置内本地 source SHA 均由自动测试核验。

## 服务器 scorer-semantics re-preflight

冻结提交 `08081cc` 以离线 archive 同步，archive SHA-256 为
`31fb7f8f5919fcfaccf6b9a5c8faefce7e9e932135c8e1eb92ecdf8bd8451764`。服务器同一 NAVSIM 环境中：

- 当前 R1 合同测试 `11/11` 通过；
- 本地 implementation hashes 与 config exact；官方 NAVSIM source hashes 与 config exact；
- `PROGRESS_MODE` 未设置时解析为 `eval`，显式 `eval` 通过，`gen_gt` 在 scorer 调用前 fail-closed；
- estimand 明确为 fixed metric-cache PDM baseline，`native_and_challenger_jointly_scored=false`；
- scorer 未调用，scientific outcomes 未访问，development/calibration/test/NavDream/target 未访问；
- semantic preflight artifact SHA-256：
  `073783d925540c1829faefa34233de8bd03eb5b6526a500e73724478def8add1`；
- local artifact：
  `F:\AORTA-Drive-artifacts\gate-b\R004H7\r004h7bs4b4r1_preflight_08081cc\semantic_preflight.json`。

旧 `9664e4b` payload interface smoke 与 roster audit 仍用于候选域证据，因为 core、membership、
candidate provider、NAVSIM execution 与 B4 dependency hashes 未变；新提交只修正 estimand 文字并在
runner/audit 增加 progress-mode fail-closed 合同。

## 服务器 provenance re-preflight

冻结提交 `53f16c1` 的最小 delta bundle SHA-256 为
`3cd759512916eea5d3e9b73207cd15dedaf60e84288168010a862b840f5c779f`。服务器同一 NAVSIM
环境中：

- R1 合同测试 `12/12` 通过；
- runner/auditor Python compile 通过；本地 strict mypy 同样通过；
- config、runner、audit hashes 与冻结值 exact；
- 当时 formal wall cap 为120分钟；现已由开封前用户指令替换为无固定 wall cutoff，待重新 preflight；
- 新机械审计会拒绝 primary/replay 的 commit、config、frozen input、implementation source、
  sealed-role 或 full-PDMS boundary 漂移；
- scorer 未调用，scientific outcomes 未访问，development/calibration/test/NavDream/target 未访问；
- preflight artifact SHA-256：
  `4477fe2799bc71145d3a8cee21d55611328db99f5f641d73e07685aaf48b9191`；
- server artifact：
  `/data/khsong/aorta_runs/r004h7bs4b4r1_preflight_53f16c1/preflight.json`；
- local mirror：
  `F:/AORTA-Drive-artifacts/gate-b/R004H7/r004h7bs4b4r1_preflight_53f16c1.json`。

## 服务器 interface smoke

历史冻结提交 `9664e4b` 已通过离线 archive 同步到服务器；source archive SHA-256 为
`5132bfa7b1d751b25e0d779275ea6b89f6af871efee4420dc65cffca8bfb9419`。服务器同一环境中：

- R1 的 9 个合成/合同测试全部通过；
- 仅按预 B4 membership 选择 anchor-0 sentinel、零 eligible、部分 eligible、全 eligible 四类
  interface cases；sentinel 与 full 恰为同一个预定义 anchor，因此实际覆盖 3 个唯一 anchors；
- zero case：`eligible=0`、`scored=1`；partial case：`eligible=4`、`scored=5`；full/sentinel：
  `eligible=6`、`scored=7`；
- 每类 `scored_indices_equal_legal_domain=true`、pose bit-exact、membership exact、
  `ineligible_candidate_score_count=0`、sentinel replay exact；
- smoke 没有写出 candidate scores、oracle gain、margin、hard outcome 或任何 aggregate；
- smoke artifact SHA-256：
  `55813b59c0985689744cb55b8ae901bcf1d92793c9bca98b24da0b6f20634ce2`；
- local artifact：
  `F:\AORTA-Drive-artifacts\gate-b\R004H7\r004h7bs4b4r1_preflight_9664e4b\smoke.json`。

随后完成 outcome-free 全 roster 审计：

- 672/672 anchors、42/42 groups × 16 anchors；panel identity、SHA、mask shape/dtype 均零失败；
- membership histogram：`0:15, 2:8, 3:4, 4:13, 5:8, 6:624`；
- observed 与 preregistered expected 完全一致：3,864 eligible challengers、657 nonzero/15 zero
  anchors、4,536 scored candidates、13,402 complete-graph pairs；
- 每组 eligible challenger 总数范围为 80–96；
- scorer 未调用、scientific outcomes 未访问、封存数据未访问；
- roster audit SHA-256：
  `035f15a3a8be5e7afdcedf06ad1fff334c5e25a10bde0c124ee2ea5b3f88828a`；
- local artifact：
  `F:\AORTA-Drive-artifacts\gate-b\R004H7\r004h7bs4b4r1_preflight_9664e4b\domain_roster_audit.json`。

## 尚未闭合

- 未验证正式 672-anchor scorer outputs；membership roster 与 candidate/pair totals 已闭合；
- 未运行 primary、replay 或 audit；
- 外部 reviewer unavailable；
- 未读取新的 source outcome，更未读取 development、calibration、test、NavDream 或 target。

## 历史冻结摘要（已被 scorer-semantics amendment 取代）

- config SHA-256：`ed0f9969f7f2aa80da94e0f2ebee8f959efd94dbf0bd76e93c1b50636a732ed2`；
- preregistration SHA-256：`2e81cfebd6d12bfa37e379387c386908c8a8d31840a6422956f77337e88fbebb`；
- core SHA-256：`4c3f73fd3cc1b964faf987081065e680567a6802d4464c4f53f24ceb9fe063c7`；
- runner SHA-256：`f263608a5522a76654226023b762b80bbffb3c58cebd3ad0b5be2fc6ddac9001`；
- audit SHA-256：`28f4395d83d1336e7f09b145530ae49a13c71df023952d974c8009511fc81c79`。

## 当前冻结摘要

- source contract commit：`53f16c1`；
- config SHA-256：`7a79e85b81c89b2741f8a6784612cf4233effbbd5fa43d75a9fa54d483ad9c27`；
- preregistration SHA-256：`9db3afe2069936cb87d8c6a074ee205fe347c952cccbd801dd42290b59ce3bab`；
- core SHA-256：`4c3f73fd3cc1b964faf987081065e680567a6802d4464c4f53f24ceb9fe063c7`；
- runner SHA-256：`6ac7e7f575631ede5a0d15c20f8cd5013419c4a19230b02db201773fa5262902`；
- audit SHA-256：`43389f6725f893b9775f53bec8c1ba393b15bef7b944b2f42b3101091ba19333`；
- 历史 maximum wall time：120分钟；正式运行使用后续无固定 wall cutoff 修订。

## 下一门

下一步是一次 primary + 一次 independent full replay + mechanical audit 的正式 R1。根据 B4 失败后
冻结的停止规则，smoke 通过不自动授权第三次正式运行；在明确授权前保持 HOLD。正式运行期间
禁止读取 partial scientific metrics 或修改 membership、threshold、bootstrap、candidate provider、
scorer 与统计协议。
