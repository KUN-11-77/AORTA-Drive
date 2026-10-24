# R004H7-BS4B4R1 implementation preflight

状态：`SERVER_INTERFACE_SMOKE_PASS__FORMAL_RUN_NOT_AUTHORIZED`。

## 已闭合

- 新 core summary 支持每 anchor 可变合法 candidate domain；不再硬编码 7 candidates / 21 pairs；
- 零 challenger anchor 保留为 native-only，gain/pair rate 为 0，不产生 NaN；
- scored indices 必须精确等于冻结 membership mask 导出的顶点集；任何 ineligible score 直接失败；
- pose reconstruction、membership artifact、domain counts、label、sentinel replay 分别作为独立门；
- implementation failure 优先于任何正向 oracle signal；
- recording group 仍为唯一统计独立单位，所有 group mean 固定包含 16 anchors；
- R1 thresholds、bootstrap 与 sensitivity margins 通过配置测试确认沿用 B4，未根据 B4 outcome 调整；
- 13 个 B4/R1 core tests 通过，其中 R1 新增 9 项，覆盖 variable panel、zero panel、mask injection、
  domain drift 与四类 fail-closed gate；
- 全库 `352 passed`，未破坏既有实验合同；
- runner/audit 通过 Python compile，配置内本地 source SHA 均由自动测试核验。

## 服务器 interface smoke

冻结提交 `9664e4b` 已通过离线 archive 同步到服务器；source archive SHA-256 为
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

## 尚未闭合

- 未验证正式 672-anchor membership histogram、candidate/pair totals 与 runner 输出；
- 未运行 primary、replay 或 audit；
- 外部 reviewer unavailable；
- 未读取新的 source outcome，更未读取 development、calibration、test、NavDream 或 target。

## 冻结摘要

- config SHA-256：`ed0f9969f7f2aa80da94e0f2ebee8f959efd94dbf0bd76e93c1b50636a732ed2`；
- preregistration SHA-256：`2e81cfebd6d12bfa37e379387c386908c8a8d31840a6422956f77337e88fbebb`；
- core SHA-256：`4c3f73fd3cc1b964faf987081065e680567a6802d4464c4f53f24ceb9fe063c7`；
- runner SHA-256：`f263608a5522a76654226023b762b80bbffb3c58cebd3ad0b5be2fc6ddac9001`；
- audit SHA-256：`28f4395d83d1336e7f09b145530ae49a13c71df023952d974c8009511fc81c79`。

## 下一门

下一步是一次 primary + 一次 independent full replay + mechanical audit 的正式 R1。根据 B4 失败后
冻结的停止规则，smoke 通过不自动授权第三次正式运行；在明确授权前保持 HOLD。正式运行期间
禁止读取 partial scientific metrics 或修改 membership、threshold、bootstrap、candidate provider、
scorer 与统计协议。
