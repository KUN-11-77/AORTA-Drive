# R004H7-BS4B4R1 implementation preflight

状态：`LOCAL_SYNTHETIC_QUALIFICATION_PASS__SERVER_SMOKE_NOT_RUN__FORMAL_RUN_NOT_AUTHORIZED`。

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

## 尚未闭合

- 未在服务器 NAVSIM 环境执行单 anchor smoke；
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

下一步只能先把冻结提交同步到服务器，运行一个不输出科学 aggregate 的资格 smoke，检查：

1. pose exact；
2. membership artifact exact；
3. scored indices 等于 legal domain；
4. ineligible score count 为 0；
5. sentinel replay exact。

smoke 失败只允许修实现错误并更新 source hash、提交与 preflight；不得读取 oracle aggregate。
smoke 通过也不自动授权第三次正式运行。
