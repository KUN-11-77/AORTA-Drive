# 投稿版理论审核：官方来源复核

日期：2026-10-25。方式：仅通过官方 arXiv/OpenReview/GitHub 直接 URL 获取；未使用搜索引擎。

| 对象 | 官方来源 | 本次状态 | 本地来源 SHA-256 / 备注 |
|---|---|---|---|
| D-JEPA | https://arxiv.org/html/2609.24749 | HTML 正文可读 | `2f04010b…3546` |
| World4Scorer | https://arxiv.org/html/2609.36438 | HTML 正文可读 | `9b7a7e75…7e52` |
| World4Scorer code | https://github.com/World4Scorer/World4Scorer | 代码与 checkpoints 页面可达；README 声明 2026/09 released | `b0da2cf2…c6e4` |
| GTRS | https://arxiv.org/abs/2506.06664 | 官方摘要页可读 | `2616621b…6e4b` |
| GTRS code | https://github.com/NVlabs/GTRS | 官方仓库可达，含训练/推理文档与 checkpoint 入口 | `c921590a…09d5` |
| FlowDrive | https://arxiv.org/abs/2509.14303 | 官方摘要页可读，v1 提交 2025-09-17 | `eb9e03bb…38e3` |
| READ | https://arxiv.org/abs/2609.12371 | 官方摘要页可读，提交 2026-09-11 | `927d0e55…2735` |
| MG-Select | https://arxiv.org/abs/2510.05681 | 官方摘要页可读，当前页为 v2 | `6985a0eb…a8fd` |
| PEA | https://openreview.net/forum?id=7kLNGaAHaw | OpenReview 返回浏览器验证页，本次无法读取正文 | `ac1cb574…1f8`；沿用此前审计，标记未重新核验 |

原始 HTML、可读文本、链接清单与 fetch metadata 保存在 `references/sources/*_recheck*`。本轮只补
来源追溯，不改变已有方法结论。

## 复核边界

- arXiv abstract 只支持标题、版本与摘要级结论；D-JEPA/World4Scorer 的实现细节来自 HTML 正文。
- World4Scorer 与 GTRS 官方仓库可达；本轮没有逐行复现代码或运行 checkpoint。
- FlowDrive、READ 的实现可用性仍按既有审计处理，不能把 paper-derived reimplementation 称为
  official reproduction。
- PEA 官方正文因 OpenReview challenge 未在本轮重新读取，必须在论文提交前通过可访问官方来源
  再核验；不能据此次 fetch 增加任何结论。
- PEA/MG-style 本地负结果只适用于已审计的 driving adaptation，不外推原方法一般无效。
