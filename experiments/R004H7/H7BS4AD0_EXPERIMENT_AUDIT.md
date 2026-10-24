# R004H7-BS4A-D0 实验完整性审查

状态：`MECHANICAL_PASS__POSTHOC_ATTRIBUTION__EXTERNAL_REVIEW_UNAVAILABLE`。

- provenance：输入绑定H7BS4A结果、原配置、H7BS1 fit result、源日志与panel hashes；
- forbidden inputs：teacher、field、DINO值、outcome及development以后数据均未读取；
- scope：42 groups、672 anchors、3864 pairs全部保留，不只审查两个失败样本；
- statistics：只报告精确集合交、差、对称差与支持计数，无自归一化性能分数或事后阈值；
- replay：独立入口重新生成全部672 ledgers与summary，完全一致，0 failure；
- external review：规定的外部reviewer不可用，未以内审冒充独立科学复核。

本结果只能否定“公共区域解释全部失败”和“difference-only直接证据门足够”这两个简单解释。
它不能证明部分可观测不确定集合方案正确，也不能授权teacher、训练或决策收益主张。
