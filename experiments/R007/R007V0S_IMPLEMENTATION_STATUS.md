# R007V0S 实现资格状态

状态：`LOCAL_MECHANICAL_CORE_PASS__REAL_FORWARD_HOLD`

已实现 `aorta_drive.vision.r007_interface_audit` 与
`aorta_drive.vision.constant_eye_runtime`：

- hybrid 模型的完整 2×2 有限差分与非加性交互闭合；
- DINO-only 模型的拓扑有序 stage interchange 合同；
- identity exactness、finite、shape、path fingerprint、donor derangement 检查；
- 对路径串扰、路径别名、非有限值、shape 错位、错误 stage 顺序、伪 full-cut 的反例测试。
- 对 Constant Eye TransFuser-DINO 的 raw tokens、projector output、decoder key/value、trajectory query
  四个完整切口做 topology-faithful capture/replacement；不虚构 native visual path。

验证：

- 四个相关测试文件合计：`19 passed`；
- Python compileall：PASS；
- 四个新增源码/测试文件 strict mypy：PASS；
- 当前环境未安装 Ruff，因此未声称 Ruff PASS。

真实 Constant Eye 两锚点 forward 仍被训练 checkpoint 缺失阻塞；该状态不构成科学实验完成。

