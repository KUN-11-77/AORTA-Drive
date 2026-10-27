# R007V0S 实现资格状态

状态：`LOCAL_MECHANICAL_CORE_PASS__REAL_FORWARD_HOLD`

已实现 `aorta_drive.vision.r007_interface_audit`、
`aorta_drive.vision.constant_eye_runtime` 与
`aorta_drive.vision.dino_evidence_layers`、
`aorta_drive.vision.r007_evidence_contract`：

- hybrid 模型的完整 2×2 有限差分与非加性交互闭合；
- DINO-only 模型的拓扑有序 stage interchange 合同；
- identity exactness、finite、shape、path fingerprint、donor derangement 检查；
- 对路径串扰、路径别名、非有限值、shape 错位、错误 stage 顺序、伪 full-cut 的反例测试。
- 对 Constant Eye TransFuser-DINO 的 raw tokens、projector output、decoder key/value、trajectory query
  四个完整切口做 topology-faithful capture/replacement；不虚构 native visual path。
- outcome-free 中间层候选由实际模型深度的固定分位点与最后四层生成；等权层组合保持 token 数和
  通道宽度，不允许通过 concat 扩容制造互补优势。
- task-context timing 四格实验必须在数据、writer切口、输出token、参数、FLOPs、步数与搜索预算上
  exact matched，并机械计算预登记的 risk difference-in-differences；
- writer-target 四臂必须共享同一预算并联合记录 feature error、tail output error 与 task utility，审计器
  只输出有符号对比，不替实验选择阈值或机制标签。

验证：

- 四个相关测试文件合计：`20 passed`；
- Python compileall：PASS；
- 八个目标源码/测试文件 strict mypy：PASS；
- R007 YAML 预注册配置可解析：PASS；
- 本次纯协议测试使用当前可用 Python 3.13.5；本机声明的 `dinov3` 环境入口失效，未修改环境，因而
  不把这轮机械测试冒充真实 DINO runtime 资格。
- 当前环境未安装 Ruff，因此未声称 Ruff PASS。

真实 Constant Eye 两锚点 forward 仍被训练 checkpoint 缺失阻塞；该状态不构成科学实验完成。

