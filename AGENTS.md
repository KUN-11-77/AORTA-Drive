# AORTA-Drive 协作入口

## 项目上下文

- 研究任务首先阅读 `docs/RESEARCH_BRIEF.md`、`docs/METHOD_SPEC.md` 与 `docs/CLAIM_EVIDENCE_MATRIX.md`。
- 当前核心对象是候选轨迹面板上的仿射任务算子，而不是完整地图重建或独立轨迹分类器。
- 当前阶段以理论资格、最小实现和可证伪实验为主；不得提前写性能结论。

## 研究约束

- 默认中文沟通，代码、配置键和论文正式术语使用英文。
- 原 planner、数据划分、官方 scorer 与目标域协议必须保持冻结；任何变化需写入 `docs/DECISIONS.md`。
- NavDream/目标域只作最终零样本评估，不用于训练、早停、阈值、增强或架构选择。
- 有意义的运行必须先登记 `experiments/registry.csv`，并在 `experiments/<run-id>/README.md` 保存问题、配置、结果、解释和下一步。
- 大文件只写入 `NAVSIM_EXP_ROOT`；Git 中只保存 manifest、摘要、表格、图和必要的小型 fixture。
- 不原地修改共享 `dinov3` conda 环境；需要新依赖时创建独立环境。
- 涉及联网时使用项目允许的 web-access 工作流；PDF 仅通过 PDF 专用工作流读取。

## 工程约束

- Python 源码采用 `src/` 布局；公共接口必须有类型标注和 docstring。
- 新功能同时添加测试；修复 bug 时先添加可复现失败测试。
- 配置、随机种子、数据 manifest 与代码 revision 必须进入结果 provenance。
- 禁止提交数据、checkpoint、缓存、密钥、账号信息或机器绝对路径。
- 不降低门禁来制造正结果；失败结果进入 claim–evidence 矩阵。

## 方法边界

- 不把 trajectory query、conformal、privileged teacher、abstention、Set Transformer 或 DINO 单独宣称为创新。
- 不把 source calibration 宣称为任意 OOD 安全保证。
- 不把 no-op/fallback 宣称为安全动作。
- 不把道路可行性证书扩展为碰撞、交通规则或闭环系统安全。

