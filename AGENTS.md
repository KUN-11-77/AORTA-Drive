# AORTA-Drive 协作入口

## 项目上下文

- 研究任务首先阅读 `docs/RESEARCH_BRIEF.md`、`docs/AORTA_FIRST_PRINCIPLES_THEORY_V1_CN.md`、`docs/BEST_PAPER_IMPLEMENTATION_STANDARD_V1_CN.md` 与 `docs/CLAIM_EVIDENCE_MATRIX.md`。
- 当前 canonical 理论入口是 `docs/AORTA_FIRST_PRINCIPLES_THEORY_V1_CN.md`。核心对象是合法 ragged 候选域上的 native-relative contextual relation、appearance-interventional differential evidence 与条件性保守仲裁；旧 graph-gradient field、SDF、scalar set-min、几何证书和 affine panel 均为历史证据或 baseline。
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
- 所有新增模块、训练、评估和论文主张必须通过 `docs/BEST_PAPER_IMPLEMENTATION_STANDARD_V1_CN.md` 的对应硬门；未通过只能标记为探索性工作。

## 方法边界

- 不把 trajectory query、conformal、privileged teacher、abstention、Set Transformer 或 DINO 单独宣称为创新。
- 不把 source calibration 宣称为任意 OOD 安全保证。
- 不把 no-op/fallback 宣称为安全动作。
- 不把道路可行性证书扩展为碰撞、交通规则或闭环系统安全。
- calibration set 不得用于 checkpoint、阈值、候选规则或架构选择；appearance target 只报告 empirical results。
- AORTA 不学习 candidate generator；候选来源、数量、时域、预算与 provenance 必须在 calibration 前冻结。

