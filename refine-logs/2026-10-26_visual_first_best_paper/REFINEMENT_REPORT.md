# Refinement Report

- 原方案风险：activation patching + Jacobian/SVD + DINO adapter 拼装。
- 修订后核心：同一数学对象贯穿 mechanism target、DINO recoverability 和 final repair。
- 复杂度：一个冻结基 + 一个空间 coefficient reader；planner/DINO 全冻结。
- 下一步：只实现 R006V0S/R006V0；正结果前不训练 CPTR。
- 真实接口复核修正：shifted点 \(J_a\)、有限干预桥、C3 status mask、C4 query语义、科学图/部署图等价门。
- 最终协议修正：精确site-mask投影与full-patch归一化SESOI，拒绝“极小但显著”的介导主张。
- 当前裁定：`DESIGN_READY 8.9/10`；理论已达到可证伪资格，不等于ICCV/Best Paper方法已成立。
