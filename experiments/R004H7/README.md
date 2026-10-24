# R004H7 — Decision-Quotient Structure Necessity

R004H7 检验 AORTA 的核心结构是否必要，而不是继续优化 R004H6 的视觉 MAE。

执行顺序固定为：

1. H7-A：数学资格与无数据 oracle；
2. H7-B：冻结模型的只读 shadow decision diagnosis；
3. H7-C：仅在 A/B 通过后启动容量匹配训练。

H7-A 已通过无数据 oracle 与机械复核。H7-B runner、候选面板、group-first bootstrap
与独立机械重算审计器已经通过实现资格，完整仓库281 tests通过；24-group development
payload 仍未读取，科学状态仍为 `HOLD`。H7-C 不得训练。完整合同见 `THEORY.md`、
`PREREGISTRATION.md`、`RESULTS_H7A.md` 与 `H7B_IMPLEMENTATION_QUALIFICATION.md`。
