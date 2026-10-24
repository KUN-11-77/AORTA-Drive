# R004H7 — Decision-Quotient Structure Necessity

R004H7 检验 AORTA 的核心结构是否必要，而不是继续优化 R004H6 的视觉 MAE。

执行顺序固定为：

1. H7-A：数学资格与无数据 oracle；
2. H7-B：冻结模型的只读 shadow decision diagnosis；
3. H7-C：仅在 A/B 通过后启动容量匹配训练。

当前只授权 H7-A 的理论审查与合成 oracle 实现。H7-B 不得访问新 payload，H7-C 不得训练。完整合同见 `THEORY.md` 与 `PREREGISTRATION.md`。
