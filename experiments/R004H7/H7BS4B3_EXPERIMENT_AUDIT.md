# R004H7-BS4B3 Experiment Audit

**Date:** 2026-10-25  
**External reviewer:** `REVIEW_UNAVAILABLE`  
**Mechanical verdict:** `PASS`  
**Scientific integrity verdict:** `PASS_WITH_NEGATIVE_SCIENTIFIC_RESULT`

- **Ground-truth provenance — PASS（限定对象）：** exact-token frozen NAVSIM forward96 teacher；
  它是 nominal-road teacher，不是 planner outcome 或现实安全真值。
- **No score normalization — PASS：** exact integer footprint sets、binary64 minima 与 exact-zero
  contrasts；没有预测依赖归一化、epsilon tie 或结果后阈值。
- **No phantom result — PASS：** primary、replay、audit、logs 与 bundle 均已保存；672 ledgers、
  summary 和 root contract exact，failure count 0。
- **Scope completeness — PASS：** 42 groups、672 anchors、3,864 pairs 与 B2 的 2,033 common-min
  identities 全部复现。
- **Design integrity — PASS：** temporal first-fork 在运行前冻结为唯一 primary；spatial 只作
  sensitivity，未因其数值更好而改写主判据。recording group 是独立统计单位。
- **Role isolation — PASS：** `forbidden_payload_accessed=false`；未读取 learned field、DINO、outcome、
  development、calibration、test、NavDream 或 target。
- **Identifiability — PASS_WITH_SCOPE:** branch contrast 对共同 gauge 不变；absolute teacher zero 仅作
  oracle context，未被写成 quotient model 可提供的安全证书。
- **Selection-bias threat — FAIL FOR SPATIAL GENERALIZATION:** spatial-exclusive 仅对 77.13% pairs
  双侧非空；不能用 complete-case 子集替代一般 panel 结论。
- **Scientific gate — FAIL:** temporal group-equal recovery CI `[0.0634,0.1029]` 未超过 0.5；spatial
  validity 未达到 0.95。按预注册规则停止当前一般 reranker 路线。

结论：实验真实性与机械完整性通过，科学假设不通过。H7C/network/OOD/decision evaluation 继续
HOLD；不得通过事后选 spatial operator、删除 nested pairs 或降低门槛制造正结论。
