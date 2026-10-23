# AORTA-Drive Roadmap

## Completed

- [x] 建立顶会风格研究仓库与 CI
- [x] 完成 AORTA-v8 五轮独立理论审查
- [x] 冻结 v8.3.1 V0 条件理论规范
- [x] 用 v8.3.2 asymmetric outer-cover/native-interior 合同修正连续几何 lifting
- [x] 明确标准工具、风险量词、geometry contract 与公平 baseline
- [x] 补齐 continuous teacher、node-label map error 与 token-to-node/candidate-provider 理论合同
- [x] 将主风险升级为所有授权干预达到 \(\gamma_{task}\) 的 relative-margin event
- [x] 将 affine panel 路线降级为历史 baseline

## Next — freeze Gate B inputs

- [x] 使用 `experiment-plan` 冻结首次资格配置
- [x] 冻结 NAVSIM ROI、resolution 与 graph size（R002G：64×64、1 m、4096 nodes、8064 edges）；\(T(Z)\) 留待 Gate B protocol freeze
- [ ] 冻结多相机/历史帧 token-to-node 投影、missing-evidence mask 与可见性协议
- [ ] 冻结可复现的 source-only candidate provider
- [x] 设计并 synthetic 验证 proof-carrying asymmetric geometry checker
- [x] 定义 continuous polygon teacher、\(\varepsilon_{map}\) 与 geometry arithmetic 的保守证明方案
- [ ] 指定 Gate B 的合法 source-only predictor 或最小 probe
- [ ] 固定具体 CDF/joint-field 最近邻
- [ ] 核验 calibration sessions 与 train/development/test 互斥且满足 \(n\ge\lceil1/\alpha\rceil-1\)

## Gate A implementation

- [x] graph uncertainty LP oracle
- [x] outward-safe multi-source min-plus implementation
- [x] common-shift、nonzero-label、zero-edge、chain、cycle、grid tests
- [x] equality/inclusion set cases
- [x] analytic sweep-cover property tests（R002C clean revision 通过）
- [x] exact polygon teacher 与 map-error ledger synthetic tests（R002D clean revision 通过）
- [x] NAVSIM cache/source/vehicle/polygon interface audit 与 rotation-free geometry arithmetic（R002E clean revision 通过）
- [x] ego-local graph node 到 global raw polygons 的 transform interval 与 \(\varepsilon_{transform}\) primitive（R002F）
- [x] 冻结 ROI/resolution 并生成完整 raw-cover node-label/error manifest（R002G：164/164 caches、671,744 nodes）

## Gate B/C qualification

- [ ] held-out source 米制 bound sharpness
- [ ] exact global-gauge 与 local graph uncertainty-only control
- [ ] meaningful continuous non-inclusion definition and audit
- [ ] 分别报告 \(S_{better}\)、\(R_{cert|better}\)、\(R_{opportunity}\) 与 \(R_{selected}\)
- [ ] stop/go decision before formal training

## Only after Gate A/B pass

- [ ] minimal node-potential decoder
- [ ] session calibration pipeline
- [ ] end-to-end legal baseline layer
- [ ] frozen planner/candidate transfer
- [ ] full NAVSIM evaluation plan
- [ ] experiment-integrity audit before claims
