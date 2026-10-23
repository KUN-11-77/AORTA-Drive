# AORTA-Drive Roadmap

## Completed

- [x] 建立顶会风格研究仓库与 CI
- [x] 完成 AORTA-v8 五轮独立理论审查
- [x] 冻结 v8.3.1 V0 条件理论规范
- [x] 明确标准工具、风险量词、geometry contract 与公平 baseline
- [x] 补齐连续 teacher/node-label map error、token-to-node evidence 与 candidate-provider contract
- [x] 将主风险升级为所有授权干预达到 \(\gamma_{task}\) 的 relative-margin event
- [x] 将 affine panel 路线降级为历史 baseline

## Next — experiment planning only

- [ ] 使用 `experiment-plan` 冻结首次资格配置
- [ ] 确定 NAVSIM ROI、resolution、graph size 与 \(T(Z)\)
- [ ] 冻结多相机/历史帧 token-to-node 投影、missing-evidence mask 与可见性协议
- [ ] 冻结可复现的 source-only candidate provider
- [ ] 设计 proof-carrying internal geometry checker
- [ ] 定义 \(\varepsilon_{map}\) 与 \(\varepsilon_{num}\) 的保守测量/证明方案
- [ ] 指定 Gate B 的合法 source-only predictor 或最小 probe
- [ ] 固定具体 CDF/joint-field 最近邻
- [ ] 核验 calibration sessions 与 train/development/test 互斥且满足 \(n\ge\lceil1/\alpha\rceil-1\)

## Gate A implementation

- [x] graph uncertainty LP oracle
- [x] outward-safe multi-source min-plus implementation
- [x] common-shift、nonzero-label、zero-edge、chain、cycle、grid tests
- [x] equality/inclusion set cases
- [ ] production geometry certificate property tests（synthetic rectangle route 已通过）

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
