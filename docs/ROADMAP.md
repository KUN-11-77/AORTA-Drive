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
- [x] 冻结 source recording-group split 与有限 \(T(Z)\)（R004A：162 groups，16 anchors/group；40 calibration sessions）
- [x] 冻结多相机/历史帧 token-to-node 投影、missing-evidence mask 与投影支持协议（R004B）
- [x] 冻结可复现的 source-only candidate provider（R004C：deterministic constant-acceleration/curvature library）
- [x] 设计并 synthetic 验证 proof-carrying asymmetric geometry checker
- [x] 定义 continuous polygon teacher、\(\varepsilon_{map}\) 与 geometry arithmetic 的保守证明方案
- [x] 指定并结构审计 Gate B 的合法 source-only quotient-field probe（R004D）
- [x] R004E-A：为 1440 个 train/development anchors 生成一一对应的 metric cache 与 node labels
- [x] R004E-B：生成冻结 DINO feature/projection artifacts 并完成 exact feature-label join
- [x] R004E-C：按唯一注册配置完成 20 epochs，并冻结 development-only epoch-1 checkpoint；保留 train objective 与 development group-tail 反向变化的负面结果
- [x] R004E-D：visual-necessity control FAIL；checkpoint 优于 train-label mean prior，但差于约 1 m 的 zero-field endpoint
- [x] R004E-E：0/24 gradient overshoot、9/24 missing-edge、15/24 fully observed；单独 Lipschitz 投影与二值观测 mask 均被否定
- [x] R004E-F：24/24 组支持“平均视觉信号存在但极值目标失配”；排除全局符号反转与完全无信号
- [ ] R004E-G：冻结 observation-conditioned heterogeneous edge envelope 的定理、架构、目标和 synthetic falsification gates
- [ ] 固定具体 CDF/joint-field 最近邻
- [x] 核验 calibration sessions 与 train/development/test 互斥且满足 \(n\ge\lceil1/\alpha\rceil-1\)（主 \(\alpha=0.10\)，次 \(\alpha=0.05\)）

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
