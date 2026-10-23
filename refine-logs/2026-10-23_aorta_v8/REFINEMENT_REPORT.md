# AORTA-v8 Refinement Report

## Goal

把此前以 affine panel、候选打分和绝对 anchor 为中心的方案，重新建立为一套数学对象、视觉架构、训练目标、统计校准、几何修正和仲裁风险彼此一致的首版系统。

## Refinement path

### Initial proposal — GCCO

使用 trajectory scalar potential 和 pair contrast 做 session calibration。审查发现固定尺度时它退化为普通 scalar relative score，learned scale 只是 normalized conformal；候选监督也无法识别局部 support potential。

### First pivot — GCMF

改为共享节点场、hard-min set operator 和 additive-gauge sup error。该版本首次取得 candidate-policy-independent 的全场事件，但全域 residual range 会被无关坏区域主导，且没有利用集合包含关系。

### Second pivot — GQCF

改用局部 graph-difference uncertainty polytope，并推导 exact worst-case set-min contrast。审查确认定理正确，同时指出它是标准 min-plus Lipschitz envelope；还发现连续 geometry 单侧扣项缺少 native 采样方向，以及逐帧尺度归一化删除整体观测质量。

### Frozen version — CMFE / AORTA-v8.3

固定一个 node-potential decoder、固定物理 edge scale、source-session calibration、一次精确 multi-source transform、proof-carrying nominal geometry gate 和 native-preserving arbitration。learned scale 与路径近似移出 V0。贡献措辞从“新图算法”降级为待验证的结构化道路仲裁接口。

### Independent amendment — AORTA-v8.3.1

独立审查后补齐五个原规范尚未闭合的合同：continuous teacher 到 node label 的 map-error budget；DINO token 到物理节点的投影与 missing-evidence 规则；冻结且可复现的 source-only candidate provider；所有授权干预达到 \(\gamma_{task}\) 的 relative-margin 风险事件；candidate supply、certificate recall、certifiable opportunity 与 final selection 的分解。meaningful non-inclusion 改用连续 footprints 的双向差面积，appearance graph-gradient stability 仅保留为经验诊断。复审结论为 **Freeze YES**，但没有运行实验，也没有提升论文原创性结论。

## Major design decisions

- 学习 candidate-free 共享物理场，不学习候选列表内的独立 scalar scores；
- 只监督 graph gradients，主动放弃不可辨识 absolute offset；
- calibration 随机对象是 source session 内所有声明时刻与边；
- 对 uncertainty set 直接求 exact relative task bound；
- continuous geometry 必须通过 sound checker，不能由有限采样经验误差代替；
- V0 采用 fixed scale，避免 learned uncertainty 先于必要性证据；
- 所有强 baseline 使用各自合法训练与 exact relative readout；
- 系统正确性、实证效用与论文原创性分开验收。

## Reviewer conclusion

五轮独立审查后的最终分数为 7.60/10。审稿人认可 V0 条件理论规范可冻结并进入 experiment-plan 与 Gate A/B，但拒绝授权顶会原创性、系统有效性、实时性或 OOD 保证。该结论被完整保留，没有把 REVISE 改写成 READY。

## Next handoff

下一阶段只执行 Gate A/B：先冻结 NAVSIM ROI/resolution/graph size、continuous teacher/node projection、candidate provider、合法 Gate B predictor、geometry/map checker 与数值误差预算。未通过前，不启动完整训练或目标域评测。

