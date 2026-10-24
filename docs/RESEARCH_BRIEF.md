# AORTA-Drive Research Brief

> 状态：AORTA-v8.3.2 V0 的 Gate A 与 R004E-A/B/C 已完成。R004H4P4E 闭合了128个 fit anchors 的官方执行接口。R004H4P5I 的8组 realized-geometry 哨兵实现门通过，但36个可计算候选的双向 non-inclusion 下界全为0。R004H4P5D 随后证明这是证书分辨率问题：36/36 前序零证据候选都落在充分 no-witness barrier 内，实际最大配对后轴距离1.447620米，而最小屏障4.495949米；机械审计通过。因此 rotation-free disk non-inclusion 被退役。当前冻结为两条并行、互不替代的验证线：R004H4P5R 只允许在相同8个 fit sentinels 上完成一次带停止规则的方向感知几何终止资格；R004H5 立即在42个 fit recording groups 上验证冻结 QuotientFieldPredictor 的同场景受控外观稳定性与任务信息保留。development、official calibration、test 与 target 继续封存。正式 coverage、真实天气/OOD、驾驶收益、Gate B 与创新性仍未成立。

## 1. 问题锚点

AORTA-Drive 研究 NAVSIM v1/v2 上的 source-only appearance robustness。对同一物理场景与驾驶意图，外观变化可能改变冻结端到端 planner 的输出，却不应改变道路可行结构。

核心问题是：

> 能否学习一个与候选无关的道路差分场，在绝对安全零点不可靠时，为任意冻结 planner 的 challenger 相对 native 提供 nominal road-margin non-degradation 证书，并只在证书成立时干预？

## 2. 核心假设

完整道路 signed-distance field 的共同 offset 可能在视觉退化下不可辨识，但局部 clearance differences 可能仍可学习。若 source session 上的 graph-gradient residual 能被同时校准，则一个 field coverage event 可以覆盖任务图内所有自适应 swept-set pairs，并经 challenger 外覆盖/native 内点的非对称几何合同，认证每次授权干预达到冻结的相对改进阈值 \(\gamma_{task}\ge0\)。

该假设可能失败；遮挡、全局 calibration maximum、长路径累积和图错设都可能令证书失去非平凡性。即使证书有效，冻结 candidate provider 也可能没有提供更优候选；candidate supply 与 certificate recall 必须分别报告。

## 3. 冻结范围

- NAVSIM v1/v2；
- frozen DINOv3-H+ 与被审计 planner；
- source-only train/development/calibration；
- 一个具有冻结多相机节点投影/缺失证据协议的 candidate-free node-potential decoder；
- V0 fixed physical scale control 与 v8.4 candidate-independent observation-and-edge scale candidate、single native、exact multi-source solver；
- 预注册 ego-centric ROI、grid、continuous teacher/map error ledger、geometry checker、candidate provider、资格层与 session time set；
- 默认保留 native，仅认证 relative nominal-road non-degradation。

## 4. Non-goals

- 不恢复完整 HD map 或绝对道路安全零点；
- 不联合训练或修改原 planner；
- 不把道路 margin 等同于碰撞、规则、舒适性、真实执行或完整闭环安全；
- 不把 source conformal coverage 写成任意 appearance OOD 保证；
- 不把 DINO、SDF、split conformal、difference constraints、Dijkstra 或 min-plus transform 单独宣称为创新；
- 不在理论/idea 阶段启动 pilot 或长跑实验。

## 5. V0 条件理论链

\[
\text{source-session edge event}
\Rightarrow
\text{all-pair discrete robust bounds}
\Rightarrow
\text{map/geometry/numerical correction}
\Rightarrow
\text{all authorized interventions reach }\gamma_{task}.
\]

既有观察只支持继续研究任务空间接口，不证明 v8.3.2 的局部差分假设、校准锐度、系统收益或创新性。

## 6. 验收门

1. **Gate A:** 数学 oracle、multi-source 实现、geometry checker 和数值误差方向全部通过；
2. **Gate B:** 合法 source-only predictor 产生可接受的 session-tail residual 与非平凡米制证书，并分离 candidate supply、certificate recall 与 selection loss；
3. **Gate C:** 收益存在于具有连续几何意义的 non-inclusion candidates；
4. **Gate D:** 在同一 field event 下迁移到未见 candidate policy/frozen planner；
5. 完成 uncertainty-only 与 end-to-end 两层公平 baseline；
6. 完成独立 novelty-check。

Gate A/B 失败则不进入完整评测；公平 baseline 或查新失败则降级论文定位，不追加模块挽救主张。

## 7. 当前裁定

- V0 条件理论规范：可冻结；
- Gate A primitive/生成器：通过；R004 anchor-aligned teacher instantiation：R004E-A 通过；
- 实际系统：未验证；
- 顶会方法原创性：未授权；
- 下一阶段 A（视觉机制）：R004H5 已预注册但尚未执行。冻结 R004H1 checkpoint、DINO、投影与 teacher，只读42个 fit groups 的原始图像。对严格配对的同一 scene/camera/history 施加三种确定性光度变换，在 edge domain 报告 gauge-invariant appearance distance、teacher task error、zero/mean-prior 与 pooled/uniform evidence probes。低 appearance distance 必须与非塌缩和任务信息保留同时成立；这些变换不冒充真实 rain/snow/dusk，也不支持 held-out/OOD 或驾驶收益主张。
- 下一阶段 B（几何终止资格）：R004H4P5R 已预注册但尚未读取数据。方向感知理论原语和合成反例已实现并通过243项仓库测试。该实验冻结相同8个 fit sentinels、候选、1米 teacher 图、阈值与 ROI；比较独立 polygon diagnostic、退役 disk control、方向矩形在固定分辨率和相位上的严格下界。若 polygon diagnostic 有差异而最细网格仍全零，停止细化并评估保守 polygon 证书；若几何通过但 teacher improvement 仍全负，停止几何工程并转查 provider/task margin；两项资格未通过前禁止扩到128 anchors。
