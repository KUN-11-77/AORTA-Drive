# AORTA-Drive Research Brief

> 状态：AORTA-v8.3.2 V0 的 Gate A 与 R004E-A/B/C 已完成。R004H2R1 与 R004H3 通过各自工程门。R004H4P2S 否定了缓存动态速度字段，R004H4P4E 在128个 fit anchors 上闭合官方0.1秒 LQR+bicycle 执行接口。R004H4P5I 的8组 realized-geometry 哨兵实现门通过：8/8 有 realized supply，6/8 完成连续几何与 teacher bound，2/8 因真实轨迹超出冻结 ROI 被拒绝；但6组共36个可计算候选的双向非包含下界全为0，teacher improvement 下界全为负。该现象尚不能解释为“没有机会”，因为 rotation-free 外接圆与后轴内切圆在1米网格上存在约4.496米的结构性无证据区。外部独立审查仍不可用。当前应先做 fit-only 证书分辨率诊断，再决定是否扩到128 anchors；不得进入 development。正式 coverage、Gate B、OOD、性能与创新性仍未成立。

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
- 下一阶段：R004H4P5D 已按 revision `f017ef3` 注册。它把 R004H4P5I 的全零 non-inclusion 作为 post-result mechanism signal，而不是方法阴性结果；在相同8个 fit sentinels 上核验 disk outer / inner-cell 组合的解析 no-witness separation barrier，并记录官方 realized candidate 与 native 的逐时刻最大后轴距离。若全部零证据候选均落在解析屏障内，则在128-anchor扩展前退役该 disk non-inclusion 架构，转向方向感知矩形外覆盖；不得调整 provider、资格阈值或读取 development、calibration、test、target-domain 数据。
