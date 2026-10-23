# AORTA-v8.4 Round 3 Protocol Amendment

Round 2 的完整 proposal 保持不变，仅冻结以下两项措辞修正。它们是 R004E-G 注册的
控制文本。

## Problem Anchor

- **Bottom-line problem:** 对同一物理道路场，仅外观变化就可能使冻结端到端 planner
  产生更差的 nominal-road margin；系统必须只在 candidate-independent 视觉证据能够
  认证 challenger 相对 native 不退化时干预。
- **Must-solve bottleneck:** R004E-F 证明当前视觉势在 24/24 development groups 上
  降低平均 edge MAE，却在 24/24 groups 上输掉 exact group maximum。统一全图尺度把
  少量高误差边传播成所有路径的共同惩罚，抹去其余区域的有效视觉信号。
- **Non-goals:** 不恢复绝对安全零点；不修改冻结 planner；不声称 collision、规则、
  舒适性、闭环安全或任意 appearance-OOD coverage；不把 conformal、heteroscedastic
  regression、difference constraints 或 Dijkstra 单独包装成创新。
- **Constraints:** NAVSIM v1/v2 source-only；冻结 DINOv3-H+、图、teacher、geometry、
  candidate provider 与官方 calibration/test/target splits；以 recording group 为独立
  单位；新训练前先通过定理和 synthetic falsification gates。
- **Success condition:** 在未读取官方 calibration 的 development qualification 中，
  candidate-independent 异质 edge envelope 必须保持 group-level tail 可靠性，同时在
  米制 graph bounds 上严格优于 zero-field 与 uniform-scale controls；否则停止该路线。

## Amendment A — Formal exchangeability scope

正式 split-conformal 定理的假设为：在全部训练、模型选择、score construction、图、
时间集合和数值协议冻结的外部信息条件下，

\[
Z_1,\ldots,Z_n,Z_{new}
\]

联合可交换，且同一确定性 upper-score 映射作用于 calibration 与 new session。仅
calibration sample 内部交换不足以推出未来 session coverage。历史 proxy-cal 不满足
本项目要求的未暴露资格，仍只作流程排练。

## Amendment B — One invalid-calibration rule

对 proxy-cal 和未来 official calibration 统一采用以下规则：

> 任一预注册 calibration group 无法产生完整、有限、合法的 upper score，整个 variant
> 的 calibration attempt 立即失败；部署阈值在记录中记为 \(+\infty\)，不得通过删除、
> 替换或依靠 \(k<n\) 的 order statistic 绕过该失败。

因此当前 \(n=16,k=16\) 和未来任意 \(n,k\) 使用同一保守合同。

## Gate Decision

- Independent review score: 8.05/10；
- Paper verdict: REVISE；
- R004E-G synthetic theorem/numerical gate: READY；
- R004H real training: not authorized until R004E-G passes and R004H is separately frozen；
- Official calibration/test/target: remain sealed.
