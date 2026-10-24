# R004H7-BS1 固定前缀支持资格结果

## 裁决

`DEVELOPMENT_SUPPORT_PASS`。按照事前冻结规则，在42个fit recording groups上选择出的最大
共同可行前缀为 `k=25`，即2.5秒；该前缀随后在24个historically exposed development
groups上一次性通过覆盖资格门。它只证明近时域查询的定义域闭合，不证明该时域保留完整
候选排序信息；H7-B2继续HOLD，先完成H7BS2候选区分充分性诊断。不授权读取outcome、训练
H7-C或声称驾驶收益。

## Fit选择

| 前缀 | 有效anchors | 达到≥12/16的groups |
|---:|---:|---:|
| 1.0 s | 657/672 | 42/42 |
| 1.5 s | 657/672 | 42/42 |
| 2.0 s | 657/672 | 42/42 |
| **2.5 s** | **622/672** | **42/42** |
| 3.0 s | 558/672 | 37/42 |
| 3.5 s | 521/672 | 29/42 |
| 4.0 s | 491/672 | 25/42 |

因此2.5秒是唯一由冻结“最大可行前缀”规则选出的结果，不是根据decision outcome调出的
时域。

## Development一次性资格

- 24/24 groups、384/384 anchors均保留在分母；
- 360/384 anchors有效；15个为`native_invalid`，9个为`no_valid_challenger`；
- 24/24 groups均达到至少12/16，最差组恰为12/16；
- 未读取视觉场、DINO、teacher、regret、ranking flip、calibration、test、NavDream或target。

## 实现事件与机械审计

首次fit执行完成672个panel artifacts后，辅助intent审计字段含`NaN`，严格JSON结果写入失败。
该失败没有产生`result.json`。恢复流程在不生成第二套科学panel的前提下，重新执行672次
official NAVSIM execution并要求与原artifact逐bit一致，同时复现完整时域intent eligibility；
全部通过后才物化结果。第二个审计脚本随后再次从672个artifacts复算7个前缀和组级选择，
0失败。

- fit result SHA-256：`41d8c39e2f83d4d232678d5e59daefb1415f9e4168709a92c2e40f0e66f6da38`；
- fit audit SHA-256：`7674d9fe36c2e8cb8c40abce620450a9962093626c75591767011e3a85f1b970`；
- development result SHA-256：`34fcb15fe995b17534b33ec9f10072834df4c9f48d7e6e03205bb956ae20e0d8`；
- development audit SHA-256：`7604f26537a38616f2879605b26a868c61ccb4f8637c34dfca3b2126fd83c59c`。

规定的跨模型reviewer backend不可用，状态为`REVIEW_UNAVAILABLE`；机械完整性审计不能
冒充独立科学复核。

## 理论边界

通过的是不同数学对象：

\[
M_{2.5}(\tau)=\inf_{q\in\mathcal B(\tau_{0:2.5s})}u(q),
\]

不是原完整4秒的 \(M_4\)。因此H7-B原始`INSUFFICIENT_PANEL_COVERAGE`保持不变；本结果
不能外推完整时域支持，也没有证明 \(M_{2.5}\) 与regret或ranking flip相关。若两条候选
在前缀上的离散footprint节点集合相同，则对任意场都有相同set-min，即使它们在2.5秒后
分叉。下一步必须先冻结并完成不读取field、teacher或outcome的H7BS2，再决定是否允许
H7-B2或转向完整时域支持架构。
