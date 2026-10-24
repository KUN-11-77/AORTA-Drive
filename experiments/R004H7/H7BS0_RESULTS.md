# R004H7-BS0 覆盖失败归因

## 裁决

`FORWARD_HORIZON_SUPPORT_MISMATCH_OBSERVED_POSTHOC`。

这是看过 H7-B coverage failure 后进行的 outcome-blind 描述诊断，不是新的确认性检验。
它支持“当前完整4秒查询超出固定前向支撑”作为本次失败的直接机制；没有发现当前
panel interface 上的坐标中心错绑症状。它不能证明所有上游投影都正确，也不能验证
decision quotient、视觉可靠性或驾驶收益。

## 关键事实

- 384/384 panel artifacts 重读成功；119 个无效 anchor 身份全部复现；
- 110 个 native 的首次越界方向全部是 `x_max`，没有 `x_min/y_min/y_max` 越界；
- 770 次 candidate ROI failure 的首次越界方向同样全部是 `x_max`；
- native 首次越界最早发生在第21个0.1秒采样点，中位为30.5，四分位区间27–34；
- native 所需前向额外支撑中位为10.016m，最大26.693m；
- 所有候选的初始平移和yaw残差最大值均为0；artifact和ledger重算失败均为0；
- 9个`no_valid_challenger` anchor中，六个challenger全部因冻结intent条件失败，而不是
  ROI或视觉证据失败；
- panel在四个appearance fields之前只构建一次，119个无效anchor的`conditions`均为
  `None`。因此appearance不能解释本次coverage差异，当前记录的invalidity也不是视觉
  置信度导致；ROI内视觉证据是否充分仍是未测量问题。

## 选择性偏差诊断

在23个同时含valid与native-invalid样本的recording groups内等权比较，native-invalid
相对valid的平均差异为：

- path length：`+24.300m`；
- terminal x：`+24.643m`；
- maximum absolute lateral displacement：`+1.274m`；
- total yaw variation：`-0.122rad`；
- maximum discrete curvature proxy：`-0.0278/m`。

这说明缺失主要集中在前向行程更长的anchor，而不是更高曲率的转弯。当前265个有效
anchors因此对短行程/较低前向reach存在明显选择性；即使在其上得到高AUROC，也不能
直接代表原384-anchor任务分布。以上是post-hoc描述，不给出p值或因果解释。

## 理论含义

若预测场只定义在 \(\Omega\)，而候选查询集合 \(F(\tau)\not\subseteq\Omega\)，则完整
set-min margin不能由域内场唯一确定。裁剪为 \(F(\tau)\cap\Omega\) 改变了任务对象；在
没有域外延拓假设时，裁剪margin与完整margin的差可以任意大。因此：

1. 当前fail-closed有效性门必须保留；
2. 缩短固定prefix意味着研究近时域决策；
3. 扩大ROI意味着重建投影、标签、网络输出和容量匹配baselines；
4. 不能用padding、常数外推、删组或降低coverage门槛修复原H7-B。

## 审计与边界

独立audit脚本重新计算车辆角点、首次越界方向、扩张量、group counts和119个无效
identity，384 artifacts全部通过，0 attribution failure。raw attribution结果保存在被Git
忽略的`artifacts/r004h7bs0/result.json`。其SHA-256为
`c6589beb33dbd67504072f8dd98bac52d13ca6f177b38c719638dab1f4e77c99`；机械审计SHA-256
为`db81ac7807298e6331b977e51c568834b39ab7c1bd4d4287bdac8512b23d695e`。

跨模型reviewer backend不可用，状态保持`REVIEW_UNAVAILABLE`。calibration、test、
NavDream和target均未访问。下一阶段固定prefix选择仍HOLD，需先审查本归因结果及其
estimand代价。
