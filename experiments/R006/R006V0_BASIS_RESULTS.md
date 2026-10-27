# R006V0 basis 阶段结果

正式裁决：`STOP_GLOBAL_SHARED_INTERFACE`。

这是一项有效的负结果。它不否定 DINOv3，也不否定 C4 trajectory-query 中存在任务敏感方向；它否定
的是本次预注册的更强命题：六个 outer folds 应通过“每折最小合格秩”规则收敛到同一个固定
layer/rank 接口，随后才能继续 outer-confirmation 干预。

## 1. 完整性

- 正式运行：`/data/khsong/aorta_runs/r006v0_basis_1d2d973_formal`；
- 实现提交：`1d2d973cffd7efc4b830373d3a27726843ed4c7b`；
- 6/6 folds 完成，主运行与 replay 逐文件 SHA-256 一致；
- 本地独立聚合审计：93/93 checks PASS；
- development、calibration、test、NavDream、target 均未访问；
- outer-confirmation endpoint 未计算，因此不得读取介导效果、DINO 可恢复性或行为收益。

## 2. 预注册选择结果

| Outer fold | Selected layer | 最小合格 rank | Group-equal coverage | Worst-group coverage |
|---:|---|---:|---:|---:|
| 0 | trajectory query secondary (C4) | 8 | 0.920055 | 0.864408 |
| 1 | trajectory query secondary (C4) | 32 | 0.914739 | 0.751026 |
| 2 | trajectory query secondary (C4) | 4 | 0.903425 | 0.817300 |
| 3 | trajectory query secondary (C4) | 32 | 0.904114 | 0.772164 |
| 4 | trajectory query secondary (C4) | 4 | 0.904861 | 0.840541 |
| 5 | trajectory query secondary (C4) | 32 | 0.925075 | 0.648451 |

层选择为 C4 `6/6`，超过预注册的 `5/6` 门；最小合格秩计数为
`rank4: 2, rank8: 1, rank16: 0, rank32: 3`，没有任何秩达到 `4/6` 门。因此裁决必须是
`STOP_GLOBAL_SHARED_INTERFACE`，不得事后把 rank 32 改成正式接口。

各 rank 在 C4 上“合格”的折数为 `rank4: 2/6, rank8: 3/6, rank16: 1/6, rank32: 6/6`；C3
所有 rank 均为 `0/6`。`rank32=6/6` 只是结果可见后的描述性事实，不改变冻结裁决。

## 3. 结果能支持什么

1. 在当前 source-fit、受控外观和 task-weighted coverage 定义下，C4 比 C3 更稳定地承载相关方向；
2. 当前“独立拟合各 rank，再选最小合格 rank”的接口没有得到跨折固定复杂度支持；
3. 需要区分“方向不共享”与“方向大体共享但有效维数/估计误差随组变化”两种解释。

## 4. 结果不能支持什么

- 不能称“低秩介导已成立”，因为 outer finite intervention 未执行；
- 不能称 rank 32 是新正式接口；
- 不能称 DINO 能恢复该子空间；
- 不能称 CPTR、NavDream、跨 planner、official utility、时延或驾驶安全有效；
- 不能把 C4 的模型内部结构直接解释为真实世界语义因果因素。

## 5. Post-hoc 子空间几何诊断

为决定下一条理论路线，使用已开封 inner bases 计算旋转不变的投影重叠，主裁决保持不变：

| Rank | 跨折归一化 projector overlap（均值） | 最小 overlap | 最大主角均值 |
|---:|---:|---:|---:|
| 4 | 0.978531 | 0.962408 | 11.48° |
| 8 | 0.982742 | 0.973354 | 13.36° |
| 16 | 0.904990 | 0.873798 | 66.13° |
| 32 | 0.956268 | 0.941707 | 45.03° |

低 rank 的主方向高度重叠，而达到 coverage 阈值所需的最小秩不稳定。这使“共享方向核心 + 不稳定/异质
谱尾”成为值得预注册检验的竞争假设，但不能把描述性 overlap 当成独立确认：六折 discovery 数据彼此
重叠，pairwise 数值不是独立统计单位；rank16 的异常也可能来自非凸优化、独立 rank 拟合或 inner
selection 方差。

机器可读依据：

- `results/r006v0_basis_1d2d973_formal/results/aggregate_audit.json`；
- `results/r006v0_basis_1d2d973_formal/results/posthoc_basis_geometry.json`。
