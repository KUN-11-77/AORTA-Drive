# R006V0F：fixed-rank STOP 的失败分解预注册

状态：`PREREGISTERED_DIAGNOSTIC__NOT_EXECUTED__NO_OUTER_ACCESS`。

## 1. 唯一目的

解释 R006V0 的 `STOP_GLOBAL_SHARED_INTERFACE` 来自哪一层：

1. Stiefel/流式求解器对初始化敏感；
2. 各 rank 独立拟合导致非嵌套、不可比较的局部解；
3. 共享方向存在，但达到固定 coverage 所需的有效维数随 group/fold 变化；
4. 全局共享方向本身不稳定。

本诊断不能修改原 STOP，也不能解锁 outer intervention、DINO、CPTR 或 NavDream。

## 2. 冻结范围

- 只读现有42个source-fit activation cache及原6折28/7 inner角色；
- outer-confirmation endpoint、development、calibration、test、NavDream、target全部封存；
- 只研究已由R006V0选择为6/6的C4 trajectory-query层；
- rank仍为`{4,8,16,32}`，coverage阈值、W度量、batch=128与样本顺序不变；
- 原提交、原六个basis与原结果作为seed-0 reference，不覆盖；
- 新初始化seed固定为`2026102701`至`2026102708`；
- 每个fold/rank/seed保存basis、discovery objective、逐组selection coverage、projector与exact replay。

聚焦C4是结果可见后的机制诊断，不是新的确认性结论；论文若使用，必须明确标注post-failure study。

## 3. 诊断量

### 3.1 优化稳定性

对同一fold/rank的seed对计算

\[
d_{seed}^2=r-\|B_{f,r,s}^{\top}B_{f,r,s'}\|_F^2,
\]

并报告objective、group-equal/worst-group coverage与qualification flip。若至少2/6 folds存在同一rank在
8 seeds中既PASS又FAIL，归因为`ESTIMATOR_OR_OPTIMIZATION_INSTABILITY_PRESENT`；这不排除同时存在
数据异质性。

### 3.2 折间方向与seed内噪声

分别报告同fold跨seed与同seed跨fold的projector distance分布。只在跨fold距离系统性大于seed内距离
时，才把方向差异解释为fold heterogeneity；否则优先归因估计器不稳定。这里“系统性”只作描述：报告
全部pair和group bootstrap，不设置事后论文PASS门。

### 3.3 最小rank稳定性

每个seed按原规则独立计算 (r_{f,s}^*)。报告：

- 每折8 seeds的rank频数；
- 同一fold的rank mode是否达到7/8；
- 达到7/8的fold之间mode是否仍不同；
- rank coverage曲线是否非单调。

若至少5/6 folds内部rank mode达到7/8但fold modes不同，标记
`STABLE_FOLD_HETEROGENEOUS_EFFECTIVE_DIMENSION`。若达不到，标记
`RANK_ESTIMATION_UNSTABLE`。

## 4. 后续路线裁决

- `RANK_ESTIMATION_UNSTABLE`：先修正求解器/估计器，禁止设计adaptive-rank reader；
- `STABLE_FOLD_HETEROGENEOUS_EFFECTIVE_DIMENSION`且低rank方向跨折稳定：允许预注册“共享有序flag +
  因果输入条件化有效维数”的新方法；
- 折间方向差异显著超过seed内噪声：全局共享basis路线停止，考虑条件化basis或放弃CPTR；
- 无论哪种结果，均不得从本诊断声称有限介导或视觉鲁棒收益。

## 5. 完整性

每个job必须same-GPU primary/replay、basis逐文件hash exact、有限/正交、输入角色与source commit一致。
任何机械失败只允许修复实现并以新目录重跑一次；不得更改rank、seed、threshold或读取outer endpoint。
