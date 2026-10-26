# R006V0 协议修正 6：轨迹度量与流式精确求解语义

状态：`FROZEN_BEFORE_FORMAL_SCIENTIFIC_ENDPOINTS`。

## 1. 为什么需要修正

原协议把 primary output metric 写为 `trajectory_xy_physical_rms`，但没有显式写出二次型常数；同时
正式 discovery 的样本量不允许长期保留全部 JVP 计算图。若实现者自行选择坐标均值或未加权 minibatch
均值，会让数学估计对象与资源实现发生隐式漂移。

## 2. 冻结的物理度量

令冻结 planner 的二维轨迹输出展平为

\[
y=(x_1,y_1,\ldots,x_T,y_T)\in\mathbb R^{2T}.
\]

R006V0 的 primary squared metric 固定为

\[
\|\Delta y\|_W^2
=\Delta y^\top W\Delta y,
\qquad W=I_{2T}/T,
\]

即每个时间步二维物理位移平方的平均；其平方根才称为 trajectory XY physical RMS。这里不使用
`I/(2T)` 的逐坐标 MSE 语义。secondary yaw/heading 只报告，不进入 layer/rank 选择。

## 3. 冻结的流式求解语义

流式 solver 只能改变计算图驻留方式，不得改变完整 discovery 集目标。对固定顺序 batch
\(\mathcal B_b\)，值与欧氏梯度都按 batch 样本数加权：

\[
L(B)=\frac{1}{N}\sum_b |\mathcal B_b|L_b(B),\qquad
\nabla L(B)=\frac{1}{N}\sum_b |\mathcal B_b|\nabla L_b(B).
\]

随后才进行同一 Stiefel 切空间投影、Armijo 搜索与 QR retraction。禁止使用随机 batch、等 batch
权重平均、随机梯度更新或提前根据 partial coverage 停止。batch size 只影响显存和耗时，将由不产生
科学 endpoint 的资源 benchmark 冻结并写入正式 manifest。

## 4. 机械资格

已加入合成反例测试：19个样本、非整除 batch size=5 时，流式 solver 与完整稠密目标达到相同
objective，并恢复同一真实投影子空间。该测试只证明实现等价，不证明真实 planner 存在低秩介导。

