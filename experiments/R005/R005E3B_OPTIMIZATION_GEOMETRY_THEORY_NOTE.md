# R005E3B：相对分数参数化与决策替代损失的优化几何

日期：2026-10-26  
状态：`PRE_RESULT_THEORY_FREEZE__DOES_NOT_CHANGE_R005E3B`

## 1. 本轮究竟隔离什么

对合法候选集合 (V)，令网络未激活读出为 (r_i)，native索引为0。R005E3A与R005E3B分别使用

\[
s_i^{A}=\tanh(r_i-r_0),
\qquad
s_i^{B}=r_i-r_0.
\]

两者都满足native gauge (s_0=0)，都不改变参数量、视觉输入、候选集合或soft selection-regret。
因此E3B只删除输出激活的Jacobian，不检验新表示、新视觉机制或新网络容量。

## 2. 梯度的精确分解

记

\[
p_i=\frac{\exp(s_i/T)}{\sum_{j\in V}\exp(s_j/T)},
\qquad
\bar u_p=\sum_i p_i u_i,
\]

soft selection-regret为

\[
\mathcal L_{SR}=u^*-\sum_i p_i u_i.
\]

对challenger (k)，identity参数化给出

\[
\frac{\partial\mathcal L_{SR}}{\partial(r_k-r_0)}
=\frac{p_k}{T}(\bar u_p-u_k),
\]

而tanh参数化给出

\[
\frac{\partial\mathcal L_{SR}}{\partial(r_k-r_0)}
=\frac{p_k}{T}(\bar u_p-u_k)(1-s_k^2).
\]

因此，只要 (u_k>\bar u_p)，两者方向都提高候选 (k) 的分数；E3B删除的只是

\[
1-s_k^2\rightarrow0
\]

这一额外饱和因子。该因果处理不删除 (p_k\rightarrow0) 造成的softmax尾部衰减。

## 3. identity仍可能出现近吸收区的严格反例

只考虑native和一个真实更优challenger：

\[
u_0=0,\quad u_1=a>0,\quad s_0=0,\quad s_1=-M.
\]

则

\[
p_1=\frac1{1+e^{M/T}},
\qquad
\mathcal L_{SR}=a(1-p_1),
\]

且

\[
\left|\frac{\partial\mathcal L_{SR}}{\partial s_1}\right|
=\frac aT p_1(1-p_1)
\xrightarrow[M/T\to\infty]{}0.
\]

所以“无界logit在总体上具有正确最优方向”不等于“有限梯度训练一定能从错误低分区恢复”。E3B若失败，
不能继续归因于tanh；它将把问题收紧到softmax概率加权梯度、正机会稀疏、有限函数类或优化预算。

## 4. E3B结果能够支持和不能支持的命题

若E3B通过，只支持：在冻结fold 0 outer-train上，删除tanh Jacobian后，当前网络与预算能够获得正的
绝对native-relative utility，并达到预注册的oracle capture门。它不证明：

- OOF组级泛化；
- correct视觉相对mismatch/removed具有决策必要性；
- Differential Query优于matched generic scorer；
- 外观鲁棒、NavDream、policy value或安全收益。

若E3B失败，则否定“只删除tanh足以恢复当前soft-regret训练”的具体解释，不否定RGB的信息论可恢复性，
也不允许直接增加网络复杂度。

## 5. 失败后唯一允许预注册的最小surrogate候选

若E3B机械有效但失败，下一层可研究utility-gap加权的pairwise logistic surrogate。对所有严格非tie对

\[
\mathcal P=\{(i,j):u_i\ne u_j\},
\quad
y_{ij}=\operatorname{sign}(u_i-u_j),
\]

定义未归一化损失

\[
\mathcal L_{PW}
=\sum_{(i,j)\in\mathcal P}|u_i-u_j|
\log\left(1+\exp\left[-\frac{y_{ij}(s_i-s_j)}T\right]\right).
\]

其对pair margin (m_{ij}=s_i-s_j) 的导数为

\[
\frac{\partial\ell_{ij}}{\partial m_{ij}}
=-\frac{|u_i-u_j|}{T}y_{ij}
\sigma\left(-\frac{y_{ij}m_{ij}}T\right).
\]

当一对候选被错排，即 (y_{ij}m_{ij}\le0) 时，梯度幅值满足

\[
\left|\frac{\partial\ell_{ij}}{\partial m_{ij}}\right|
\ge\frac{|u_i-u_j|}{2T}.
\]

它不会因为“错误候选当前概率很小”而失去纠错梯度。

进一步，令 (hat i\in\arg\max_i s_i)，(i^*\in\arg\max_i u_i)。若选择错误，则

\[
s_{\hat i}\ge s_{i^*},
\]

所以pair ((i^*,\hat i)) 的logistic项至少为

\[
(u^*-u_{\hat i})\log2.
\]

因此逐anchor有

\[
\boxed{
R_{hard}=u^*-u_{\hat i}
\le\frac{\mathcal L_{PW}}{\log2}
}.
\]

这个界给出比“换一个loss试试”更明确的理由：该surrogate既对错排候选保留非消失纠错梯度，又直接
上界hard selection regret。它仍只是可能的优化工具，不是AORTA的视觉或架构创新；只有E3B失败后，
才允许另行预注册单因素比较，且不得与网络、采样、温度或数据改动同时进行。

## 6. 理论—实验闭环

当前闭环为：

1. E3A观察到`tanh`边界与all-native collapse；
2. 梯度分解提出“activation Jacobian是必要贡献因素”的可证伪假设；
3. E3B只删除该因子并检查绝对机会捕获；
4. 若通过，进入完整OOF确认；
5. 若失败，由两候选反例把原因推进到softmax尾部；此时才允许检验具有错排非消失梯度与hard-regret界的
   pairwise surrogate。

每一步都删除一种竞争解释，而不是用更复杂系统提高通过概率。
