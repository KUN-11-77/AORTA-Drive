# R004H7-BS4A-D0：候选公共/差异区域视觉支持归因

状态：`PREREGISTERED_POSTHOC_NOT_RUN`。这是H7BS4A严格失败后的单次机制归因，不能把
原始`STOP_OR_REDESIGN_VISUAL_SUPPORT_INTERFACE`改成PASS。

## 问题与数学对象

对每个anchor的native节点集\(F_0\)和合法challenger节点集\(F_j\)，精确定义

\[
C_{0j}=F_0\cap F_j,\qquad
E_0=F_0\setminus F_j,\qquad
E_j=F_j\setminus F_0,\qquad
D_{0j}=E_0\cup E_j.
\]

其中\(C_{0j}\)是候选公共查询区，\(D_{0j}\)是离散几何上能够区分两者的对称差区域。
本实验只把冻结projective-evidence mask与这些集合相交，不读取teacher、预测场、DINO值、
决策outcome或任何development/test/NavDream数据。

## 冻结问题

1. H7BS4A中两个零证据native的所有未支持节点，是否对每一个合法challenger都位于
   \(C_{0j}\)？
2. 是否存在\(D_{0j}\neq\varnothing\)但其中没有任何直接projective evidence的候选对？
3. 全部42 fit groups、672 anchors上的失败结构是公共区域、差异区域还是混合归因？

不设置支持比例阈值，只报告空集、部分支持与全部支持的精确计数，避免从结果反向选门槛。

## 预先限定的解释

- 若存在非空\(D_{0j}\)且其支持节点为0，则候选差异接口本身也存在直接证据盲区；
- 若两个blind native的未支持节点对所有pair均是公共节点，且所有非空差异区域都有证据，
  只支持“原始失败主要归因于候选公共区域”的描述；
- 其他情况记为mixed attribution。

即使第二种情况成立，也**不证明决策充分性**。对于set-min，公共区域中的未知低值仍可能同时
控制两个候选分数：

\[
R(F_0)=\min(c,e_0),\qquad R(F_j)=\min(c,e_j),
\]

未知公共项\(c\)不会一般性地从两个最小值中相消。因此D0只能决定下一版接口假设是否值得
正式预注册，不能授权teacher、训练、H7-B2或H7-C。

## 完整性

运行覆盖整个fit panel而非只查看两个失败样本；输入、实现、日志和panel artifact均hash绑定。
正式输出必须由独立入口机械重放。规定的外部科学reviewer若不可用，应明确标记，不得用内部
复算冒充独立复核。
