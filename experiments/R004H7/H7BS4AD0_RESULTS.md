# R004H7-BS4A-D0：候选公共/差异区域视觉支持归因结果

## 裁决

`CONTRAST_INTERFACE_ALSO_HAS_ZERO_EVIDENCE_PAIRS`。H7BS4A原始`STOP`保持不变；D0没有
支持“只审查候选对称差即可修复接口”的假设。

## 结果

- 42 fit groups、672 anchors、3864个native–eligible challenger pairs全部纳入；
- 3678对离散几何不同，186对离散节点集完全相同；
- 8/3678个几何不同pair的非空对称差区域没有任何直接projective evidence；
- 8对分布在5 groups、5 anchors；其对称差只有1–2个节点；
- 这些pair的公共区域仍有5–25个直接支持节点；
- H7BS4A的两个零证据native都位于没有合法challenger的anchor，因此没有pair可以支持
  “blindness来自候选公共区域”的归因。

这暴露了两个不同问题。第一，原all-candidate门把“没有比较对象的候选供给状态”和“视觉
仲裁接口”绑定在一起；后续必须分账。第二，即使只看真实决策pair，差异区域也可能完全无
直接视觉证据，不能把difference-only query当作充分修复。

## 理论含义

视觉支持不应只是删除未直接观测节点的二值门。更合适的候选是：把视觉场视为部分可观测
条件下的不确定集合\(\mathcal U_\theta(I,O)\)，由共享的同一个\(u\)计算候选差异：

\[
\mathcal D_{ij}=\left\{
\min_{q\in F_i}u_q-\min_{q\in F_j}u_q:\;u\in\mathcal U_\theta(I,O)
\right\}.
\]

只有当整个差异集合与0严格分离且支持/校准条件满足时才允许仲裁，否则拒判。该形式保留了
公共节点造成的相关性，避免把两个candidate interval独立相减或假设公共最小值必然相消。
这只是下一版理论目标，尚未得到teacher-criticality、校准或决策收益验证。

## 完整性与边界

结果由独立入口重放全部672 ledgers与summary，0失败：

- result SHA-256：`aa40df441cc73800b64002cd8ac15159f4304f20afb75d20cc396a60fdea3ebf`；
- audit SHA-256：`1fe1642a45185ff7796891098117d7d4e480e8f044669a550dc6ae6228080284`；
- runtime：`70.07s`；
- external reviewer：`REVIEW_UNAVAILABLE`。

本实验未读取teacher、field、DINO feature values、outcome、development、calibration、test、
NavDream或target。它不证明teacher关键性、任务正确性、网络精度、决策收益、时延或安全。
训练、H7-B2与H7-C继续HOLD。
