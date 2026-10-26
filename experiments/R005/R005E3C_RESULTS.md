# R005E3C 结果：Soft-Regret 梯度质量只读诊断

## 裁决

`MECHANICAL_PASS_WITH_ARCHIVE_LIMITATION__DESCRIPTIVE_ONLY`

本轮没有训练、没有改权重，不产生方法PASS。它只回答R005E3B native-only塌缩的局部优化几何。

## 冻结范围

- source-fit fold-0 outer-train：35组、560 anchors；
- 合法challengers：3235，其中正收益623、非正收益2612；
- 有正收益机会的anchors：236/560；
- 同初始化、同温度、GPU0 primary/replay；
- development/calibration/test/NavDream/target均未访问。

## 主要观察

| 量 | 结果 | 严格解释 |
|---|---:|---|
| opportunity anchor比例 | 42.14% | 机会并非极罕见，但不代表可学习 |
| positive challenger比例 | 19.26% | 正收益候选为少数 |
| 初始化正收益raise方向 | 94.54% | 大多数、但不是全部方向正确 |
| 初始化正/非正参数梯度cosine | -0.9999899 | 聚合梯度近乎反向；仅为局部描述 |
| 非正/正参数梯度范数比 | 2.6682 | 非正机会聚合梯度更强 |
| 正收益score-gradient绝对均值（初始） | 0.1429245 | 初始仍有纠错信号 |
| 正收益score-gradient绝对均值（终点） | 1.6981e-9 | 约8.42e7倍衰减 |

## 可以与不可以主张的内容

可以主张：当前soft-regret实现中，遗漏正收益候选被压入softmax概率尾部后，score级纠错信号几乎消失；
初始化聚合参数梯度同时显示强冲突。

不可以主张：上述现象是唯一根因；线性或完整scorer不可实现；视觉无信息；proper mean必然成功；
Differential Query、外观鲁棒、行为或安全收益已获得支持。

## 完整性说明

primary/replay、输入/代码/config/runner hash、模型未更新、有限值与角色封存均通过。强化审计从冻结E3B
prediction独立重算终点score-gradient并精确一致。紧凑归档未保存初始化prediction和参数梯度张量，
因此初始化梯度几何无法从归档独立重算；外部审查为`WARN`而非无条件PASS。

下一轮R005E3D是独立预注册的一因素资格实验：保持identity scorer与全部训练合同不变，只将目标替换为
native-relative proper mean MSE。它不是由E3C数值事后选出来的，也不是论文核心创新。
