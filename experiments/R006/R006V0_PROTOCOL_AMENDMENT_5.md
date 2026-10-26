# R006V0 协议修正 5：wrong-scene donor 的outcome-free唯一映射

状态：`FROZEN_BEFORE_FORMAL_SCIENCE`。

## 原协议缺口

原预注册只规定donor来自outer-train、不同recording group且不使用RGB/activation/output/label，没有唯一
定义“条件匹配”。随意抽取场景会把时间位置差异混入A2，使wrong-scene成为过弱对照。

## 冻结映射

对每个outer-confirmation recipient anchor：

1. donor只来自该fold的35个outer-train groups；
2. donor必须具有完全相同的`anchor_index`；
3. 各group内部以`session_position`的min/max归一化到`[0,1]`，优先选择与recipient归一化位置最接近者；
4. 距离相同时，以salt `aorta-r006v0-wrong-scene-donor-v1`对recipient/donor token作SHA-256排序，最后
   以donor token确定性决胜；
5. 运行时使用donor在**同一appearance condition**下自身的`clean−shift` delta，而不是donor绝对clean
   activation；
6. 将donor delta投影到同一个已选择basis，并逐recipient anchor精确匹配A1 correction Frobenius能量；
7. 映射构建只读取group、anchor index、session position和scene token，不读取RGB、activation、trajectory、
   utility或科学endpoint。

42组×16 anchors共672个recipient均有唯一合法donor；生成与独立audit均PASS。manifest SHA-256为
`16769ffff83dca52c9a2e0ce1ae8cc222a23c29affb03fa96841f0b7557d3b98`。

## 边界

该匹配只控制记录内相对时间位置，不能保证道路拓扑、交通状态或语义完全匹配。因此A2必须与A4/A5
共同解释；A1只有同时可靠胜过三者才能支持scene specificity。
