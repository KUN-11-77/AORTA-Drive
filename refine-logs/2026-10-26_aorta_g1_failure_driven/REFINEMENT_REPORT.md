# Refinement report

## 从初稿到冻结方案的变化

- 删除“pairwise surrogate总体一致”的过强主张；
- 删除结果后由E3C分支选择loss的自由度；
- 把E3B失败从“线性不可实现”改为“冻结资格门失败”；
- 把capacity ladder改为只有严格包含关系的exact-head probe；
- 将下一处理冻结为identity+proper mean MSE，保持其余合同完全一致；
- 将成功端点从loss下降改为实际native-relative gain和oracle capture；
- 增加same-GPU replay、独立指标复算、control SHA和逐项失败标签。

## 仍未解决

训练域资格不等于组外泛化；组外泛化不等于视觉条件增量；视觉增量不等于Differential Query必要性；
离线选择收益不等于NavDream闭环或安全。上述边界已写入预注册和claim matrix。
