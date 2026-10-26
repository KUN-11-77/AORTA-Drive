# R006V0R 流式 JVP 资源资格结果

状态：`MECHANICAL_PASS__BATCH_128_FROZEN__NO_SCIENCE_CLAIM`。

## 协议

资源benchmark只使用一个已审计cache shard的shifted activation作为shape-correct base，并使用固定seed
合成tangent；不读取clean-shift delta，不输出objective、coverage或restoration。rank固定为最坏字典值32，
每个batch重复3次；冻结规则为两层都低于18 GiB时选择登记字典`[16,32,64,128]`中的最大batch。

## 结果

| stage | batch | median step | samples/s | peak allocated |
|---|---:|---:|---:|---:|
| C3 planner memory | 16 | 42.66 ms | 375.1 | 0.36 GiB |
| C3 planner memory | 32 | 44.98 ms | 711.4 | 0.44 GiB |
| C3 planner memory | 64 | 44.57 ms | 1435.9 | 0.61 GiB |
| C3 planner memory | 128 | 49.40 ms | 2590.9 | 0.94 GiB |
| C4 trajectory query | 16 | 6.45 ms | 2481.4 | 0.27 GiB |
| C4 trajectory query | 32 | 4.82 ms | 6645.7 | 0.27 GiB |
| C4 trajectory query | 64 | 4.78 ms | 13392.7 | 0.27 GiB |
| C4 trajectory query | 128 | 4.68 ms | 27345.4 | 0.28 GiB |

runner与audit exit code均为0，38/38机械检查PASS，封存角色未访问且未产生科学指标。因此正式solver
的`streaming_batch_size`冻结为128。

这些是单步JVP资源数字，不是最终CPTR batch-1推理时延；不得用于实时方法主张。

