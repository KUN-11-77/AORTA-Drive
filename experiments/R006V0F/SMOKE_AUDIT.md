# R006V0F 真实 GPU 烟测审计

状态：`PASS_MECHANICAL_ONLY__NO_SCIENTIFIC_CLAIM`。

## 资格结果

- 源码提交：`a0675d60f5093540fff3954ba6802c30b9c6a6a7`；
- 服务器产物：`/data/khsong/aorta_runs/r006v0f_a0675d6_smoke/smoke.json`；
- GPU：冻结正式 roster 内的物理 GPU 0，运行时映射为`cuda:0`；
- 输入：一个已登记 discovery group、一个 selection group、C4、rank 4、seed `2026102701`；
- 优化：random-only、2 steps、2 line-search steps，primary/replay同进程同GPU顺序运行；
- 结果：basis shape、有限性、正交性全部通过，primary/replay逐元素完全一致；
- 数据边界：未访问 outer-confirmation、development、calibration、test、NavDream 或 target；
- `scientific_metrics_emitted=false`。

## Provenance

- runner SHA-256：`0a75e24d486a1dbd814ced07bf3b9a306a0a5d9d125abb5542bc14530084a28e`；
- checkpoint SHA-256：`d08e97ba6beb937412697d619171553041622859e026a06935a2e0e118e6309d`，
  与 cache 记录完全一致；
- cache manifest SHA-256：`f3658ea5b0a85547129587ed39691bda6ba1853e83e0c83cf569854bfc0f01bb`；
- nested manifest SHA-256：`449578209aa2621c932124236b1740b4930d5321c467777e02d9e3cc40a7ced7`；
- runtime：Python 3.10.18、PyTorch 2.7.1+cu118、CUDA 11.8。

## 结论边界

该烟测只证明正式计算路径具备执行资格。它不证明rank稳定、共享方向、有限介导、DINO可恢复性、
外观鲁棒性、驾驶收益或安全性，也不改变父实验`STOP_GLOBAL_SHARED_INTERFACE`。
