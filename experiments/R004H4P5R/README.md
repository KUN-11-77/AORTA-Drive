# R004H4P5R — 定向几何终止资格实验

本实验冻结 R004H4P5I 的 8 个 fit sentinels、候选、1 m teacher 图、阈值与官方执行轨迹，只回答几何表示是否同时满足 sound、nontrivial、computable。它是扩大到 128 anchors 前最后一次几何表示迭代。

连续 polygon 结果仅是独立高分辨率诊断参考，不称为精确真值。论文级正面积存在性必须来自保守 cell witness。方向感知矩形证书在预注册的四个分辨率和四个网格相位上完整报告，主判断只使用 canonical phase，禁止结果后择相位。

终止裁定分三层：几何 soundness、几何 nontriviality、teacher improvement。前两者通过而 teacher 仍无收益时，停止提高几何精度，转向候选生成或任务裕量；reference 有差异而细网格仍全零时，停止细化网格，评估有保守误差界的 polygon 证书。

配置：`configs/experiment/r004h4p5r_oriented_terminal.yaml`。

