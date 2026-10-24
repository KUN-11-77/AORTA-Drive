# R004H4P5R — 定向几何终止资格实验

本实验冻结 R004H4P5I 的 8 个 fit sentinels、候选、1 m teacher 图、阈值与官方执行轨迹，只回答几何表示是否同时满足 sound、nontrivial、computable。它是扩大到 128 anchors 前最后一次几何表示迭代。

连续 polygon 结果仅是独立高分辨率诊断参考，不称为精确真值，也不作为证书 soundness 的上界 oracle；采样矩形并集本身不保证给出连续 set-difference 面积上界。论文级正面积存在性必须来自保守 cell witness。方向感知矩形证书在预注册的四个分辨率和四个网格相位上完整报告，主判断只使用 canonical phase，禁止结果后择相位。

终止裁定分三层：几何 soundness、几何 nontriviality、teacher improvement。前两者通过而 teacher 仍无收益时，停止提高几何精度，转向候选生成或任务裕量；reference 有差异而细网格仍全零时，停止细化网格，评估有保守误差界的 polygon 证书。

配置：`configs/experiment/r004h4p5r_oriented_terminal.yaml`。

## 执行裁定（2026-10-24）

第二次合法启动在预注册的 30 分钟 wall-time 上限内没有完成第一个哨兵，因而被外部终止并记为 `STOPPED_RESOURCE_CAP`。没有生成任何完整哨兵记录，也没有可解释的 polygon reference、正式非包含下界或 teacher improvement 结果。该次运行只能证明当前逐单元精确有理实现不满足本资格实验的可计算性要求，不能据此判断连续非包含面积存在或不存在。

按照终止规则，本阶段不继续细化网格、修改候选、调整阈值或开放式重构几何证书，也不启动 128-anchor 扩展。原始停止记录见 `experiments/results/R004H4P5R_RESOURCE_STOP.json`，标准输出仅含 `nohup` 启动行。

