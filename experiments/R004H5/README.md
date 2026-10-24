# R004H5 — fit-only 配对外观机制诊断

本实验冻结 R004H1 checkpoint、DINO、投影、teacher 与 42 个 fit recording groups。每个 group 的 16 个 anchors 在严格相同场景、相机与历史帧上施加预注册的确定性光度变换，检验预测差分场是否同时具备外观稳定性与任务信息。

主距离在 edge domain 计算，因此不受势函数共同常数偏移影响。低外观距离只有在 task error 仍优于 zero field、teacher edge correlation 为正且场未塌缩时才可解释为机制证据。recording group 是独立单位，anchor 和外观条件都是组内重复测量。

这些条件是受控 photometric stress，不命名为真实 rain、snow 或 dusk，也不产生 OOD、coverage、驾驶收益或安全主张。若原图哈希/配对无法重放，实验直接拒绝；禁止用 feature noise 代替图像外观实验。

配置：`configs/experiment/r004h5_fit_appearance_mechanism.yaml`。

输入资格状态：`PASS`。42 groups、672 anchors、8064 unique raw images 已逐文件重放，0 missing、0 hash mismatch；详见 `experiments/results/R004H5_INPUT.md`。该结果只授权冻结模型推理。

