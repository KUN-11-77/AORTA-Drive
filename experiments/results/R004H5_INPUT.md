# R004H5 input replay preflight

## Outcome

`PASS`。冻结的42个 fit recording groups 共672个 anchors，引用8064个唯一原始图像槽位。服务器逐文件重算 SHA-256 后得到：

- missing raw images: `0`
- hash mismatches: `0`
- prohibited payload accessed: `false`
- raw image bytes: `1,661,211,801`

这关闭的只是原始图像身份、scene/camera/history 配对和 fit-only 输入可用性。它不包含模型推理，不支持外观稳定性、任务准确性、held-out/OOD、驾驶收益或安全结论。

原始机器可读结果见 `R004H5_INPUT.json`。主 R004H5 推理只有在此资格门通过后才可启动。

