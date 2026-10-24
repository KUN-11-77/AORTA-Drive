# R004H5 input replay preflight

## Outcome

`PASS`。冻结的42个 fit recording groups 共672个 anchors，引用8064个唯一原始图像槽位。服务器逐文件重算 SHA-256 后得到：

- missing raw images: `0`
- hash mismatches: `0`
- prohibited payload accessed: `false`
- raw image bytes: `1,661,211,801`

这关闭的只是原始图像身份、scene/camera/history 配对和 fit-only 输入可用性。它不包含模型推理，不支持外观稳定性、任务准确性、held-out/OOD、驾驶收益或安全结论。

原始机器可读结果见 `R004H5_INPUT.json`。DINO batch-cardinality 合同修复后，以最终配置和 implementation `51ff17207637c69845976a99888a7d9bc2b12853` 再次完整重放，结果仍为 42 groups、672 anchors、8064 images、0 missing、0 hash mismatch；对应记录为 `R004H5_INPUT_R2.json`。主 R004H5 推理只使用第二次资格门授权的冻结输入。

