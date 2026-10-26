# R006V0S 结果：冻结 LTF 因果接口机械资格

状态：`MECHANICAL_PASS__NO_SCIENTIFIC_EFFECT_CLAIM`。

## 结论

R006V0S 在 2 个不同 source-fit recording groups 上完成 primary 与 same-GPU replay，二者 JSON
逐字节一致（SHA-256 均为
`48b46ff8696078410283c07b8e268a9e4a194cb61ccaf334fb8ae3246fb3983a`）。55 项机械审计全部通过；
development、calibration、test、NavDream 与 target 均未访问，且未输出科学效果指标。

## 失败驱动的接口修正

旧原型把 C1 挂在图像编码器最后一个 return layer。此时早期 fusion 已把视觉扰动复制进 latent BEV
支路，因此该张量不是完整因果切口：两组 full clean patch 的恢复 RMS 分别为 0.1256 m 与 0.1568 m。
该失败被保留为 `smoke_attempt2_incomplete_cut_fail.json`，没有通过放宽阈值处理。

C1 随后被严格改为 **第一次图像编码输出、任何跨模态 fusion 之前**。这是一个完整视觉因果切口：
所有后续视觉路径都经过该张量，同时仍属于 learned visual representation。修正后 C1–C4 的 same-scene
full patch 在两组上均精确恢复 clean 轨迹（RMS 0），反向 patch 的轨迹扰动分别为 0.1225 m 与
0.0997 m。后者只证明 hook 非退化，不是总体因果效应估计。

## 可复算性与来源

- checkpoint：`ltf_seed_0.ckpt`，SHA-256
  `d08e97ba6beb937412697d619171553041622859e026a06935a2e0e118e6309d`；
- 实际 planner 源码：`/home/jbwang/cogatedrive`，commit
  `b0798122597ec8d9f3d09c0a5d019d6d06e2414f`；
- model 与 preprocessing 文件 SHA 已写入结果；由于跨用户 Git ownership，dirty-state 无法可靠读取，
  正式 R006V0 必须继续以逐文件哈希冻结，而不能把 dirty-state 误报为 clean；
- deterministic CUDA：`CUBLAS_WORKSPACE_CONFIG=:4096:8`；Python 3.10.18、PyTorch 2.7.1+cu118；
- synthetic dense-Jacobian oracle 通过：objective 与 gradient relative error 均为 0，projector distance
  `3.8511e-16`。

## 主张边界

本结果只说明：冻结 LTF 的四个登记层已经具有可重复、可全量替换、数值有限的 activation-intervention
接口，并且 task-weighted basis 求解器通过合成 oracle。它不证明低秩介导、DINO 可恢复、外观鲁棒、
NavDream 泛化、official utility、驾驶安全或 CPTR 有效。

紧凑证据位于 `experiments/R006/results/r006v0s_e6208cd/`。
