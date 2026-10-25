# R005E2A matched-scorer synthetic resource preflight

状态：`PASS_SCORER_SHAPE_AND_MATCHED_COMPUTE_ONLY__NO_SCIENTIFIC_TRAINING`。

- server run：`/data/khsong/aorta_runs/r005e2a_32f179f`；
- scorer/runner source commit：`32f179f`；auditor commit：`5c0fcc4`；
- source archive SHA-256：
  `731019857859c1cec6072cd7401af10b06c34c58481a5fc14e5f7d50fe9cc7fe`；
- exact config SHA-256：
  `79e8a1f8cda626d11ad640b463e0668cebbcc86c0d263d6b57fdebf5e50da04b`；
- GPU/runtime：RTX 3090、PyTorch 2.7.1+cu118、BF16、
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`、deterministic algorithms、math SDPA；
- input shape：batch 1、7 candidates、41 trajectory samples、
  `[1,12,18,32,1280]` frozen visual features，共6,912 visual tokens；
- V0/V1/V2均为4,749,313 trainable parameters，parameter ratio `1.0`；
- profiler记录三臂均为7,623,183,799 FLOPs，ratio `1.0 <= 1.01`；
- scorer-only P95：V0 `16.97ms`、V1 `17.16ms`、V2 `16.12ms`；
- 三臂peak allocated memory均为244,919,296 bytes；
- 同输入两次推理bit-exact、native margin精确0、全部输出有限；
- preflight result SHA-256：
  `cf0f158b7e55d7497b355beb2a6e857ef1ad4e7ff309b9b40921b7981f620c1f`；
- independent audit SHA-256：
  `ad80ff974a39a0eaaea8527101124222ea43a9f66b2e243c23adf352ff532141`。

这些数字只覆盖scorer forward，不含冻结DINO backbone、feature I/O、训练backward、fold refit或完整系统
仲裁时延，因此不能作为最终实时性结论。该preflight未访问科学label或封存角色，不提供视觉可恢复性、
架构优越性、外观鲁棒性或驾驶收益证据。
