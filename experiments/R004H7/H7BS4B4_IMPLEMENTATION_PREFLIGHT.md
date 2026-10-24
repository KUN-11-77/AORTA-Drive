# R004H7-BS4B4 implementation preflight ledger

## Preflight 1 — failed before scientific output

- frozen protocol commit: `10a01e6`;
- scope touched: first source-fit anchor only；未访问 development、calibration、test、NavDream 或 target；
- emitted scientific metrics: none；仅请求输出 candidate count 与三个实现布尔量，但在其生成前中止；
- failure: `navsim.common.dataclasses.Trajectory` 拒绝 native 的 9 个 poses，因为其 4 s / 0.5 s
  sampling 合同要求 8 个未来 poses；
- cause: `sample_cached_native_trajectory` 有意返回 `[current origin + 8 future poses]`，challenger provider
  有意排除 current origin；B4 runner 对 challenger 使用正确的 8-point contract，却未对 native 去掉 origin；
- permitted repair: 仅将 native scorer input 改为 `native.poses[1:]`；
- frozen invariants: estimand、candidate provider、42×16 anchors、official scorer、traffic policy、统计单位、
  bootstrap、GO/HOLD/STOP 门和封存数据边界全部不变；
- scientific interpretation: none。该失败是接口形状错误，不支持或否定 preference hypothesis。

修复后必须更新 runner SHA、提交并推送，随后用相同第一 anchor 重做不输出科学数值的资格预检。

## Full materialization — reproducible but scientifically invalid

- frozen protocol commit：`599a2a0`；primary 与完整 replay 均完成 672/672 anchors；
- mechanical audit：两次 anchor ledgers、summary 与 root contract exact，`failure_count=0`；
- formal decision：`INVALID_PREFERENCE_LABEL_INTERFACE`；
- exact cause：48 anchors 的冻结 `challenger_eligible` 非全真，但 runner 一方面把
  `pose_exact AND all_eligible` 合并命名为 `panel_reconstruction_bit_exact`，另一方面又将 168 个
  intent-ineligible challengers 纳入 oracle summary；
- failure-only attribution：48/48 的 realized poses 实际逐 bit exact、最大误差 0；15 anchors
  全部 challenger 不合格，33 anchors 部分不合格，影响 27 groups；
- action：不解释正向 oracle diagnostics，不自动第三次正式运行，不授权训练。若继续必须以新的
  R1 预注册统一 panel membership、pair domain 与统计分母。
