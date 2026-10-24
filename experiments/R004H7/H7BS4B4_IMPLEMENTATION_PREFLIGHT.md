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
