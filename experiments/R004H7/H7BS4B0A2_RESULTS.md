# R004H7-BS4B0A2：精确 fit-token forward96 teacher 结果

状态：`PASS_FIT_ARTIFACT_QUALIFICATION_ONLY`。

## 结果

- H7BS4A 与 R004E 通过精确 `(log_name, scene_token)` 连接 672/672 anchors；
- 仅包含 42/42 train fit groups，无 development/calibration/test/NavDream/target；
- 生成 4,128,768 个节点标签（672×6144）；
- 672/672 artifact 内部确定性字节回放通过；
- 新 96×64 工件的四个数组，在相同 token 的前 64 列与旧资格工件逐位一致；
- 独立机械 audit 重新打开 672/672 artifact，hash、形状、有限性、误差方向和前缀检查失败均为 0；
- 构建用时 1560.843 s；机械审计 `PASS`；外部 reviewer `REVIEW_UNAVAILABLE`。

正式 manifest SHA-256：
`65e686e06522fddf4950f1fe772fbcbea0547cec3d54fb383a8ee33d208b9fd0`。

## 裁决

B0A2 修复了 B0A 的 token 角色错位，授权在这 672 个 fit anchors 上开展另行预注册的
teacher-critical observability 诊断。它本身不读取视觉预测、DINO 值、planner outcome 或
驾驶评分，所以不能证明盲区关键、视觉模型有效或决策收益。H7BS4A 的 STOP 保持不变。
