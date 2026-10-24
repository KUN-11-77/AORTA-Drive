# R004H7-BS4B0A2：672 个精确 fit token 的 forward96 teacher

状态：`PREREGISTERED_NOT_RUN`。

B0A 已证明生成器在 164 个资格 cache 上可重复，但那些 token 与 H7BS4A-D0 不同。B0A2
修复这个角色错位：用 H7BS4A 结果中冻结的 `(log_name, scene_token)` 与 R004E 1440-cache
manifest 精确连接，只选取 42 个 train fit groups 的 672 个 anchor。任何 development、近邻
scene、token 替代或标签跨场景复用都立即失败。

每个 96×64 标签从对应 metric cache 的原始 polygon 重新计算，不把旧 64×64 标签插值到前向
区域。旧工件唯一合法用途是同一 token 的前 64 列逐位回放：对 `label_m`、transform、distance
和 total-error 四个数组均要求 bit-exact。该检查同时验证新定义域没有悄悄改变已资格区域语义。

## 成功与停止规则

- 精确连接 672/672 anchors、42/42 groups，且全部为 train；
- 生成 4,128,768 个节点，672 个 artifact 内部字节回放通过；
- 独立 audit 重新打开全部 artifact，四数组有限、误差方向正确、旧前缀逐位相同；
- 任一身份、hash、角色、前缀或数值错误即 FAIL；120 分钟墙钟上限，禁止调参后重跑。

PASS 仅授权另行冻结 B0B 的 teacher-critical observability 公式与代码。它仍不回答哪些盲区会
控制 set-min，更不证明视觉、决策、OOD、实时性或驾驶收益。
