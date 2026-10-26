# R006V0C 正式 activation cache 结果（第一次完整运行）

状态：`MECHANICAL_PASS__PLANNER_DIRTY_STATE_UNKNOWN__RERUN_REQUIRED_BEFORE_SCIENCE`。

## 结果

- 运行目录：`/data/khsong/aorta_runs/r006v0c_be5dc7d_formal`；
- 冻结实现提交：`be5dc7d`；
- runner exit code：0；audit exit code：0；
- 42/42 source-fit groups、672/672 anchors、42/42 shards完成；
- 四种条件：clean、low exposure、low contrast、cool cast；
- C3=`[1,65,256]`、C4=`[1,1,256]`，registered site语义一致；
- 机械审计261/261 PASS；封存角色未访问，未产生科学指标；
- manifest SHA-256：`55ffc094c8c7f7a62173eccbe33377b1ef923a35f08232689ef318d103733b89`；
- audit SHA-256：`4020fea0e05c3e870c71c7042aeca001b07a59a01c56a8baf8f129381a5bdb09`。

## 不能忽略的 provenance 缺口

manifest记录了planner commit `b0798122597ec8d9f3d09c0a5d019d6d06e2414f`，但因跨用户checkout的
Git ownership保护，运行器回退后把`dirty`记录为`null`。运行后只读检查显示整个仓库有8个研究文档类
改动，但实际执行的`navsim/`子树在该commit上为clean；这提高了可信度，却不能严格重建运行开始时的
dirty state。

因此该缓存不用于正式科学endpoint。运行器和审计器已升级为显式记录并强制检查`navsim/`执行子树
clean及其Git tree object；将从修复后的冻结提交重新生成一次完整缓存。旧结果保留，不覆盖、不删除。

## 主张边界

本结果只说明第一次完整缓存的shape、hash、角色与site机械合同闭合。它不支持低秩介导、DINO可恢复性、
CPTR、NavDream、行为、时延或安全主张。

