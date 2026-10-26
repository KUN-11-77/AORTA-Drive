# R006V0 协议修正 7：fold-basis 运行、独立重算与停止语义

日期：2026-10-26。状态：`FROZEN_BEFORE_ANY_FOLD_COVERAGE_RESULT`。

## 1. 为什么需要本修正

修正6冻结了数学目标和流式精确归约，但尚未把“谁可用于发现、谁可用于选择、谁必须封存”落实到
可审计的正式运行器。若先观察某一fold的coverage再补审计或调整rank，会重新引入研究者自由度。
因此，本修正在任何真实fold coverage可见前冻结第一科学阶段的执行与失败规则；它不改变问题、数据、
层、rank、阈值或最终R006V0裁决。

## 2. 每折唯一数据流

- basis discovery：该fold固定的28个recording groups，共`28×16×3=1344`个shifted-point样本；
- layer/rank selection：固定的7个groups，共`7×16×3=336`个样本；
- outer confirmation：7个groups的身份可写入角色账本，但其activation、trajectory与任何endpoint均不得
  在本阶段读取或输出；
- 顺序固定为group字典序、anchor `0..15`、appearance
  `low_exposure→low_contrast→cool_cast`。

每个样本在shifted activation处计算tail JVP。目标是修正6冻结的完整样本均值；batch=128只改变内存
驻留，不构成随机小批量优化。C3只允许前64个BEV sites被投影，status token漂移保留在uncovered
residual；C4允许唯一trajectory-query site。

## 3. 求解与选择

- layer：`planner_memory`、`trajectory_query_secondary`；rank：`4,8,16,32`；
- 三个初始化、QR retraction、Armijo参数、最多500步、patience=20均沿用冻结config；
- Armijo单步最多50次回溯，在科学结果可见前写入config；
- discovery objective最低的restart进入selection；
- selection逐样本计算未裁剪coverage；组内平均后再对7组等权；
- 合格门：group-equal mean `>=0.90`且worst-group mean `>=0.50`；
- 唯一选择规则沿用修正3。没有任何候选合格时必须输出
  `STOP_NO_QUALIFIED_INTERFACE`，不得扩大rank、改层、降阈值或挑选有利fold。

## 4. 独立机械审计

审计器不信任运行器写出的summary，必须：

1. 从nested manifest重建28/7/7角色并验证完全一致、互斥；
2. 验证每个basis为`[256,r]`、有限、正交、文件SHA一致；
3. 验证每个候选恰有336个唯一selection样本且只来自selection groups；
4. 从逐样本total/uncovered task drift重新计算全部group coverage与资格门；
5. 重新执行唯一layer/rank选择或STOP；
6. 对primary/replay要求结果JSON与basis文件逐字节一致；
7. 核验checkpoint、正式cache、nested manifest、config、runner和数学模块哈希。

任何机械失败均禁止读取或解释科学结论。六折basis阶段只回答低秩接口是否在inner-held-out组上达到
预注册coverage，不回答outer-confirmation介导、DINO recoverability、CPTR收益、NavDream、驾驶行为、
在线时延或安全性。

## 5. 正式资源与复现合同

正式GPU roster冻结为物理卡`[0,1,2,3]`，最多四个不同fold并发；单卡同一时刻只执行一个fold。
fold按`fold_id mod 4`唯一绑定GPU，且同一fold严格按primary→replay→audit串行执行，因此重放不改变
硬件。调度只并行独立fold，不改变任何fold内部顺序或统计对象。运行健康时只允许读取不含科学数值的
progress字段；六折审计全部结束后方可汇总科学裁决。
