# R006V0 协议修正 3：nested split 与唯一 layer/rank 选择规则

状态：`FROZEN_BEFORE_FORMAL_SCIENCE`。

## 原协议缺口

原预注册规定42组六折outer confirmation、outer-train内固定hash划分discovery/selection，以及最小合格
rank，但没有冻结inner组数、hash salt，也没有规定C3/C4同时合格时如何唯一选择。正式结果可见后再决定
这些细节会引入研究者自由度，因此在完整缓存和任何R006V0科学endpoint产生前补齐。

## 冻结规则

1. outer 6折，每折7组confirmation；outer salt为`aorta-r006v0-outer-v1`；
2. 每折35个outer-train groups中，28组basis discovery、7组layer/rank selection；inner salt为
   `aorta-r006v0-inner-selection-v1`；
3. split只接收recording-group标识，不接收RGB、activation、trajectory、utility或任何科学结果；
4. coverage不裁剪，先在组内平均，再对selection groups等权；合格门保持mean至少0.90、最差组至少0.50；
5. 每层先选rank字典`{4,8,16,32}`中最小合格rank；
6. C3/C4均合格时依次按：更小rank、更高最差组coverage、更高组等权mean coverage、固定层顺序
   `C3 planner memory → C4 trajectory query`选择；
7. 无候选合格时该fold fail-closed，禁止读取outer-confirmation补救；
8. outer-confirmation只评价冻结选择程序，不参与split、basis、layer、rank、阈值或tie-break。

## 未改变

层字典、rank字典、appearance、task metric、A0–A7、SESOI、六折稳定性、bootstrap与GO/STOP均不变。
该修正只消除协议歧义，不提供任何低秩介导或方法效果证据。
