# Round 2 External Method Review

Reviewer: `gpt-6.1-sol`, reasoning `xhigh`（目标GPT-5.4当前不可用，降级已记录）  
Verdict: `REVISE`  
Score: `8.40 / 10`

## 已解决

- 撤回utility-gap all-pair logistic的一般Bayes一致性主张；
- 明确hindsight、Bayes、函数类、优化和估计五层不可互相替代；
- capacity probe只在“post-GELU representation + 最后一层线性权重”处主张严格子类；
- E3C被限制为局部描述，不再结果后选择处理；
- R005E3D改为统计一致的proper conditional-mean目标，且唯一改变loss；
- 失败统一标为冻结资格门失败并逐项报告，不冒充不可实现性定理。

## 剩余阻塞

R005E3D实验设计本身没有剩余冻结阻塞，但整个AORTA仍缺少组外确认、正确视觉条件增量、
Differential Query相对matched scorer的结构优势，以及NavDream/行为收益。因此总体仍为REVISE，不能因
G1资格实验设计严谨就升级为ICCV方法成功。
