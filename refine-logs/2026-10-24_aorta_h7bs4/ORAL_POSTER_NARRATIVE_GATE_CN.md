# AORTA Oral / Poster 叙事资格门

状态：`FROZEN_NARRATIVE_GATE__SCIENTIFIC_CLAIMS_PENDING`  
目的：约束方法设计和实验顺序；禁止在结果出来后把互不相干的模块拼成“创新故事”。

## 1. 一句话记忆点

> **天气改变了像素，却没有改变道路；真正需要跨外观保持的不是整幅特征图，而是足以区分冻结
> planner 候选行为的任务算子。**

论文的中心名词固定为 **decision-sufficient operator invariance**。DINO、ROI、SDF、Hodge、
uncertainty 与 reranking 都只能作为实现或对照，不能各自成为一条主线。

## 2. 反常现象

同一物理场景的外观改变后，已有模型可能同时出现：

1. 全局特征仍高度相似；
2. 稠密道路场平均误差不大；
3. 但候选轨迹的安全排序发生翻转。

反方向也成立：一个跨外观几乎不变的输出可能只是稳定数据先验，不包含正确场景信息。因此，
`feature invariance` 与 `field invariance` 都不是驾驶决策鲁棒性的充分条件。

## 3. 一个数学对象

对冻结候选面板 \(\Pi=\{\tau_0,\ldots,\tau_K\}\)，令 \(F_j\) 为候选 \(\tau_j\) 的完整时域
扫掠节点集合，\(d\) 为视觉条件任务势。定义

\[
M_d(\tau_j)=\min_{q\in F_j}d_q-m_0,
\qquad
\Delta_{j0}(d)=M_d(\tau_j)-M_d(\tau_0).
\]

方法只需恢复可支持候选决策的 \(\Delta_{j0}\)，而不是声称重建完整世界。由于
\(\Delta_{j0}(d+c\mathbf 1)=\Delta_{j0}(d)\)，可辨识对象属于共同 offset 的商空间。由于相机
部分可观测，算法还必须给出与视觉支持共同定义的可行势集合 \(\mathcal U(X,O)\)，最终决策量为

\[
L_{j0}(X)=\inf_{d\in\mathcal U(X,O)}\Delta_{j0}(d).
\]

这三个式子分别对应论文的表示对象、可辨识性与选择性决策规则；不得再增加第二套互不闭合的
核心数学对象。

## 4. 被问题迫使产生的三项设计

1. **Quotient potential：**共同 offset 不可辨识，所以网络学习 mean-zero 节点势及其 exact edge
   differences；其必要性必须胜过容量匹配的 direct-edge、direct-pair 与普通 dense-field head。
2. **Support-conditioned shared uncertainty：**缺失视觉证据可能控制 set-min，所以缺失区域必须
   扩大共享潜在场集合或触发拒答；其必要性必须胜过 shared node box 与 candidate-independent
   intervals。
3. **Native-preserving robust arbitration：**视觉证据不足时不应强制替换冻结 planner；只有
   \(L_{j0}\) 超过预注册阈值并满足完整时域、候选合法性及时延合同时才允许干预。

若任一结构不能在相同输入、监督、容量和候选合同下胜过更简单对照，它必须从主贡献删除。

## 5. Figure 1 只允许三栏

1. **Paradox：**同一路、不同天气；全局表征相似，但 native/challenger 排序翻转。
2. **Failure of old object：**普通 feature/dense field 忽略 candidate-critical support，短前缀又删除
   晚期分叉。
3. **AORTA closure：**候选联合图上的 quotient operator → support-conditioned shared set →
   worst-case native-preserving arbitration。

Figure 1 禁止加入训练细节、所有 loss、几何证书历史或完整系统框图；这些进入后续图表。

## 6. 30 秒与 2 分钟口述合同

### 30 秒

> 外观鲁棒驾驶通常让特征或稠密地图在不同天气下保持一致，但我们发现高表征相似度并不能
> 阻止候选安全排序翻转。AORTA 把鲁棒性对象改成候选面板上的相对任务算子：在候选联合域上
> 学习可积的 quotient potential，用视觉支持条件化一个共享不确定集合，并且只在所有兼容势场
> 都认为 challenger 优于 native 时才干预。这样，外观稳定性、可观测性和驾驶决策由同一个对象
> 闭合。

### 2 分钟

1. 用一组真实失败展示“feature stable, decision unstable”的矛盾；
2. 用 gauge 反例说明绝对场不是必要对象，用 unsupported minimizer 反例说明稳定场也可能缺少
   决策关键证据；
3. 从两个反例推出 quotient potential 与 support-conditioned shared set；
4. 给出 \(L_{j0}\) 的鲁棒仲裁规则，并说明看不清时保持 native；
5. 只报告四类决定性结果：正确图像依赖、结构必要性、NavDream/跨 planner 决策收益、端到端时延。

## 7. Oral 级与 Poster 级证据门

### Poster 竞争力最低门

- 未见组与 NavDream 上同时满足 task correctness 和 operator stability；
- correct image 明显优于 mismatch、removed-image 与 source prior；
- quotient、shared uncertainty 与完整时域至少各有一项容量匹配消融支持；
- 冻结 planner 的排序或官方指标有可复现收益，并报告 clean harm、abstention 与最差组；
- P50/P95/P99 时延与峰值显存满足冻结预算。

### Oral 候选附加门

- 反常现象跨至少两种 planner 架构重复；
- 方法不仅提高均值，还显著修复原生方法的典型 failure mode，且有双向/反事实证据；
- 一项核心结果能被一句话复述，并直接排除最强竞争解释；
- 理论对象、架构和主要实证共用同一符号与同一评价单位，而不是三个平行故事。

是否被选为 Oral 由审稿和会议决定；本门只定义应争取的证据质量，不作承诺。

## 8. Rebuttal 前置问题

投稿前必须能用一张表回答：

1. 为什么不是 DINO 更强或数据先验？
2. 为什么不是普通 SDF/BEV、FlowDrive-style 或 READ-style field？
3. 为什么 quotient structure 不只是参数化差异？
4. 为什么联合集合优于节点边际或独立候选区间？
5. 部分可观测区域为何不会被错误插值成安全？
6. 为什么短时重规划不能替代完整时域？
7. 对 NavDream 的证据是稳定且正确，还是仅仅不敏感？
8. 改善的是候选排序、官方驾驶行为，还是仅视觉代理指标？
9. clean harm、拒答率、最差组和选择偏差是什么？
10. 增加的性能是否值得端到端时延与显存成本？

任一关键问题没有预注册的实验回答，相关内容不得写成主贡献。

## 9. 当前真实性边界

现有证据支持：问题真实、fit 视觉机制非平凡、固定支持域存在 teacher-critical 盲区。尚未证明：
joint uncertainty 的必要性、quotient head 的匹配容量优势、NavDream 泛化、跨 planner 决策收益、
clean 无害和实时部署。当前故事具有一致性，但还不是已完成的 ICCV 论文。

### B1 后修正

H7BS4B1 的预注册 non-vacuity 门未通过：joint Lipschitz 虽缩窄部分区间，却没有比 shared-node
box 增加 sign-identifiable pairs，且组级覆盖不足。故第 4 节中的 shared uncertainty 当前仅是
被检验但未获支持的候选设计，不能进入 Oral/Poster 核心口述。下一版故事是否转向候选共同前缀
与分支后缀的算子分解，必须由 H7BS4B2 的零差异归因决定。
