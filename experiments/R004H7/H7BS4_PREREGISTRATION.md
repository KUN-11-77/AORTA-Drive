# R004H7-BS4 预注册：视觉任务支持与时延资格链

状态：`H7BS4A_PREREGISTERED_NOT_RUN`。H7BS3R1只证明96×64包含完整候选查询；本阶段
禁止直接训练，先区分“落在空间域内”“获得视觉证据”“teacher关键点可识别”和“实时可算”。

## 数学分层

对anchor (a)、候选\(\tau\)，令\(F_a(\tau)\subset\Omega_{96}\)为完整4秒查询节点，
\(V_a(q)\in\{0,1\}\)表示节点是否至少落入一个合法camera-history patch slot。定义

\[
r_a(\tau)=\frac{1}{|F_a(\tau)|}\sum_{q\in F_a(\tau)}V_a(q).
\]

H7BS4A只审计\(r_a\)和零证据事件，不读取teacher。\(r_a>0\)只是“正确图像可能直接影响
该候选查询”的最小预条件，不是任务可识别性定理；卷积上下文也可能利用查询集合外证据，
而projective support没有处理遮挡。

H7BS4B随后使用带误差账本的96×64 teacher。对节点区间
\([y_q-e_q,y_q+e_q]\)，可能成为set-min的集合定义为

\[
C_a(\tau)=\{q\in F_a(\tau):y_q-e_q\le \min_{p\in F_a(\tau)}(y_p+e_p)\}.
\]

该集合由teacher数值误差导出，不使用事后米制阈值。B报告\(C_a\)中无、部分、全部具有
projective support的候选，并决定缺失证据不确定性/abstention是否必须成为架构组成。

H7BS4C只测计算合同：同GPU、同BF16、同batch、同步CUDA，比较冻结64×64、前向96×64
和128×128控制，分别报告预处理、DINO、投影采样/融合、decoder、candidate query、仲裁、
端到端P50/P95、峰值显存与吞吐。绝对门来自冻结控制周期；若该系统尚无可核验周期预算，
只允许作相对资格，不得事后用观测到的时延制定PASS线。

## H7BS4A 冻结设计

- 数据：42 fit recording groups×16 anchors；group是独立单位；
- 候选：native及全部full-horizon intent-eligible challengers；
- 网格：前向96×64、1m、完整4秒，禁止修改；
- 视觉合同：R004B三相机×四历史帧、畸变模型、half-pixel resize、patch-center凸包；
- 硬失败：任一合法候选的全部查询节点均无直接projective evidence；
- 组级门：继承H7的42/42 groups各至少12/16 anchors，不从本次结果选取；
- 其余支持比例只作描述，不从结果反向设置阈值。

通过A只授权B的teacher缓存与审计；失败则停止当前直接查询接口，分析需要视锥自适应域、
显式空间先验还是拒答，不自动改阈值。development、calibration、test、NavDream与target
继续封存。
