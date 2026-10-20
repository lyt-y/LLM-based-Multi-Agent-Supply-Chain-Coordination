# Methodology

## Updated implementation and active settings

| Setting | Current definition |
|---|---|
| Environment calibration | First 130 source weeks; approved evaluation is 130 periods |
| Opening inventory | max(1, ceil(linear 95th percentile of two-week rolling demand sums)) |
| New-order capacity | max(1, ceil(linear 95th percentile of weekly demand, including zeros)) |
| Capacity scope | New orders only; fulfillment is inventory-limited |
| Lead times | Two weeks at every stage |
| Holding/backlog/fixed-order coefficients | One each |
| s,S training | First 26 weeks; sample standard deviation, ddof=1 |
| Synthetic input | Revision 2; smooth p=1, k=25, theta=0.8 |
| Gemini | gemini-3.8-flash; temperature 0; low thinking |
| Current prompt versions | no communication v7; adjacent v10; no forecast v6; controlled adjacent v1 |

The source in `src/` is unchanged. Runtime calibration is computed from frozen
demand. The obsolete rho-based calibration sweep is excluded from active artifacts,
with only its archive location/hash retained. It is not an alternative default.
Exact current provider options and per-run configurations are in the
[manifest](../results/manifest.json). Abandoned historical settings are excluded from the public experiment records.

## Implementation, evaluated evidence and incomplete work

All five methods are implemented. s,S, no communication, adjacent communication and
no forecast have retained experimental results. Controlled adjacent has offline
regression coverage but no retained completed experimental results; there is no
empirical performance claim for it.

The [summary](../results/summary.json) contains 32 valid completed runs: 16 M5
and 16 synthetic revision-2 runs, all evaluated over weeks 1-130 under the current
protocol. Eligibility requires current calibration, frozen inputs, policy/prompt
settings and explicit successful completion; horizon alone is insufficient.
The abandoned 277-period M5 runs and four abandoned reference rows are excluded.
A separate intermittent job is paused at 19/130 periods with two queued jobs.
These jobs and unevaluated controlled adjacent are excluded from completed counts.
No result has been truncated, relabelled or fabricated.

Stable run IDs link every row to its original run identity, configuration, policy,
source path and available checksum. Full per-run KPI values and resource accounting
are preserved without new aggregation. Retained provider configurations match the current
reference below. Figure reproduction from summaries does not independently audit
trajectory KPIs or recreate paid responses. That requires original external evidence;
remote backup has not been verified.

## Publication layout

Two files, `results/summary.json` and `results/manifest.json`, replace overlapping
summary, batch and per-item projection files. The four retained PDFs are in
`results/figures/`. All four README plotting commands are offline;
PNG/table exports are disposable. New experiments use a separate output root and
retain the simulator's normal per-method result/checkpoint schema. Historical paths
are provenance, not local paths to rewrite or pass to an execution supervisor.

The M5 demand figure retains `FOODS_1_188_WI_1_evaluation` as its intermittent
series; the updated comparison uses `FOODS_2_368_TX_2_evaluation`. Preserving this
distinction leaves the existing plotted data unchanged.

## Scientific definitions

The following scientific sections preserve the original methodological decisions.
Repeated source listings, machine snapshots, setup instructions and handoff notes
are omitted; implementation details remain inspectable in the source.

## 1. 研究问题与比较范围

研究对象为单产品、四级、离散时间、多周期补货系统。研究比较固定规则基线与同一 Gemini 模型在不同局部信息、外部预测输入、相邻信息共享机制下的运营表现和计算资源开销。

可研究的问题包括：

- 在相同需求序列、相同初始条件和物理/成本参数下，LLM 局部补货策略与固定 `(s,S)` 基线的成本、服务及订单波动有何差异？
- 增加相邻节点的期初状态与局部 insight 后，系统表现是否变化？
- 删除显式外部需求预测输入后，局部决策是否变化？
- 相同方法在真实 M5 需求和独立 Bernoulli–Gamma 合成需求上是否呈现不同表现？
- 若使用自主选择通信的协议，消息交付量与独立通信选择调用的资源代价如何变化？

这是策略/信息协议的仿真对照，不是求解全局最优库存控制问题，也不是多代理强化学习训练。各 LLM 节点收到“帮助降低全链运营成本”的目标，但实际仅看到允许提供的局部/邻接信息，没有全局优化器、联合规划或保证达到全局最优的机制。

## 2. 程序架构与实验单位

| 模块 | 实际责任 |
|---|---|
| `src/demand/m5_demand.py` | M5 周聚合、需求类型刻画、代表序列选择、冻结输入及旧预测文件导出 |
| `src/demand/synthetic_demand.py` | 读取冻结合成序列；验证并附加来源路径/哈希 |
| `src/demand/demand_data.py` | 非负整数需求验证、库存和订单上限校准 |
| `src/demand/forecast.py` | 根据各节点已完成的下游需求构建在线移动平均 |
| `src/simulation/config.py` | 两种数据集各自独立校准、四级环境配置 |
| `src/simulation/env.py` | 确定性收货、需求、履约、库存、积压和成本更新 |
| `src/simulation/experiment_interface.py` | 顺序决策、观察、通信路由、校验、轨迹、检查点及重放 |
| `src/policies/ss.py` | 固定 `(s,S)` 阈值和动作 |
| `src/agents/gemini_*.py` | 英文提示词、SDK 调用、响应解析、协议与调用诊断 |
| `src/evaluation/experiment_results.py` | 运营、服务、牛鞭指标、详细结果原子保存 |
| `src/evaluation/metrics.py` | 调用、Token、消息、通信轮次、运行计时 |
| `src/evaluation/summary.py` | 运行编号、汇总和来源记录 |
| `scripts/run_experiments.py` | 新实验、恢复及延长的统一入口 |
| `scripts/run_m5_smooth_batch.py` | 当前 M5 smooth 三策略串行批次、网络失败重连 |

实验单位为“数据集 × 需求序列 × 策略 × 运行编号 × 评估周数 × 参数版本”。每个单位执行一次从初始状态开始的完整 episode，或由兼容的检查点复用前缀继续执行。

Run IDs are not seeds. Both dataset inputs retain 277 source weeks; the approved evaluation uses only weeks 1-130.

## 3. 需求数据

### 3.1 M5：真实日销量转为周需求

原始预处理入口读取 `sales_train_evaluation.csv`，要求 `id`、`item_id`、`store_id` 及 `d_1` 到 `d_1941`。每条商品—门店序列作为一个候选需求记录，日值转换为非负整数；负值会拒绝。

周需求定义为：

\[
D_t=\sum_{j=7(t-1)+1}^{7t}d_j,\qquad t=1,\ldots,277.
\]

使用不重叠的连续七天块，保留 `d_1…d_1939`，丢弃 `d_1940,d_1941`。第 1 周不是按日历 ISO 周重新分组。主实验仅用销量序列，不使用售价、节假日、促销、日历特征、门店间流动或产品间替代关系。

冻结的 `selection_manifest.json` 记录原始候选 30,490 条，合格 30,490 条，排除 0 条；分类数量为 smooth 8,725、erratic 1,144、intermittent 17,260、lumpy 3,361。这些数量来自已保存的预处理清单，并不意味着本次重新执行了全部 CSV 预处理。

原始 CSV 默认解析路径为项目父目录的 `Data/sales_train_evaluation.csv`。日常实验直接读取 `artifacts/m5/representatives.json`，不要求再次读原始 CSV。本项目加载器要求每条 M5 冻结记录恰有 277 个非负整数观测；评估时再取前缀。

### 3.2 需求分类

设完整序列长度 `N=277`，正需求周集合 `P={t:D_t>0}`，数量 `n_+=|P|`。所有分类统计使用完整 277 周，不是仅前 26 或前 130 周。

\[
ADI=\frac{277}{n_+},\qquad
CV_+^2=\left(\frac{s_+}{\bar D_+}\right)^2,
\]

其中 `s_+` 是正需求子样本的样本标准差，`ddof=1`。ADI 中零需求周参与总长度，CV² 中仅正需求参与均值/标准差。

| 类型 | ADI | 正需求 CV² |
|---|---|---|
| smooth | ≤ 1.32 | ≤ 0.49 |
| erratic | ≤ 1.32 | > 0.49 |
| intermittent | > 1.32 | ≤ 0.49 |
| lumpy | > 1.32 | > 0.49 |

边界等于阈值时归入低侧。若正需求周少于 2，记录排除；若滞后 1 周自相关的分母为 0，也排除。自相关为 `D_1…D_276` 与 `D_2…D_277` 的 Pearson 相关，包含零值，各子序列分别中心化。

“smooth”只代表 ADI/正需求 CV² 的统计分类，不表示序列没有趋势、结构变化、季节性或峰值。整段 smooth 也不保证前 130 周或前 26 周有相同均值与分类。

### 3.3 代表序列选择

每个需求类别独立选择 30 个真实 medoid，共 120 条冻结记录。特征为：

\[
\mathbf f=(\log(1+\bar D),ADI,CV_+^2,\rho_1).
\]

在各类别内，对每个特征以中位数中心化并除以 IQR；25%/75% 分位使用线性插值；IQR 为 0 时尺度设为 1。距离为标准化空间的欧氏距离。

实现为确定性的 alternating k-medoids，不是 PAM，也不保证全局最优：候选按 ID 排序；首个 medoid 是距逐分量中位数最近的记录；其余 medoid 以 farthest-first 选取；分配到最近 medoid 后，在每簇选择到簇内成员距离总和最小的成员；并列更新保留原 medoid；medoid 自己归属自己的簇，即使特征重复。最多 300 次迭代；距离计算更新块大小 256；未收敛会报错。输出按记录索引/ID 顺序，而不是按“代表性最好”排序。

当前 smooth 研究记录为 `FOODS_1_002_WI_1_evaluation`。它是已选代表记录之一；代码没有单独证明它是 smooth 类最典型、最接近类别中心或对所有 smooth 的无偏代表。`--all` 对 M5 会运行全部 120 条，而不是默认四条。

### 3.4 合成需求：Bernoulli–Gamma

对每个类别生成一条 277 周序列，重复次数为 1。冻结数据集标识为 `bernoulli_gamma_v1`。

\[
A_t\sim Bernoulli(p),\quad X_t\sim Gamma(k,\theta),\quad
D_t=A_t\lceil X_t\rceil.
\]

Gamma 使用 shape–scale 参数化；发生标记与需求大小按生成约定独立抽样，时间上没有显式趋势、自相关或季节性机制。即便 `A_t=0`，生成流程也抽取该周的 Gamma 大小。整数化只向上取整，没有裁剪、截尾、重抽样或按实际均值归一化。

设连续总体目标均值为 20，即 `p*k*theta=20`：

| 类别 | p | k | θ | 种子向量 |
|---|---:|---:|---:|---|
| smooth (revision 2) | 1.0 | 25 | 0.8 | [42,0,0] |
| erratic | 0.95 | 1 | 20/0.95≈21.052632 | [42,1,0] |
| intermittent | 0.40 | 5 | 10 | [42,2,0] |
| lumpy | 0.40 | 1 | 50 | [42,3,0] |

生成约定：NumPy 1.26.4；`default_rng(SeedSequence([42,regime_index,0]))`，PCG64；类别索引依次 smooth、erratic、intermittent、lumpy，均为零基索引。先一次抽取全部 277 个 `binomial(1,p)`，再一次抽取全部 277 个 Gamma 大小，最后整数化。交换抽样顺序将改变序列。

20 is the continuous population mean before ceiling. In regime order, current frozen 277-week means are 20.4982, 23.4079, 21.3755, 20.0469; zero-week counts are 0, 18, 163, 166. These describe source inputs, not 277-period experiments. The loader does not regenerate these series during evaluation.

### 3.5 数据冻结与来源

M5 加载器为记录附加 `dataset_source='m5'`、冻结 JSON 的绝对路径和文件 SHA-256；合成加载器为记录附加 `dataset_source='synthetic'`、相同类型的来源信息，并检查重复 ID 与需求整数契约。环境工厂拒绝显式标注为另一数据源的记录。

结果中的 `input_sha256` 是所加载冻结 JSON 整个文件的哈希，不是单个商品序列的哈希，也不是原 CSV 哈希。原始 CSV 哈希位于 M5 选择清单。检查点另存运行需求前缀的 SHA-256。冻结合成生成清单的环境校准文件属于历史快照，当前环境工厂不读取它来覆盖运行参数。

## 4. 时间窗口与环境校准

### 4.1 三个窗口必须分开

| 目的 | 当前窗口 | 与主评估是否重叠 |
|---|---|---|
| M5 分类与选择 | 完整 277 周 | 是 |
| 初始库存、订单上限校准 | 原始冻结序列前 130 周 | 是 |
| `(s,S)` 阈值校准 | 前 26 周 | 是 |
| 当前主要评估 | 第 1～130 周 | — |
| LLM 在线移动平均 | 决策前最多 26 个已完成周 | 只用截至 t−1 的本级下游需求 |

The generic runner calibrates from the full source record before selecting an evaluation prefix. Only 130-period evaluation is approved here; always pass --periods 130. Frozen inputs retain 277 source weeks.

### 4.2 线性分位数的精确定义

对排序后的 `n` 个值 `x_(0)…x_(n−1)`，令 `r=(n−1)q`，`j=floor(r)`，`a=r−j`。分位数为相邻排序值的线性插值 `Q_q=(1−a)x_(j)+a*x_(j+1)`；端点按有效下标处理。这对应 NumPy `quantile(...,method='linear')`。

这里的 95% 是分位位置，不是把需求乘以 0.95。零需求保留在分位数样本中。

### 4.3 初始库存

设 `n_c=min(130,N)`，在前 `n_c` 周构造 `n_c−1` 个两周重叠需求和：

\[
W_u=D_u+D_{u+1},\quad u=1,\ldots,n_c-1;
\qquad I^{init}=\max(1,\lceil Q_{0.95}(W)\rceil).
\]

四级均初始化为该值，初始积压、订单、历史销售和在途管道为 0。两周滚动和使用 `valid` 卷积，不填补不存在的第 0 周或第 131 周需求。取分位后向上取整。

### 4.4 新订单上限

\[
c=\max(1,\lceil Q_{0.95}(D_1,\ldots,D_{n_c})\rceil).
\]

四级固定使用相同 `c`，整个 episode 不随预测或需求变化。变量名 `prod_capacities` 是历史兼容名称，实际语义为 new-order quantity limit。

| 记录 | 每级初始库存 | 每级新订单上限 |
|---|---:|---:|
| M5 smooth `FOODS_1_002_WI_1_evaluation` | 17 | 11 |
| 合成 `smooth_001` | 65 | 37 |
| 合成 `erratic_001` | 96 | 60 |
| 合成 `intermittent_001` | 115 | 86 |
| 合成 `lumpy_001` | 173 | 111 |

这些数值应以文末机器导出快照复核。M5 smooth 旧结果上限为 7，不能当作这里的上限。两套数据使用相同规则，而非相同数值。

## 5. 供应链模型与状态转移

### 5.1 假设和符号

阶段编号 `m=0,1,2,3` 分别为零售商、批发商、分销商、制造商；周编号 `t=1,…,H`。信息与订单从零售商向制造商传递，实物流向相反。

| 符号 | 含义 |
|---|---|
| D_t | 第 t 周真实客户需求 |
| O_(m,t) | 第 m 级本周已接受的新补货订单 |
| Y_(m,t) | 本周发往第 m 级的补货货物（发运量） |
| F_(m,t) | 第 m 级本周向下游履约/销售量 |
| A_(m,t) | 本周到货后、履约前的可用期初库存 |
| I_(m,t) | 周末库存 |
| B_(m,t) | 周末本级欠下游的积压 |
| U_(m,t) | 决策时本级尚未被上游发出的旧补货欠货 |
| L_m | 发运后到货提前期，当前均为 2 周 |
| c_m | 每周新订单上限，当前同一序列四级相同 |

`arriving_orders[m,t]` 在代码里保存的是第 t 周发往节点 m 的发运量 `Y_(m,t)`；不是“第 t 周已经实际收到的量”。实际到货需要再偏移 `L_m` 周。

模型为单一非易腐产品，单位转化比 1:1，没有随机损耗、批量舍入、运输成本、销售收入、变量订货成本、产能投资、价格决策、运输随机延迟、取消订单或失销。未满足需求全部结转积压。没有独立发货/生产/处理吞吐上限。

物理环境在给定冻结需求和合法订单后是确定性的。环境模块默认设置 NumPy seed=0，空间初始化 seed 默认也为 0，但当前转移没有额外随机抽样；不能将这两个 seed 写成 Gemini 的随机种子。没有对历史销量作缺货截断/潜在需求恢复，所以 M5 使用的是观测销量作为实验需求代理，不是识别出的无约束真实需求。

### 5.2 期初收货

\[
A_{m,t}=I_{m,t-1}+Y_{m,t-L_m},
\]

其中当 `t−L_m<1` 时对应发运量为 0。订单不是立即变为现货；两周提前期从实际发运开始，而不是从提交订单开始。旧订单因上游缺货可能先进入未发运欠货，随后发运后再等两周。

观察中的 inventory 已包含本周期初到货，deliveries 仅包含之后各周已发运的货物。不能把本周已到账货物再作为在途加一次。

### 5.3 当周需求与履约

定义本级当周新下游需求：

\[
d_{m,t}=\begin{cases}D_t,&m=0,\\O_{m-1,t},&m=1,2,3.\end{cases}
\]

所有四级订单接受后，统一执行物理更新：

\[
F_{m,t}=\min(B_{m,t-1}+d_{m,t},A_{m,t}).
\]

对内部节点 `m=0,1,2`：

\[
Y_{m,t}=F_{m+1,t}.
\]

制造商外部供给边界：

\[
Y_{3,t}=O_{3,t}.
\]

外部供给按制造商本周接受订单全额发运，两周后到账，不产生外部未发运 backlog。制造商对分销商的履约依然受其本级现货约束。

当库存足够时，`F_(m,t)` 可以超过 `c_m`，例如同时清理旧 backlog 与本周需求。订单上限只要求 `O_(m,t)∈{0,…,c_m}`。实际履约没有再次截断到 `c_m`。

### 5.4 周末状态

\[
B_{m,t}=B_{m,t-1}+d_{m,t}-F_{m,t},\qquad
I_{m,t}=A_{m,t}-F_{m,t}.
\]

数值保持非负整数。代码将旧 backlog 与新需求合并计算总履约，没有保存逐个客户订单的队列、年龄或等待时间分布；因此不能从结果声称测量了单笔订单的 FIFO 等待时间或按时履约率。

同周内部发运不会同周供给下一节点消费，因为提前期为 2。四级履约基于各自本周期初可用库存，并非从制造商开始让货物在同周穿越全链。

### 5.5 库存位置中的欠货与在途

对 `m<3`，决策时 `U_(m,t)=B_(m+1,t−1)`；对制造商 `U_(3,t)=0`。这是本级向上游提交过但尚未发运的旧订单余额，不是本级对下游的欠货，也不是供应商完整私人状态。

在固定 `L=2` 下，未来到货列表在决策周 t 表示本级在 t+1、t+2 周已承诺到货；只有在以前已完成周发出的货物计入。实际上固定两周提前期且在当周订单前观察时，第二个未来槽通常为 0，因为要在 t+2 到货需 t 周发运，而该发运尚未发生。仍保留两槽统一观察结构。

内部节点的在途库存一经上游发运，已经不在上游现货中；运输期间没有额外在途持有成本。

### 5.6 边界与终止

第 0 周库存为校准库存，其他状态为 0。完成 H 周后终止，不自动增加清算周期。没有末期库存残值、末期在途估值、未完成订单注销或另加终端 backlog 罚金。末周只按同样周末库存/积压收费。剩余 backlog、在途和库存由轨迹/重放状态解释，而不是在模型外补作隐含清算。

## 6. 运营成本与目标

每级每周成本：

\[
C_{m,t}=h_m I_{m,t}+b_m B_{m,t}+K_m\mathbf1(O_{m,t}>0),
\qquad h_m=b_m=K_m=1.
\]

全链总运营成本及奖励：

\[
C^{op}=\sum_{t=1}^{H}\sum_{m=0}^{3}C_{m,t},\qquad
r_{m,t}=-C_{m,t},\qquad R^{episode}=-C^{op}.
\]

持有与积压按周末状态计算；固定订货费只看接受的新订单是否为正，即便上游暂时不能发货也收费。没有订货数量比例费、采购成本、需求收益、折现、API 价格换算或 Token 与运营成本的加权目标。

单次 episode 的运营指标来自不可变的逐周成本快照，不应从之后被到货更新的观察倒推。资源开销与运营表现分开报告。当前成本权重是归一化研究设置，不是根据 M5 真实利润/罚金估计的货币成本。

初始库存没有单独的采购/投入成本，只在运行周末仍剩余时产生持有成本。环境没有额外仓储容量上限；observation_space 的数值上界是编码空间声明，不是限制库存的物理容量规则。

## 7. 决策顺序与允许的信息

### 7.1 一周完整流程

1. 投影/接收已到期补货，构造四级期初观察。
2. 若策略需要通信，按协议完成期初共享或通信选择；这时没有任何本周订单。
3. 零售商决策并验证 `O_(0,t)`。
4. 批发商在本级观察中收到已接受的本周零售商订单，决策 `O_(1,t)`。
5. 分销商收到本周批发商订单，决策 `O_(2,t)`。
6. 制造商收到本周分销商订单，决策 `O_(3,t)`。
7. 四个合法动作一起传给 `env.step`；客户真实需求实现，计算履约、发运、积压、期末库存、成本。
8. 只有整周完成，才提交该周动作轨迹和本级历史，更新预测历史及累计成本。
9. 保存检查点并进入下一周。

零售商决策时看不到本周真实客户需求 `D_t`，其 `latest_order` 是 `D_(t−1)`，第一周为 null。上游的 `latest_order` 是当前周刚接受的直接下游订单 `O_(m−1,t)`。旧 backlog 尚不含该新订单，不能重复扣减。即便是无通信实验，上游也必须接收真实业务订单；“无通信”表示无额外状态/insight 消息，而不是业务订单隐藏。

### 7.2 LLM 可见的局部观察白名单

| 提示词字段 | 精确定义 |
|---|---|
| `inventory_level` | 本周期初到货后的可用库存 A |
| `backlog` | 上周期末本级对下游欠货 B，未含本周新需求 |
| `upstream_backlog` | 上周期末本级旧订单尚未被上游发出的量 U |
| `latest_order` | 零售商为上周客户需求；其他级为本周直接下游订单 |
| `in_transit_orders` | 已发运、未来周到达的本级货物，当前两个槽 |
| `total_cost_so_far` | 本级到上周末的累计运营成本 |

提示词另说明供应链角色、提前期、订单上限、成本、观察含义、信息限制和输出契约。`total_cost_so_far` 是本级，不是全链累计成本。无通信节点不获得其他级库存、完整订单列表、其他级预测或全链成本。

当前模板没有显式提供完整实验总周数 H、剩余周数、数据集名称/商品 ID、需求类别标签、ADI、CV²、全序列统计或环境校准所用完整 130 周需求。节点知道当前决策周及预测目标周，不能因此假定模型知道何时实验终止，或使用明确的有限期末优化策略。

模拟器观察还保存 `sales`、成本参数、容量等物理字段；白名单提示词不直接渲染 `sales` 列表，也不将完整环境或完整 context 发给模型。当前上游决策的 context 在程序中可能有多个已接受下游订单，但提示词只选择直接下游 `latest_order`，不能把程序中持有的信息都写成 LLM 可见信息。

### 7.3 最近 10 周历史

每个节点保存最多 10 个已完成周的“决策时局部快照”，由上述六个局部字段组成，按从旧到新排列。第一周为空；第 t 周不含当前周快照。它不是完整对话历史、不是全链历史，也不是期末状态历史。

快照不显式加入该节点自己的当周订单、模型原始响应、insight、当周已实现需求或期末成本。不过某些订单信息通过历史 `latest_order`、在途、欠货等间接存在。零售商历史快照中的 `latest_order` 在第 u 周为 D_(u−1)，所以历史并非直接的最近 10 个真实需求数组。

历史在四级动作全部成功且物理更新完成后才提交；中断周的已验证动作仅在检查点缓存，不提前成为已完成历史。

### 7.4 局部预测

第 m 级预测需求源：零售商为客户 D；其他级为直接下游订单。令已完成周源序列为 `d_(m,u)`。决策周 t 的预测：

\[
\widehat d_{m,t+1\mid t-1}=\frac{1}{n_t}\sum_{u=\max(1,t-26)}^{t-1}d_{m,u},
\quad n_t=\min(26,t-1),\quad t\ge2.
\]

第一周无历史，预测为 null；不足 26 周用已有已完成周的扩展均值；包含零值。代码和提示词把目标标为第 t+1 周，但输入只到 t−1 周，当前客户需求未观察、当前下游新订单也不进入均值。必须保留该时间对齐，不能写成“使用截至 t 周预测 t+1”或“用 t−1 预测当前 t”。

只向模型渲染一个浮点预测值和目标含义，不渲染完整 26 周需求数组。预测不是置信区间、提前期累计需求或订货建议。上游预测目标不是客户需求、历史销售、发运或含旧 backlog 的总待履约量。

冻结 M5 的 `demand_forecasts_26.json` 是保存的客户需求预测文件；当前运行的上游预测由实际该策略下游订单历史在线重算，不直接使用该冻结文件替代。不同策略产生不同上游订单历史，因此不能声称上游预测输入在不同策略中始终数值相同。

## 8. 策略定义

### 8.1 固定规则 `(s,S)`

使用需求记录前 26 周客户需求估计：

\[
\mu=\frac1{26}\sum_{t=1}^{26}D_t,\quad
\sigma=\sqrt{\frac1{25}\sum_{t=1}^{26}(D_t-\mu)^2},\quad
z=\Phi^{-1}(0.95)\approx1.64485362695.
\]

各级阈值：

\[
s_m=\lceil\mu L_m+z\sigma\sqrt{L_m}\rceil,\quad
Q_m^{EOQ}=\sqrt{2K_m\mu/h_m},\quad
S_m=\max(s_m,\lceil s_m+Q_m^{EOQ}\rceil).
\]

特别注意：S 在已经取整的 s 上加 EOQ 增量，再向上取整。不能改写为对未取整的原表达式统一取整。当 μ=σ=0 时显式边界规则为 `s=0,S=1`，不是从未来需求修正。

节点库存位置：

\[
IP_{m,t}=A_{m,t}+U_{m,t}+\sum_{j=1}^{2}P_{m,t,j}-B_{m,t-1}.
\]

策略为：

\[
O_{m,t}=\begin{cases}\min(c_m,\max(0,S_m-IP_{m,t})),&IP_{m,t}\le s_m,\\0,&IP_{m,t}>s_m.\end{cases}
\]

四级都使用同一客户需求前 26 周校准，只因不同提前期/成本才可能阈值不同；当前这些参数相同，所以四级阈值相同。没有按各级内生下游订单单独训练、滚动重新估计或最优阈值搜索。

基线 provider 虽然收到相同框架观察/context，但动作函数只用库存、上游欠货、在途、本级旧 backlog、容量和固定阈值，**不使用 latest_order、当前下游新订单、在线预测或最近 10 周历史**。因此不能声称基线与 LLM 利用完全相同的信息；只能说共享环境和观察框架。基线没有额外通信。

M5 smooth 的当前参数：μ≈2.576923、σ≈1.879853、s=10、S=13；其前 130 周客户均值≈4.938462。固定校准与后续需求均值差异是可能解释结果的机制，但不是已经因果验证的结论。

95% 是阈值公式的名义服务参数，不是实现出的按时服务保证；网络联动、非正态需求、订货上限及样本变化会影响实际服务。

阈值在第 1 周运行前由完整的前 26 周校准数据离线算出，因此早期第 1～25 周策略参数已经受尚未实现的客户需求影响；LLM 在线预测/历史则按完成周生成。这一不对称也是回顾性设计的一部分。所谓“在线决策未看到未来需求”仅指 LLM 输入/运行时序，不能扩展成所有策略校准无未来信息。

### 8.2 Gemini 无通信（`no_communication`）

每周四次顺序局部订货决策。模型看到局部观察、最近 10 个决策快照、单一外部移动平均预测；上游看到直接下游当前订单；没有额外邻居状态、insight 或全链私人信息。

每个调用是独立 `generate_content` 请求，复用 SDK client 但不复用模型对话记忆。输出严格为包在三重方括号中的整数，例如 `[[[2]]]`；整个响应去除首尾空白后必须匹配正则 `\[\[\[(-?[0-9]+)\]\]\]`。允许在语法解析阶段出现负整数，但范围校验随后拒绝。除首尾空白外不接受解释、代码块、JSON 替代格式或额外文本。

### 8.3 自动相邻通信（`adjacent`）

这是“双向期初状态共享 + 顺序向上游订单/insight”的单次决策调用协议，不是双向多轮议价。

阶段 A：四级期初收货后、所有订单前，各节点向两个直接邻居（存在时）自动发送冻结快照：`inventory_level,backlog,in_transit_orders`。不包含 upstream_backlog、预测、累计成本、订单或 insight；六条有向消息，发送不需要 LLM 调用。

阶段 B：零售商到制造商顺序决策。节点同时看到本级 P1–P5 信息、直接上游期初快照、直接下游期初快照，以及直接下游刚生成的订单/insight（零售商无此消息）。上游尚未订货，不提供上游决策 insight。

前三个节点在同一决策响应中返回两行：

```text
[[[2]]]
{"current_order_to_upstream":2,"planning_insight":"One short local insight."}
```

制造商第二行为 `null`。消息中的订单必须与第一行及已接受动作完全相同；不是额外需求。JSON 严格只有两个指定字段，拒绝重复字段和非 JSON 常数。insight 必须是非空字符串，长度≤200 个字符，不含换行/回车。代码检查结构、类型、长度和订单一致性，**不会自动证明 insight 的事实正确性、不转述邻居信息或语义上恰为一句话**；这些是提示词约束。

动作与消息通过校验后，消息只发给直接上游，在其决策前交付；不得通过程序转发非邻接状态/旧消息。提示词明确要求 insight 基于发送者本级状态、历史、预测和选定订单，不复制邻居私人信息或承诺发货。但发送者决策受邻接信息影响，因此不能声称所有间接信息影响被数学隔绝。

一个成功四级周：4 次决策调用、0 次独立通信调用、6 条期初状态 + 3 条订单/insight = 9 条交付；6 条活动有向链路，1 个周通信轮次。130 个完整周为 520 次基本决策、1,170 条消息。因失败周已经交付的消息也保留，带中断重试的累计交付量可能不严格等于已完成周数×9；重放已有相同交付不重复计数。

### 8.4 无外部预测（`no_forecast`）

这是无通信策略的预测输入消融：保留角色、成本、物理规则、局部观察、最近 10 个快照、当前直接下游订单和相同模型设置；去除整个显式预测段，声明无外部预测提供。`uses_forecast_input=False` 使框架不为此策略构建/暴露预测值与 26 周需求数组；本级已完成下游需求仍内部维护以支持公共流程。

无预测不表示模型不会自行估计未来需求，也不表示删除 latest_order 或历史。实际消融还改变提示词段落编号和文本长度，所以不能把差异严格归因于“只变一个数值字段”。

### 8.5 自主选择相邻通信（`controlled_adjacent`）

已实现的扩展协议，本次 M5 smooth 三策略批次不包含它。每周先做四次通信选择调用，再四次顺序订货调用。

通信选择阶段仅提供本级期初观察、已完成历史和预测，没有邻居当前状态或本周订单。四级的 latest_order 在这一阶段都为上周已完成的下游需求/订单，第一周为 null。各节点输出严格 JSON `{"upstream":null或insight,"downstream":null或insight}`；对不存在的邻居必须为 null，两个方向独立决定。

模拟器为选中的消息填写 `source_period,inventory_level,backlog,upstream_backlog,in_transit_orders,planning_insight`，数值不由模型编造。选中消息仅交付给邻居，全部通信选择完成后开始订货。未发送消息不等于无风险，不用旧消息填补 silence。订货仍看到当前直接下游业务订单，不由是否通信决定是否接受。

本协议不再额外发送顺序订单/insight 消息。每周基本 4 个 coordination 调用 + 4 个 decision 调用；0～6 条有向消息；代码每周记录一个通信轮次，即便该周没有消息。通信选择共享与订货各自检查点缓存。

选择性共享减少的是消息数量，不自动减少 API 次数；它的基本调用量是相邻自动协议的两倍。由于消息字段、发送时机、是否传递当前订单 insight 都不同，它不只是对同一协议进行消息开关，比较时需承认这些共同变化。

### 8.6 方法与版本汇总

| CLI 方法 | 结果目录 | 提示词版本 | 适配器版本 | 每周基本调用/消息 |
|---|---|---|---|---|
| ss | rule_based_sS_baseline | 不适用 | 不适用 | 0 / 0 |
| no_communication | gemini_no_communication | gemini_no_communication_v7 | gemini_no_communication_adapter_v2 | 4 / 0 |
| adjacent | gemini_local_communication | gemini_local_communication_v10 | gemini_local_communication_adapter_v4 | 4 / 9 |
| no_forecast | gemini_no_communication_without_forecast_input | gemini_no_communication_without_forecast_input_v6 | gemini_no_communication_without_forecast_input_adapter_v1 | 4 / 0 |
| controlled_adjacent | gemini_controlled_adjacent_communication | gemini_controlled_adjacent_v1 | controlled_adjacent_adapter_v1 | 8 / 0～6 |

上述调用数不包含重试。自动相邻 schema 为 `adjacent_opening_state_then_order_insight_v3`；自主选择 schema 为 `optional_bidirectional_opening_insights_v1`；普通订货 schema 为 `decision_action_v2`。这些是本地输出/协议标识，不是新的模型名称。

## 9. LLM 实现、输出校验和失败处理

### 9.1 模型请求设置

| 项目 | 当前实现 |
|---|---|
| provider | Google |
| 模型字符串 | gemini-3.8-flash |
| SDK | google-genai；实际 sdk_version 记录在结果 |
| API | v1beta，vertexai=False |
| temperature | 0.0 |
| thinking_level | low |
| include_thoughts | False |
| 普通订货响应 MIME | text/plain |
| 自主通信选择响应 MIME | application/json |
| 自动函数调用 | 禁用 |
| SDK HTTP attempts | 1 |
| 外部工具/RAG/搜索 | 当前调用未配置 |
| 对话状态 | 单请求 stateless；client 复用 |
| 模型 seed、top_p、top_k、输出 Token 上限、显式 HTTP timeout | 代码未显式指定，不能编造值 |

英文系统提示与用户模板由代码固定，文末附录保存原文。配置字符串可验证调用设置，但不应声称锁定了供应商后端具体权重版本或无限期不变。模型无本地训练；没有提供隐式全序列需求或 `(s,S)` 公式要求其模仿基线。

密钥从项目 `.env` 或已有环境变量的 `GEMINI_API_KEY` 读取，环境变量优先；不把密钥写入 metadata/checkpoint。SDK client 构建发生在每个新运行或恢复尝试，close 在 finally 中执行。模块导入本身不调用 API。

### 9.2 两层动作契约

第一层 provider 解析原始模型响应并保存诊断。第二层统一 `validate_action` 检查动作必须为整数（拒绝 bool）且 `0≤O≤c`，然后将 proposed 与 executed 设为同一整数。

正式入口不做非法 LLM 动作裁剪、不修正数值、不自动切换成 `(s,S)` 或零订单。越界动作可被 provider 标为 invalid_action，随后由框架抛出 ActionValidationError；它不是可自动重新提示的格式错误。公共接口存在可选 fallback_handler，但当前统一正式 runner 未启用，不能写成有策略兜底。

若任一级动作未成功，整周不执行物理更新；此前已经成功的同周决策仍保留在检查点，API 与消息资源仍保留，但它们尚不进入 completed-period action_trace/成本。

### 9.3 单次决策的适配器重试

adapter_retries=2，因此一次决策最多 3 次尝试。统一循环同时覆盖 API/transport 异常与格式/消息解析异常；不是各自另加 3 次形成 9 次。每次使用同一观察和同一提示词，没有添加“你刚才格式错误”对话、反馈式 reprompt 或改变状态。

传输/API 异常在前两次失败后分别等 1 秒、2 秒；格式错误在循环内立即重新请求。第三次失败终止该次 episode。元数据中 transport_retries、malformed_output_reprompts 等字段要结合这一个实际循环解释，不应相乘。

### 9.4 批次级自动重连

`run_m5_smooth_batch.py` 串行执行 M5 smooth 无通信、相邻通信、无预测，各 130 周。前一个完整成功后才运行下一个；不是三组同时并行。

适配器耗尽尝试后，若最后诊断为 api_transport_failure，并属于连接/超时/协议异常，或诊断消息包含 429、500、502、503、504、RESOURCE_EXHAUSTED、UNAVAILABLE，批次从同一检查点重启。等待依次 10、20、40、60 秒，之后保持 60 秒；没有固定总重连次数上限。不会对所有模型/鉴权/动作错误无限重试。连接型 WinError 10013 若被归为 ConnectError 也符合该分类。

批次采用最后保存的错误类别及异常字符串匹配，而不是另一个完整 HTTP 状态分类器。鉴权、不存在模型、验证/配置错误及其他不满足条件的错误停止批次。各次失败已观察到的调用保留。后台 supervisor 退出后重新用同一 manifest 启动，已成功任务跳过，未完成任务使用检查点；没有自动重启 supervisor 进程的 OS 服务。

### 9.5 诊断与可审计性

逐次 invocation 保存周期、阶段、尝试号、prompt/system prompt SHA-256、局部观察、原始响应文本、解析状态、解析订单、finish reason、候选/安全相关状态、失败类型以及供应商 usage。相邻协议另保存收到的消息和输出消息，自主通信保存 phase/kind/选择。

prompt_hash 不是完整提示词文本备份；模板和原状态需要同时保留才能重构。原始模型响应不等于隐式思维链：`include_thoughts=False`，仅记录供应商返回的 thoughts Token 数，不请求/保存隐藏思考文本。错误文本中的已知 GEMINI_API_KEY/GOOGLE_API_KEY 字符串会替换为 REDACTED。

`action_trace.attempt_count` 是公共框架的 provider/fallback 层次数，正式无 fallback 时通常为 1；不表示 provider 内部 HTTP 重试次数。重试应查 invocation_trace 或调用数。

## 10. 评价指标及准确口径

所有物理/运营指标仅使用已完成的 n 周；完整成功时 n=H，失败时 n<H。不能把失败前缀当作完整 H 周与成功运行直接比较。LLM/消息资源可能包含未完成周，分母与运营指标不同。

### 10.1 成本、库存、积压与订货

- holding_cost、backlog_cost、fixed_ordering_cost：对完成周和节点对应项求和。
- global_operational_cost：四级三项成本总和。
- stage total_order_quantity：该节点完成周接受订单总量。
- positive_order_periods：该节点完成周订单为正的次数，当前等于其固定订货成本。
- average_inventory：本级周末库存均值；由 holding_cost/h 还原，不是期初库存均值。
- average_backlog：本级周末 backlog 均值。
- backlog_frequency_pct：`100×mean(B_(m,t)>0)`，是周末欠货频率，不是需求单位缺货率。
- capacity_saturation_pct：`100×mean(O_(m,t)=c_m)`，是接受新订单达到上限的频率，不代表已证明期望订单超过上限。
- supply_shortage_pct：`100×mean(A_(m,t)<B_(m,t−1)+d_(m,t) 且待履约量>0)`。当前库存限制和整量履约下通常等于周末 backlog 频率。
- strict_capacity_binding_pct：当前为 null，因为没有旧意义的履约产能上限；不能解释成 0%。

完整 detail 的 stage_metrics 对四级均保存上述指标；顶层 service_inventory 仅是零售商对应指标。

### 10.2 客户累计履约比例

\[
FR^{cum}=\frac{\sum_{t=1}^{n}F_{0,t}}{\sum_{t=1}^{n}D_t}.
\]

分母为 0 时 null。包含对旧 backlog 的迟交，不是即时 fill rate、按时服务率、周期服务水平或缺货概率。初始客户 backlog=0 时：

\[
FR^{cum}=1-\frac{B_{0,n}}{\sum_{t=1}^{n}D_t}.
\]

因此累计履约 100% 可与很多周欠货并存，只表明评估末尾客户 backlog 为 0。必须同时报告平均积压、欠货周占比或轨迹，不能仅凭 100% 宣称无缺货。

### 10.3 各级相对客户需求的方差比

\[
BW_m^{var}=\frac{Var_0(O_{m,1:n})}{Var_0(D_{1:n})}.
\]

`Var_0` 使用总体方差 ddof=0，包含零订单/零需求。四级分母全部是客户需求，不是各级下游订单。当 n=0 或客户需求方差为 0 时 null；订单方差为 0 而分母正时可为 0。低方差可能源于长期饱和/不补货，不自动表示优质协调。

### 10.4 相邻需求的 CV 比

定义 `CV_0(x)=std_0(x)/mean(x)`：

\[
BW_m^{CV}=\frac{CV_0(O_{m,1:n})}{CV_0(d_{m,1:n})}.
\]

零售商分母为客户 D；其他级分母为直接下游订单 O_(m−1)。该比值**不平方**，不能等同方差比。空序列、均值非正、需求 CV 为 0 等情况返回 null。代码注释使用 Fransoo–Wouters 标签；论文若作理论归属需另外核对原始文献，不能把该注释当作外部文献引用。

### 10.5 项目定义的全局牛鞭指标

\[
BW^{global}_{project}=\prod_{m=0}^{3}BW_m^{var}.
\]

这是四个“相对同一客户需求”的方差比乘积。不是相邻方差比的连乘；不会望远镜抵消为制造商/客户方差比。它与端到端 `Var(O_3)/Var(D)` 不同，并会放大四级波动影响。任何分量未定义/非有限或乘积非有限时返回 null，不删除分量凑乘积。

论文应给出此确切公式并称为项目定义的乘积指标，同时报告每级 BW 和 CV 比。不要把 global_bullwhip 直接描述为公认端到端指标。若另算端到端指标，应单独标注新增分析，不悄悄改写保存的 global_bullwhip。

### 10.6 LLM 调用与 Token

decision 和 coordination 分别统计调用次数、输入/输出/总 Token；总调用和 Token 为两类相加。失败尝试计入调用数，没有供应商 usage 的尝试计入 unavailable。

Gemini 输入 Token 取 prompt_token_count。输出优先为 `total_token_count−prompt_token_count`（两者是非负整数且总量≥输入）；否则在 candidates 和 thoughts 都可用时取二者之和。这样将 thinking 包含一次。单独 thinking_tokens 是供应商 thoughts 报告的和，不再加到 total_tokens；缺失报告数量另计。

输入/输出 usage 任一缺失、非整数、bool 或负值时，该调用已知/未知 Token 都不进入聚合 Token 和，增加对应 usage_unavailable_call_count；不能把这类 0 当作确认零开销。provider.usage_reports 可保存部分字段，但 metrics 总数仍遵循完整输入/输出契约。Token 不靠 tokenizer 估计，不自动换算货币，不宣称等同账单。

调用数的单位是程序观察到的 client invocation，不是独立抓包证明的网络请求；SDK attempts=1 限制内部重试，但供应商内部行为未观察。恢复保留原尝试计数；同一有效决策从缓存重放不会再增加 API 调用。

自动相邻的订单+insight 混合调用全部计 decision，因此 coordination_total_tokens=0 不表示通信毫无计算开销。状态/insight 增加的提示词、输出均包含在 decision Tokens。需要比较 matched 条件下 total_tokens、llm_call_count、wall_clock 的差值，不能只比较 coordination Tokens。

### 10.7 消息与轮次

一条消息为一次实际程序内有向节点间交付，不是 TCP 报文、物理订单、模型提示词或本地日志。A→B 与 B→A 为不同链路；同链路重复交付增加消息数，不增加 distinct links。

记录 coordination_message_count、coordination_round_count、active_communication_links、per_link_message_counts。显式周 ID 定义通信轮次；无通信不隐式创建轮次；自动相邻和自主选择均每周显式创建轮次。已发送但整周失败的消息仍计入，恢复用 period/sender/receiver/phase 识别已保存交付，内容必须一致，不重复加计。

### 10.8 时间

wall_clock_seconds 使用 perf_counter 测量 run_episode 内 reset、决策、adapter 内等待、物理更新、轨迹构造和过程中检查点写入。导入、provider/client 构建、结束后的 KPI 计算和详细结果/汇总持久化不在主体 timer 内。

恢复/延长保留先前 wall_clock 并增加本次 run_episode 时间，本次本地重放也进入时间。supervisor 在 episode 外的 10～60 秒重连等待、进程未运行期间的停机时间不加入此数，所以它不是从批次开始到完成的全部日历时间，也不是纯模型 latency、纯通信时间或纯推理时间。温度/硬件/网络不同的时间不能直接当作策略计算复杂度。

## 11. 公平比较、跨数据集分析与统计范围

同一序列的策略对照应确保：相同冻结 JSON/hash、同一前缀需求、H、初始库存、提前期、新订单上限、履约规则、成本、预测实现、局部信息语义、历史长度、模型字符串、采样参数、提示词/适配器版本与重试制度。通信策略按设计改变信息和输出协议；这不是所有提示词逐字相同的对照。

跨 M5 与合成不要求容量/库存数值一致，而要求相同校准规则。两者客户需求尺度和时间结构不同，原始总成本不能单独回答跨数据优劣。可额外计算每单位需求运营成本 `C^op/sum(D)`、每周成本 `C^op/H`、相对同序列基线改善 `(C_ss−C_method)/C_ss`（C_ss>0）等；这些是建议的派生分析，当前标准 summary 没有自动保存全部这些字段。

M5 与合成 smooth 不是尺度匹配、趋势匹配或相同噪声的随机配对试验。真实 M5 保留其时间顺序；合成独立需求没有显式趋势。M5 前 26 周校准与后续均值差异可能使固定基线吃亏；不能把更低成本全部归因于通信或 LLM 的一般智能。

当前每类合成只一条 realization，很多策略每条只有一次成功运行，不能声称已作多种子 Monte Carlo、显著性检验、置信区间、稳健性或统计普遍性验证。若后续增加重复，应每条序列做策略配对比较并在序列/重复层面计算差值，而非把同一 episode 的 130 周当作 130 个独立实验。

同一远程 LLM 连续调用可能存在输出波动和时间漂移；不同策略批次串行运行也不是随机化执行顺序。复用检查点只保证缓存的成功决策不重算，不能保证新尾部与某次从头远程重跑得到完全相同输出。

成本下降可以伴随库存上升、积压下降、牛鞭上升；这些是不同目标的取舍，不是自动判定程序出错的依据。通信可能帮助或伤害，不保证单调改善。没有高成本/低成本自动排除规则，应保留失败与异常运行证据。

## 12. 结果格式、保存、恢复与延长

This section describes newly executed runs and checkpoint compatibility. The publication copies use the consolidated summary/manifest layout described above.


### 12.1 输出目录和命名

```text
results/
  m5/<method_directory>/<record_id>_run_NNN.json
                         <record_id>_run_NNN.checkpoint.json
                         summary.json              # LLM
                         summary_runs.json         # 新基线
  synthetic/bernoulli_gamma_v1/<method_directory>/...
```

New M5 and synthetic runs share the detailed-result/summary interface. The runner preserves existing historical summaries when operating on an external legacy output directory; those overlapping publication files are consolidated in this submission.

新运行编号高于已有 result/checkpoint 及汇总预留身份；默认不覆盖旧 detail。失败运行也保留身份和摘要，不因成功后另一个编号而删除。

### 12.2 详细结果字段

| 字段 | 意义 |
|---|---|
| run_status | success 或 failed；检查点另可为 in_progress |
| completed_periods | 真正完成物理更新的周数 |
| error_type/error_message | 最终失败或 null |
| run_metadata | 数据、窗口、环境、初始值、模型协议、运行身份、可选延长来源 |
| action_trace | 按周、按阶段排列的完成周决策和成本快照 |
| episode_reward | 完成周奖励总和 |
| coordination_metrics | 完成与未完成周的已观察资源累计 |
| evaluation_summary | 成本、服务、牛鞭、四级指标、资源和时间 |
| provider_invocation_trace | LLM 每次尝试诊断（基线不要求） |
| communication_trace | 有通信策略的实际交付（无通信无此字段） |
| policy | `(s,S)` 校准参数和公式（LLM 无此字段） |

action_trace 保存 period、stage、observation、raw_output、proposed/executed order、validation_status/error、reward、四项成本、decision_summary、attempt_count。完整 130 周有 520 行。正式 Gemini 通常不产生单独 decision_summary；相邻 insight 在通信/调用轨迹中。

详细结果没有直接保存环境全部 inventory/order/sales/arriving_orders 矩阵或客户完整需求数组；需求从冻结输入加载，完整物理序列可据接受动作和一致环境重放。正文/分析若需要这些矩阵，必须明确是重建或新增派生文件，而不是声称 detail 已有全部矩阵。

汇总为 JSON list，每个 record/run 一行，按身份 upsert；保留成功/失败、参数与 KPI、source_result_file。summary 中 input_file/绝对路径可用于本地溯源，跨机器移动路径会影响严格恢复兼容校验。

### 12.3 检查点与原子写入

LLM 开始运行时保存初始检查点；每个合法阶段决策、完整周、失败诊断更新和结束时保存。自主通信还保存 sharing_decisions。检查点包含 checkpoint_version=1、运行需求 SHA-256、runner 身份、结果快照、各阶段 decisions/observation hash、通信 round IDs、调用资源/诊断。

写入临时文件后 flush、fsync，再原子 replace。Windows 暂时被阅读器锁住时 replace 最多 20 次，每次失败等 0.05 秒；之前的有效文件保留。详细结果新身份使用原子 create-if-absent；恢复失败身份可原子替换。不要多个进程同时运行同一 checkpoint/manifest。

检查点中的 result 不一定含最终完整 evaluation_summary/service_inventory；它是恢复状态，不是最终论文结果。失败中断周期的已合法动作位于 decisions，但未加入完成周 action_trace。

### 12.4 恢复中断（resume）

`--resume` 使用原 checkpoint 的数据、方法、窗口、run identity，不允许同时重选 record、all、periods 或 output_root。校验版本、环境/来源/窗口 metadata、需求 hash、模型/提示词/适配器/schema/温度/thinking/重试/预测与通信配置。自主通信另校验共享提示片段 hash。

恢复从初始状态本地重放完成周缓存订单，验证观察 hash、动作/成本轨迹，重建库存、积压、在途、最近快照、预测历史和本级累计成本；重放不调用 LLM，也不重复通信计数。中断周已验证的顺序前缀复用，仅缺少的阶段重新请求。

成功最终 result 不允许 resume 覆盖。若供应商已处理/计费但本地尚未取得并验证/保存响应，恢复可能再次调用；不保证供应商 exactly-once billing。原失败尝试资源保留，无法获知供应商 usage 的费用不估计。

严格 metadata/version 校验不是所有源码字节都由恢复器比较。普通 provider 恢复主要按版本/配置字段，自主共享另比较片段 hash；本文件附录源码哈希为审计补充，不能说代码任意变更都一定被恢复器拒绝。提示词变更应同步更新版本标识。

### 12.5 延长已完成运行（extend）

`--extend <successful_result_or_checkpoint> --periods H_new`，H_new 是新总周数，必须大于旧 H 且不超过原输入长度。仅接受完整成功的来源；未完成者使用 resume。

LLM 用检查点缓存恢复旧轨迹。若传 result JSON，加载相邻同名 checkpoint；历史 LLM result 无兼容 checkpoint 不能直接延长。先验证旧窗口环境/输入和 runner，校验需求前缀 hash，再复制 checkpoint、适配新窗口，模型配置仍需一致。

The generic extension capability is outside the approved submission protocol. Only H=130 is evaluated here; no extended M5 experiment is retained.

输出新 run_NNN，旧结果与旧检查点不覆盖。新 detail 覆盖完整第 1～H_new 周，资源保留前缀 + 新调用；extension metadata 保存来源绝对路径/SHA、previous_num_periods、continuation_first_period。不是只保存新增后半段 KPI。可以由新的成功延长结果再次延长，失败延长使用新的 checkpoint resume。

`(s,S)` 延长从成功当前 result 重算确定性全程，并校验原 policy 和 action_trace 前缀完全相同，再保存新身份；没有 API 费用，但不是从对象内存直接推进第 131 周。

### 12.6 预检查

--preflight 校验输入、参数、记录选择、输出身份以及 LLM provider 初始化；不调用 API、不写结果。它不证明远程 API 连通、密钥被远程接受、模型可用或配额足够。extension preflight 还验证来源与 provider 配置，但不进行完整物理前缀重放，因此不能当作已验证所有轨迹缓存。

## Research limitations

1. 筛选与环境/策略校准和评估重叠，是回顾性设置，不是纯 out-of-sample。
2. s/S 固定于前 26 周，四级以客户而非各级内生订单校准，且不利用本周直接下游订单；不应推广为所有传统最优库存算法。
3. LLM 预测仅均值，使用到 t−1 却标为 t+1，与两周运输需求不是同一目标；没有不确定性/趋势/季节性预测模型。
4. 合成独立过程只有每类一次 realization，不代表真实时间结构；M5 实验也不是所有已选 120 条均已完成。
5. 单次 LLM 结果、温度 0 和 checkpoint 缓存不足以建立统计稳定性或供应商后端可重现性。
6. 归一化成本和订单上限的分位数设计不是实测行业成本与吞吐约束，结果对成本比、提前期、校准和上限可能敏感。
7. 终止无额外清算/残值，部分成本/服务指标依赖选定终点；累计 fill 包含迟交，不能替代即时服务。
8. 全局牛鞭是项目乘积指标，必须与端到端/相邻指标区分；capacity 截断也可能压低订单波动。
9. 自动相邻共享包含数值状态与 LLM advice 两类变化，不能直接分离两者因果贡献；自主通信与自动协议还有消息内容与时序差异。
10. 格式校验不等于 insight 语义校验，提示词信息限制是行为约束，未证明模型严格遵守每句语义限制。
11. 运营与 API 费用分开；缺失 usage、不观测的供应商计费/缓存行为限制成本推断。
12. 失败运行/未完成周保留资源，不应只挑低成本成功记录，或把短失败前缀混入完整比较。

# Current Coordination Prompts

Extracted from the current source templates; placeholders represent runtime values.
No experiment was run. Existing result files retain their original prompt versions.
The adjacent JSON example is shown with single braces; source templates escape them.

## Controlled Adjacent Coordination With Forecast

Source: `src/agents/gemini_controlled_adjacent.py`

Prompt version: `gemini_controlled_adjacent_v1`.

This independent method uses two prompts per stage per week. Both reuse the
no-communication P1-P5 environment, costs, local state, history and forecast.
The sharing prompt changes P3 latest_order to last week's completed downstream
demand/order for every stage; the ordering prompt retains normal sequential P3.
The authoritative appended templates are `SHARING_RULES` and `DECISION_RULES`.

### Communication Selection

System prompt:

```text
Decide independently whether your local information could materially affect each immediate neighbor's replenishment decision.
```

The LLM selects upstream and downstream independently before any ordering.
Share only when local information could materially change a neighbor's order
timing, quantity or supply-risk assessment, not merely because a neighbor exists.
Signals include stockout risk, persistent backlog, changes in demand/shipments
or recovery. A continuing risk can warrant a message even without a new change.
No guarantee of semantic relevance is enforced by the parser: selection is an
LLM judgment; only schema, boundaries and insight length are validated.

Return only JSON, with each direction either null or a brief insight string:

```json
{"upstream": "Persistent demand requires continued replenishment.", "downstream": null}
```

Retailer must return null for downstream; Manufacturer must return null for
upstream. The simulator attaches the sender's own opening inventory_level,
backlog, upstream_backlog, in_transit_orders and source_period to each selected
insight. No neighbor states or messages are available during selection, and no
current order or fulfillment outcome is included.

### Ordering

System prompt is the no-communication decision system prompt. P6 supplies only
the selected current-period upstream/downstream messages. null means no message
or no neighbor, never an inference of zero stock or no risk. Neighbor inventory
is not the recipient's inventory; advice is not a shipment promise. No messages
are forwarded or carried over. Current downstream orders remain visible in P3
whether or not optional information was shared.

P7 returns only the new order as a triple-bracket integer within local capacity.
No additional insight is generated during ordering. Four coordination plus
four decision calls occur per successful week, before retries; actual message
deliveries range from zero to six. The sharing choices are checkpointed for
resume. Existing adjacent prompts and historical results remain unchanged.

## No Communication With Forecast

Source: `src/agents/gemini_provider.py`

Prompt version: `gemini_no_communication_v7`

### System Prompt

```text
Make one replenishment decision for the stated supply-chain stage using only the supplied local information.
```

### User Prompt Template

```text
P1 Environment

You are the {stage_name} in a sequential four-stage supply chain:
Retailer, Wholesaler, Distributor, Manufacturer.

Your objective is to help minimize total supply-chain operating cost over time.
Use only the local information supplied below.

Each week, due shipments arrive before ordering decisions.
Stages then decide sequentially from Retailer to Manufacturer.
Customer demand and downstream orders are fulfilled after these decisions.
Unfulfilled demand is carried forward as backlog.

A new replenishment order does not immediately increase available inventory.
Orders may be fulfilled partially. Unshipped quantities remain outstanding.
Shipments dispatched in week t arrive at the beginning of week t+2.
The lead time starts when goods are dispatched, not when an order is placed.

Your stage's new-order capacity is {order_capacity} units per week.
It limits only your new replenishment order, not your downstream fulfillment.
Fulfillment is limited by available inventory and total pending downstream
demand (last week's backlog plus this week's new demand or downstream order).
You may fulfill more than your order capacity when inventory and pending demand
allow it. There is no separate production, handling or shipment capacity limit.
At the Manufacturer, external replenishment orders are fully dispatched
in the ordering week and arrive two weeks later.

P2 Objective function

Total supply-chain cost is the sum of all stages' operating costs.

Your stage's cost in a week is:
end-of-week inventory
+ end-of-week backlog
+ 1 if order quantity > 0

The three terms represent:
- holding cost,
- backlog cost,
- fixed ordering cost.

Each cost coefficient is 1. There is no variable ordering cost.

Costs are measured after fulfillment, not from the opening observation.
The fixed ordering cost applies even if the supplier cannot ship immediately.
Consider the effect of your order on future inventory, backlog and costs.

P3 Current period information

Your current observations are:

{local_json}

Definitions:

inventory_level:
The stock available at the beginning of this week, including last week's
remaining stock and all shipments due at the beginning of this week.
These opening receipts have already been included in inventory_level.

backlog:
Unfulfilled downstream demand carried forward from last week's end.
It does not yet include this week's new customer demand or downstream order.

upstream_backlog:
Your previously placed replenishment orders that your immediate upstream
supplier has not yet shipped. This is your outstanding order quantity, not
your own backlog and not stock already in transit. Do not count it as available
inventory or as a new order to submit again. It has no guaranteed arrival date.
For the Manufacturer this value is zero at the external-supplier boundary.

latest_order:
{latest_order_definition}

in_transit_orders:
Committed shipments scheduled to arrive at the beginning of next week and
the week after next, respectively. These quantities are not included in
inventory_level. Do not add this week's opening receipts again.

total_cost_so_far:
Your accumulated operating cost through the end of last week.

P4 Recent observations

The following observations cover up to ten completed periods, oldest to newest.
They are the decision-time snapshots from those earlier periods. The current
period's observation and newly decided orders are excluded. They are not
end-of-week outcomes. The list is empty
in the first period. Current-period information is provided separately in P3.

{recent_observations_json}

P5 Demand forecast

The demand forecasting module estimates your stage's downstream demand of:
{forecast_value_statement}

for week {forecast_target_period}, the week after the current decision week.

The prediction target is:
{forecast_target_definition}

This forecast is generated from previously realized demand using a trailing
26-week simple moving average of this stage's own downstream demand, including
zero-demand weeks. With fewer than 26 completed weeks, the tool uses the available-history
mean. The history ends last week: this week's customer demand or downstream
order is not included. A current downstream order in P3 is a separate known
order, not an observation used to calculate this forecast.

If no completed downstream demand history exists, no forecast is available.
This is an uncertain one-week-ahead estimate, not a guaranteed order
and not a forecast of total demand over the two-week shipment lead time.

Consider it together with your current inventory, backlog, incoming orders,
and recent observations.

P6 Information and communication rule: local agent

This is a no-communication, local-information setting.

Use only the information provided in P3-P5.

You do not receive messages or private information about other agents'
inventory, backlog, forecasts, costs, or reasoning.

You do not send messages to other agents.

Your own outstanding orders and incoming shipments are local order-status
information, not access to the supplier's complete internal state.

You may reason from your supplied observations, but do not treat estimates
of another stage's private state as known facts.

P7 Decision

Decide how many units to order now.

Your order must be one integer from:
0 to {order_capacity}

inclusive.

Return only the integer enclosed in triple brackets.

Example:

[[[2]]]

Do not include any explanation, text, calculation, or additional output.
```

## Adjacent Communication With Forecast

Source: `src/agents/gemini_local_communication.py`

Prompt version: `gemini_local_communication_v10`

### System Prompt

```text
Make one replenishment decision and generate one local planning insight for your immediate upstream neighbor using only the supplied local and adjacent information.
```

### User Prompt Template

```text
P1 Environment

You are the {stage_name} in a sequential four-stage supply chain:
Retailer, Wholesaler, Distributor, Manufacturer.

Your objective is to help minimize total supply-chain operating cost over time.
Use only the local information supplied below.

Each week, due shipments arrive before ordering decisions.
Stages then decide sequentially from Retailer to Manufacturer.
Customer demand and downstream orders are fulfilled after these decisions.
Unfulfilled demand is carried forward as backlog.

A new replenishment order does not immediately increase available inventory.
Orders may be fulfilled partially. Unshipped quantities remain outstanding.
Shipments dispatched in week t arrive at the beginning of week t+2.
The lead time starts when goods are dispatched, not when an order is placed.

Your stage's new-order capacity is {order_capacity} units per week.
It limits only your new replenishment order, not your downstream fulfillment.
Fulfillment is limited by available inventory and total pending downstream
demand (last week's backlog plus this week's new demand or downstream order).
You may fulfill more than your order capacity when inventory and pending demand
allow it. There is no separate production, handling or shipment capacity limit.
At the Manufacturer, external replenishment orders are fully dispatched
in the ordering week and arrive two weeks later.

P2 Objective function

Total supply-chain cost is the sum of all stages' operating costs.

Your stage's cost in a week is:
end-of-week inventory
+ end-of-week backlog
+ 1 if order quantity > 0

The three terms represent:
- holding cost,
- backlog cost,
- fixed ordering cost.

Each cost coefficient is 1. There is no variable ordering cost.

Costs are measured after fulfillment, not from the opening observation.
The fixed ordering cost applies even if the supplier cannot ship immediately.
Consider the effect of your order on future inventory, backlog and costs.

P3 Current period information

Your current observations are:

{local_json}

Definitions:

inventory_level:
The stock available at the beginning of this week, including last week's
remaining stock and all shipments due at the beginning of this week.
These opening receipts have already been included in inventory_level.

backlog:
Unfulfilled downstream demand carried forward from last week's end.
It does not yet include this week's new customer demand or downstream order.

upstream_backlog:
Your previously placed replenishment orders that your immediate upstream
supplier has not yet shipped. This is your outstanding order quantity, not
your own backlog and not stock already in transit. Do not count it as available
inventory or as a new order to submit again. It has no guaranteed arrival date.
For the Manufacturer this value is zero at the external-supplier boundary.

latest_order:
{latest_order_definition}

in_transit_orders:
Committed shipments scheduled to arrive at the beginning of next week and
the week after next, respectively. These quantities are not included in
inventory_level. Do not add this week's opening receipts again.

total_cost_so_far:
Your accumulated operating cost through the end of last week.

P4 Recent observations

The following observations cover up to ten completed periods, oldest to newest.
They are the decision-time snapshots from those earlier periods. The current
period's observation and newly decided orders are excluded. They are not
end-of-week outcomes. The list is empty
in the first period. Current-period information is provided separately in P3.

{recent_observations_json}

P5 Demand forecast

The demand forecasting module estimates your stage's downstream demand of:
{forecast_value_statement}

for week {forecast_target_period}, the week after the current decision week.

The prediction target is:
{forecast_target_definition}

This forecast is generated from previously realized demand using a trailing
26-week simple moving average of this stage's own downstream demand, including
zero-demand weeks. With fewer than 26 completed weeks, the tool uses the available-history
mean. The history ends last week: this week's customer demand or downstream
order is not included. A current downstream order in P3 is a separate known
order, not an observation used to calculate this forecast.

If no completed downstream demand history exists, no forecast is available.
This is an uncertain one-week-ahead estimate, not a guaranteed order
and not a forecast of total demand over the two-week shipment lead time.

Consider it together with your current inventory, backlog, incoming orders,
and recent observations.

P6 Information and communication rule: adjacent agents

At the beginning of this week, after due receipts and before any ordering
decisions, each stage shared its own inventory_level, backlog and
in_transit_orders with both immediate neighbors, where they exist.
The simulator delivered all these opening snapshots without any LLM call.
Opening snapshots contain no insight, current-week order or fulfillment outcome.

Your immediate upstream neighbor's opening snapshot is:

{upstream_message}

Your immediate downstream neighbor's opening snapshot is:

{downstream_opening_message}

The upstream snapshot is null for Manufacturer. The downstream snapshot is null
for Retailer. Neither the external supplier nor the end customer shares a state.

In each snapshot, inventory_level already includes that neighbor's receipts due
this week. backlog is its unfulfilled demand carried forward from last week's
end. in_transit_orders lists shipments due to arrive at that neighbor next week
and the week after next. These are separate components of the neighbor's opening
inventory status, not your own inventory position or available stock.
Do not count either neighbor's inventory or incoming shipments as your own.
An upstream snapshot is not a shipment promise or guaranteed arrival schedule
for your own outstanding orders.

Ordering decisions now proceed from Retailer to Manufacturer.
Your immediate downstream neighbor has already decided its current order and
sent this order-and-insight message to you:

{downstream_message}

For Retailer this message is null. Otherwise, current_order_to_upstream is the
accepted new order from your direct downstream neighbor for this week.
It is the SAME order shown as latest_order in P3, not additional demand.
It is not yet included in your opening backlog and is not a guarantee that
you can fulfill it completely this week.
planning_insight was generated together with that neighbor's current order
using its current local observations, forecast and chosen order. Treat it as
advice, not a guaranteed future order, shipment or forecast.

Use both opening snapshots and the downstream order-and-insight message
together with your own observations and forecast when choosing your order.
You do not receive an upstream insight: that stage has not decided its order yet.
Do not assume access to any nonadjacent agent's private information.

In the SAME response as your order, generate one brief planning_insight for
your immediate upstream neighbor. Base it on your own opening observations,
completed history, forecast and chosen order, explaining whether your own
replenishment needs suggest increasing, maintaining or reducing replenishment.
Do not copy, summarize or forward either neighbor's private information,
received message or insight. Do not promise fulfillment or future orders.

Your order-and-insight message is sent only to your immediate upstream neighbor
after your order is validated and before that neighbor decides. There is no
separate insight-generation LLM call. Manufacturer has no upstream agent:
it returns null for its outbound message, but its opening state was already
shared with Distributor.

P7 Decision and outbound insight

Decide how many units to order now.

Your order must be one integer from:
0 to {order_capacity}

inclusive.

Return exactly two lines.

Line 1: Return the integer enclosed in triple brackets.

Example:

[[[2]]]

Line 2: If you have an upstream neighbor, return one valid JSON object with
exactly these fields:

{"current_order_to_upstream": <same integer as line 1>, "planning_insight": "<one brief sentence of at most 200 characters>"}

If you are Manufacturer, return null on line 2.
Do not include any explanation, calculation or additional output.
```

## No Communication Without Forecast

Source: `src/agents/gemini_no_forecast.py`

Prompt version: `gemini_no_communication_without_forecast_input_v6`

### System Prompt

```text
Make one replenishment decision for the stated supply-chain stage using only the supplied local information.
```

### User Prompt Template

```text
P1 Environment

You are the {stage_name} in a sequential four-stage supply chain:
Retailer, Wholesaler, Distributor, Manufacturer.

Your objective is to help minimize total supply-chain operating cost over time.
Use only the local information supplied below.

Each week, due shipments arrive before ordering decisions.
Stages then decide sequentially from Retailer to Manufacturer.
Customer demand and downstream orders are fulfilled after these decisions.
Unfulfilled demand is carried forward as backlog.

A new replenishment order does not immediately increase available inventory.
Orders may be fulfilled partially. Unshipped quantities remain outstanding.
Shipments dispatched in week t arrive at the beginning of week t+2.
The lead time starts when goods are dispatched, not when an order is placed.

Your stage's new-order capacity is {order_capacity} units per week.
It limits only your new replenishment order, not your downstream fulfillment.
Fulfillment is limited by available inventory and total pending downstream
demand (last week's backlog plus this week's new demand or downstream order).
You may fulfill more than your order capacity when inventory and pending demand
allow it. There is no separate production, handling or shipment capacity limit.
At the Manufacturer, external replenishment orders are fully dispatched
in the ordering week and arrive two weeks later.

P2 Objective function

Total supply-chain cost is the sum of all stages' operating costs.

Your stage's cost in a week is:
end-of-week inventory
+ end-of-week backlog
+ 1 if order quantity > 0

The three terms represent:
- holding cost,
- backlog cost,
- fixed ordering cost.

Each cost coefficient is 1. There is no variable ordering cost.

Costs are measured after fulfillment, not from the opening observation.
The fixed ordering cost applies even if the supplier cannot ship immediately.
Consider the effect of your order on future inventory, backlog and costs.

P3 Current period information

Your current observations are:

{local_json}

Definitions:

inventory_level:
The stock available at the beginning of this week, including last week's
remaining stock and all shipments due at the beginning of this week.
These opening receipts have already been included in inventory_level.

backlog:
Unfulfilled downstream demand carried forward from last week's end.
It does not yet include this week's new customer demand or downstream order.

upstream_backlog:
Your previously placed replenishment orders that your immediate upstream
supplier has not yet shipped. This is your outstanding order quantity, not
your own backlog and not stock already in transit. Do not count it as available
inventory or as a new order to submit again. It has no guaranteed arrival date.
For the Manufacturer this value is zero at the external-supplier boundary.

latest_order:
{latest_order_definition}

in_transit_orders:
Committed shipments scheduled to arrive at the beginning of next week and
the week after next, respectively. These quantities are not included in
inventory_level. Do not add this week's opening receipts again.

total_cost_so_far:
Your accumulated operating cost through the end of last week.

P4 Recent observations

The following observations cover up to ten completed periods, oldest to newest.
They are the decision-time snapshots from those earlier periods. The current
period's observation and newly decided orders are excluded. They are not
end-of-week outcomes. The list is empty
in the first period. Current-period information is provided separately in P3.

{recent_observations_json}

P5 Information and communication rule: local agent

This is a no-communication, local-information setting.

Use only the information provided in P3-P4.

No external demand forecast is supplied.

You do not receive messages or private information about other agents'
inventory, backlog, forecasts, costs, or reasoning.

You do not send messages to other agents.

Your own outstanding orders and incoming shipments are local order-status
information, not access to the supplier's complete internal state.

You may reason from your supplied observations, but do not treat estimates
of another stage's private state as known facts.

P6 Decision

Decide how many units to order now.

Your order must be one integer from:
0 to {order_capacity}

inclusive.

Return only the integer enclosed in triple brackets.

Example:

[[[2]]]

Do not include any explanation, text, calculation, or additional output.
```

## Runtime Inputs

- `{stage_name}`: Retailer, Wholesaler, Distributor or Manufacturer.
- `{local_json}`: local inventory_level, backlog, upstream_backlog, latest_order, in_transit_orders and total_cost_so_far. upstream_backlog is this stage's unshipped outstanding replenishment orders, distinct from its own downstream backlog and from shipments already in transit.
- `{recent_observations_json}`: decision-time snapshots from up to ten completed earlier periods, oldest first; excludes the current period and is empty initially.
- `{latest_order_definition}`: retailer sees last week's realized customer demand, not new demand to fulfill again; other stages see the current accepted new order, not yet counted in opening backlog and not guaranteed to be completely fulfilled.
- `{forecast_value_statement}`: trailing 26-week mean followed by "units", or an explicit no-forecast-available statement. Expanding mean before 26 completed observations; excludes current-period demand/orders. Omitted in the no-forecast variant.
- `{forecast_target_period}`: current decision week plus one.
- `{forecast_target_definition}`: customer demand for Retailer; direct downstream new orders for the other stages, never sales/shipments or accumulated backlog.
- `{order_capacity}`: this stage's new-order limit: max(1, ceil(Q_0.95(first 130 source weeks))), using linear quantile interpolation and including zero-demand weeks. If an explicitly supplied source has fewer than 130 weeks, use all available source weeks. All four stages receive the same fixed value. The calibration window is independent of the simulation/evaluation horizon: generic shorter or longer evaluations retain the same source-prefix calibration; only 130-week evaluation is approved for this submission. It does not limit fulfillment, shipments or inventory.
- `{upstream_message}`: null for Manufacturer; otherwise the immediate upstream stage's frozen opening inventory_level, backlog and in_transit_orders, without orders or insights.
- `{downstream_opening_message}`: null for Retailer; otherwise the immediate downstream stage's frozen opening inventory_level, backlog and in_transit_orders.
- `{downstream_message}`: null for Retailer; otherwise the direct downstream stage's current accepted order (current_order_to_upstream) and planning_insight generated in the same decision call. Its order equals P3 latest_order, not additional demand.

Adjacent communication has one prompt and one decision LLM call per stage, without retries. After opening receipts, the simulator exchanges six directed state-only snapshots between immediate neighbors. Decisions then run Retailer to Manufacturer. Each non-Manufacturer response contains the order and a local planning_insight in two lines; after validation, its order-and-insight message is delivered to the immediate upstream stage before that stage decides. Manufacturer returns null on line 2. A successful four-stage week has four decision LLM calls, zero separate coordination LLM calls, nine actual message deliveries (six snapshots plus three order-and-insight messages), six active directed links and one weekly coordination round. Combined-response tokens are counted as decision usage; message counts still measure communication. No upstream insight or previous-week insight is used. All accepted orders enter simulation together, followed by fulfillment and end-of-week cost accounting. No centralized/full-information prompt is implemented. Rule-based s,S uses no LLM prompt. Historical saved runs retain their original protocol and records.


## Capacity and Result Compatibility

Current capacity applies only to newly placed orders. Fulfillment equals min(opening inventory, previous backlog + current downstream demand), with no throughput cap. New result metadata records capacity_quantile=0.95 and capacity_scope=new_order_quantity_only. capacity_saturation_pct now measures accepted orders equal to the order limit; strict_capacity_binding_pct is null because the old fulfillment-cap metric is no longer applicable. supply_shortage_pct compares opening inventory with total pending demand. Historical result files and calibration artifacts retain their original formulas and are not directly comparable to new runs without rerunning them.
