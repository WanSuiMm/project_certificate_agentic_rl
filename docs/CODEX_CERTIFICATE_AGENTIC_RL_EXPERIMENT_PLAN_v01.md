# Codex 执行方案：程序语义证书监督的 Agentic RL 价值学习

**版本：v0.1 · 2026-09-21**
**项目工作名：Certificate-Supervised Agentic Value Learning（CSAVL，内部名称，不是已确认新算法）**
**当前阶段：已有有限世界理论／代数审计；尚无真实工具代理的学习收益证据。**

> 给 Codex 的核心任务：在一个有明确规范的、多文件代码修复环境中，运行真实 LLM 工具代理，先固定 actor 收集轨迹，检验程序语义证书是否改善 critic；只有预注册条件成立后，才启动小规模在线 PPO。不能用另一轮有限状态表实验代替真实代理实验，也不能用“奖励更稠密”代替学习收益。
>
> 本文是实验执行规格，不是已完成的实现或新实验报告。文中的新增 CLI、模块和配置需要你实现。本文规定的数量与阈值是本轮建议默认值，不是用户已经承诺的算力、也不是理论常数。

---

## 0. 执行边界与优先级

### 0.1 这轮做什么

按顺序完成：

1. `P0_EVIDENCE`：读取并复现已有两个实验包，建立环境清单和数据／成本口径。
2. `P1_RUNTIME`：实现闭合程序语义、真实工具环境、成对证书和终端验证；跑 LLM smoke。
3. `P2_DATA`：冻结 actor、任务划分、采样协议，收集真实多轮轨迹和训练用证书。
4. `P3_CRITIC`：在固定数据上完成配平对照、独立评估、误差诊断与裁决。
5. `P4_ONLINE`：仅在 P3 通过、实现审计通过且配置明确给出在线预算时运行小规模 PPO。

默认必须把 P0–P3 推进到实际结果或具体 blocker，不能只交目录骨架、伪代码或训练命令。P4 是门控阶段，不是无条件启动。资源不足时交付已完成产物、日志和准确缺口，不编造结果。

若只收到本文、没有两份旧实验包，先在用户授权工作区查找同名文件；找不到则记录 `MISSING_PRIOR_ARTIFACTS`，不得伪造复现。可以继续独立实现本方案，但复现状态必须保留为缺失。完整交接 ZIP 已包含两包。

### 0.2 禁止偏离

- 不继续发展通用“软件数值分析”大理论；只补当前实现所需的语义和正确性证明。
- 不先上完整 SWE-bench、任意 Python 仓库、并发／网络／外部服务或自动理解任意自然语言意图。
- 不重新发明大型 agent 框架；使用薄工具循环，优先复用本机已有基础设施。
- 不把 Relative Correctness、关系验证、potential shaping 或 PPO 宣称成我们的新理论。
- 不把 `BETTER → +1`、`WORSE → -1` 作为主方法；不按语义负方向硬剪枝。
- 不把 `UNKNOWN` 当成“真实变化为零”；不把 syntax/type checking 通过当成任务完成。
- 不从其他项目继承 GPU 型号、20 小时额度、数据、checkpoint 或目录假设。
- 不访问支付接口，不购买云计算，不自动调用收费模型；外部 API 默认关闭。
- 不修改用户其他项目、不删除既有数据、不终止他人进程；原始实验包只读保存。

### 0.3 本地启动与资源清单

先读取工作目录的 `AGENTS.md`、`PROJECT.md` 等实际存在的项目指令。若当前目录不是本项目，在用户给定工作区下新建 `project_certificate_agentic_rl/`；不要猜测 D 盘路径。

产出 `reports/ENVIRONMENT.md`，包括：OS、Python、CPU、可用内存、磁盘、GPU、可用显存、现有模型、模型 license、已有训练框架和版本。记录软件版本与 git commit，不直接相信文档中的配置字段名称。

优先使用已有可本地运行、可导出 log-probabilities 的代码／工具 instruct 模型。模型和 tokenizer 路径在 `resolved_config.yaml` 中冻结。没有已授权模型或隔离运行环境时，完成能做的 P0／语义测试，分别标记 `BLOCKED_MODEL` 或 `BLOCKED_SANDBOX`；不得拿脚本策略伪装 LLM 轨迹。

---

## 1. 研究问题与目前已有证据

### 1.1 本轮主假说

在相同真实代理轨迹、相同输入访问、可比模型容量及总计算预算下：

\[
\widehat V(h)=\Phi_{\mathrm{cert}}(P,\mathcal S,\mu)+\widehat W(h)
\]

是否比仅从任务 return 学习的 critic，更准确地预测当前策略的任务完成价值？

这里研究的是**程序语义证书作为价值学习的额外监督／归纳偏置**，不是“证书已经认证了 RL advantage”。

分开检验三个命题：

| 命题 | 要比较什么 | 通过后最多能说什么 |
|---|---|---|
| H1：证书信息有价值 | 证书方法 vs return-only／公共测试信息 | 在本任务分布中，额外语义监督有用 |
| H2：固定加法分解有价值 | `CERT_DECOMP` vs 使用同一证书的 `CERT_FEATURE` | 分解方式有额外收益，不只是多了信息 |
| H3：收益能变成在线效率 | 门控后的 PPO、计算与交互预算配平 | 在所测规模下改善学习效率 |

H1 成立不代表 H2 成立；H1/H2 成立不代表 H3 成立。离线失败也不能被无限外推成“证书对所有 agent RL 都没用”。

### 1.2 输入实验包

随交接包附带：

- `prior_artifacts/semantic_certificate_lab.zip`
- `prior_artifacts/certificate_rl_adapter.zip`

原始 SHA-256：

```text
semantic_certificate_lab.zip
  e4a3252d97daebf44581f7c9f7edec01709506e965d1213aa163ee8cea2d22cc
certificate_rl_adapter.zip
  391451e895a491c3e52f34d871f2144def69a3589033d2dc2ead778753461e46
```

安全解压到 `baselines/prior/`，检查 zip 路径不能逃逸目标目录。阅读顺序：

```text
semantic_certificate_lab/REPORT.md
semantic_certificate_lab/THEORY.md
semantic_certificate_lab/experiment.py
semantic_certificate_lab/independent_check.py
semantic_certificate_lab/cost_check.py
certificate_rl_adapter/THEORY.md
certificate_rl_adapter/check_adapter.py
```

原包自带复现命令；在副本中运行，结果写入新的运行目录，不覆盖历史原始输出。[F1][F2]

### 1.3 已有结论：必须保留正、负两面

- 64 状态有限程序世界中，成对区间可利用共同部分抵消不确定性；不能据此声称真实软件可扩展。
- 已有反例证明：两个可靠误差上界相减，可能把真实退步判成进步。
- 已有反例证明：即时程序误差下降与长期修复价值不是同一对象。
- 旧有限程序基准中，精确后缀缓存比抽象证书更快；当前没有系统加速证据。
- 有限决策树已检查势函数／critic 正确变换后的 return、TD 和 GAE 等价；不是 RL 训练实验。
- 旧语料是开发／机制审计数据，不能重新标记为本轮独立测试集。

原报告中约 75.51% 的方向覆盖率，分母是具有真实非零作用的编辑；全部编辑覆盖率约 47.73%。不能混写。[F1]

---

## 2. 冻结数学定义：实现不得偷换

### 2.1 状态、目标、两个时间尺度

`i` 表示候选程序内部的执行步骤；`t` 表示代理工具回合；在线 token 级 PPO 另用 `k`。三者必须分别记录。

\(h_t\) 表示动作前信息状态，包括可见历史、当前程序、规范、预算及必要的 runtime 状态。固定 actor 为 \(\pi_0\)。

本轮默认有限回合、\(\gamma=1\)，原始奖励只有提交或真正终止时的任务检查：

\[
r_t=0\ (t<T-1),\qquad r_{T-1}=\mathbf 1[\text{当前程序满足冻结任务规范}].
\]

在真实预算耗尽时，也检查当前程序；预算用完但程序已经满足规范，不能一律判为失败。格式错误、语义违规、拒绝非法工具等按确定性环境规则处理并记录；基础设施崩溃不冒充语义失败。

任务的有限输入全集 \(\mathcal X\)、输入权重 \(\mu\) 和规范 \(\mathcal S\) 在任务创建时冻结。定义：

\[
E_{\mathcal S,\mu}(P)=\mathbb E_{x\sim\mu}[\ell_{\mathcal S}(P,x)],\qquad \ell\in\{0,1\}.
\]

主实验取均匀有限输入，所有输入质量为正。此时 \(E=0\) 才等价于整个**声明的有界输入域**均正确，不等价于任意部署输入下正确。

### 2.2 证书对象

\[
[L_t,U_t]\ni E(P_t)-E(P_{t+1}).
\]

- `BETTER`：\(L_t>0\)。
- `WORSE`：\(U_t<0\)。
- `EQUAL`：区间精确为 `[0,0]` 且有证明／精确执行依据。
- `UNKNOWN`：其他包含零的区间；包括分析不支持、超时或精度不足。

`INCOMPARABLE` 是定性 relative-correctness 关系的状态，不等于定量净变化为零；需要时另设字段。

### 2.3 从证书监督单值势函数

\[
q_\phi(P,\mathcal S,\mu)\in[-1,0],\quad
\mathcal L_{\rm cert}=\sum_j w_j\operatorname{dist}\big(q_\phi(P'_j)-q_\phi(P_j),[L_j,U_j]\big)^2.
\]

其中 \(\operatorname{dist}(z,[L,U])=\max(L-z,0,z-U)\)。只使用训练任务中的证书。

规范和输入测度必须作为输入或明确的任务上下文；不能把不同规范下同一代码的效应约束混在一起。宽区间不提供假精确标签；不能默认取 midpoint 当真值。

真实 \(-E\) 是可靠区间约束的一种可行势函数。但**学到的** \(q_\phi\) 不是自动获得证明的误差估计器；其输出不是被认证的“距离成功还剩多少”。单值函数只保证环路差分和为零。

没有锚点的差分图存在加法自由度。训练中可用已被终端验证为正确的训练程序锚定 `q=0`；没有锚点时记录这一事实，不伪造 `q=-E` 的绝对校准。不得用测试任务的证书图做 transductive 拟合。

只有符号的消融中，不得凭空要求统一 margin。对于均匀 N 个输入、二值损失，**已严格认证为正**才可利用最小非零效应 `1/N`；一般连续测度不能照搬。

### 2.4 接入 RL 的固定形式

非终端：\(\Phi(h_t)=\beta q_\phi(P_t,\mathcal S,\mu)\)；所有真正终端：\(\Phi(h_T)=0\)。

\[
r_t^\Phi=r_t+\gamma\Phi(h_{t+1})-\Phi(h_t),\qquad
W^\pi(h)=V^\pi(h)-\Phi(h).
\]

因此：

\[
G_t^\Phi=G_t-\Phi(h_t),\quad
\widehat V=\Phi+\widehat W,
\]

\[
r_t^\Phi+\gamma\widehat W_{t+1}-\widehat W_t
=r_t+\gamma\widehat V_{t+1}-\widehat V_t.
\]

同一数据、同一 mask、同一 bootstrap 下，两边 GAE 必须一致。[F2][R1][R2]

实现硬约束：

- `W` 不接受语义方向的符号约束；不要加正则把 `W` 强压到零；输出层不要用 `[0,1]` sigmoid 限制 W。
- 冻结 q 的**整个计算路径**，不只是最后一层。W 训练若改变 q 共用的 encoder，q 就没有被冻结。
- 每一轮 actor 采样／更新中固定 q，停止 actor 经 q 反向传播；更新 q 后重新计算相关缓存。
- q 在时刻 t 不看动作 \(a_t\)、未来工具反馈或本条轨迹终端结果。
- 真终止与 minibatch／上下文切块截断分开；后者必须 bootstrap。
- 失败终端的势函数结算不解释成“失败带来语义进步”。
- 只重写 reward 而 critic 总函数不变，应当得到相同 TD/GAE；这是必须通过的负控制。

本方案不使用任意事后 reward redistribution，不把动作相关证书无校正地当 baseline。[F2][R3][R4]

---

## 3. 任务环境：真实工具代理，闭合的程序语义

### 3.1 主环境的最小规格

构造可实际读写和运行的 Python 小仓库：3–6 个源文件、若干跨模块依赖、明确用户 issue、公共测试、只读任务规范。程序核心限制在一个有精确语义的子集，而不是支持任意 Python。

首版优先实现：有界整数字段／固定宽度 bit-vector、布尔条件、纯函数、固定元组、有限函数调用、显式定长循环或展开。明确 signed/unsigned、溢出、位移、异常、导入和执行步数语义。每个 operator 都需独立执行器与符号翻译的一致性测试。Z3 的 bit-vector 支持只是后端能力，不代表我们的 Python 翻译器自动正确。[R5]

不支持 eval、动态导入、反射、不受界递归、动态 heap、浮点近似语义、网络和文件系统副作用进入被认证核心。代理运行 shell 是工具环境的一部分，不意味着 shell 本身被形式认证。

域外编辑应当保存在真实工作区供代理修复，但分析器返回 `UNSUPPORTED/UNKNOWN`，不能静默当成 no-op 或正确。违反公开的任务实现约束，可由终端规范判为未完成；该规则须所有方法相同。

### 3.2 任务类型

至少覆盖三类不同语义骨架，每类包含多个独立程序模板：

| 类型 | 示例内容 | 主要检验 |
|---|---|---|
| 输入验证／变换管道 | 有界记录解码、合法性检查、标准化、输出编码 | 跨模块条件和输出契约 |
| 配置／权限决策组合 | flags、角色／资源字段、分层规则、兼容返回码 | 局部改善与其他输入上的回归 |
| 小型有界状态机 | 长度受限的事件序列、状态转换、终态断言 | 调用／时间依赖与中间状态 |

任务可以由正确生成器生成再注入 bug，但 gold program、mutation id、注入位置、最短修复脚本均只在审计侧；不进入 actor/critic 输入。不能把 mutation 数当“真实修复距离”。生成器自带正确版本也不是唯一正确实现。

默认 max turns 为 64；扩展实验为 32/96。想观察 20–100 回合的有效探索，但不得强行补空动作、禁止提前成功或人为拆开可一次完成的补丁来制造 long horizon。实际中位回合若很短，就报告短回合任务，不能包装成长程效果。

### 3.3 专门诊断的结构

必须保留：有信息价值的读／测试动作；多文件联合修改；公共测试进步但全域规范退步；revert；不改变语义的格式修改；中间语法不合法；分析器不支持的代码；回到旧版本；某些成功轨迹中的暂时语义退步。

“必须暂时退步”只在动作集合被明确限制、并已证明的有限回归单测中使用这一措辞。真实代码代理可能一次提交联合补丁，不能因为存在一条先坏后好的路径就宣称所有成功路径都必须如此。

### 3.4 工具与隔离

最少支持：

```text
list_files(path)
read_file(path, range)
search(pattern, path)
apply_patch(unified_diff)
run_public_tests(target)
shell(command)                 # 隔离容器内的受控命令
submit()
```

提供真正文件、实际进程退出码、stdout/stderr 和 traceback；不得预写工具观察充当执行。无复杂 planner、无 oracle edit chooser。

使用隔离容器／等价安全沙箱：无网络、无宿主凭证、非 root、资源配额、只允许工作区写入。公共测试和规范在只读位置；私有终端评估器与 gold fixtures 不挂载到代理。单独的 subprocess 不是安全边界。禁止代理修改 reward、runner、certificate cache 或日志。

### 3.5 Snapshot 的完整性

快照至少包含文件内容、运行目录、任务与规范哈希、完整会话、剩余回合／token 预算、采样配置、必要随机状态和工具运行状态。异步进程先完成／终止再快照；未支持的后台进程禁止产生。

同一快照 restore 后，确定性工具响应应重放一致；新的 LLM continuation 使用新采样 seed。只复制代码、不复制会话与预算，不是同一 RL 状态。

---

## 4. 证书实现：先用可信的最小后端，不伪装现成 DAC

### 4.1 默认的有限域分区成对证书

没有可直接接入的现成 relational verifier 时，实现一个明确受限的参考后端 `paired_partition_smt`。它是新工程实现，不等于已集成 DAC/SymDiff，也不宣称方法新颖。

对同一个输入 x，同时翻译 old/new 程序与固定规范，构造：

\[
d(x)=\ell(P,x)-\ell(P',x)\in\{-1,0,1\}.
\]

把有限输入域分成互不相交、覆盖全集且质量可精确计算的 bit-prefix cells \(C_j\)。均匀 bit-vector 域中，固定 b 个输入位的 cell 质量为 \(2^{-b}\)。

对每个 cell，查询 `d == -1`、`d == 0`、`d == +1` 的可满足性。只在获得 `UNSAT` 时排除一个取值；`UNKNOWN/TIMEOUT` 仍保留该取值。令未排除值的 min/max 为 \([l_j,u_j]\)，得到：

\[
L=\sum_j\mu(C_j)l_j,\qquad U=\sum_j\mu(C_j)u_j.
\]

这是按 cell 包围真实差值后取期望，不需要声称已完成一般 #SMT/model counting。SAT 见证说明存在，不说明该类输入占多大比例。

同一完整语义状态的共同后缀、或已证明任务损失相同的区域，可消去为 `[0,0]`；不能凭文本相似、变量同名或仅某个局部变量相等就消去。

### 4.2 Refinement 与成本

按 `cell_mass × interval_width` 的确定性优先级细分，逐步增加预算。children 联合界与旧 parent 界取交；若出现空交，视为语义／实现错误，立即停止，不选一个好看的结果。

每个证书预算点固定为建议的 `100 / 500 / 2000 ms`，实际求解器调用需保留余量。初始化、AST 翻译、缓存构建、进程启动、失败查询、全局校验均计费；这些是实验预算点，不是承诺能够达到的延迟。

缓存键必须包括 old/new 语义哈希、spec hash、mu hash、语义版本、solver version 和分析设置。跨任务同名函数不允许错误复用。

### 4.3 可信边界

标记 `guarantee_type=deterministic_bound_under_tcb`：证明以语言翻译器、规范编码、分区覆盖和 SMT 后端为可信基础。若未生成／检查独立 proof artifact，不声称 Lean 级内核证明。

小域必须用另一份独立 scalar interpreter 穷举真值，审计每个区间；不能两边共用同一 evaluator 函数。较大域抽查只能提供测试证据，不能替代符号分析的 soundness 论证。

浮点区间需要向外舍入；首版使用有理数／整数分子分母，避免假 soundness。unsupported 不降格成“经验上可信”。

### 4.4 两层对照必需

**验证层：**独立 old/new 区间差；公共测试计数；相同 CPU 预算的差分测试；增量精确执行／缓存；paired SMT。不能只比较最弱的独立上界。

**学习层：**比较证书监督是否帮助 critic；它和验证器是否加速是两个命题。若证书更贵但学习有效，只能报告信息收益及代价。

精确域可设为 2^10、2^14、2^18 的分级开发实验，按资源只完成可行范围；不可为了让证书赢而人为拖慢精确基线、去掉其缓存或只给一方预处理。


## 5. 信息权限与泄漏防线

### 5.1 三个平面

| 平面 | 能读取什么 | 不能读取什么 |
|---|---|---|
| Actor | issue、公开规范说明、工具历史、通过工具取得的文件／公共测试 | gold patch、私有测试输入、证书数值、未来结果 |
| 所有 critic 方法 | 同一份当前历史、当前仓库快照／编码、公开规范、剩余预算 | gold patch、mutation 身份、未来动作／反馈、私有终端标签作为输入 |
| Trainer／evaluator | 当前程序证书、训练 terminal labels、审计信息 | 不得把隐藏字段序列化进模型 prompt／feature |

critic 可在训练时读取完整当前程序快照，即 asymmetric critic；但**所有对照必须相同**。这种输入并不是 actor 已经通过阅读知道的全部内容，需要明确报告。不能让我们的方法看到完整仓库，而 plain critic 只能看短对话。

证书训练标签比公共测试更强，这正是实验变量；不得同时偷偷改变 critic 的原始输入。`CERT_FEATURE` 使用与主方法完全相同的证书、q 网络和当前信息。

### 5.2 自动泄漏检查

为每个数据字段定义 `actor_visible / critic_visible / training_label / audit_only`。输出可打印的输入样本，检查路径、任务名称、comment、文件名、模板序号、mutation id、剩余 bug 数和 future return 没有混入输入。

同一状态的 critic 输入 hash，在追加未来轨迹后必须不变。当前值函数输入不能包含本回合动作后的证书；历史已经发生的程序快照是当前状态，但主实验不向任何 critic 直接暴露原始证书标量。

证书缓存、q 输出缓存、编码缓存均按任务与 split 隔离。缓存命中不能使测试标签进入训练；禁止在测试任务上再拟合 q。

---

## 6. P2：真实代理轨迹与冻结数据集

### 6.1 先校准环境，再冻结正式划分

只在独立 smoke/dev tasks 上调工具格式、模型、上下文长度、任务难度。actor 必须实际输出工具调用并修改代码；记录模型哈希、tokenizer、chat template、temperature、top-p、stop 条件。

若成功率几乎全为 0 或全为 1，不能靠学习曲线宣称方法失败／有效。先报告 `POLICY_TASK_MISMATCH`，允许一次只用 smoke/dev 的任务／模型校准，之后重新冻结正式任务清单。不能观察 test 结果后不断修改生成器。

不强制某个现成模型名称；优先能在现有显存内正常工具调用、能够本地训练的模型。若只使用远端 actor 收集数据，必须已有明确授权，并承认后续本地 actor PPO 需要另一轮数据，不能假装是同一 policy。

### 6.2 默认规模

| Profile | train / dev / test tasks | 每任务初始轨迹 | critic seeds | 用途 |
|---|---:|---:|---:|---|
| smoke | 4 / 2 / 2 | 1 | 1 | 工程检查，不作科学证据 |
| pilot | 48 / 16 / 24 | 4 | 3 | 首轮可判断的固定 actor 实验 |

数量是上限式默认值。预算不足时保留完整较小 profile 并标注分母，不能复制轨迹凑数。相同 snapshot 的独立 stochastic continuation 用于价值估计，有统计用途；它们不是新的独立任务。

正式按**程序语义模板／原始任务来源组**切分，所有派生 bug、初始程序、轨迹、版本对留在同一 split。仅随机更换常数或文件名不算新语义模板。至少三类任务，test 包含训练未见的语义组合／调用骨架；报告模板数，不能把 24 个实例当 24 个独立 family。

先保存 split 和生成器版本 hash，再生成正式轨迹。任何修改都生成新版本，不覆盖冻结语料。

### 6.3 轨迹记录

每个 tool turn 保存：

```text
task_id, task_group_id, split, trajectory_id, policy_version
turn_id, pre_history_hash, pre_snapshot_hash, post_snapshot_hash
actor_messages_prefix_ref, action_text, tool_call, tool_result_ref
assistant_token_ids, behavior_logprobs, response_mask, sampling_config
program_changed, syntax_status, supported_semantics_status
remaining_turns, remaining_generated_tokens, done, termination_reason
terminal_reward, wall_ms, cpu_ms, input_tokens, output_tokens, gpu_ms
certificate_ref                         # 标签索引，不进入模型输入
```

对每个代码变化的版本对做后台批处理式**当前实验内离线标注**，不反馈给冻结 actor。不修改代码的工具动作可记录 semantic delta = 0，但不要把“无代码变化”的样本大量复制成语义训练主数据。

所有方法使用同一组已保存轨迹。训练数据包含真实成功、失败、回退、格式错误和未支持状态；不只保留可认证的成功编辑。

### 6.4 保持对话与训练分布一致

同一模型和 chat template 处理 rollout 与训练。优先 temperature=1、top-p=1、无额外 repetition penalty，方便在线阶段使采样分布与计算 logprob 的 policy 一致。若 smoke 需要其他采样，必须冻结并明确实际行为分布；不能用原始 logits 的 logprob 冒充经截断／重加权后的采样概率。

上下文截断策略固定，所有方法相同；保存真实送入模型的 token prefix。不得在训练时偷偷补回推理时被截掉的文件。工具输出不是 actor 生成 token。verl 的多轮文档专门讨论这一 mask／tokenization 问题，实际集成必须加严格检查。[R6]

---

## 7. P3：critic 实验，不先更新 actor

### 7.1 两种输入编码必须配平

可以先用同一 frozen code-capable encoder 提取历史与程序特征，训练小型 critic heads，作为便宜的第一层证据。所有方法共用相同 encoder、packing、context cap 和 feature cache。

head-only 通过后再做小型 LoRA critic 验证；head-only 失败不能直接证明全参数 critic 不需要证书，head-only 成功也不是完整 LLM RL 成功。不要为赶结果把有结构程序压成 task ID embedding。

`q` 接收当前程序＋规范＋测度描述；`W`/V 接收相同历史＋程序＋规范＋预算。程序超过上下文时使用固定的、非标签驱动的打包／截断，报告被省略比例。

### 7.2 必做对照，不重复计算同一种 PPO

| ID | 方法 | 输出／训练方式 | 区分的问题 |
|---|---|---|---|
| B0 | `RETURN_ONLY` | 普通 V(h)，仅训练真实 return | 核心基线 |
| B1 | `SPLIT_NO_CERT` | V=βq+W，但无真实证书监督；用 return 训练 | 双头／参数化本身是否带来收益 |
| B2 | `TEST_FEATURE` | 公共测试得分监督一个辅助网络，供普通 V 使用 | 是否普通可执行反馈已经足够 |
| B3 | `CERT_FEATURE` | 与 M 相同的证书预训练 q；冻结后让普通 V(h,q) 自由使用 | 证书信息本身 vs 固定加法分解 |
| M | `CERT_DECOMP` | q 由区间监督；冻结 q；W 拟合 G−βq | 主方法 |

B3 必须真的能用到证书监督后的表征／q 输出，不能接一个不影响 value 的装饰性辅助头。M 与 B3 共用同一训练 q checkpoint；其差别只在 value 的消费方式。

冻结 encoder、trainable 参数、q 预训练、return 更新次数、数据条数、总 forward/backward 成本分别计数。可比 active 模型容量尽量在 5% 内；不能用未参与计算的 dummy 参数凑数。若严格配平不能实现，就报告差异并加入相同计算预算版本，不写“完全公平”。

`Terminal-only PPO` 与 `plain critic PPO` 在这里不是两个独立 baseline；二者通常是同一个 B0 在线实现。[R7]

辅助诊断（不替代上述对照）：

- `ORACLE_E`：在小型可穷举任务用精确 −E 代替 q，只作昂贵上限诊断，不能混作廉价主方法。
- `SHUFFLED_CERT`：在冻结训练集内按任务类型／区间宽度桶打乱语义对应关系，保持数量与预算；它不再是 sound certificate。记录约束冲突，不能把不可能拟合的随机标签当成唯一控制。
- `BETA_ZERO` 与 algebraic-equivalence：检查没有语义成分时应退回相应 baseline。
- naive sign reward 仅在既有反例中证明错误，不优先为一个已知会变目标的方案花大规模 PPO 预算。

### 7.3 训练流程

1. 只从 train split 的版本对训练 q；dev 选固定 checkpoint、权重和 β。
2. 默认 β 候选 `[0, 0.25, 1.0]`，相同 dev 调参预算；不在 test 上选 β。
3. 报告 q 的非零方差、区间损失、有效方向标签数、正／负方向分母、相连分量与锚点数量。零输出也能满足大量宽区间，不能把低 loss 当学到了语义。
4. 冻结 q。对每条真实轨迹计算 Monte Carlo terminal return G。
5. M 的目标为 `G - Phi`；B0/B3 的目标为 G。主实验用同类 MSE/Brier 目标、同样训练样本权重，不向 M 偷加未来 rollout。
6. 比较全部回合的 `V_total=Phi+W`，不是只比较 W 自己的 MSE。M 的 W 通常需要输出大于 1 的值。
7. 所有模型值预测在 held-out prefix 上冻结后，再运行参考 continuation。之后不能根据参考结果修改模型。

若 q 与 W 共享可训练特征，先独立冻结 q 分支或复制出固定 teacher。训练 W 不能让已缓存的 Phi 失效。跨 online iteration 更新 q 时重做这个过程。

### 7.4 额外输入与额外计算分别计账

报告两种预算口径：

- **交互配平**：同样 actor 轨迹／token 数；允许额外证书计算，但完整展示该成本。
- **总成本配平**：证书的 CPU 时间、q 训练、额外 encoder 和缓存也计入；给普通 critic 相当的额外训练或数据预算作对照。

不能把第一种结果写成“总体训练加速”。硬件异构时先并列 GPU-hours、CPU-hours、输入／输出 tokens、wall time；没有真实费率不要捏造统一美元成本。

---

## 8. 独立评价：真实 LLM 的 V/A 没有精确 oracle

### 8.1 第一主指标：冻结 policy 的 held-out return prediction

对 test 任务的独立轨迹，在动作前 prefix 预测最终 return。按任务／来源组聚合 Brier/MSE 与校准，不把同一轨迹 50 个 prefix 当成 50 个独立样本。

二值终端回报下：

\[
\mathbb E[(\widehat V(h)-R)^2\mid h]
=(\widehat V(h)-V^{\pi_0}(h))^2+V^{\pi_0}(h)(1-V^{\pi_0}(h)).
\]

因此，对同一批 held-out 状态和结果的**配对损失差**，能比较 value 拟合；单次 outcome 不是该状态的真实 V。保留噪声项，不把 Brier 直接称为 exact value error。

不剪裁原始 V 来制造主指标改善；可另报 clipped-to-[0,1] 校准诊断。所有方法口径相同。

### 8.2 参考 continuation：有限预算的小面板

在看各 critic 的预测和错误前，以固定 seed 从 dev/test 的预声明任务／回合分层抽取中间状态。默认 dev 8 个、test 16 个，总计 24 个 anchor，覆盖至少 12 个任务；预算允许可扩至 48 个。

每个 anchor 固定 16 次独立 continuation，得到 `V_MC`；使用**同一个冻结 actor、同样完整状态、同样剩余预算**。训练这些 critic 的 return 不得来自这组评估 continuations。

选取其中 8 个 anchors 作小型动作面板：固定一个由同一行为 policy 真实采样的动作，执行后另做 16 次 continuation，得到 `Q_MC`；与独立 V_MC 比较。必要时增加第二个候选动作，必须单独计入预算。

16 次估计可能很粗；输出 successes/n、置信区间。区间无法区分 A 的符号就标为 `UNRESOLVED_MC`，不硬造“真实 advantage 标签”。采用固定样本量；若后续自适应追加，先实现有效的 anytime/多重查看校正，不能反复查看普通置信区间直到显著。

大量 continuation 是评估成本，不是主方法可以免费访问的 oracle；所有方法复用同一参考面板。

### 8.3 指标表

| 层 | 主报告 | 不允许的替代说法 |
|---|---|---|
| 证书 | 区间违规、宽度、方向分母、成本、unknown 原因 | “零测试误判 = 全语言证明” |
| q | held-out 区间一致性、已认证方向覆盖、锚点／规范泛化 | “q 就是剩余修复距离” |
| critic | 配对 held-out Brier/MSE、MC reference 误差与不确定性 | “预测成功率已被形式认证” |
| advantage | 参考可辨别子集的方向／排序，GAE 偏差与方差诊断 | “GAE variance 小就一定更好” |
| online | 终端成功率 vs 交互、vs 总成本、回归与预算消耗 | “非零奖励数多 = credit 改善” |

真实 LLM 梯度无法穷举。可在固定 LoRA 参数子集或预注册随机投影上估计 gradient variance／与独立高样本参考的偏差，必须写作 sampled/sketched diagnostic；不叫 exact gradient audit。不以低方差单独作为 gate，零梯度也有低方差。

### 8.4 统计与 horizon

主统计单位为独立任务来源组，嵌套报告模板、实例、轨迹和 prefix 数。critic seeds 不是新的任务。用 paired task/group bootstrap 报 95% 区间，至少 3 个训练 seeds；pilot 小样本不冒充最终论文证据。

horizon 分析按事先定义的依赖深度／任务结构或随机分配的预算上限分层。按训练后实际回合数分桶只能作描述，不能把“模型更弱所以跑得久”的选择偏差说成长程因果效应。

不预设收益必须随 horizon 单调增加；这不是前面理论的推论。

---

## 9. 阶段裁决：先确认可观测，再确认有效

### G0：正确性与数据完整性，必须通过

- 原包复现或者准确记录不可复现原因；禁止以旧日志冒充新运行。
- 有界执行器／符号翻译、区间、分区覆盖、snapshot 和信息权限检查通过。
- 任何已确认的 certificate 包围违规或输入泄漏：`INVALID`，先修复并重新生成受影响数据。
- 采样／训练 tokenization、done/truncation、q 冻结与 GAE 等价性检查通过。

### G1：真实 actor 与证书具有可测支持

pilot 至少来自 8 个独立 train tasks 的 32 个有信息版本对，且能看到至少 10 个认证改善、10 个认证退步；这是最低诊断门槛，不代表样本已充分。标签偏向某一方向须报告，不人工往 test 注入所需比例。

工具环境产生真实读／测／改过程，任务存在成功与失败；大部分状态值都不可区分或 q 近似常数时，标记 `INSUFFICIENT_SIGNAL` 或 `POLICY_TASK_MISMATCH`，不能直接宣布 H0 赢。

报告证书支持率：按所有回合、所有实际编辑、可翻译编辑分别列分母。只在可分析编辑上很强，不代表覆盖了整个 agent 轨迹。

### G2：P3 是否值得上小规模 PPO

建议预注册的推进标准：M 或 B3 对 B0 和 B2，在 dev 上有至少 5% 的相对平均 Brier/MSE 改善，方向在 3 个 seeds 中一致；冻结后 test 的配对主指标差为正，并且 95% group bootstrap 区间下界大于 0。若 B0 loss 几乎为零，改用预先声明的绝对差而不是不稳定百分比，并承认 ceiling。

这些是本轮预算分配标准，不是物理规律。必须在打开 test 前冻结。

- `READY_FOR_ONLINE`：完整性与信号门槛通过，证书相对 return／测试基线有独立改善。
- `CERT_SIGNAL_ONLY`：B3 与 M 均有效但 M 未优于 B3；支持证书信息，不支持“固定加法分解是贡献”。在线需保留 B3，不能隐藏。
- `INCONCLUSIVE`：区间过宽／样本不足／reference 太噪。停止扩大 GPU 实验，交付证据和所缺分母；不宣布普遍否定。
- `NOT_SUPPORTED_IN_TESTED_SETTING`：数据有效且预注册范围内没有收益或明显变差；保留全部负结果。
- `INFO_GAIN_BUT_COSTLY`：交互配平有效、总成本不占优。可作限定的小 online 诊断，但不得给出 infra 加速结论。

`gate_decision.json` 必须含每个检查的证据路径、阈值、实测值、CI、日期和 config hash；不是一句手写 PASS。


## 10. P4：小规模在线 PPO 的接入规格

### 10.1 进入条件和最小组别

只有 `gate_decision.json` 有效、完整性检查通过，并在本地配置明确给出 `online_gpu_hours > 0`，才执行。默认值为 0：先交付固定 actor 的真实实验结果，不自动消耗新一轮预算。

至少保留：

- B0：原始 terminal objective + 普通 critic 的 PPO。
- B3：同一 q 的 certificate-feature critic PPO。
- M：certificate decomposition PPO。

如 P3 表明公共测试基线同样强，再保留 B2。不要把早已被代数证明不增加信息的纯 reward 改写做成主方法；它只用于数值一致性测试。

同一 actor 初始 checkpoint、tokenizer、任务分布、rollout 上限、PPO hyperparameters、KL 设置、更新次数、clip／mask；critic 结构与额外成本如实记录。各方法更新后 policy 不同，不能继续声称训练轨迹“完全相同”；只能配平采样计划、初始条件和预算。

### 10.2 默认用现有 actor–critic 实现，不用 GRPO 冒充

优先复用已有可运行的 PPO/GAE 框架。verl 的官方文档提供 PPO 以及有状态多轮工具接口，可作为候选；其具体配置随版本变化，必须检查当前安装源码／schema，不照抄文档字段。[R6][R7]

如果只能跑 GRPO，则记录 `BLOCKED_CRITIC_BACKEND`，不把训练 objective 偷换后继续声称测试了本方案。

第一轮在线固定 q 整个 pilot，不做复杂的逐轮 q 自更新；新增证书只做旁路审计。后续 policy distribution shift 要单独报告。若未来研究刷新 q，必须在 rollout 边界进行、让 B3/M 同步刷新、重新计算 value/GAE 并计入费用。

### 10.3 不能混淆 token PPO 与 turn PPO

首选已有框架的 **assistant-token 级 PPO**。实现一份明确的 `token_alignment.md`：

- `k` 只索引实际由 actor 采样的 assistant token；工具输出／用户文本不参与 actor logprob loss。
- 状态 \(h_k\) 是当前 token 生成前的 prefix，不包含该 token 或未来工具结果。
- 一个工具回合生成期间，程序尚未改变，Phi 使用动作前程序；工具实际执行后，下一生成状态才采用新程序的 Phi。
- 将环境奖励／势函数变化放在与该工具转移对应的最后一个 assistant token 转移处；终端检查同理。
- GAE 的下一 value 索引是下一**决策状态**，可能跨过很多工具文本；不能简单取拼接文本的相邻 token value。
- 使用 `gamma=1` 的有限 token 决策过程保持本轮 terminal 目标；done 与仅被 packing 截断分开。
- 同样实现 W 的 token-prefix value；只在 tool turn 边界训练过的 critic 不能未经验证就冒充所有 token 上的 critic。在线前需要 token 级 smoke 和因果输入检查。

若改做 agent-turn PPO，必须使用与整段动作概率相对应的目标和 ratio，并记录序列长度的数值问题。把 turn advantage 重复到每个 token 后分别 clipping，不等价于严格的 sequence/turn PPO；不允许在报告里混称。

### 10.4 Actor/critic 更新

每批：

```python
# 概念伪代码；实现时必须把 done、bootstrap、token indexing 明确对齐。
phi = frozen_potential(current_programs, spec, mu)
phi[true_terminal_states] = 0
value_total = phi + continuation_critic(prefixes)
advantages = gae(original_rewards, value_total, done, gamma=1.0, lam=0.95)

# 同时计算变换形式作为只读审计，而不是再加一遍 shaping。
shaped = original_rewards + gamma * phi_next - phi
advantages_check = gae(shaped, continuation_values, done, gamma, lam)
assert_close(advantages, advantages_check)

actor_loss = ppo_clipped_loss(old_logprobs, new_logprobs, stopgrad(advantages), masks)
critic_loss = mse(value_total, detached_value_targets)
```

可以直接用原 reward + V_total 的接口，减少 reward pipeline 改动。不能同时让 reward 加一次 Phi 差、critic 又错误输出 V_total 而重复扣加。

old logprobs 来自实际行为 policy；所有方法同样处理 KL、advantage normalization、value clipping、entropy 和 padding。若 KL 是附加正则，报告其改变了怎样的优化目标；它不属于程序证书理论。

默认起点：GAE λ=0.95，clip=0.2，每次 rollout 后 1 个 PPO epoch，每批 8 个完整 episodes；学习率和微批按已有稳定本地基线在 dev 上选定后统一冻结。不能把“跑了 32 次 optimizer update”写成“跑了 32 个数据 epochs”。

### 10.5 在线评价的独立性

P3 test 用来决定是否继续，不能再充当不受任何选择影响的最终 online test。P4 另封存新的任务来源组／语义组合，或使用事先从未打开的第二测试划分。默认新增 24 个任务，所有方法共用。

最少 3 个独立训练 seeds 才能报告正式 pilot 稳定性；预算只能跑 1 seed 时写工程 smoke，不下稳定收益结论。

报告成功率随 generated tokens、prefill tokens、工具回合、GPU-hours、CPU verification time、总 wall time 的曲线。同步报告 syntax/unsupported 率、临时退步后的恢复、回退循环、公共测试过拟合、长度和模型 KL。

不要只报告某一个中间最好 checkpoint；预先固定评估时间点与主结果 checkpoint。

---

## 11. 必须实现的单元与回归测试

以下检查在任何正式 GPU 训练前运行，失败即退出，不“先训练看看”。

| 测试 | 应当验证的事情 |
|---|---|
| `upper_bound_difference_counterexample` | 可靠 U 下降但真实 E 上升；不产生错误正标签 |
| `paired_certificate_contains_truth` | 独立解释器小域真值始终在成对区间内 |
| `partition_cover_and_weights` | cells 不交、覆盖全集、质量精确加和为 1 |
| `smt_unknown_is_not_unsat` | solver unknown/timeout 不排除可能损失 |
| `refinement_is_nested` | 同一版本对／规范下增加预算不会无依据扩大或错缩界 |
| `unsupported_not_equal` | 不支持的代码不能被标成 EQUAL／正确 |
| `invalid_syntax_semantics` | 若规范明确要求可执行，语法错误按冻结规则处理，且与 analyzer unsupported 区分 |
| `shared_state_cancellation` | 消去要求完整后续相关状态／等损失证明 |
| `sign_cycle_counterexample` | +1/-1 符号环可赚 reward；单值 Phi 差分环为零 |
| `future_reward_leak_counterexample` | 任意未来依赖的事后 reward 分配可破坏 reward-to-go 梯度 |
| `temporary_regression_counterexample` | 语义退步与正长期 advantage 能同时成立 |
| `terminal_settlement` | 成功、失败、真实超时都正确清零 Phi |
| `truncation_bootstrap` | packing/minibatch 截断不冒充终止 |
| `td_gae_equivalence` | reward 变换形式与原 reward+V_total 数值相同 |
| `phi_frozen_entire_path` | W 更新不改变同一输入 q 输出 |
| `action_independent_baseline` | 当前 baseline 输入不含当前动作或未来证书 |
| `tool_tokens_masked` | 工具/user token 不进入 actor loss；value 索引无错位 |
| `sampling_logprob_agreement` | 采样分布、teacher forcing logprob、chat template 一致 |
| `snapshot_replay` | 确定性工具与终端 evaluator 重放一致 |
| `gold_and_split_leakage` | gold、mutation、future、test labels 均不出现在输入或训练 cache |

浮点代数审计在 CPU float64 用 `1e-10` 量级阈值；BF16/FP16 训练检查另报实测数值误差与精度设置，不能用过宽 tolerance 掩盖索引或符号错误。

新增反例、发现的错误版本和失败 seed 都保留为 fixtures，不通过删除困难样本让测试通过。

---

## 12. 数据结构与目录规格

### 12.1 核心接口

```python
# 需要实现的接口；不是宣称已经提供了这些模块。
reset(task_id: str, seed: int) -> Observation
step(action: AgentAction) -> StepResult
snapshot() -> SnapshotRef
restore(snapshot: SnapshotRef) -> Observation
terminal_verify(snapshot: SnapshotRef) -> TerminalResult
compare(old: SnapshotRef, new: SnapshotRef, spec: SpecRef,
        measure: MeasureRef, budget_ms: int) -> DirectionCertificate
```

`terminal_verify` 不是 actor 可以无限查询的隐藏测试 API；主回合只在 submit/终止调用。public tests 是另一个接口。

### 12.2 Certificate schema

```json
{
  "old_snapshot_hash": "...",
  "new_snapshot_hash": "...",
  "spec_hash": "...",
  "measure_hash": "...",
  "semantics_version": "bounded_py_v1",
  "lower": {"numerator": 0, "denominator": 1},
  "upper": {"numerator": 1, "denominator": 1},
  "direction": "UNKNOWN",
  "guarantee_type": "deterministic_bound_under_tcb",
  "reason": "budget_exhausted",
  "backend": "paired_partition_smt",
  "solver_version": "record_actual_version",
  "assumptions": [],
  "affected_region_refs": [],
  "partition_ref": "...",
  "query_log_ref": "...",
  "budget_ms": 500,
  "actual_wall_ms": 0,
  "actual_cpu_ms": 0,
  "preprocess_ms": 0,
  "cache_hit": false,
  "created_at": "..."
}
```

这是 schema 示例，零耗时是占位，真实输出必须实测。无需伪造 `confidence=0.99`；确定性保证、统计置信度、启发式 score 使用不同字段和类型。

### 12.3 目录

```text
project_certificate_agentic_rl/
  PROJECT.md
  pyproject.toml
  configs/
    smoke.yaml
    pilot.yaml
    online.yaml
  baselines/prior/                         # 原包只读副本
  src/csavl/
    cli.py
    env/{tasks,tools,sandbox,snapshot,terminal}.py
    semantics/{syntax,interpreter,smt_translation}.py
    certificates/{types,paired,independent,refinement,cache}.py
    data/{splits,rollouts,labels,leakage,schemas}.py
    learning/{encoders,potential,critics,gae,ppo_adapter}.py
    evaluation/{reference_rollouts,metrics,statistics,costs}.py
  tests/
  data/{manifests,train,dev,test_critic,test_online_sealed}/
  runs/<run_id>/{config,logs,checkpoints,predictions,certificates}/
  reports/
  artifacts/
```

可以根据实际已有仓库调整实现，但输出接口、数学语义、split 和验收口径不可偷偷改变。

---

## 13. 默认配置与预算停止规则

以下是本项目的**建议初始配置**。模型路径、设备和实际框架必须由 P0 检测填入。没有提供模型时允许扫描已授权本地缓存，不允许为了找到“最好模型”自动下载几十 GB 或调用付费 API。

```yaml
project:
  name: csavl
  protocol_version: "0.1"
  mode: offline_gate_first
  root: ./project_certificate_agentic_rl
  require_clean_test_split: true

resources:
  model_path: null
  tokenizer_path: null
  device: auto_available_authorized_device
  allow_paid_api: false
  allow_cloud_jobs: false
  allow_model_download: false
  local_gpu_hours_p0_p3_cap: 6.0
  local_cpu_hours_p0_p3_cap: 16.0
  generated_tokens_total_cap: 3000000
  prefill_tokens_total_cap: 60000000
  new_disk_gb_cap: 30
  online_gpu_hours_cap: 0.0
  checkpoint_before_budget_exhaustion: true

profile:
  name: pilot
  train_tasks: 48
  dev_tasks: 16
  test_critic_tasks: 24
  trajectories_per_task: 4
  critic_seeds: [11, 23, 47]
  split_seed: 20260921
  group_split: semantic_template_source

environment:
  max_turns: 64
  max_generated_tokens_per_episode: 8192
  max_new_tokens_per_turn: 256
  max_context_tokens: 16384
  terminal_only_reward: true
  gamma: 1.0
  network_in_sandbox: false
  gold_visible_to_actor: false
  certificate_visible_to_actor: false

certificate:
  backend: paired_partition_smt
  budget_ms_grid: [100, 500, 2000]
  primary_budget_ms: 500
  exact_rational_bounds: true
  preserve_unknown: true
  independent_interpreter_audit: true
  fit_on_test_graph: false

potential:
  range: [-1.0, 0.0]
  loss: interval_distance_squared
  beta_grid: [0.0, 0.25, 1.0]
  freeze_for_critic_training: true
  freeze_for_first_online_pilot: true
  terminal_override: 0.0

critic:
  methods: [RETURN_ONLY, SPLIT_NO_CERT, TEST_FEATURE, CERT_FEATURE, CERT_DECOMP]
  pilot_encoder: frozen_local_code_encoder
  labels: observed_terminal_return
  shared_input_access: true
  hyperparameter_selection: dev_only

reference:
  dev_anchors: 8
  test_anchors: 16
  continuations_per_anchor: 16
  q_anchor_subset: 8
  fixed_action_continuations: 16
  fixed_sample_size: true
  actor_version: same_frozen_actor

online:
  enabled: false
  require_gate: READY_FOR_ONLINE_or_CERT_SIGNAL_ONLY
  backend: reuse_installed_ppo_or_verl
  assistant_token_level: true
  gae_lambda: 0.95
  clip_ratio: 0.2
  rollout_episodes_per_update: 8
  ppo_epochs_per_rollout: 1
  pilot_updates_per_seed: 32
  seeds: [11, 23, 47]
  test_online_new_tasks: 24
  certificate_hard_pruning: false
```

这些 cap 不意味着任务一定能在其中完成。P1 实测每条轨迹与 continuation 成本后，先预算整个 P2–P3，至少为独立评价和 reference 保留 35% 的 generated-token 额度；预计超额则在冻结正式划分前缩小 profile，不先把预算全部花在 train 轨迹。逻辑 prompt tokens、真正执行的 uncached prefill tokens 和 cache hit 分别记录。

所有阶段共享总额；达到上限即 checkpoint 并交付 `BUDGET_EXHAUSTED`，不要无限追加 continuation。资源不足时优先保留：完整性、真实轨迹、B0/B3/M、独立评价；缩小 profile 前写明计划变化，不在结果出来后挑删对照。

在线预算为 0 与 `enabled=false` 是刻意设计：本次交接默认先完成 critic 门槛。未来明确允许在线预算并开启后，程序可依据同一 gate 自动接着执行，不需要重新推理论。

---

## 14. Codex 必须实现并实际运行的命令

以下为待实现 CLI 合约。若已有相似工具，可映射到现有命令，但在报告中给出真实调用。

```bash
# P0：清点资源、检查输入包、冻结环境
python -m csavl.cli inventory --config configs/smoke.yaml
python -m csavl.cli reproduce-prior --out runs/prior_recheck
python -m pytest tests/ -q

# P1：环境与证书开发，真实 LLM smoke
python -m csavl.cli build-tasks --config configs/smoke.yaml
python -m csavl.cli audit-semantics --config configs/smoke.yaml
python -m csavl.cli collect --config configs/smoke.yaml
python -m csavl.cli certify --config configs/smoke.yaml
python -m csavl.cli audit-runtime --config configs/smoke.yaml

# P2：只能在 smoke/dev 校准后冻结正式数据
python -m csavl.cli freeze-protocol --config configs/pilot.yaml
python -m csavl.cli build-tasks --config configs/pilot.yaml
python -m csavl.cli collect --config configs/pilot.yaml
python -m csavl.cli certify --config configs/pilot.yaml
python -m csavl.cli audit-data --config configs/pilot.yaml

# P3：q、对照 critics、独立结果与裁决
python -m csavl.cli train-potential --config configs/pilot.yaml
python -m csavl.cli train-critics --config configs/pilot.yaml --all-methods
python -m csavl.cli freeze-predictions --config configs/pilot.yaml
python -m csavl.cli collect-reference --config configs/pilot.yaml
python -m csavl.cli evaluate --config configs/pilot.yaml
python -m csavl.cli decide --config configs/pilot.yaml
python -m csavl.cli report --config configs/pilot.yaml

# P4：只有 gate + online enabled + 正在线预算同时成立才允许运行
python -m csavl.cli online --config configs/online.yaml --require-gate
```

任何命令没有执行，就标为 `NOT_RUN`；不能用预期表格、硬编码 metrics、占位 loss 或随机数替代真实训练输出。`pytest` 通过不等于整个研究假说通过。

---

## 15. 必交付的结果，不只交论文式总结

最终生成：

```text
reports/ENVIRONMENT.md
reports/PROTOCOL_FROZEN.md
reports/PRIOR_REPRODUCTION.md
reports/SEMANTICS_AND_TCB.md
reports/INPUT_VISIBILITY_AUDIT.md
reports/CERTIFICATE_AUDIT.md
reports/CRITIC_COMPARISON.md
reports/RESOURCE_ACCOUNTING.md
reports/FINAL_DECISION.md
reports/NEXT_ACTION.md
artifacts/gate_decision.json
artifacts/config_resolved.yaml
artifacts/manifest.sha256
```

此外保存任务 split、真实原始轨迹、程序快照、证书查询证据、terminal outcomes、prediction CSV/Parquet、reference continuation 计数、训练曲线、完整日志、checkpoint 和复现命令。损坏／中止轨迹不得静默消失；在分母表单列。

主表至少包含：

```text
method, seed, task_group_count, task_count, trajectory_count, prefix_count,
train_return_labels, train_certificate_pairs, cert_positive, cert_negative,
brier_task_macro, paired_gain_vs_return, paired_gain_vs_test, confidence_interval,
mc_reference_n, mc_unresolved_rate, cert_supported_edit_rate, cert_sign_coverage,
actor_output_tokens, actor_prefill_tokens, q_train_gpu_hours, critic_gpu_hours,
verifier_cpu_hours, total_wall_time, parameters_active, checkpoint_hash
```

至少输出以下图（可以 CSV + matplotlib）：held-out value loss vs 训练数据量；证书有效方向覆盖 vs CPU 预算；各方法计算成本；预声明任务深度分层结果。没有 online 就不画伪 online learning curve。

`FINAL_DECISION.md` 首段只回答四件事：实际做完了什么；主假说得到什么证据；最大未解决限制；是否达到 online 门槛。负结果和成本劣势放正文，不藏附录。

---

## 16. 给 Codex 的最后执行指令

先实施和运行，不要继续产出另一份泛泛计划。保持本文件的核心问题不变：**在真实工具代理轨迹上，程序语义证书能否改善原任务目标下的价值学习？**

先判断信息是否存在，再判断怎样消费信息。能可靠认证当前编辑，不等于能认证未来策略价值；能预测价值，不等于已经改进在线 RL；同样交互更好，也不等于总计算更省。

若普通 critic 已经一样好，或 `CERT_FEATURE` 与主方法一样好，应如实输出对应结论。不要靠改名称、加模块、换测试集或只展示成功 seed 把负结果改写成成功。

---

## 附录 A. 文献、来源与本方案的关系

**引用原则：** `[F1][F2]` 是随包已有材料；`[R*]` 是外部原始论文或官方文档。技术实现默认以已安装版本源码为准。本方案的任务、超参数、阈值和阶段设计是建议，不是引用文献已经证明的结果。

### 已有项目材料

- **[F1] Semantic Certificate Lab v0.1**：`prior_artifacts/semantic_certificate_lab.zip`。成对程序差分、反例、有限审计、精确缓存更快的负结果。不是自然软件实验。
- **[F2] Certificate-compatible RL adapter**：`prior_artifacts/certificate_rl_adapter.zip`。势函数＋W 的推导、GAE 等价性、未来泄漏与必要退步反例。不是已训练的 RL 算法。

### 本次核对过的外部来源

- **[R1] Wiewiora (2003), Potential-Based Shaping and Q-Value Initialization are Equivalent.** 用于说明 reward 变换与价值参数化之间的等价性，不支持“非零 reward 更多必然更好”。
- **[R2] Schulman et al., High-Dimensional Continuous Control Using Generalized Advantage Estimation.** GAE 的原始论文；支持 TD/advantage 的估计框架，而不是程序证书的效用结论。
- **[R3] Arjona-Medina et al., RUDDER: Return Decomposition for Delayed Rewards.** 不能把 return-equivalence 简化成任意事后 credit 可以直接送入 PPO。
- **[R4] Tucker et al. (ICML 2018), The Mirage of Action-Dependent Baselines in Reinforcement Learning.** 用于控制 baseline/方差论证和实现差异，不据此否定所有 auxiliary signal。
- **[R5] Microsoft Online Z3 Guide, Bitvectors.** 后端 bit-vector 语义能力；不提供本项目翻译器的自动 soundness。
- **[R6] verl official documentation, Multi-turn Rollout Support.** 工具生命周期、assistant mask、tokenization 注意事项。部署必须 pin commit。
- **[R7] verl official documentation, Proximal Policy Optimization (PPO).** 现有 actor–critic 接口参考；不是把 terminal-only 与 plain critic 分成两个 baseline 的理由。
- **[R8] Schulman et al., Proximal Policy Optimization Algorithms.** 在线 PPO 的原始算法参考。

可复制地址（公开来源核对日期：2026-09-21）：

```text
[R1] https://jair.org/index.php/jair/article/view/10338
[R2] https://arxiv.org/abs/1506.02438
[R3] https://arxiv.org/abs/1806.07857
[R4] https://proceedings.mlr.press/v80/tucker18a.html
[R5] https://microsoft.github.io/z3guide/docs/theories/Bitvectors/
[R6] https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html
[R7] https://verl.readthedocs.io/en/latest/algo/ppo.html
[R8] https://arxiv.org/abs/1707.06347
```

原始势函数策略不变性的经典出处：Ng, Harada, Russell, *Policy Invariance under Reward Transformations*, ICML 1999。其作者稿地址在 [F2] 中；本轮直接使用已写出的望远镜恒等式和前提，不把它作为新定理。

不将前序聊天中未经此次核对的“最新 agent 论文”、性能数字或框架状态写成实验前提；本轮不声称已排除所有相邻工作或已确认论文新颖性。

---

## 附录 B. 判定范围速查

| 观察到的结果 | 允许的结论 | 不允许的结论 |
|---|---|---|
| finite 证书／代数检查通过 | 实现符合所测有界语义与恒等式 | 已改善真实 agent RL |
| M 胜 B0，但不胜 B3 | 证书信息在当前设置有用 | 固定加法分解是独有创新 |
| held-out critic 更好 | 额外监督帮助当前 policy 的价值预测 | 已解决探索／长期因果 credit |
| rollout 相同但总成本更高 | 信息—计算 trade-off | infra 净加速 |
| 部分 source/语法有效 | 当前支持子集有效 | 任意 GitHub issue 都有严格证书 |
| q 在训练对上拟合很好 | 学到了部分区间约束 | q 的泛化输出被形式认证 |
| 更多回合出现 shaped reward | 中间数值更稠密 | 原任务策略梯度信息自动增加 |
| exact oracle 只在小域胜出 | 有限世界仍可精确解 | 大域可扩展性已证明／已否定 |

**本轮最终的可证伪目标只有一个：程序语义证书是否在可承受成本下，让真实多轮代码代理的价值学习更有效。**
