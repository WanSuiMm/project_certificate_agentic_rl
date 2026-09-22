# 程序方向证书接入 RL：势函数、后续价值与梯度审计

## 结论范围

本文给出一个数学上自洽的接入方式：

**成对语义证书 → 单值语义势函数 → 原任务回报不变的奖励变换 → 后续价值修正。**

它不证明证书必然改善探索、价值估计或训练效率。核心恒等式属于既有 potential-based reward shaping / actor-critic 理论；潜在研究贡献应来自证书如何构造可学习的价值分解，以及在真实预算下是否降低价值估计误差。

`check_adapter.py` 是新写的有限决策树代数审计，未使用或重跑早先程序语义实验，也没有训练 RL agent。

## 1. 设定与两个不同的时间尺度

令 h_t 为动作前的完整历史（可以包括程序、工具输出、已获得的信息、剩余预算）。它给出有限时域决策过程的 Markov 化描述。a_t 产生 h_{t+1} 和环境奖励 r_t；t=0,...,T-1。所有终止（包括失败、超时）均以终端状态处理。原始目标为

\[
J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t r_t\right],\quad 0<\gamma\le1.
\]

例如只有最后一次转移给任务完成奖励。规范 S 和输入测度 mu 固定。程序 P 的语义违反损失 E(P) 属于 [0,1]。成对证书提供

\[
L_t\le E(P_t)-E(P_{t+1})\le U_t.
\]

程序执行中一次修改对输出损失的作用，与 agent 未来还要进行许多次编辑的价值，不是同一个量。动作还可能仅仅获取信息，程序本身完全不变。

RL 的相关量是

\[
Q^\pi(h,a)=\mathbb E[G_t\mid h_t=h,a_t=a],\quad
V^\pi(h)=\mathbb E[G_t\mid h_t=h],\quad
A^\pi=Q^\pi-V^\pi.
\]

语义方向证书没有一般性的能力决定 A 的符号。

## 2. 为什么只有 return-equivalence 还不够

逐轨迹满足 sum_t rtilde_t = R，确实保持该整段回报的目标。但是把任意事后分配的 rtilde 直接用作普通 reward-to-go，可能给出错误梯度。

两步反例：第一步是固定占位动作；第二步 a~Bernoulli(p)，终端 R=a。把 R 全搬到第一步，第二步奖励设零，逐轨迹总奖励不变。但是第二步的 reward-to-go 为零；由它计算的梯度为零，而真实 logit 梯度是 p(1-p)。p=1/2 时为 1/4。

问题是伪造的“过去奖励”依赖未来动作。标准 reward-to-go 去掉过去奖励时使用的因果性失效。用完整轨迹回报的 score-function estimator 仍然可以正确，但不能再声称这种任意重分配自动带来更好的信用分配。

RUDDER 确实建立了 return-equivalence 并研究适配的估计器；不能把其结论简化为“任意归因满足加总约束就能直接塞进 PPO”。

## 3. 从成对区间构造单值状态势函数

理想语义势函数 q(P)=-E(P)，从而

\[
q(P')-q(P)=E(P)-E(P').
\]

在已获得证书的图上，可以约束

\[
L_{ij}\le q(P_j)-q(P_i)\le U_{ij},\quad q(P_i)\in[-1,0].
\]

真实 -E 是这些约束的可行解，前提是规范、输入测度一致，且所有区间确实可靠。实际神经模型可用如下损失训练：

\[
\mathcal L_{\rm cert}(\phi)=\sum_{(i,j)}w_{ij}\,\operatorname{dist}\big(q_\phi(P_j)-q_\phi(P_i),[L_{ij},U_{ij}]\big)^2.
\]

其中 dist(z,[L,U])=max(L-z,0,z-U)。没有有用证书的样本可以跳过，而不是当成真实零效果。只有符号时不要擅自假设统一的正 margin。

神经网络在训练集上拟合这个损失，不等于在未见程序上获得数学证书。严格语义保证依旧来自原始分析器或显式验证过的约束。

为何需要单值势函数：独立选择每条边的区间中点，未必构成同一个函数的差分。在 E=(0.9,0.6,0.3) 的三状态环上，真实变化为 (0.3,0.3,-0.6)，和为零；符号奖励之和却为 +1。有效区间 [0.2,0.6]、[0.2,0.4]、[-0.7,-0.5] 的中点之和也为 +0.1。单值 q 的循环差分必然为零。

## 4. 能保持原任务目标的奖励形态

取固定强度 alpha>=0。对非终端历史设 Phi(h_t)=alpha*q_phi(P_t)，对所有终端历史强制 Phi(h_T)=0。定义

\[
r_t^\Phi=r_t+\gamma\Phi(h_{t+1})-\Phi(h_t).
\]

对任何轨迹和任意 t，有望远镜恒等式

\[
G_t^\Phi=\sum_{k=t}^{T-1}\gamma^{k-t}r_k^\Phi=G_t-\Phi(h_t).
\]

因此同一个初始历史下，所有策略的回报只减去同一个常数；最优策略和回报排序保持不变。条件价值满足

\[
V^{\pi,\Phi}(h)=V^\pi(h)-\Phi(h),\quad
Q^{\pi,\Phi}(h,a)=Q^\pi(h,a)-\Phi(h),\quad
A^{\pi,\Phi}=A^\pi.
\]

这些是已有 potential-based shaping 的结论，不是本文的新理论。

即使 q 并不精确，以上恒等式也成立。证书精度决定势函数是否提供有用归纳偏置，而不决定奖励变换是否保持原目标。

实施条件：q 在一次采样/更新周期冻结，actor 更新不经 q 反向传播；Phi 不能窥视未来动作/终端结果；失败和真实时间截止也需要终端补偿。若只是训练 minibatch 截断而非环境真正终止，需要正确 bootstrap，不能错误清零。

当 gamma=1 且下一状态非终端，shaping 项为 alpha*(q(P_{t+1})-q(P_t))。当 gamma<1，必须保留 gamma；不能用裸差分替代。终端补偿是结算，不应当被解释成新的语义进步。

## 5. 接入 agentic RL 的关键：后续价值修正

定义

\[
W^\pi(h)=V^\pi(h)-\Phi(h).
\]

于是

\[
\delta_t^\Phi=r_t^\Phi+\gamma W^\pi(h_{t+1})-W^\pi(h_t)
=r_t+\gamma V^\pi(h_{t+1})-V^\pi(h_t).
\]

条件期望 E[delta_t^Phi|h_t,a_t]=A^pi(h_t,a_t)。

当 gamma=1、Phi=-E、当前转移非终端时，结构为

\[
\delta_t^\Phi=r_t+\underbrace{(E(P_t)-E(P_{t+1}))}_{\text{即时语义变化}}+
\underbrace{(W^\pi(h_{t+1})-W^\pi(h_t))}_{\text{后续价值修正}}.
\]

W 不受语义方向的符号约束，也不应被强行压到零。它负责信息获取、未来可修复性、剩余预算等 E(P) 没有描述的因素。

例子：初始 E=0.8，有两种动作。重构使 E=1，但后续可以成功，V_refactor=1；表面改善使 E=0.4，但在剩余预算内无法成功，V_cosmetic=0。策略各选一半时 V_start=0.5。Phi=-E，故 W_start=1.3、W_refactor=2、W_cosmetic=0.4。第一步的 advantage 分解为

- 重构：-0.2 + 0.7 = +0.5；
- 表面改善：+0.4 - 0.9 = -0.5。

精确语义方向与正确策略更新方向可以相反。这不是证书错误，是两者回答不同问题。

## 6. 近似 critic 的偏差来自哪里

设 What=W+e，且终端 e=0。采用固定策略的 on-policy 一步 TD 估计，则

\[
\mathbb E[\widehat\delta_t^\Phi|h,a]-A^\pi(h,a)
=\gamma\mathbb E[e(h_{t+1})|h,a]-e(h).
\]

如果 ||e||_infty<=epsilon，条件 advantage 误差不超过 (1+gamma)*epsilon。令 S_t=grad_theta log pi_theta(a_t|h_t)，则 actor 梯度中当前状态项消去，得到

\[
\left\|\mathbb E\left[\sum_t\gamma^t S_t\widehat\delta_t^\Phi\right]-\nabla J\right\|
\le\gamma\epsilon\sum_t\gamma^t\mathbb E\|S_t\|.
\]

这不是可直接获得的数值保证，因为学习出来的 W 通常没有已知 uniform epsilon。它说明：语义证书即便完美，也没有自动控制 continuation critic 的误差。

Monte Carlo 形式 G_t-Phi(h_t)-What(h_t) 是动作前 baseline 形式（网络冻结、无未来信息泄漏时），不因为 baseline 不精确而引入这种 TD bootstrap 偏差，但方差仍可能很大或变差。

## 7. 严格等价性审计：不能把 dense numbers 当成额外信用

对任意近似值函数，若 Vhat=Phi+What，则逐转移恒等式成立：

\[
r_t^\Phi+\gamma\widehat W(h_{t+1})-\widehat W(h_t)
=r_t+\gamma\widehat V(h_{t+1})-\widehat V(h_t).
\]

因此在同一轨迹、同一截断处理下，所有 GAE(lambda) 数值也完全相同。只换奖励写法而不改变值函数的信息、拟合能力、初始化或训练过程，不会创造新的梯度信号。

若采用纯 terminal Monte Carlo、q/critic 都仅充当 baseline，预期梯度仍然是原任务梯度。额外语义监督的可能价值是让有限样本下的 Vhat=Phi+What 更容易学习，而不是提供已经认证的长期 advantage。

Potential-based shaping 与某些 Q 初始化方法的等价性已有文献；本结果不应宣称新颖。

## 8. 什么时候语义方向可以升级为长期方向

只考虑 gamma=1、确定性候选下一状态、当前环境奖励都为零。比较同一 h 下的两个动作 a,b，有

\[
Q^\pi(h,a)-Q^\pi(h,b)=\Phi(h_a)-\Phi(h_b)+W^\pi(h_a)-W^\pi(h_b).
\]

若语义差的证书为 [L_ab,U_ab]，并且另有可靠 continuation bound

\[
|W^\pi(h_a)-W^\pi(h_b)|\le\epsilon_{ab},
\]

则长期动作差在 [L_ab-epsilon_ab,U_ab+epsilon_ab] 内。只有 L_ab>epsilon_ab 等条件，才足以认证长期方向。

原来的 compiler / relative verifier 一般并不提供 epsilon_ab。它需要关于未来策略、剩余动作、预算的进一步分析。若用 Monte Carlo continuation 估计它，那通常是统计保证，不是原始的确定性程序证书。不能偷换二者。

## 9. 动作相关 control variate 的单独警告

对有限动作空间，固定 h，令 c(h,a) 为冻结的动作相关估计，S=grad log pi。有

\[
\mathbb E_a[S Q]=\mathbb E_a[S(Q-c)]+\nabla_\theta\sum_a\pi_\theta(a|h)c(h,a).
\]

只减去 c 而不加第二项，一般有偏。若 Q(a)=c(a)=a，a~Bernoulli(1/2)，仅相减得到零，而正确梯度为 1/4。大模型开放文本动作空间中的第二项也不便宜。它不是免费替换 reward 的捷径。

## 10. 最小实现

1. 从固定规范下的版本对收集可靠 [L,U]，训练有界的程序语义势函数 q。
2. 冻结 q，用真实 terminal reward 收集 trajectories，不硬剪掉语义负方向/未知方向。
3. 构造 Phi 和 shaped reward，训练 W 去拟合 G-Phi 或对应 TD/GAE 目标。
4. actor 只接收到由 r^Phi 和 W 正确构造的 advantage；不要把证书方向当作 advantage sign constraint。
5. 检查失败终止、超时、截断 bootstrap、old-policy ratios、certificate latency 与额外算力。

理论针对精确 on-policy gradient / objective。PPO clipping、GAE lambda<1、价值逼近、off-policy reuse 仍有各自误差；不能把目标等价宣称为整个实现每次更新都无偏。

## 11. 实验范围与结果

seed=20260921，500 个随机二叉有限时域决策树，深度 2--5，gamma 取 1、0.93、0.7。全部枚举 7,164 条轨迹，30,192 个路径上的转移事件。

- 回报望远镜恒等式最大误差：6.661338147750939e-16。
- 原始/势函数变换 Monte Carlo 期望梯度最大差：1.5265566588595902e-16。
- 精确 TD 恒等式最大差：4.440892098500626e-16。
- 近似 critic 正确变换后的 GAE 最大差：6.661338147750939e-16。
- 独立有限差分与枚举梯度最大差：5.461048280253067e-12。
- 上述 residual critic 梯度偏差界：未出现数值违规。

另验证：未来泄漏重分配、必要暂时退步、符号循环奖励、动作相关 baseline 缺少校正的反例。

这些是有限浮点计算的审计，不是训练效率或可扩展性实验证据，也不是定理的替代证明。

## 12. 下一道真实检验

同样的采样、critic 参数量和总计算预算下，比较普通 V、证书 Phi+W、无证书的辅助表示，以及直接符号 shaping。主动加入信息获取、必要暂时退步、无关语义改善、反复改回等任务。

最先看价值/advantage 误差和精确梯度方差，再看任务成功率。若只得到更多非零 reward 而没有这些改善，就尚未证明证书解决了 long-horizon credit。

## 参考文献

1. Ng, Harada, Russell. Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping. ICML 1999. 官方作者稿： https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
2. Arjona-Medina et al. RUDDER: Return Decomposition for Delayed Rewards. NeurIPS 2019. https://arxiv.org/abs/1806.07857
3. Schulman et al. High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR 2016. https://arxiv.org/abs/1506.02438
4. Wiewiora. Potential-Based Shaping and Q-Value Initialization are Equivalent. JAIR 19, 2003. https://jair.org/index.php/jair/article/view/10338
5. Liu et al. Action-dependent Control Variates for Policy Optimization via Stein Identity. ICLR 2018. https://openreview.net/forum?id=H1mCp-ZRZ
6. Tucker et al. The Mirage of Action-Dependent Baselines in Reinforcement Learning. ICML 2018. https://proceedings.mlr.press/v80/tucker18a.html
7. Greensmith, Bartlett, Baxter. Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning. JMLR 5, 2004. https://jmlr.csail.mit.edu/papers/v5/greensmith04a

核心模板来自以上理论；本文的数学推导和有限审计用于界定证书适配的正确形态，不主张一个新颖性已经确认的 RL 算法。
