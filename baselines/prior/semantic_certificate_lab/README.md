# Semantic Certificate Lab v0.1

这是一次实际执行的理论与代码审计：有限程序任务误差、成对编辑效果区间、反例、独立重放和成本对照。

**当前结论：** 界在所定义有限世界中可靠；保留共同状态的成对证书明显比独立误差区间相减更有方向辨识力；但上界下降不是正确的真实进步标签，而且本例中精确后缀缓存比抽象证书更快。没有训练 RL/LLM，没有宣称真实软件上的性能提升。

## 阅读顺序

- `REPORT.md`：实际结果、分母、反例、成本与局限。
- `THEORY.md`：完整定义、证明、与已有理论的关系。
- `experiment.py`：数据生成、分析器、证书、全部断言与原始数据导出。
- `independent_check.py`：不依赖 NumPy、不导入分析器的独立标量解释器重放。
- `cost_check.py`：含初始化成本的精确缓存/抽象证书成本对照。

## 复现

Python 3.10+；主实验只依赖 NumPy，独立检查器仅依赖标准库。不要以 `python -O` 运行，因为实验使用断言做 fail-fast 审计。

```bash
python -m pip install numpy
python experiment.py --out results
python independent_check.py results
python cost_check.py results --repeats 7
```

默认：240 个程序、4,800 次候选编辑、5 个分析精度，共 24,000 个编辑—精度配置。随机种子 `20260920`。修改 `--programs-per-cell`、`--edits-per-program` 或 `--seed` 可扩展压力测试。程序生成器不依赖 GPU 或网络。

## 数值口径

每个程序输入是 0..63 的全部 64 个数，概率均匀。任务要求最终输出的高两位等于输入高两位，低四位不受约束。程序深度 4/8/16；有局部混合与强混合两个 family。所用模块是有限整数字函数，可以包含带守卫赋值；不是自然 GitHub 仓库。

`edits.csv` 中所有 `_num` 分数的分母为 64；`tests_delta_num_out_of_8` 的分母为 8。方向 +1 表示真实任务违反率下降；-1 表示上升；证书方向 0 表示弃权，不等于真实作用为 0。

`block=1` 是精确状态划分；不能把该行结果当成便宜近似验证器的证据。所有区间包含检查及残差恒等式检查均用整数分子，避免浮点 tolerance 掩盖错误。

## 保留的结果

`results/summary.json` 有配置、所有聚合分母和检查计数；`edits.csv` / `residuals.csv` 是原始记录；`fixtures.json` 含全部被测试程序和候选编辑；`counterexamples.json` 含反例；`independent_audit.json` 与 `cost_check.json` 是独立检查和成本对照。

该实验是机制开发数据，不是预注册的独立确认集。统计仅描述该生成器及该随机种子下的有限语料。
