# Certificate-compatible RL adapter: theory and exact audit

- `THEORY.md`: 中文推导、反例、假设与结论边界。
- `check_adapter.py`: 可复现的有限树穷举和梯度审计。
- `results.json`: 默认种子下的原始结果。

## 运行

```bash
python -m pip install numpy
python check_adapter.py --trials 500 --seed 20260921
```

Python 3.10+；仅依赖 NumPy。无需 GPU。该脚本未训练 RL 模型；所有策略梯度期望通过完整路径枚举得到。浮点数容差检查不替代数学证明。

这是一份新建的适配审计，不是早先 semantic_certificate_lab 的续跑，未导入其代码或数据。
