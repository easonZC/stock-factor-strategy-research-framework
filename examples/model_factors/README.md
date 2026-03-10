# 神经网络模型因子示例

本目录给出当前仓库推荐的第一类神经网络因子：`mlp` 截面模型因子。

## 推荐起点

- 因子类型：`cross_section`
- 模型：`sklearn.neural_network.MLPRegressor`
- 目标：基于 `date x asset` 面板预测未来 `label_horizon` 收益对应的排序分数
- 评估：OOF `RankIC`、分层收益与研究报告

先做截面而不是时序，原因很简单：

- 当前栈的模型因子工作流天然就是面板 + OOF + 截面评估
- 截面 MLP 已经能直接复用现有研究、报告、实验登记与模型落盘能力
- 真正严肃的时序神经网络通常要引入滞后窗口、序列样本构造，甚至单独的 sequence 模型，不适合塞进第一版基线

## 数据集选择

推荐分三层理解：

- `engineering_demo`：内置高信噪比 benchmark panel（推荐 `24 assets x 220 days`，约 `5,280` 行）
- `research_realistic`：内置更接近真实量级的 benchmark panel（推荐 `40 assets x 260 days`，约 `10,400` 行）
- `local_real_data`：你自己的本地 `parquet/csv` 面板，至少包含 `date, asset, close`

若要让 MLP 因子更像真实研究数据，建议面板尽量补齐：

- 价格列：`open, high, low, close`
- 成交列：`volume`
- 横截面辅助列：`industry`, `mkt_cap`
- 手工特征列：`momentum_20, volatility_20, liquidity_shock, size`

## Canonical 命令

### 1. 生成 `engineering_demo` 数据集

```bash
python3 - <<'PY'
from factorlab.config import SyntheticConfig
from factorlab.data import generate_model_factor_benchmark_panel

panel = generate_model_factor_benchmark_panel(
    SyntheticConfig(n_assets=24, n_days=220, seed=11, start_date="2021-01-01"),
    tier="engineering_demo",
)
panel.to_parquet("data/mlp_benchmark_panel.parquet", index=False)
print(panel.shape)
PY
```

这个 benchmark 是刻意为截面模型因子设计的：

- 数据量不大，适合快速回归与本地开发
- 信号是非线性的，`mlp` 比线性模型更有发挥空间
- 用途是证明 NN 因子链路真的打通，而不是模拟真实市场 IC

### 2. 生成 `research_realistic` 数据集

```bash
python3 - <<'PY'
from factorlab.config import SyntheticConfig
from factorlab.data import generate_model_factor_benchmark_panel

panel = generate_model_factor_benchmark_panel(
    SyntheticConfig(n_assets=40, n_days=260, seed=11, start_date="2021-01-01"),
    tier="research_realistic",
)
panel.to_parquet("data/mlp_realistic_panel.parquet", index=False)
print(panel.shape)
PY
```

这个 tier 仍然是 synthetic，但目标更严肃：

- `rank_ic_mean` 处在更像真实研究的量级，而不是 0.6 这种 demo 级结果
- 更适合做回归门槛、论文风格 sanity check 与参数迭代
- 不要求 `mlp` 每次都大幅碾压线性模型，而是要求信号为正且量级合理

### 3. 截面神经网络因子基准

```bash
python -m factorlab run-model-benchmark \
  --panel data/mlp_realistic_panel.parquet \
  --models ridge,mlp \
  --feature-cols momentum_20,volatility_20,liquidity_shock,size \
  --label-horizon 5 \
  --evaluation-axis cross_section \
  --name mlp_cs_factor_v1
```

输出的核心 OOF 因子列为：

- `model_factor_oof_mlp`

### 4. 快速训练并保存一个 MLP 因子模型

```bash
python -m factorlab train-model-factor \
  --model mlp \
  --out artifacts/models/mlp_factor.joblib
```

训练完成后，可通过 `ModelFactor` 以模型因子形式复用该 artifact。

### 5. 第三层：本地真实数据模板

如果你已经有自己的 panel，请直接看：

- `examples/model_factors/local_real_data/README.md`

其中包含：

- 推荐命令
- 指标预期
- 输出位置
- 一个小型 `mlp` 参数网格模板

## 后续再做什么

如果下一步要扩展到 temporal factor，建议单独开一条更严格的路线：

- 明确滞后窗口和样本构造方式
- 把评估轴切到 `time`
- 再决定继续用 lagged-MLP，还是升级到 RNN / TCN / Transformer 一类序列模型
