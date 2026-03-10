# 本地真实数据模型因子模板

这是第三层 benchmark：**local real data**。

目标不是再造一个 synthetic demo，而是把你自己的本地股票面板直接接进 `mlp` / `ridge` 模型因子基准流程。

## 适用场景

- 你已经有自己的 `parquet/csv` 股票面板
- 你想比较线性模型与 `mlp` 的截面选股能力
- 你想把 synthetic 的“工程验证”升级到更像真实研究的本地实验

## 数据约定

至少包含：

- `date`
- `asset`
- `close`

强烈建议同时包含：

- `open`, `high`, `low`
- `volume`
- `mkt_cap`
- `industry`

如果面板里还没有模型特征，当前 benchmark 工作流会自动补内置可计算特征：

- `momentum_20`
- `volatility_20`
- `liquidity_shock`
- `size`

## 推荐命令

```bash
python -m factorlab run-model-benchmark \
  --panel data/your_panel.parquet \
  --models ridge,mlp \
  --feature-cols momentum_20,volatility_20,liquidity_shock,size \
  --label-horizon 5 \
  --train-days 252 \
  --valid-days 21 \
  --step-days 21 \
  --split-mode rolling \
  --min-train-rows 2000 \
  --min-valid-rows 400 \
  --evaluation-axis cross_section \
  --neutralize both \
  --winsorize quantile \
  --model-param-grid-dir examples/model_factors/local_real_data/model_param_grids \
  --save-model-artifacts \
  --name mlp_local_real_data
```

## 指标预期

对真实横截面股票因子，不要期待 `RankIC mean = 0.6` 这种数值。

更合理的经验区间是：

- `rank_ic_mean` 约 `0.02 ~ 0.05`：已经不错
- `rank_ic_mean` 约 `0.05 ~ 0.10`：很强，值得重点复核
- `ICIR` 约 `0.5 ~ 1.0`：比较强
- 明显高于 `0.10` 的长期稳定 RankIC：通常需要特别警惕泄漏、过拟合或标签构造问题

## 因子在哪里

运行结束后，重点看这几个文件：

- 因子名：`model_factor_oof_mlp`
- OOF 预测：`<out_dir>/model_mlp/oof_predictions.csv`
- 模型对比：`<out_dir>/model_factor_comparison.csv`
- 研究汇总：`<out_dir>/tables/summary.csv`
- HTML 报告：`<out_dir>/index.html`
- 运行元数据：`<out_dir>/run_meta.json`

如果加了 `--save-model-artifacts`，可复用的模型 artifact 默认会落在：

- `artifacts/models/model_factor_benchmark/model_factor_oof_mlp.joblib`

后续你可以把这个 artifact 接到 `ModelFactor` 中继续使用。

## 参数网格

本目录附带了一个小型 `mlp` 参数网格：

- `examples/model_factors/local_real_data/model_param_grids/mlp.json`

刻意保持很小，避免把本地研究搞成低效的超参搜索工程。
