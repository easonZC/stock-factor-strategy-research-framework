# TS/CS 因子与 ML/NN 示例（可复现）

本文给出一组可直接执行的示例，用于：

- 设计并验证自定义因子（TS + CS）
- 生成研究报告（HTML + 图表 + 表格）
- 探索机器学习与神经网络模型因子（ridge / rf / mlp）

## 0. 依赖安装

```bash
python3 -m pip install -r requirements.txt
```

## 1. CS 因子研究（自定义因子插件）

```bash
python apps/run_from_config.py \
  --config configs/cs_factor.yaml \
  --set data.adapter=synthetic \
  --set data.synthetic.n_assets=80 \
  --set data.synthetic.n_days=320 \
  --set data.synthetic.seed=2026 \
  --set factor.auto_discover=true \
  --set factor.plugin_dirs='[examples/plugins/factors]' \
  --set factor.plugin_on_error=raise \
  --set factor.names='[trend_breakout_30,volume_price_pressure_15,range_reversal_5,volatility_regime_shift_10_40]' \
  --set run.stop_after=research \
  --set backtest.enabled=false \
  --out outputs/research/factor/examples/cs_custom_factors_20260302
```

## 2. TS 因子研究（同一组因子做时序评估）

```bash
python apps/run_from_config.py \
  --config configs/ts_factor.yaml \
  --set data.adapter=synthetic \
  --set data.synthetic.n_assets=20 \
  --set data.synthetic.n_days=380 \
  --set data.synthetic.seed=2026 \
  --set factor.auto_discover=true \
  --set factor.plugin_dirs='[examples/plugins/factors]' \
  --set factor.plugin_on_error=raise \
  --set factor.names='[trend_breakout_30,range_reversal_5,volatility_regime_shift_10_40]' \
  --set run.stop_after=research \
  --set backtest.enabled=false \
  --out outputs/research/factor/examples/ts_custom_factors_20260302
```

## 3. 生成模型基准输入面板（含自定义因子）

```bash
python3 - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, "core")

from factorlab.config import SyntheticConfig
from factorlab.data import generate_synthetic_panel
from factorlab.factors import apply_factors, build_factor_registry

panel = generate_synthetic_panel(SyntheticConfig(n_assets=60, n_days=420, seed=2026))
registry = build_factor_registry(plugin_dirs=["examples/plugins/factors"], on_plugin_error="raise")
factor_names = [
    "trend_breakout_30",
    "volume_price_pressure_15",
    "range_reversal_5",
    "volatility_regime_shift_10_40",
    "momentum_20",
    "volatility_20",
    "liquidity_shock",
    "size",
]
panel = apply_factors(panel=panel, factor_names=factor_names, registry=registry)
out = Path("outputs/research/model_factor/examples/synthetic_panel_with_factors.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
panel.to_parquet(out, index=False)
print(out)
PY
```

## 4. ML/NN 模型因子基准（ridge + rf + mlp）

```bash
python apps/run_model_factor_benchmark.py \
  --panel outputs/research/model_factor/examples/synthetic_panel_with_factors.parquet \
  --models ridge,rf,mlp \
  --feature-cols trend_breakout_30,volume_price_pressure_15,range_reversal_5,volatility_regime_shift_10_40,momentum_20,volatility_20 \
  --label-horizon 5 \
  --train-days 180 \
  --valid-days 20 \
  --step-days 20 \
  --horizons 1 5 10 20 \
  --neutralize both \
  --winsorize quantile \
  --out outputs/research/model_factor/examples/ml_nn_benchmark_20260302
```

## 5. 关键产物位置

- CS 报告：`outputs/research/factor/examples/cs_custom_factors_20260302/index.html`
- TS 报告：`outputs/research/factor/examples/ts_custom_factors_20260302/index.html`
- 模型基准：`outputs/research/model_factor/examples/ml_nn_benchmark_20260302/model_factor_comparison.csv`

## 6. `adapter` 与 `synthetic` 是什么

- `adapter`：数据加载方式。日常本地研究常用 `data.path` 自动推断，不必手写适配器。
- `synthetic`：可复现合成数据源，适合无真实数据时做功能验证、CI 冒烟、方法对比。

