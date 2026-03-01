# SSF v2: GitHub-Safe Stock Factor + Strategy Research Framework

## English Overview
SSF v2 is a reusable quant research framework designed for clean-machine reproducibility and GitHub safety. The codebase separates thin scripts from reusable package logic under `src/ssf`, removes proprietary local paths, avoids committing data artifacts, and provides a fully synthetic end-to-end workflow. The primary goal is report-grade factor research: multi-horizon forward returns, IC/RankIC and ICIR, IC decay, Newey–West significance testing, quantile portfolio analytics, turnover diagnostics, coverage/missing/outlier monitoring, factor stability analysis, neutralization A/B comparison, and multi-factor correlation heatmaps.

## 中文说明
SSF v2 是一个可复用、可迁移、可在干净环境直接运行的量化研究框架。核心目标是把“因子研究”升级到可交付报告级别：支持多周期收益预测、IC/RankIC 与 ICIR、IC 衰减、Newey–West 显著性检验、分位数组合收益与多空净值、换手分析、覆盖率/缺失率/异常值监控、稳定性分析、raw vs neutralized A/B 对比，以及多因子相关性热力图。脚本层仅做参数入口，核心逻辑全部放入 `src/ssf`，便于二次开发和团队协作。

## Layout
```text
src/ssf/
  config.py
  data/
  workflows/
  preprocess/
  factors/
  strategies/
  backtest/
  research/
  plotting/
  models/
scripts/
factor_library/
model_library/
examples/legacy/
docs/
tests/
.github/workflows/ci.yml
```

## Quickstart
```bash
pip install -r requirements.txt
```

### A) Synthetic factor research run
```bash
python scripts/run_synthetic_factor_research.py --out outputs/factor_research_synthetic --run-strategy-check
```

### B) Panel input
```bash
python scripts/run_factor_research.py \
  --panel <path> \
  --factors momentum_20,volatility_20,size \
  --horizons 1 5 10 20 \
  --apply-universe-filter \
  --min-history-days 20 \
  --out outputs/factor_report
```

### C) Sina folder adapter
```bash
python scripts/prepare_data.py --adapter sina --data-dir /stock_sina_update --out data/panel.parquet
python scripts/run_factor_research.py --panel data/panel.parquet --out outputs/factor_report_sina
```

### D) Synthetic strategy run (factor + walk-forward)
```bash
python scripts/run_synthetic_strategy_backtest.py \
  --mode both \
  --strategy flex \
  --model lgbm \
  --enable-tradability-constraints \
  --max-participation-rate 0.1 \
  --benchmark-mode cross_sectional_mean \
  --out outputs/strategy_backtest_synthetic
```

### E) OOF model-factor research (neural network by default)
```bash
python scripts/run_model_factor_research.py \
  --panel data/panel.parquet \
  --model mlp \
  --feature-cols momentum_20,volatility_20,liquidity_shock \
  --factor-name model_factor_oof_mlp \
  --out outputs/model_factor_research
```

### F) Model-factor benchmark (ML vs DL side-by-side)
```bash
python scripts/run_model_factor_benchmark.py \
  --panel data/panel.parquet \
  --models lgbm,mlp \
  --feature-cols momentum_20,volatility_20,liquidity_shock,size \
  --start-date 2018-01-01 \
  --max-assets 300 \
  --out outputs/model_factor_benchmark
```

This workflow now writes both `run_meta.json` and `run_manifest.json` for reproducibility.

### G) Config-driven one-click run (TS / CS split)
```bash
python scripts/run_from_config.py --config configs/cs_factor_demo.yaml --out outputs/cs_factor_demo
python scripts/run_from_config.py --config configs/ts_factor_demo.yaml --out outputs/ts_factor_demo
```

Config highlights:
- `run.factor_scope`: `ts` or `cs`
- `run.eval_axis`: `time` or `cross_section`
- `run.standardization`: TS (`ts_rolling_zscore|zscore|none`), CS (`cs_zscore|cs_rank|none`)
- `data.mode`: `single_asset|panel`
- `data.adapter`: `synthetic|sina|parquet|csv`
- `data.fields_required`: explicit input dependency declaration

## Legacy quarantine
Legacy internship-specific code/data references are moved under `examples/legacy/` for reference only.
Legacy hand-written factor scripts are additionally indexed under `factor_library/manual_legacy_cn/`.

## Safety rules
- No raw data committed.
- No hard-coded local paths/tokens.
- Use CLI args + env vars + dataclass configs.

## Notes
See `docs/data_processing.md` for leakage-safe preprocessing rules and `docs/dl_feature_norm.md` for LayerNorm/BatchNorm concepts.
