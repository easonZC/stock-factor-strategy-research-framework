# FactorLab v2: Stock Factor + Strategy Research Framework

## English Overview
FactorLab v2 is a GitHub-safe, reusable quant research framework focused on report-grade factor research. The architecture keeps CLI scripts thin and places reusable logic in `core/factorlab`, with dataclass-based configuration, synthetic end-to-end reproducibility, and no proprietary local paths. The framework supports both cross-sectional (CS) and time-series (TS) factor scopes, runs a leakage-aware preprocessing and evaluation pipeline, and produces auditable HTML reports with figures and CSV tables. In addition to research, FactorLab v2 includes lightweight strategy backtesting, model-factor training hooks, and CI-ready tests so the project can be used as a real engineering baseline instead of a one-off notebook bundle.

## 中文说明
FactorLab v2 是一个面向实盘前研究流程的、可复用且 GitHub 安全的量化研究框架，核心目标是把因子研究提升到“可交付报告级别”。项目坚持“脚本薄、库层厚”的工程原则：命令行入口放在 `apps/`，可复用逻辑统一放在 `core/factorlab/`；配置以 dataclass 与 YAML 为核心，不依赖本地私有路径，不提交原始数据，支持在干净机器上通过 synthetic 数据完整跑通。框架提供 CS/TS 双因子研究路径、泄露防护的数据处理、可视化与统计检验、可选回测与模型因子扩展，并配套测试与 CI，使其适合作为团队化、长期维护的研究基建。

## Core Capabilities
- Factor interfaces: `Factor` base + built-in factors (`momentum_20`, `volatility_20`, `liquidity_shock`, `size`)
- Strategy interfaces: `Strategy` base + `TopKLongStrategy`, `LongShortQuantileStrategy`, `FlexibleLongShortStrategy`
- Backtesting: turnover-aware costs, equity curve, summary metrics
- Factor research (primary):
  - multi-horizon forward returns (`1,5,10,20`, configurable)
  - daily IC + RankIC, rolling IC mean, ICIR
  - IC decay across horizons
  - Newey-West t-stat + p-value (IC / RankIC / long-short)
  - quantile returns, quantile NAV, long-short, quantile turnover
  - diagnostics: coverage, missing rate, outlier before/after, stability (lag1/lag5 + drift)
  - neutralization A/B: raw vs neutralized
  - multi-factor Pearson/Spearman correlation + heatmap
- Config-driven one-click workflow:
  - explicit scope split: `factor_scope=cs|ts`
  - explicit eval axis: `eval_axis=cross_section|time`
  - explicit standardization selection

## Project Layout
```text
core/factorlab/
  backtest/      # Backtest engine and result objects
  config.py      # Dataclass configs shared across modules
  data/          # IO, synthetic data, Sina adapter, universe filters
  factors/       # Factor base + built-in factors + model factor
  models/        # Model registry + OOF training helpers
  plotting/      # Unified chart style and plotting functions
  preprocess/    # Winsorize/standardize/missing/neutralize transforms
  research/      # CS/TS research pipelines, statistics, report renderer
  strategies/    # Strategy base + implementations
  workflows/     # Config runner and benchmark workflows
apps/         # Thin CLI entrypoints
configs/         # Config templates (TS/CS demos)
examples/legacy/ # Quarantined legacy/internship reference code only
docs/            # Processing and methodology notes
tests/           # Unit and smoke tests
.github/workflows/ci.yml
```

## Installation
```bash
python3 -m pip install -r requirements.txt
```

If your environment exposes `python` directly, you can use `python` instead of `python3` in all commands below.

## Required Commands

### A) Synthetic report (must run end-to-end)
```bash
python apps/run_factor_research_synthetic.py --out outputs/research/factor/synthetic_report
```

### B) Panel input
```bash
python apps/run_factor_research.py \
  --panel <path> \
  --factors momentum_20,volatility_20,liquidity_shock \
  --horizons 1 5 10 20 \
  --out outputs/research/factor/panel_report
```

### C) Sina folder adapter
```bash
python apps/prepare_data.py --adapter sina --data-dir /stock_sina_update --out data/panel.parquet
python apps/run_factor_research.py --panel data/panel.parquet --out outputs/research/factor/sina_report
```

## Config-Driven One-Click Runs (TS/CS)

### CS example
```bash
python apps/run_from_config.py --config configs/cs_factor_demo.yaml --out outputs/research/factor/config_cs
```

### TS example
```bash
python apps/run_from_config.py --config configs/ts_factor_demo.yaml --out outputs/research/factor/config_ts
```

### Required config keys
- `run.factor_scope`: `cs` or `ts`
- `run.eval_axis`: `cross_section` or `time`
- `run.standardization`: CS (`cs_zscore|cs_rank|none`), TS (`ts_rolling_zscore|zscore|none`)

## Typical Report Output Tree
```text
outputs/research/factor/panel_report/
  index.html
  config.json
  assets/
    factor_corr_spearman.png
    outlier_before_after.png
    factors/
      <factor_name>/
        raw/
          ic.png
          ic_decay.png
          quantile_nav.png
          turnover.png
          coverage.png
          stability.png
        neutralized/
          ...
      <factor_name>/ts/
        ...
  tables/
    summary.csv
    missing_rates.csv
    factor_corr_spearman.csv
    factor_corr_pearson.csv
    factors/
      <factor_name>/
        raw/
          ic_daily_h1.csv
          ic_daily_h5.csv
          ic_decay.csv
          quantile_daily.csv
          quantile_nav.csv
          turnover.csv
          coverage.csv
          stability.csv
          ...
        neutralized/
          ...
```

## Data Safety Policy
- No raw data committed.
- No hard-coded local paths or tokens.
- Use CLI args, env vars, and config templates.
- Legacy/internship code is isolated under `examples/legacy/`.

## Testing and CI
```bash
ruff check core apps tests
pytest -q
```

CI (`.github/workflows/ci.yml`) runs lint + pytest + synthetic smoke artifact checks.
