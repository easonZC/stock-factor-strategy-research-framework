# FactorLab v2: Stock Factor + Strategy Research Framework

## English Overview
FactorLab v2 is a GitHub-safe, reusable quant research framework focused on report-grade factor research. The architecture keeps CLI scripts thin and places reusable logic in `core/factorlab`, with dataclass-based configuration, synthetic end-to-end reproducibility, and no proprietary local paths. The framework supports both cross-sectional (CS) and time-series (TS) factor scopes, runs a leakage-aware preprocessing and evaluation pipeline, and produces auditable HTML reports with figures and CSV tables. In addition to research, FactorLab v2 includes lightweight strategy backtesting, model-factor training hooks, and CI-ready tests so the project can be used as a real engineering baseline instead of a one-off notebook bundle.

## 中文说明
FactorLab v2 是一个面向实盘前研究流程的、可复用且 GitHub 安全的量化研究框架，核心目标是把因子研究提升到“可交付报告级别”。项目坚持“脚本薄、库层厚”的工程原则：命令行入口放在 `apps/`，可复用逻辑统一放在 `core/factorlab/`；配置以 dataclass 与 YAML 为核心，不依赖本地私有路径，不提交原始数据，支持在干净机器上通过 synthetic 数据完整跑通。框架提供 CS/TS 双因子研究路径、泄露防护的数据处理、可视化与统计检验、可选回测与模型因子扩展，并配套测试与 CI，使其适合作为团队化、长期维护的研究基建。

## Core Capabilities
- Factor interfaces: `Factor` base + built-in factors (`momentum_20`, `volatility_20`, `liquidity_shock`, `size`)
- Strategy interfaces: `Strategy` base + `TopKLongStrategy`, `LongShortQuantileStrategy`, `FlexibleLongShortStrategy`, `MeanVarianceOptimizerStrategy`
- Backtesting: turnover-aware costs, equity curve, summary metrics, configurable risk constraints (`max_turnover`, `max_abs_weight`, `max_gross_exposure`, `max_net_exposure`), optional industry neutrality, and benchmark alpha/beta
- Factor research (primary):
  - multi-horizon forward returns (`1,5,10,20`, configurable)
  - daily IC + RankIC, rolling IC mean, ICIR
  - IC decay across horizons
  - Newey-West t-stat + p-value (IC / RankIC / long-short)
  - quantile returns, quantile NAV, long-short, quantile turnover
  - diagnostics: coverage, missing rate, outlier before/after, stability (lag1/lag5 + drift)
  - neutralization A/B: raw vs neutralized
  - cross-sectional regression diagnostics: Fama-MacBeth beta/t-stat/p-value
  - group decomposition: industry/style long-short attribution
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
  ops/           # Retention/cleanup operations utilities
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

### D) Public adapter (Stooq, no proprietary data)
```bash
python apps/prepare_data.py --adapter stooq --symbols aapl,msft,googl,amzn --start-date 2022-01-01 --out data/panel.parquet
python apps/run_factor_research.py --panel data/panel.parquet --out outputs/research/factor/stooq_report
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

### CS public-data example (Stooq)
```bash
python apps/run_from_config.py --config configs/cs_stooq_demo.yaml --out outputs/research/factor/config_stooq
```

### Multi-config merge + temporary overrides
```bash
python apps/run_from_config.py \
  --config configs/cs_factor_demo.yaml \
  --config configs/local_override.yaml \
  --set research.horizons='[1,5,10]' \
  --set research.quantiles=10 \
  --out outputs/research/factor/config_cs_merged \
  --save-effective-config
```

### Model-factor benchmark (OOF with split/scoring controls)
```bash
python apps/run_model_factor_benchmark.py \
  --panel data/panel.parquet \
  --models ridge,rf,lgbm \
  --label-horizon 5 \
  --split-mode expanding \
  --purge-days 2 \
  --scoring-metric rank_ic \
  --evaluation-axis cross_section \
  --out outputs/research/model_factor/benchmark
```

### Generate config template (recommended)
```bash
python apps/generate_run_config.py --scope cs --adapter synthetic --out configs/generated_cs.yaml
python apps/generate_run_config.py --scope ts --adapter parquet --set data.path=data/panel.parquet --out configs/generated_ts.yaml
```

### Output retention cleanup (recommended)
```bash
python apps/cleanup_outputs.py --root outputs/research --older-than-days 14 --keep-latest 20
python apps/cleanup_outputs.py --root outputs/research --older-than-days 14 --keep-latest 20 --dry-run
```

`apps/run_from_config.py` also supports post-run cleanup:
```bash
python apps/run_from_config.py \
  --config configs/cs_factor_demo.yaml \
  --out outputs/research/factor/config_cs \
  --cleanup-old-outputs \
  --cleanup-days 14 \
  --cleanup-keep 30
```

### Required config keys
- `run.factor_scope`: `cs` or `ts`
- `run.eval_axis`: `cross_section` or `time`
- `run.standardization`: CS (`cs_zscore|cs_rank|cs_robust_zscore|none`), TS (`ts_rolling_zscore|zscore|none`)

### Flexible research knobs (recommended for exploratory research)
- `factor.on_missing`: `raise` (strict) or `warn_skip` (skip unresolved factors with warnings)
- `factor.auto_discover` + `factor.plugin_dirs`: auto-discover custom `Factor` classes from plugin folders
- `factor.plugins`: load plugin modules/class paths explicitly
- `factor.plugin_on_error`: `raise` or `warn_skip` for plugin load conflicts/errors
- `research.missing_policy`: `drop|fill_zero|ffill_by_asset|cs_median_by_date|keep`
- `research.preprocess_steps`: ordered list from `winsorize|standardize|neutralize`
- `run.config_mode`: `strict|warn|compat` (strict blocks auto-corrections; warn records and continues)
- `run.fail_on_autocorrect`: fail the run whenever normalization changed user-provided values
- `run.leakage_guard_mode`: `strict|warn|off` to guard forbidden future/label references
- `backtest.strategy.mode`: built-in `sign|topk|longshort|flex|meanvar` or plugin-defined mode
- `backtest` risk controls: `max_turnover|max_abs_weight|max_gross_exposure|max_net_exposure`
- `backtest.benchmark_mode`: `none|cross_sectional_mean|panel_column`
- `data.adapter`: supports `synthetic|parquet|csv|sina|stooq`
- data-adapter plugins:
  - `data.adapter`: custom adapter name
  - `data.adapter_auto_discover` + `data.adapter_plugin_dirs` for folder discovery
  - `data.adapter_plugins` for explicit module/class-path loading
  - `data.adapter_plugin_on_error`: `raise|warn_skip`
- CLI `run_from_config` supports repeated `--config` deep-merge and repeated `--set key.path=value` overrides.
- CLI `run_from_config` pre-validates config schema by default; use `--skip-schema-validation` to bypass.
- `run_meta.json` includes `config_governance` and `leakage_guard` sections for run auditability.
- CLI logging: major entrypoints support `--log-level` and `--log-file`; env `FACTORLAB_LOG_LEVEL` is also supported.
  - run metadata includes `warning_summary` for benign/actionable warning audit.
- CLI output retention:
  - `--cleanup-old-outputs` + `--cleanup-root|--cleanup-days|--cleanup-keep|--cleanup-dry-run`

### Plugin Factor Example
```yaml
factor:
  names: [simple_reversal]
  on_missing: raise
  auto_discover: true
  plugin_dirs:
    - plugins/factors
  plugin_on_error: raise
```

Your plugin file can define a `Factor` subclass (or `get_factor_registry` / `FACTOR_REGISTRY`) and it will be discoverable.

### Factor Expression Composer Example
```yaml
factor:
  names: [mom_minus_vol]
  expressions:
    mom_minus_vol: "momentum_20 - volatility_20"
  expression_on_error: raise
```

Expressions are evaluated with a safe parser (no arbitrary code execution). Supported operations:
- arithmetic: `+ - * / **`
- unary: `+x`, `-x`
- functions: `abs(x)`, `log1p(x)`, `exp(x)`, `sqrt(x)`, `clip(x, lo, hi)`

### Multi-factor Combination Example
```yaml
factor:
  names: [combo_mom_vol]
  combinations:
    - name: combo_mom_vol
      weights:
        momentum_20: 1.0
        volatility_20: -0.5
      standardization: cs_zscore
      orthogonalize_to: [size]
  combination_on_error: raise
```

This lets you define reusable weighted composite factors without hard-coding research scripts.

### Strategy Plugin Example
```yaml
backtest:
  enabled: true
  strategy:
    mode: turnover_guarded_ls
    auto_discover: true
    plugin_dirs:
      - plugins/strategies
    plugin_on_error: raise
```

Custom strategy mode is allowed when strategy plugins are configured (`plugin_dirs` or `plugins`).

### Data Adapter Plugin Example
```yaml
data:
  adapter: my_adapter
  adapter_auto_discover: true
  adapter_plugin_dirs:
    - plugins/data_adapters
  adapter_plugin_on_error: raise
```

Plugin module can expose:
- `DATA_ADAPTER_REGISTRY` dict
- `get_data_adapter_registry()`
- or `prepare_<name>_panel(config)` function
- `DATA_ADAPTER_VALIDATORS` dict
- `get_data_adapter_validators()`
- or `validate_<name>_config(config)` function

Reference templates:
- `examples/plugins/data_adapters/README.md`
- `examples/plugins/data_adapters/mock_feed.py`

`run_meta.json` now includes structured adapter load profile:
- `data.load_report.adapter_load_seconds`
- `data.load_report.panel_profile` (rows/assets/date-range/columns/source)
- `data.adapter_validation_report` (pre-load config validation hook report)
- `data.adapter_audit_tables` (quality audit CSV pointers)

### Custom Transform Plugin Example
```yaml
research:
  transform_auto_discover: true
  transform_plugin_dirs:
    - examples/plugins/transforms
  transform_plugin_on_error: raise
  custom_transforms:
    - name: robust_clip
      kwargs:
        lower_q: 0.02
        upper_q: 0.98
    - name: signed_log
      on_error: warn_skip
```

Transform plugin module can expose:
- `TRANSFORM_REGISTRY` dict
- `get_transform_registry()`
- or functions `transform_<name>(panel, factor_col, **kwargs)`

Reference templates:
- `examples/plugins/transforms/README.md`
- `examples/plugins/transforms/custom_transforms.py`

`run_meta.json` includes transform execution details:
- `research.custom_transform_report`
- `research.transform_plugin_config`

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
    data/
      adapter_quality_audit.csv
      field_missing_rates.csv
      asset_row_counts.csv
      date_coverage.csv
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
          fama_macbeth_daily.csv
          fama_macbeth_summary.csv
          industry_decomposition_daily.csv
          industry_decomposition_summary.csv
          style_decomposition_daily.csv
          style_decomposition_summary.csv
          ...
        neutralized/
          ...
```

## Data Safety Policy
- No raw data committed.
- No hard-coded local paths or tokens.
- Use CLI args, env vars, and config templates.
- Legacy/internship code is isolated under `examples/legacy/`.

## Data Directory Convention
- `data/`: 本地中间数据目录（推荐存放清洗后的 panel，如 `data/panel.parquet`）。
- `stock_data/`: 你现在的原始 A 股/北交所 CSV 数据目录（仓库内可用，但建议迁移到 `data/raw/stock_data/` 统一管理）。
- 由于 `.gitignore` 已忽略 `/data/` 与 `*.csv`，把 `stock_data` 放进 `data/` 是安全且推荐的。

## Testing and CI
```bash
ruff check core apps tests
pytest -q
```

CI (`.github/workflows/ci.yml`) runs lint + pytest + synthetic smoke artifact checks.
