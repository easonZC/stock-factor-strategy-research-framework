# Architecture Overview

## Design objective
SSF v2 separates orchestration from reusable research logic:
- `scripts/` only parse CLI args and call service APIs.
- `src/ssf/` contains testable domain modules.

## Runtime flow
1. Load panel data:
- `ssf.data.read_panel(...)` for parquet/csv
- `ssf.data.prepare_sina_panel(...)` for Sina folder
- `ssf.data.generate_synthetic_panel(...)` for demos/tests

2. Compute factors:
- `ssf.factors.apply_factors(...)`
- Supports built-in and model-based factors (`ModelFactor`)

3. Preprocess:
- winsorize (quantile/MAD)
- standardize (CS rank/zscore, TS rolling zscore)
- missing handling (`drop`)
- neutralization (size/industry/both/none)

4. Research and reporting:
- CS pipeline: `ssf.research.FactorResearchPipeline`
- TS pipeline: `ssf.research.TimeSeriesFactorResearchPipeline`
- Outputs:
  - `index.html`
  - `assets/*.png`
  - `tables/*.csv`
  - `config.json`

5. Optional backtest:
- Strategy weights from `ssf.strategies.*`
- Performance from `ssf.backtest.run_backtest(...)`

## Config-driven one-click mode
`ssf.workflows.run_from_config(...)` enforces explicit setup:
- `factor_scope`: `cs` or `ts`
- `eval_axis`: `cross_section` or `time`
- `standardization`: scope-aware choice
- optional universe filter + optional backtest
- run metadata/manifests for reproducibility
