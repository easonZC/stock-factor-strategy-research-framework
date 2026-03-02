# Architecture Overview

## Design objective
FactorLab v2 separates orchestration from reusable research logic:
- `apps/` only parse CLI args and call service APIs.
- `core/factorlab/` contains testable domain modules.

## Runtime flow
1. Load panel data:
- `factorlab.data.read_panel(...)` for parquet/csv
- `factorlab.data.prepare_sina_panel(...)` for Sina folder
- `factorlab.data.generate_synthetic_panel(...)` for demos/tests

2. Compute factors:
- `factorlab.factors.apply_factors(...)`
- Supports built-in and model-based factors (`ModelFactor`)

3. Preprocess:
- winsorize (quantile/MAD)
- standardize (CS rank/zscore, TS rolling zscore)
- missing handling (`drop`)
- neutralization (size/industry/both/none)

4. Research and reporting:
- CS pipeline: `factorlab.research.FactorResearchPipeline`
- TS pipeline: `factorlab.research.TimeSeriesFactorResearchPipeline`
- Outputs:
  - `index.html`
  - `assets/factors/<factor>/<variant>/*.png`
  - `tables/factors/<factor>/<variant>/*.csv`
  - global summary/diagnostics at `assets/*` and `tables/*`
  - `config.json`

5. Optional backtest:
- Strategy weights from `factorlab.strategies.*`
- Performance from `factorlab.backtest.run_backtest(...)`

## Config-driven one-click mode
`factorlab.workflows.run_from_config(...)` enforces explicit setup:
- `factor_scope`: `cs` or `ts`
- `eval_axis`: `cross_section` or `time`
- `standardization`: scope-aware choice
- optional universe filter + optional backtest
- run metadata/manifests for reproducibility
