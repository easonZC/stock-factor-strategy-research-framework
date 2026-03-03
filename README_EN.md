# Stock Factor Strategy Research Framework

A GitHub-safe and reusable quantitative research framework for stock factor research, report generation, and optional strategy backtesting.

## What This Project Solves
- One stable runtime entrypoint for end-to-end research: `apps/run_from_config.py`
- A reusable core library under `core/factorlab/` (factor, data, research, backtest, model workflows)
- Config-driven execution with two primary templates:
  - `configs/cs_factor.yaml` for cross-sectional research
  - `configs/ts_factor.yaml` for time-series research
- Audit-ready outputs (`run_meta.json`, `run_manifest.json`, adapter quality tables)

## Core Principles
- No hard-coded local/proprietary paths or tokens
- Data path first: set `data.path`, run directly
- Reproducible and auditable outputs by default
- Keep scripts thin, keep logic in core modules

## Installation
```bash
python3 -m pip install -r requirements.txt
```

## Quick Start
### 1) Cross-sectional run (CS)
```bash
python apps/run_from_config.py \
  --config configs/cs_factor.yaml \
  --set data.path=data/raw \
  --set factor.names='[factor_a,factor_b]'
```

### 2) Time-series run (TS)
```bash
python apps/run_from_config.py \
  --config configs/ts_factor.yaml \
  --set data.path=data/raw/000001.csv \
  --set factor.names='[factor_ts]'
```

### 3) Validate config only
```bash
python apps/lint_config.py --config configs/cs_factor.yaml --set data.path=data/raw
```

### 4) Model-factor benchmark (ML/NN)
```bash
python apps/run_model_factor_benchmark.py \
  --panel data/panel.parquet \
  --models ridge,rf,mlp \
  --name model_benchmark_v1
```

## Data Contract
`data.path` is the default data contract:
- File path (`.parquet` / `.csv`) -> single panel read
- Directory path -> auto-merge `*.parquet,*.csv`

`data.adapter=synthetic` exists for smoke tests, CI, and reproducible experiments when real data is unavailable.

## Output Interpretation (First 30 Seconds)
For each run directory, open in order:
1. `README_FIRST.md`
2. `overview/factor_insights.csv`
3. `overview/factor_scorecard.csv`
4. `assets/key/`

Deep details remain in:
- `tables/detail/`
- `assets/detail/`

## Main CLI Entrypoints
- `apps/run_from_config.py`: unified config-driven entrypoint
- `apps/lint_config.py`: config diagnostics
- `apps/run_factor_research.py`: quick panel research wrapper
- `apps/prepare_data.py`: prepare panel via adapters
- `apps/run_model_factor_benchmark.py`: OOF model-factor benchmark
- `apps/cleanup_outputs.py`: output cleanup (`--purge-all` supported)

## Documentation
- `docs/user_guide.md`: command cookbook and practical usage
- `docs/architecture.md`: architecture, metric tiers, output structure

## Git Safety
- No raw data committed
- `data/`, `outputs/`, `artifacts/` and common data files are git-ignored
- Recommended workflow: one branch per PR, no direct push to `main`
