# Stock Factor Strategy Research Framework

## English
This repository is a reusable quant research framework for stock factor research, report generation, and optional strategy backtesting. The project is organized around one stable runtime entry (`apps/run_from_config.py`) and one reusable core library (`core/factorlab/`). Configuration is intentionally reduced to two primary templates: one for cross-sectional research (`configs/cs_factor.yaml`) and one for time-series research (`configs/ts_factor.yaml`). The design goal is practical research efficiency: put your dataset path in config, run one command, and get auditable outputs (`index.html`, tables, plots, run metadata).

`data.path` is the default data contract:
- If `data.path` points to `.parquet` or `.csv`, it is loaded as a single panel file.
- If `data.path` points to a directory, the framework auto-merges `*.parquet,*.csv`.
- No hard-coded local proprietary paths are required.

Factor selection is also simplified:
- `factor.names` supports explicit factor names.
- Placeholder names like `factor_name` are ignored intentionally.
- If names are empty (or placeholders only), the pipeline auto-discovers factor columns from the panel.

## 中文说明
这是一个面向股票因子研究的可复用工程框架，核心目标是把“数据读取 -> 因子处理 -> 研究报告 -> 可选回测”变成稳定、可审计、可持续迭代的工程流程。项目入口收敛为 `apps/run_from_config.py`，核心逻辑集中在 `core/factorlab/`，配置文件只保留两份主模板：`configs/cs_factor.yaml`（截面）和 `configs/ts_factor.yaml`（时序）。你只需要在配置里写 `data.path`，就可以直接读取你的本地数据并运行。

`data.path` 是统一数据入口：
- 指向 `.parquet/.csv`：按单文件面板读取。
- 指向目录：自动合并目录下 `*.parquet,*.csv`。
- 不依赖私有硬编码路径，不需要改源码才能切数据集。

因子配置也做了简化：
- `factor.names` 可显式指定因子。
- `factor_name` 这类占位名会被自动忽略。
- 若未显式给出因子名，框架会自动从面板列中发现可研究因子列。

## Install
```bash
python3 -m pip install -r requirements.txt
```

## Quick Start

### 1) Cross-sectional research (CS)
```bash
python apps/run_from_config.py \
  --config configs/cs_factor.yaml \
  --set data.path=data/raw \
  --set factor.names='[factor_a,factor_b]'
```

### 2) Time-series research (TS)
```bash
python apps/run_from_config.py \
  --config configs/ts_factor.yaml \
  --set data.path=data/raw/000001.csv \
  --set factor.names='[factor_ts]'
```

### 3) Validate config before run
```bash
python apps/lint_config.py --config configs/cs_factor.yaml --set data.path=data/raw
```

### 4) Benchmark model factors
```bash
python apps/run_model_factor_benchmark.py \
  --panel data/panel.parquet \
  --models ridge,rf,lgbm \
  --name benchmark_v1
```

### 5) Optional: fast panel research wrapper
```bash
python apps/run_factor_research.py \
  --panel data/panel.parquet \
  --factors factor_a,factor_b
```

`apps/run_from_config.py`、`apps/run_factor_research.py`、`apps/run_model_factor_benchmark.py` 在未传 `--out` 时会自动生成输出目录（带时间戳），也支持 `--name` 指定更易读的运行名。

## Why `adapter` and `synthetic` still exist

- `adapter` means **data loading mode**, not strategy logic.
- In daily use you can ignore explicit adapter setup and only set `data.path`.
- `synthetic` is for reproducible smoke tests/CI when no real data is available.

Example synthetic smoke:
```bash
python apps/run_from_config.py \
  --config configs/cs_factor.yaml \
  --set data.adapter=synthetic \
  --set factor.names='[momentum_20,volatility_20,liquidity_shock]' \
  --set backtest.enabled=false \
  --out outputs/research/factor/ci_smoke
```

## Main Entry Points

- `apps/run_from_config.py`: 配置驱动主入口（推荐）
- `apps/lint_config.py`: 配置体检
- `apps/run_factor_research.py`: 面板文件快速研究（主入口薄封装）
- `apps/prepare_data.py`: 通过适配器准备面板数据
- `apps/run_model_factor_benchmark.py`: 模型因子 OOF 基准评测
- `apps/cleanup_outputs.py`: 输出目录清理
- `docs/cli_quickstart.md`: CLI 使用路径与常见场景速查
- `docs/example_factor_and_model_runs.md`: TS/CS 自定义因子 + ML/NN 基准完整示例

## Output Tree
```text
outputs/research/factor/<run_name>/
  index.html
  config.json
  run_meta.json
  run_manifest.json
  assets/
  tables/
    summary.csv
    data/
      adapter_quality_audit.csv
      field_missing_rates.csv
      asset_row_counts.csv
      date_coverage.csv
```

## Data Safety
- No raw data committed.
- No hard-coded tokens or proprietary paths.
- Data/output/artifact directories are ignored by Git.
