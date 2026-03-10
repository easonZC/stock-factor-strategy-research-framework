# Stock Factor Strategy Research Framework

English | [中文](#中文完整版)

---

## English Full Guide

### 1. Overview
This repository is a GitHub-safe, reusable quantitative research framework for:
- stock factor research
- report generation
- optional strategy backtesting
- model-factor benchmark workflows

The design target is practical, repeatable research engineering:
- run from config
- avoid hard-coded local paths/tokens
- keep outputs auditable
- keep the public surface small and core logic reusable

### 2. Project Structure
```text
scripts/              # Canonical repo entrypoints
examples/workflows/   # Canonical workflow configs
examples/model_factors/ # Canonical neural/model-factor recipes
apps/                 # Legacy compatibility wrappers
configs/              # Compatibility config shims
factorlab/            # Core reusable library
  cli/                # Packaged CLI entrypoint and helpers
  reporting/          # Report rendering, artifact catalog, figure attribution
docs/                 # Minimal docs (user guide + architecture)
examples/             # Workflow configs + plugin examples + legacy archive
tests/                # Test suite
```

### 3. Installation
```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

### 4. Main Entrypoints
- `factorlab run`: unified config-driven entrypoint
- `factorlab run-panel-research`: quick panel research workflow
- `factorlab run-model-benchmark`: OOF model-factor benchmark
- `factorlab prepare-data`: adapter-based panel preparation
- `factorlab cleanup-outputs`: retention / purge operations
- `factorlab train-model-factor`: quick synthetic training helper
- `factorlab lint-config`: config diagnostics
- `factorlab list-factors`: inspect available factors, required columns, formulas
- `factorlab list-strategies`: inspect available strategies, constraints, parameters
- `python -m factorlab ...`: module entrypoint
- `python scripts/factorlab.py ...`: repo-local fallback

### 5. Quick Start
#### 5.1 Cross-sectional (CS)
```bash
factorlab run \
  --config examples/workflows/cs_factor.yaml \
  --set data.path=data/raw \
  --set factor.names='[factor_a,factor_b]'
```

#### 5.2 Time-series (TS)
```bash
factorlab run \
  --config examples/workflows/ts_factor.yaml \
  --set data.path=data/raw/000001.csv \
  --set factor.names='[factor_ts]'
```

#### 5.3 Validate config only
```bash
factorlab lint-config --config examples/workflows/cs_factor.yaml --set data.path=data/raw
```

#### 5.4 Cross-sectional neural factor (MLP)
```bash
python3 - <<'PY'
from factorlab.config import SyntheticConfig
from factorlab.data import generate_model_factor_benchmark_panel

generate_model_factor_benchmark_panel(
    SyntheticConfig(n_assets=24, n_days=220, seed=11, start_date="2021-01-01"),
    tier="engineering_demo",
).to_parquet("data/mlp_benchmark_panel.parquet", index=False)
PY

python3 - <<'PY'
from factorlab.config import SyntheticConfig
from factorlab.data import generate_model_factor_benchmark_panel

generate_model_factor_benchmark_panel(
    SyntheticConfig(n_assets=40, n_days=260, seed=11, start_date="2021-01-01"),
    tier="research_realistic",
).to_parquet("data/mlp_realistic_panel.parquet", index=False)
PY

python -m factorlab run-model-benchmark \
  --panel data/mlp_realistic_panel.parquet \
  --models ridge,mlp \
  --feature-cols momentum_20,volatility_20,liquidity_shock,size \
  --label-horizon 5 \
  --evaluation-axis cross_section \
  --name mlp_cs_factor_v1
```

For a real local panel template, see `examples/model_factors/local_real_data/README.md`.

#### 5.5 Inspect factor definitions
```bash
factorlab list-factors --name volume_price_pressure_20 --json
```

#### 5.6 Inspect strategy definitions
```bash
factorlab list-strategies --name meanvar --json
```

### 6. Data Contract
`data.path` is the primary data contract:
- file path (`.parquet` / `.csv`) -> single panel read
- directory path -> auto-merge `*.parquet,*.csv`

`data.adapter=synthetic` is for reproducible smoke tests / CI / no-real-data validation.

Canonical workflow configs now live in `examples/workflows/`; `configs/` remains as compatibility shims.

### 7. Output Reading Order (First 30 Seconds)
For each run directory:
1. `README_FIRST.md`
2. `overview/README.md`
3. `tables/overview/quick_summary.csv`
4. `tables/overview/figure_attribution.csv`
5. `data_lineage.json`
6. `tables/overview/factor_definitions.csv`
7. `tables/overview/strategy_definitions.csv`
8. `artifact_catalog.json`
9. `experiment_registry.json`

For deep details:
- `tables/detail/`
- `assets/detail/`

Canonical overview tables live only in `tables/overview/`; `overview/` is now a navigation-only layer to avoid redundant CSV snapshots and copied key images.
Runs also include `tables/overview/factor_definitions.csv/json`, `tables/overview/strategy_definitions.csv/json`, `data_lineage.json`, and `experiment_registry.json` so each run carries auditable factor/strategy contracts, data fingerprints, and experiment registration metadata.

### 8. Docs
- `docs/user_guide.md`: commands and usage patterns
- `docs/architecture.md`: architecture, metric tiers, output layout, reporting governance
- `examples/model_factors/README.md`: first neural/model-factor recipe
- `CONTRIBUTING.md`: contributor workflow, layout rules, validation commands

### 9. Safety and Git Policy
- No raw data committed
- `data/`, `outputs/`, `artifacts/` and common data outputs are git-ignored
- Recommended workflow: one branch per PR, no direct push to `main`

---

## 中文完整版

### 1. 项目简介
这是一个 GitHub-safe、可复用的量化研究工程，覆盖：
- 股票因子研究
- 报告生成
- 可选策略回测
- 模型因子基准评估

目标是把研究流程做成“可长期维护、可复现、可审计”的工程体系。

### 2. 目录结构
```text
scripts/              # 推荐命令入口
examples/workflows/   # 推荐工作流配置
examples/model_factors/ # 推荐神经网络/模型因子示例
apps/                 # 旧命令兼容包装层
configs/              # 旧配置兼容 shim
factorlab/            # 核心可复用库
  cli/                # 打包后的统一 CLI
docs/                 # 精简文档（使用手册 + 架构）
examples/             # 工作流配置 + 插件示例 + 旧代码归档
tests/                # 测试集
```

### 3. 安装
```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

### 4. 主要入口
- `factorlab run`：统一配置驱动主入口
- `factorlab run-panel-research`：面板快速研究
- `factorlab run-model-benchmark`：模型因子基准
- `factorlab prepare-data`：适配器数据准备
- `factorlab cleanup-outputs`：输出清理
- `factorlab train-model-factor`：快速训练示例模型因子
- `factorlab lint-config`：配置诊断
- `factorlab list-factors`：查看因子定义、输入列与公式
- `factorlab list-strategies`：查看策略定义、约束与参数
- `python -m factorlab ...`：模块入口
- `python scripts/factorlab.py ...`：仓库内回退入口

### 5. 快速开始
#### 5.1 截面研究（CS）
```bash
factorlab run \
  --config examples/workflows/cs_factor.yaml \
  --set data.path=data/raw \
  --set factor.names='[factor_a,factor_b]'
```

#### 5.2 时序研究（TS）
```bash
factorlab run \
  --config examples/workflows/ts_factor.yaml \
  --set data.path=data/raw/000001.csv \
  --set factor.names='[factor_ts]'
```

#### 5.3 仅配置体检
```bash
factorlab lint-config --config examples/workflows/cs_factor.yaml --set data.path=data/raw
```

#### 5.4 截面神经网络因子（MLP）
```bash
python3 - <<'PY'
from factorlab.config import SyntheticConfig
from factorlab.data import generate_model_factor_benchmark_panel

generate_model_factor_benchmark_panel(
    SyntheticConfig(n_assets=24, n_days=220, seed=11, start_date="2021-01-01"),
    tier="engineering_demo",
).to_parquet("data/mlp_benchmark_panel.parquet", index=False)
PY

python3 - <<'PY'
from factorlab.config import SyntheticConfig
from factorlab.data import generate_model_factor_benchmark_panel

generate_model_factor_benchmark_panel(
    SyntheticConfig(n_assets=40, n_days=260, seed=11, start_date="2021-01-01"),
    tier="research_realistic",
).to_parquet("data/mlp_realistic_panel.parquet", index=False)
PY

python -m factorlab run-model-benchmark \
  --panel data/mlp_realistic_panel.parquet \
  --models ridge,mlp \
  --feature-cols momentum_20,volatility_20,liquidity_shock,size \
  --label-horizon 5 \
  --evaluation-axis cross_section \
  --name mlp_cs_factor_v1
```

如果你要直接接本地真实 panel，请看 `examples/model_factors/local_real_data/README.md`。

#### 5.5 查看因子定义
```bash
factorlab list-factors --name volume_price_pressure_20 --json
```

#### 5.6 查看策略定义
```bash
factorlab list-strategies --name meanvar --json
```

### 6. 数据约定
`data.path` 是默认统一入口：
- 指向 `.parquet/.csv` 文件：按单文件读取
- 指向目录：自动合并目录下 `*.parquet,*.csv`

`data.adapter=synthetic` 用于 CI 冒烟、无真实数据时的可复现实验。

canonical 运行配置已迁移到 `examples/workflows/`；`configs/` 仅保留兼容导入层。

### 7. 结果阅读顺序（30 秒）
每次运行先看：
1. `README_FIRST.md`
2. `overview/README.md`
3. `tables/overview/quick_summary.csv`
4. `tables/overview/figure_attribution.csv`
5. `data_lineage.json`
6. `tables/overview/factor_definitions.csv`
7. `tables/overview/strategy_definitions.csv`
8. `artifact_catalog.json`
9. `experiment_registry.json`

深层明细再看：
- `tables/detail/`
- `assets/detail/`

说明：
- `overview/` 现在只保留导航文件，不再重复保存 overview CSV
- 关键图直接引用 `assets/detail/` 中的 canonical 图表，不再额外复制
- 图表来源可在 `tables/overview/figure_attribution.csv` 中追溯
- 因子/策略定义分别在 `tables/overview/factor_definitions.csv` 与 `tables/overview/strategy_definitions.csv` 中审计
- `data_lineage.json` 与 `experiment_registry.json` 用于记录输入指纹、配置哈希与实验注册信息

### 8. 文档入口
- `docs/user_guide.md`：命令手册与运行范式
- `CONTRIBUTING.md`：开发约定、目录规则与回归校验
- `docs/architecture.md`：架构、指标分层、输出结构
- `examples/model_factors/README.md`：首个神经网络/模型因子示例

### 9. 安全与 Git 规则
- 不提交原始数据
- `data/`、`outputs/`、`artifacts/` 及常见数据产物默认忽略
- 建议流程：一个分支一个 PR，禁止直接推 `main`
