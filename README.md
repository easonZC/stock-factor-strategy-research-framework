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
- keep scripts thin and core logic reusable

### 2. Project Structure
```text
apps/                 # CLI entrypoints (thin wrappers)
configs/              # Main config templates (CS/TS)
core/factorlab/       # Core reusable library
docs/                 # Minimal docs (user guide + architecture)
examples/             # Legacy archive + plugin examples
tests/                # Test suite
```

### 3. Installation
```bash
python3 -m pip install -r requirements.txt
```

### 4. Main Entrypoints
- `apps/run_from_config.py`: unified config-driven entrypoint
- `apps/run_factor_research.py`: panel quick-research wrapper
- `apps/run_model_factor_benchmark.py`: OOF model-factor benchmark
- `apps/lint_config.py`: config diagnostics
- `apps/prepare_data.py`: adapter-based data preparation
- `apps/cleanup_outputs.py`: output cleanup (supports `--purge-all`)

### 5. Quick Start
#### 5.1 Cross-sectional (CS)
```bash
python apps/run_from_config.py \
  --config configs/cs_factor.yaml \
  --set data.path=data/raw \
  --set factor.names='[factor_a,factor_b]'
```

#### 5.2 Time-series (TS)
```bash
python apps/run_from_config.py \
  --config configs/ts_factor.yaml \
  --set data.path=data/raw/000001.csv \
  --set factor.names='[factor_ts]'
```

#### 5.3 Validate config only
```bash
python apps/lint_config.py --config configs/cs_factor.yaml --set data.path=data/raw
```

#### 5.4 Model-factor benchmark (ML/NN)
```bash
python apps/run_model_factor_benchmark.py \
  --panel data/panel.parquet \
  --models ridge,rf,mlp \
  --name model_benchmark_v1
```

### 6. Data Contract
`data.path` is the primary data contract:
- file path (`.parquet` / `.csv`) -> single panel read
- directory path -> auto-merge `*.parquet,*.csv`

`data.adapter=synthetic` is for reproducible smoke tests / CI / no-real-data validation.

### 7. Output Reading Order (First 30 Seconds)
For each run directory:
1. `README_FIRST.md`
2. `overview/factor_insights.csv`
3. `overview/factor_scorecard.csv`
4. `assets/key/`

For deep details:
- `tables/detail/`
- `assets/detail/`

### 8. Docs
- `docs/user_guide.md`: commands and usage patterns
- `docs/architecture.md`: architecture, metric tiers, output layout

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
apps/                 # 命令行入口（薄封装）
configs/              # 主配置模板（CS/TS）
core/factorlab/       # 核心可复用库
docs/                 # 精简文档（使用手册 + 架构）
examples/             # 旧代码归档 + 插件示例
tests/                # 测试集
```

### 3. 安装
```bash
python3 -m pip install -r requirements.txt
```

### 4. 主要入口
- `apps/run_from_config.py`：统一配置驱动主入口
- `apps/run_factor_research.py`：面板快速研究封装
- `apps/run_model_factor_benchmark.py`：OOF 模型因子基准
- `apps/lint_config.py`：配置诊断
- `apps/prepare_data.py`：适配器数据准备
- `apps/cleanup_outputs.py`：输出清理（支持 `--purge-all`）

### 5. 快速开始
#### 5.1 截面研究（CS）
```bash
python apps/run_from_config.py \
  --config configs/cs_factor.yaml \
  --set data.path=data/raw \
  --set factor.names='[factor_a,factor_b]'
```

#### 5.2 时序研究（TS）
```bash
python apps/run_from_config.py \
  --config configs/ts_factor.yaml \
  --set data.path=data/raw/000001.csv \
  --set factor.names='[factor_ts]'
```

#### 5.3 仅配置体检
```bash
python apps/lint_config.py --config configs/cs_factor.yaml --set data.path=data/raw
```

#### 5.4 模型因子基准（ML/NN）
```bash
python apps/run_model_factor_benchmark.py \
  --panel data/panel.parquet \
  --models ridge,rf,mlp \
  --name model_benchmark_v1
```

### 6. 数据约定
`data.path` 是默认统一入口：
- 指向 `.parquet/.csv` 文件：按单文件读取
- 指向目录：自动合并目录下 `*.parquet,*.csv`

`data.adapter=synthetic` 用于 CI 冒烟、无真实数据时的可复现实验。

### 7. 结果阅读顺序（30 秒）
每次运行先看：
1. `README_FIRST.md`
2. `overview/factor_insights.csv`
3. `overview/factor_scorecard.csv`
4. `assets/key/`

深层明细再看：
- `tables/detail/`
- `assets/detail/`

### 8. 文档入口
- `docs/user_guide.md`：命令手册与运行范式
- `docs/architecture.md`：架构、指标分层、输出结构

### 9. 安全与 Git 规则
- 不提交原始数据
- `data/`、`outputs/`、`artifacts/` 及常见数据产物默认忽略
- 建议流程：一个分支一个 PR，禁止直接推 `main`
