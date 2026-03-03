# 股票因子与策略研究框架

这是一个 GitHub-safe、可复用的量化研究工程，用于股票因子研究、报告生成与可选回测。

## 这个项目解决什么问题
- 一个稳定入口跑通全流程：`apps/run_from_config.py`
- 核心逻辑沉淀在 `core/factorlab/`（数据、因子、研究、回测、模型）
- 配置驱动，默认只维护两份主模板：
  - `configs/cs_factor.yaml`：截面研究
  - `configs/ts_factor.yaml`：时序研究
- 每次运行自动产出可审计文件（`run_meta.json`、`run_manifest.json`、数据质量审计表）

## 设计原则
- 禁止硬编码私有路径、密钥与本地特殊依赖
- 优先 `data.path` 数据约定，最少配置即可运行
- 结果可复现、过程可追踪
- 脚本层尽量薄，复杂逻辑沉淀到核心模块

## 安装
```bash
python3 -m pip install -r requirements.txt
```

## 快速开始
### 1) 截面研究（CS）
```bash
python apps/run_from_config.py \
  --config configs/cs_factor.yaml \
  --set data.path=data/raw \
  --set factor.names='[factor_a,factor_b]'
```

### 2) 时序研究（TS）
```bash
python apps/run_from_config.py \
  --config configs/ts_factor.yaml \
  --set data.path=data/raw/000001.csv \
  --set factor.names='[factor_ts]'
```

### 3) 只做配置体检
```bash
python apps/lint_config.py --config configs/cs_factor.yaml --set data.path=data/raw
```

### 4) 模型因子基准（ML/NN）
```bash
python apps/run_model_factor_benchmark.py \
  --panel data/panel.parquet \
  --models ridge,rf,mlp \
  --name model_benchmark_v1
```

## 数据约定
`data.path` 是默认统一入口：
- 指向 `.parquet/.csv` 文件：按单文件读取
- 指向目录：自动合并目录下 `*.parquet,*.csv`

`data.adapter=synthetic` 用于 CI 冒烟、无真实数据时的可复现实验，不是业务策略逻辑本身。

## 结果阅读顺序（30 秒）
每次运行先看：
1. `README_FIRST.md`
2. `overview/factor_insights.csv`
3. `overview/factor_scorecard.csv`
4. `assets/key/`

再深入看：
- `tables/detail/`
- `assets/detail/`

## 主要命令入口
- `apps/run_from_config.py`：配置驱动主入口
- `apps/lint_config.py`：配置诊断
- `apps/run_factor_research.py`：面板快速研究封装
- `apps/prepare_data.py`：适配器数据准备
- `apps/run_model_factor_benchmark.py`：OOF 模型因子基准
- `apps/cleanup_outputs.py`：结果清理（支持 `--purge-all`）

## 文档入口
- `docs/user_guide.md`：命令手册与常见运行范式
- `docs/architecture.md`：架构、指标分层、输出结构

## Git 与安全原则
- 不提交原始数据
- `data/`、`outputs/`、`artifacts/` 与常见数据文件默认忽略
- 建议流程：一个分支一个 PR，禁止直接推 `main`
