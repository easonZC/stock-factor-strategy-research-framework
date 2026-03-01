# SSF v2: GitHub-Safe Stock Factor + Strategy Research Framework

## English Overview
SSF v2 is a reusable quant research framework designed for clean-machine reproducibility and GitHub safety. The codebase separates thin scripts from reusable package logic under `src/ssf`, removes proprietary local paths, avoids committing data artifacts, and provides a fully synthetic end-to-end demo. The primary goal is report-grade factor research: multi-horizon forward returns, IC/RankIC and ICIR, IC decay, Newey–West significance testing, quantile portfolio analytics, turnover diagnostics, coverage/missing/outlier monitoring, factor stability analysis, neutralization A/B comparison, and multi-factor correlation heatmaps.

## 中文说明
SSF v2 是一个可复用、可迁移、可在干净环境直接运行的量化研究框架。核心目标是把“因子研究”升级到可交付报告级别：支持多周期收益预测、IC/RankIC 与 ICIR、IC 衰减、Newey–West 显著性检验、分位数组合收益与多空净值、换手分析、覆盖率/缺失率/异常值监控、稳定性分析、raw vs neutralized A/B 对比，以及多因子相关性热力图。脚本层仅做参数入口，核心逻辑全部放入 `src/ssf`，便于二次开发和团队协作。

## Layout
```text
src/ssf/
  config.py
  data/
  preprocess/
  factors/
  strategies/
  backtest/
  research/
  plotting/
  models/
scripts/
examples/legacy/
docs/
tests/
.github/workflows/ci.yml
```

## Quickstart
```bash
pip install -r requirements.txt
```

### A) Synthetic report (must pass)
```bash
python scripts/demo_factor_research.py --out outputs/factor_report_demo
```

### B) Panel input
```bash
python scripts/run_factor_research.py --panel <path> --factors momentum_20,volatility_20 --horizons 1 5 10 20 --out outputs/factor_report
```

### C) Sina folder adapter
```bash
python scripts/prepare_data.py --adapter sina --data-dir /stock_sina_update --out data/panel.parquet
python scripts/run_factor_research.py --panel data/panel.parquet --out outputs/factor_report_sina
```

## Legacy quarantine
Legacy internship-specific code/data references are moved under `examples/legacy/` for reference only.

## Safety rules
- No raw data committed.
- No hard-coded local paths/tokens.
- Use CLI args + env vars + dataclass configs.

## Notes
See `docs/data_processing.md` for leakage-safe preprocessing rules and `docs/dl_feature_norm.md` for LayerNorm/BatchNorm concepts.
