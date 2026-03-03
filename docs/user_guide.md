# User Guide

本页是项目唯一使用手册，覆盖命令行、配置覆盖、示例运行与输出解读。

## 1. 安装
```bash
python3 -m pip install -r requirements.txt
```

## 2. 最常用命令
### 2.1 配置驱动主入口（推荐）
```bash
python apps/run_from_config.py \
  --config configs/cs_factor.yaml \
  --set data.path=data/raw \
  --set factor.names='[factor_a,factor_b]'
```

### 2.2 TS 研究
```bash
python apps/run_from_config.py \
  --config configs/ts_factor.yaml \
  --set data.path=data/raw/000001.csv \
  --set factor.names='[factor_ts]'
```

### 2.3 配置体检
```bash
python apps/lint_config.py --config configs/cs_factor.yaml --set data.path=data/raw
```

### 2.4 模型因子基准
```bash
python apps/run_model_factor_benchmark.py \
  --panel data/panel.parquet \
  --models ridge,rf,mlp \
  --name model_benchmark_v1
```

## 3. `--set` 覆盖语义
`apps/run_from_config.py` 支持多次 `--set`，按顺序生效。

### 3.1 操作符
- `=`：替换
- `+=`：追加/合并
- `-=`：删除

### 3.2 示例
```bash
# 替换
--set run.standardization=cs_rank
--set research.quantiles=10

# 追加
--set research.horizons+=20
--set factor.names+='[factor_c,factor_d]'

# 删除
--set research.horizons-=1
--set factor.names-='[factor_x,factor_y]'
```

## 4. 典型可复现实验（TS/CS + ML）
### 4.1 CS 因子研究（自定义插件）
```bash
python apps/run_from_config.py \
  --config configs/cs_factor.yaml \
  --set data.adapter=synthetic \
  --set data.synthetic.n_assets=80 \
  --set data.synthetic.n_days=320 \
  --set factor.auto_discover=true \
  --set factor.plugin_dirs='[examples/plugins/factors]' \
  --set factor.names='[trend_breakout_30,volume_price_pressure_15,range_reversal_5,volatility_regime_shift_10_40]' \
  --set run.stop_after=research \
  --set backtest.enabled=false
```

### 4.2 TS 因子研究
```bash
python apps/run_from_config.py \
  --config configs/ts_factor.yaml \
  --set data.adapter=synthetic \
  --set data.synthetic.n_assets=20 \
  --set data.synthetic.n_days=380 \
  --set factor.auto_discover=true \
  --set factor.plugin_dirs='[examples/plugins/factors]' \
  --set factor.names='[trend_breakout_30,range_reversal_5,volatility_regime_shift_10_40]' \
  --set run.stop_after=research \
  --set backtest.enabled=false
```

### 4.3 ML/NN 模型因子基准
```bash
python apps/run_model_factor_benchmark.py \
  --panel outputs/research/model_factor/examples/synthetic_panel_with_factors.parquet \
  --models ridge,rf,mlp \
  --feature-cols trend_breakout_30,volume_price_pressure_15,range_reversal_5,volatility_regime_shift_10_40,momentum_20,volatility_20 \
  --label-horizon 5 \
  --train-days 180 \
  --valid-days 20 \
  --step-days 20
```

## 5. 输出怎么读（最短路径）
每次运行目录先看：
1. `README_FIRST.md`
2. `overview/factor_insights.csv`
3. `overview/factor_scorecard.csv`
4. `assets/key/`

再看明细：
- `tables/detail/`
- `assets/detail/`

## 6. 清理输出
### 6.1 全清
```bash
python apps/cleanup_outputs.py --root outputs --purge-all
```

### 6.2 按保留策略清理
```bash
python apps/cleanup_outputs.py \
  --root outputs/research \
  --older-than-days 14 \
  --keep-latest 20
```

## 7. 常见问题
### Q1: 不想写 adapter，能直接读本地数据吗？
可以。只设 `data.path` 即可。文件自动识别，目录自动合并。

### Q2: `synthetic` 的意义是什么？
用于无真实数据时做可复现实验、CI 冒烟和流程回归测试。

### Q3: 因子名不填会怎样？
如果 `factor.names` 为空或占位名，系统会自动从 panel 列发现候选因子。
