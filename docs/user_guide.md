# User Guide

本页是项目唯一使用手册，覆盖命令行、配置覆盖、示例运行与输出解读。

## 1. 安装
```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## 2. 最常用命令
### 2.1 配置驱动主入口（推荐）
```bash
factorlab run \
  --config examples/workflows/cs_factor.yaml \
  --set data.path=data/raw \
  --set factor.names='[factor_a,factor_b]'
```

### 2.2 TS 研究
```bash
factorlab run \
  --config examples/workflows/ts_factor.yaml \
  --set data.path=data/raw/000001.csv \
  --set factor.names='[factor_ts]'
```

### 2.3 配置体检
```bash
factorlab lint-config --config examples/workflows/cs_factor.yaml --set data.path=data/raw
```

### 2.4 模型因子基准
```bash
python -m factorlab run-model-benchmark \
  --panel data/panel.parquet \
  --models ridge,rf,mlp \
  --name model_benchmark_v1
```

### 2.5 查看可用因子目录
```bash
factorlab list-factors --name volume_price_pressure_20 --json
```

### 2.6 查看可用策略目录
```bash
factorlab list-strategies --name meanvar --json
```

补充：
- 安装后可直接使用 `factorlab run ...`
- 也支持 `python -m factorlab ...`
- `python scripts/factorlab.py ...` 可作为仓库内回退入口
- `apps/*.py` 和 `configs/*.yaml` 仍兼容，但不再是公开入口

## 3. `--set` 覆盖语义
`factorlab run` 支持多次 `--set`，按顺序生效。

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
factorlab run \
  --config examples/workflows/cs_factor.yaml \
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
factorlab run \
  --config examples/workflows/ts_factor.yaml \
  --set data.adapter=synthetic \
  --set data.synthetic.n_assets=20 \
  --set data.synthetic.n_days=380 \
  --set factor.auto_discover=true \
  --set factor.plugin_dirs='[examples/plugins/factors]' \
  --set factor.names='[trend_breakout_30,range_reversal_5,volatility_regime_shift_10_40]' \
  --set run.stop_after=research \
  --set backtest.enabled=false
```

### 4.3 MLP 截面神经网络因子（推荐起点）

推荐先做 `cross_section`，不先做 `time`：

- 当前模型因子工作流天然适配 `date x asset` 面板 OOF 训练
- 当前研究评估默认就是 RankIC / 分层收益这类截面研究指标
- 真正严肃的 temporal neural factor 往往需要额外的滞后窗口与序列样本构造，应当单独设计

推荐数据集分三层：

- 工程回归：`engineering_demo`，建议 `24 assets x 220 days`
- 研究 sanity check：`research_realistic`，建议 `40 assets x 260 days`
- 生产研究：`local_real_data`，即你自己的本地 `parquet/csv` 面板

基线特征建议从最稳妥的一组开始：

- `momentum_20`
- `volatility_20`
- `liquidity_shock`
- `size`

```bash
python3 - <<'PY'
from factorlab.config import SyntheticConfig
from factorlab.data import generate_model_factor_benchmark_panel

panel = generate_model_factor_benchmark_panel(
    SyntheticConfig(n_assets=24, n_days=220, seed=11, start_date="2021-01-01"),
    tier="engineering_demo",
)
panel.to_parquet("data/mlp_benchmark_panel.parquet", index=False)
PY

python3 - <<'PY'
from factorlab.config import SyntheticConfig
from factorlab.data import generate_model_factor_benchmark_panel

panel = generate_model_factor_benchmark_panel(
    SyntheticConfig(n_assets=40, n_days=260, seed=11, start_date="2021-01-01"),
    tier="research_realistic",
)
panel.to_parquet("data/mlp_realistic_panel.parquet", index=False)
PY

python -m factorlab run-model-benchmark \
  --panel data/mlp_realistic_panel.parquet \
  --models ridge,mlp \
  --feature-cols momentum_20,volatility_20,liquidity_shock,size \
  --label-horizon 5 \
  --evaluation-axis cross_section \
  --train-days 180 \
  --valid-days 20 \
  --step-days 20 \
  --name mlp_cs_factor_v1
```

当前基线分成两层：

- `engineering_demo`：验证 NN 因子链路、OOF、报告、artifact 全部打通
- `research_realistic`：把 `rank_ic_mean` 与 `ICIR` 压回更接近真实研究的量级

如果你要直接上自己的数据，请看：

- `examples/model_factors/local_real_data/README.md`

如果只想快速落一个可复用的神经网络模型因子 artifact：

```bash
python -m factorlab train-model-factor \
  --model mlp \
  --out artifacts/models/mlp_factor.joblib
```

## 5. 输出怎么读（最短路径）
每次运行目录先看：
1. `README_FIRST.md`
2. `overview/README.md`
3. `tables/overview/quick_summary.csv`
4. `tables/overview/figure_attribution.csv`
5. `data_lineage.json`
6. `tables/overview/factor_definitions.csv`
7. `tables/overview/strategy_definitions.csv`
8. `artifact_catalog.json`
9. `experiment_registry.json`

再看明细：
- `tables/detail/`
- `assets/detail/`

补充说明：
- `overview/` 只保留导航文件，不再重复保存 overview CSV
- 关键图不再复制到单独目录，而是直接引用 `assets/detail/` 中的 canonical 图表
- 每张图的来源表可在 `tables/overview/figure_attribution.csv` 中追溯
- 每个实际使用的因子定义可在 `tables/overview/factor_definitions.csv` 中查看
- 每个实际使用的策略定义可在 `tables/overview/strategy_definitions.csv` 中查看
- `data_lineage.json` 用于追踪输入数据来源与稳定指纹，`experiment_registry.json` 用于登记实验元数据

## 6. 清理输出
### 6.1 全清
```bash
python -m factorlab cleanup-outputs --root outputs --purge-all
```

### 6.2 按保留策略清理
```bash
python -m factorlab cleanup-outputs \
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
