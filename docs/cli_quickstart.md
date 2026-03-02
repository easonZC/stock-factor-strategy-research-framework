# CLI Quickstart（人类友好版）

本页目标：给你最短路径，不看源码也能跑起来。

## 1. 先确认配置有效
```bash
python apps/lint_config.py --config configs/cs_factor.yaml --set data.path=data/raw
```

## 2. 推荐主入口（配置驱动）
```bash
python apps/run_from_config.py \
  --config configs/cs_factor.yaml \
  --set data.path=data/raw \
  --set factor.names='[factor_a,factor_b]'
```

说明：
- 不写 `--out` 会自动生成时间戳目录。
- 可加 `--name my_run` 自定义目录名。
- 可加 `--save-effective-config` 固化实际运行配置。

## 3. 只研究面板（薄封装入口）
```bash
python apps/run_factor_research.py \
  --panel data/panel.parquet \
  --factors factor_a,factor_b
```

适用场景：
- 你只想快速验证一个面板文件；
- 不想先编辑 YAML；
- 不需要复杂分层配置合并。

## 4. 跑模型因子基准
```bash
python apps/run_model_factor_benchmark.py \
  --panel data/panel.parquet \
  --models ridge,rf,lgbm \
  --name model_bench_v1
```

产出重点：
- `model_factor_comparison.csv`
- `index.html`
- `run_meta.json`

## 5. 清理历史输出
```bash
python apps/cleanup_outputs.py \
  --root outputs/research \
  --older-than-days 14 \
  --keep-latest 30 \
  --dry-run
```

## 常见问题

### Q1: 不想写 adapter，能直接读本地文件/目录吗？
可以。只写 `data.path` 即可：
- 指向 `.parquet/.csv`：按单文件读取
- 指向目录：自动合并 `*.parquet,*.csv`

### Q2: 因子名没写会怎样？
- `factor.names` 为空或仅占位名（如 `factor_name`）时，会自动从面板列发现可研究因子。

### Q3: 如何提前看最终合并后的配置？
```bash
python apps/run_from_config.py \
  --config configs/cs_factor.yaml \
  --set data.path=data/raw \
  --show-effective-config \
  --validate-only
```

