# Workflow Examples

该目录是当前仓库的 canonical 运行配置入口，风格参考 Qlib 一类项目的 `examples/` 布局：

- `_base.yaml`：公共研究基线
- `cs_factor.yaml`：截面研究
- `ts_factor.yaml`：时序研究

推荐命令：

```bash
python scripts/factorlab.py run --config examples/workflows/cs_factor.yaml --set data.path=data/raw
python scripts/factorlab.py run --config examples/workflows/ts_factor.yaml --set data.path=data/raw/000001.csv
python scripts/factorlab.py lint-config --config examples/workflows/cs_factor.yaml
```

兼容说明：
- `configs/cs_factor.yaml`
- `configs/ts_factor.yaml`

这两个旧路径仍可使用，但现在只是导入到本目录的兼容 shim。

如果你要跑 `mlp` 一类模型因子，请优先看 `examples/model_factors/README.md`。
