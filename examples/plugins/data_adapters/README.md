# Data Adapter 插件模板

本目录提供可直接复制的自定义数据适配器模板，支持“加载函数 + 配置校验钩子”。

## 注册方式

1. 自动发现加载函数
- `prepare_<name>_panel(config: AdapterConfig) -> pd.DataFrame`

2. 显式加载注册表
- `DATA_ADAPTER_REGISTRY = {"my_adapter": my_fn}`
- `get_data_adapter_registry() -> dict[str, Callable]`

3. 自动发现配置校验钩子
- `validate_<name>_config(config: AdapterConfig) -> None | str | list[str]`

4. 显式校验注册表
- `DATA_ADAPTER_VALIDATORS = {"my_adapter": my_validator}`
- `get_data_adapter_validators() -> dict[str, Callable]`

## 输出列约定

适配器输出至少包含：
- `date`
- `asset`
- `close`

建议附带：
- `open`, `high`, `low`, `volume`, `mkt_cap`, `industry`

## 使用示例

### `apps/prepare_data.py`

```bash
python apps/prepare_data.py \
  --adapter mock_feed \
  --adapter-plugin-dir examples/plugins/data_adapters \
  --out data/panel.parquet
```

### `apps/run_from_config.py`

```yaml
data:
  adapter: mock_feed
  adapter_auto_discover: true
  adapter_plugin_dirs:
    - examples/plugins/data_adapters
  adapter_plugin_on_error: raise
```
