# Data Adapter Plugin Templates

This folder provides minimal, copy-ready templates for custom data adapters.

## Supported Registration Patterns

1. Auto-detected function name:
- `prepare_<name>_panel(config: AdapterConfig) -> pd.DataFrame`

2. Explicit registry object:
- `DATA_ADAPTER_REGISTRY = {"my_adapter": my_fn}`

3. Explicit registry function:
- `get_data_adapter_registry() -> dict[str, Callable]`

## Canonical Output Columns

Adapter should return a panel dataframe that includes at least:
- `date`
- `asset`
- `close`

Recommended additional columns:
- `open`, `high`, `low`, `volume`, `mkt_cap`, `industry`

## Usage

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

