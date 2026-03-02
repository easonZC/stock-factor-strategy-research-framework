# Transform Plugin Templates

This folder provides a copy-ready template for custom preprocess transforms used in config runs.

## Auto-discovery contract
- Put one or more `*.py` files in a plugin directory.
- Provide either:
  - `TRANSFORM_REGISTRY: dict[str, callable]`, or
  - `get_transform_registry() -> dict[str, callable]`, or
  - functions named `transform_<name>(panel, factor_col, **kwargs)`.

## Callable signature
A transform callable must return a pandas `Series` aligned to the input panel rows:

```python
def transform_example(panel: pd.DataFrame, factor_col: str, **kwargs) -> pd.Series:
    ...
```

## Config example

```yaml
research:
  transform_auto_discover: true
  transform_plugin_dirs:
    - examples/plugins/transforms
  transform_plugin_on_error: raise
  custom_transforms:
    - name: robust_clip
      kwargs:
        lower_q: 0.02
        upper_q: 0.98
    - name: signed_log
      on_error: warn_skip
```

## Notes
- Built-in transform names include `clip`, `signed_log1p`, `ts_rolling_zscore`, `cs_rank`, `cs_zscore`.
- Use `on_error: warn_skip` per transform item to keep long research runs resilient.
