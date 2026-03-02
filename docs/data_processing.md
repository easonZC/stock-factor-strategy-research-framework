# Leakage-Safe Data Processing Notes

This framework enforces conservative preprocessing defaults for factor research:

1. Winsorization
- `quantile`: clip each date's cross-section by quantiles.
- `mad`: clip by median +/- k * MAD.

2. Standardization
- Cross-sectional z-score (`cs_zscore`)
- Cross-sectional rank (`cs_rank`)
- Cross-sectional robust z-score (`cs_robust_zscore`, median/MAD)
- Time-series rolling z-score per asset (`ts_rolling_zscore`)

3. Missing handling
- Default policy is strict `drop` to avoid hidden assumptions.
- Optional policies for exploratory research:
  - `fill_zero`
  - `ffill_by_asset` (past-only forward fill per asset)
  - `cs_median_by_date` (same-date cross-sectional median fill)
  - `keep` (preserve missing values for downstream handling)

4. Configurable preprocess order
- CS pipeline supports ordered steps via `research.preprocess_steps`:
  - `winsorize`
  - `standardize`
  - `neutralize`
- This allows flexible A/B experiments without changing code.

5. Neutralization
- Supports `size`, `industry`, `both`, `none`.
- Residualization is performed cross-sectionally per date.
- No-lookahead guarantee: only same-date exposures are used.

6. Forward returns
- Computed by per-asset future price shift.
- Strategy backtest shifts weights by one day to avoid lookahead.

7. TS / CS split defaults
- TS scope:
  - default standardization: `ts_rolling_zscore`
  - evaluation axis: `time`
  - quantiles are assigned in a rolling time window per asset.
- CS scope:
  - default standardization: `cs_zscore`
  - evaluation axis: `cross_section`
  - optional per-date winsorization and size/industry neutralization.

8. Custom transform plugins
- Config-driven custom transforms are supported via:
  - `research.transform_auto_discover`
  - `research.transform_plugin_dirs` / `research.transform_plugins`
  - `research.custom_transforms` list
- Each transform callable must return a `Series` aligned to panel rows.
- Per-item failure policy is explicit: `on_error: raise|warn_skip`.
- Built-in transforms include `clip`, `signed_log1p`, `ts_rolling_zscore`, `cs_rank`, `cs_zscore`.
