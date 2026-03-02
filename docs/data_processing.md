# Leakage-Safe Data Processing Notes

This framework enforces conservative preprocessing defaults for factor research:

1. Winsorization
- `quantile`: clip each date's cross-section by quantiles.
- `mad`: clip by median +/- k * MAD.

2. Standardization
- Cross-sectional z-score (`cs_zscore`)
- Cross-sectional rank (`cs_rank`)
- Time-series rolling z-score per asset (`ts_rolling_zscore`)

3. Missing handling
- Default policy is strict `drop` to avoid hidden assumptions.

4. Neutralization
- Supports `size`, `industry`, `both`, `none`.
- Residualization is performed cross-sectionally per date.
- No-lookahead guarantee: only same-date exposures are used.

5. Forward returns
- Computed by per-asset future price shift.
- Strategy backtest shifts weights by one day to avoid lookahead.

6. TS / CS split defaults
- TS scope:
  - default standardization: `ts_rolling_zscore`
  - evaluation axis: `time`
  - quantiles are assigned in a rolling time window per asset.
- CS scope:
  - default standardization: `cs_zscore`
  - evaluation axis: `cross_section`
  - optional per-date winsorization and size/industry neutralization.
