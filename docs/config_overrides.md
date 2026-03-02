## Config Override Semantics (`--set`)

`apps/run_from_config.py` supports repeated `--set` overrides.  
Syntax: `path.to.key<op>value`

- `=`: replace value
- `+=`: append/merge
- `-=`: remove

### 1) Replace (`=`)

```bash
--set research.quantiles=10
--set run.standardization=cs_rank
--set research.horizons='[1,5,10,20]'
```

### 2) Append / Merge (`+=`)

- list append / extend:

```bash
--set research.horizons+=20
--set factor.names+='[momentum_60,volatility_60]'
```

- dict deep merge:

```bash
--set research.winsorize+='{lower_q: 0.02, upper_q: 0.98}'
```

### 3) Remove (`-=`)

- list remove:

```bash
--set research.horizons-=1
--set factor.names-='[size,liquidity_shock]'
```

- dict key remove:

```bash
--set research.winsorize-=method
--set research.winsorize-='[method,upper_q]'
```

## Notes

- Overrides are applied in order; later entries can modify earlier results.
- For `-=` operations, the target path must already exist.
- For `+=` on dict targets, value must be an object.
- YAML parsing is enabled for values, so booleans/lists/dicts can be passed directly.
