# `--set` 覆盖语义

`apps/run_from_config.py` 支持多次 `--set`，用于临时覆盖 YAML。

## 操作符
- `=`：直接替换
- `+=`：追加/合并
- `-=`：删除

## 示例

### 1) 替换
```bash
--set run.standardization=cs_rank
--set research.quantiles=10
--set run.stop_after=research
```

### 2) 追加/合并
```bash
--set research.horizons+=20
--set factor.names+='[factor_c,factor_d]'
--set research.winsorize+='{lower_q: 0.02, upper_q: 0.98}'
```

### 3) 删除
```bash
--set research.horizons-=1
--set factor.names-='[factor_x,factor_y]'
--set research.winsorize-=method
```

## 说明
- 覆盖按顺序执行，后面的覆盖会覆盖前面的结果。
- `-=` 目标路径必须已存在。
- 值使用 YAML 解析，支持布尔、数组、对象。
