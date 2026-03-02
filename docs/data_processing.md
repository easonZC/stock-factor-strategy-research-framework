# 数据处理与防泄露说明

## 1. 去极值
- `quantile`：按分位点截断
- `mad`：按中位数与 MAD 截断

## 2. 标准化
- 截面：`cs_zscore` / `cs_rank` / `cs_robust_zscore`
- 时序：`ts_rolling_zscore`

## 3. 缺失值处理
- 默认：`drop`
- 可选：`fill_zero` / `ffill_by_asset` / `cs_median_by_date` / `keep`

## 4. 处理中序
- `research.preprocess_steps` 支持有序配置：
  - `winsorize`
  - `standardize`
  - `neutralize`

## 5. 中性化
- 支持 `none/size/industry/both`
- 按交易日做截面回归残差化
- 不使用未来信息

## 6. 前瞻收益与回测
- 前瞻收益按资产未来价格移位计算
- 回测默认有执行延迟，避免同日未来函数

## 7. TS / CS 默认行为
- TS：默认 `ts_rolling_zscore`，评估轴 `time`
- CS：默认 `cs_zscore`，评估轴 `cross_section`
