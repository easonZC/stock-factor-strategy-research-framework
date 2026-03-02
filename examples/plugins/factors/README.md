# 因子插件示例（TS/CS 通用）

本目录提供可直接复用的自定义因子插件，用于：

- 截面研究（CS）：按日期做 IC / 分层组合等评估。
- 时序研究（TS）：按资产做时间序列信号评估。

## 文件说明

- `advanced_factors.py`：一组可直接注册的实战因子示例。

## 在配置中启用

```yaml
factor:
  names: [trend_breakout_30, volume_price_pressure_15, range_reversal_5]
  auto_discover: true
  plugin_dirs:
    - examples/plugins/factors
  plugin_on_error: raise
```

## 设计原则

- 只依赖标准面板字段：`date, asset, open, high, low, close, volume, mkt_cap`。
- 缺字段时返回空信号并保留流程可解释性（不会静默伪造数据）。
- 因子实现保持轻量，便于你后续二次修改。
