# Architecture

## 设计目标
- 入口少：统一以 `apps/run_from_config.py` 为主。
- 配置少：`configs/` 只保留 `cs_factor.yaml` 与 `ts_factor.yaml` 两份主配置。
- 逻辑复用：业务能力集中在 `core/factorlab/`，脚本仅做参数解析。
- 数据自由：`data.path` 支持文件与目录，避免为不同数据源反复写脚本。

## 运行主链路
1. 配置读取与归一化（支持别名、CLI 覆盖、基础校验）
2. 数据加载（`data.path` 自动推断文件/目录读取）
3. 因子处理（内置因子、表达式因子、组合因子）
4. 研究评估（CS/TS 双管线）
5. 可选回测
6. 报告与审计产物落盘

## 因子配置策略
- 显式配置：`factor.names: [factor_a, factor_b]`
- 占位名：`factor_name` 会被自动忽略
- 自动发现：当未显式给因子名时，从 panel 列中自动发现可研究因子

## 数据加载策略
- `data.path=*.parquet/*.csv`：单文件读取
- `data.path=<directory>`：目录合并读取（`raw_pattern` 控制文件匹配）
- `data.adapter=synthetic`：用于 CI/冒烟，不依赖真实数据

## 关键审计输出
- `run_meta.json`：配置归一化、插件注册、因子解析、告警汇总
- `run_manifest.json`：运行时环境信息
- `tables/data/adapter_quality_audit.csv`：数据质量审计摘要

## 报告目录分层（减嵌套）
- `overview/`：一站式入口层（README + 评分卡 + 自动解读）
- `tables/overview/`：核心决策层（`factor_scorecard.csv`, `metric_inventory.csv`）
- `tables/detail/`：全量明细层（每个因子与变体的细分表）
- `assets/key/`：关键图层（Top 因子主图）
- `assets/detail/`：全量图层（按因子/变体）
