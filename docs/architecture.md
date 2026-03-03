# Architecture

## 1. 设计目标
- 入口稳定：统一以 `apps/run_from_config.py` 为主入口。
- 配置简洁：默认只维护 `configs/cs_factor.yaml` 与 `configs/ts_factor.yaml`。
- 核心复用：业务逻辑集中在 `core/factorlab/`，脚本层只做参数组装。
- 数据自由：使用 `data.path` 统一接入本地文件或目录，避免硬编码路径。
- 工程可审计：每次运行都落盘可追踪元数据与质量审计结果。

## 2. 运行链路
1. 配置合成与归一化（含别名兼容、CLI `--set` 覆盖、校验）
2. 数据加载（`data.path` 自动推断，或显式 `adapter`）
3. 因子计算（内置、插件、表达式、组合）
4. 预处理（去极值、标准化、中性化、缺失处理）
5. 研究评估（CS/TS）
6. 可选回测
7. 报告与审计产物输出

## 3. 配置与数据策略
### 3.1 因子配置
- 显式指定：`factor.names: [factor_a, factor_b]`
- 占位名自动忽略：`factor_name` 不会被当作真实因子
- 自动发现：未显式配置时从 panel 列中自动识别可研究因子

### 3.2 数据接入
- `data.path=*.parquet/*.csv`：按单文件读取
- `data.path=<directory>`：按目录合并读取（`raw_pattern` 控制）
- `data.adapter=synthetic`：用于 CI、冒烟、无真实数据时验证流程

### 3.3 预处理与防泄露
- 去极值：`quantile` / `mad`
- 标准化：
  - 截面：`cs_zscore` / `cs_rank` / `cs_robust_zscore`
  - 时序：`ts_rolling_zscore`
- 缺失处理：默认 `drop`，可切换 `fill_zero` / `ffill_by_asset` / `cs_median_by_date` / `keep`
- 中性化：`none/size/industry/both`，按交易日截面回归，不使用未来信息
- 前瞻收益与回测：使用未来价格移位，默认执行延迟，避免同日未来函数

## 4. 指标分层（不删指标）
### 4.1 Core（决策优先）
- CS：`rank_ic_mean`, `rank_icir`, `nw_p_rank_ic`, `ls_sharpe`, `ls_max_drawdown`
- TS：`ic_mean` / `signal_lag0_ic_mean`, `icir`, `nw_p_ic`, `ls_sharpe`, `ls_max_drawdown`

### 4.2 Diagnostic（解释与排障）
- `fmb_factor_beta_mean`, `fmb_factor_nw_p`
- `industry_top_group_mean_ls`, `style_top_group_mean_ls`
- `quantile_monotonicity_mean`, `rank_autocorr_lag1_mean`
- `nw_t_long_short`, `nw_p_long_short`

## 5. 报告输出分层
- `overview/`：一站式入口层（README + 评分卡 + 自动解读）
- `tables/overview/`：核心决策层（`factor_scorecard.csv`, `factor_insights.csv`, `metric_inventory.csv`）
- `tables/detail/`：全量明细层（按因子与变体的细分表）
- `assets/key/`：关键图层（Top 因子主图）
- `assets/detail/`：全量图层（按因子与变体）

## 6. 关键审计产物
- `run_meta.json`：配置归一化、插件注册、因子解析、告警汇总、阶段耗时
- `run_manifest.json`：运行时环境信息
- `tables/data/adapter_quality_audit.csv`：数据质量审计摘要

## 7. DL 特征归一化建议
- BatchNorm：大批量、分布稳定场景更友好
- LayerNorm：小批量/时序模型更稳健
- 原则：训练与验证必须时间切分，禁止随机打散泄露未来信息
