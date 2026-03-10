# Architecture

## 1. 设计目标
- 入口稳定：统一以 `factorlab ...` / `scripts/factorlab.py ...` 为主入口。
- 配置简洁：canonical 配置集中在 `examples/workflows/`，`configs/` 仅保留兼容 shim。
- 核心复用：业务逻辑集中在 `factorlab/`，脚本层只做参数组装。
- 边界清晰：`research/` 负责分析计算，`reporting/` 负责报告、产物目录与归因。
- 数据自由：使用 `data.path` 统一接入本地文件或目录，避免硬编码路径。
- 工程可审计：每次运行都落盘可追踪元数据与质量审计结果。

## 2. 运行链路
1. CLI / scripts 入口解析（统一子命令）
2. 配置合成与归一化（含 `_base.yaml` 继承、别名兼容、CLI `--set` 覆盖、校验）
3. 数据加载（`data.path` 自动推断，或显式 `adapter`）
4. 因子计算（内置、插件、表达式、组合）
5. 预处理（去极值、标准化、中性化、缺失处理）
6. 研究评估（CS/TS）
7. 可选回测
8. 报告与审计产物输出

## 3. 配置与数据策略
### 3.1 因子配置
- 显式指定：`factor.names: [factor_a, factor_b]`
- 占位名自动忽略：`factor_name` 不会被当作真实因子
- 自动发现：未显式配置时从 panel 列中自动识别可研究因子
- 因子定义治理：运行时输出 `factor_definitions.csv/json`，记录实际使用因子的输入契约、公式、参数与实现位置
- 策略定义治理：回测启用时输出 `strategy_definitions.csv/json`，记录实际使用策略的约束、参数与实现位置

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
- `overview/`：导航层（`README.md`, `manifest.json`），不再复制 overview CSV
- `tables/overview/`：核心决策层（`quick_summary.csv`, `factor_scorecard.csv`, `factor_insights.csv`, `metric_inventory.csv`, `figure_attribution.csv`）
- `tables/detail/`：全量明细层（按因子与变体的细分表）
- `assets/detail/`：全量图层（按因子与变体，关键图直接引用 canonical 文件）

## 5.2 仓库结构约定
- `factorlab/`：唯一核心源码包，风格接近 Qlib / vectorbt 的顶层 package
- `scripts/`：仓库内推荐入口，便于不安装直接使用
- `python -m factorlab`：与安装态一致的模块入口
- `examples/workflows/`：可直接运行的 canonical 配置
- `apps/`：历史包装层，仅为兼容保留
- `configs/`：历史配置路径，仅通过 `imports` 指向 `examples/workflows/`

### 5.1 报告治理升级
- 不再把 overview 表重复复制到 `overview/`
- 不再把关键图复制到 `assets/key/`
- 新增 `artifact_catalog.json`：运行级产物目录，便于脚本读取和自动化审计
- 新增 `tables/overview/figure_attribution.csv`：每张图对应的来源表、因子、变体与说明
- 新增 `tables/overview/factor_definitions.csv/json`：当前运行实际因子的定义目录
- 新增 `tables/overview/strategy_definitions.csv/json`：当前运行实际策略的定义目录
- 新增 `data_lineage.json`：输入数据来源、面板规模与稳定指纹
- `overview/manifest.json` 与 `report_navigation.json` 统一指向 canonical 路径

## 6. 关键审计产物
- `run_meta.json`：配置归一化、插件注册、因子解析、告警汇总、阶段耗时
- `run_manifest.json`：运行时环境信息
- `experiment_registry.json`：实验注册信息（配置哈希、数据指纹、策略/因子契约、关键输出）
- `artifact_catalog.json`：报告与研究产物目录（机器可读）
- `tables/overview/figure_attribution.csv`：图表来源归因
- `data_lineage.json`：输入数据血缘与指纹
- `tables/data/adapter_quality_audit.csv`：数据质量审计摘要

## 7. DL 特征归一化建议
- BatchNorm：大批量、分布稳定场景更友好
- LayerNorm：小批量/时序模型更稳健
- 原则：训练与验证必须时间切分，禁止随机打散泄露未来信息
