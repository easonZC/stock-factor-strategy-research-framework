# 因子评估指标分层（先看什么）

目标：不删指标，但先区分“决策必看”与“解释辅助”。

## Core（优先）

### CS（截面）
- `rank_ic_mean`：方向与强度主指标
- `rank_icir`：稳定性主指标
- `nw_p_rank_ic`：显著性（是否只是噪声）
- `ls_sharpe`：组合可交易性
- `ls_max_drawdown`：风险约束

### TS（时序）
- `ic_mean` / `signal_lag0_ic_mean`：当期信号有效性
- `icir`：时序稳定性
- `nw_p_ic`：显著性
- `ls_sharpe`、`ls_max_drawdown`：策略可执行性

## Diagnostic（诊断）
- `fmb_factor_beta_mean`、`fmb_factor_nw_p`：回归解释
- `industry_top_group_mean_ls`、`style_top_group_mean_ls`：收益来源拆解
- `quantile_monotonicity_mean`：分层一致性
- `rank_autocorr_lag1_mean`：信号衰减/拥挤度线索

这些指标用于解释和排障，不建议单独作为是否上线的首要依据。

## 当前输出位置
- `tables/overview/factor_scorecard.csv`：每个因子的核心评分卡（建议动作）
- `tables/overview/metric_inventory.csv`：指标分层清单与可用率
- `tables/quick_summary.csv`：Top 因子速览
- `tables/summary.csv`：完整原始指标，不做删除

