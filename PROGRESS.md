# 项目进度与下一步（持续更新）

## 当前状态
- 仓库主线已完成 `Should Improve` 相关功能增强并合并到 `main`。
- 本文件恢复到项目根目录，用于记录当前进度、规则和下一步方向。
- 注释规范：**代码注释与 docstring 使用中文**（面向自用维护，优先可读性）。

## Git 流程规则（强制）
1. 规则 1：PR squash 合并后，立刻删除分支  
   - 仓库已开启 GitHub `Automatically delete head branches`。  
   - 本地检查结果：`delete_branch_on_merge = true`（已可用）。

2. 规则 2：一个分支只服务一个 PR，禁止在同一分支二次开发  
   - 新需求 = 从最新 `main` 新开分支。  
   - 绝不在已被 squash 合并过的分支继续提交。  
   - 目的：避免重复遇到 squash 后的提交身份冲突（冲突来源已验证过）。

## 执行清单（每次开发都做）
- 先 `git switch main && git pull --ff-only origin main`。
- 再 `git switch -c <feature-or-fix-branch>`。
- 开发完成后先本地校验：`python3 -m pytest -q`、`python3 -m ruff check .`。
- 推送分支并创建 PR，等待 CI 通过后再合并。

## 下一步改进方向
- 性能：继续优化大面板场景下的回测与研究阶段内存占用（减少不必要复制）。
- 研究灵活性：补充更多可配置评估项（分层统计阈值、可选输出粒度）。
- 可维护性：继续收敛接口文档，保持“配置驱动 + OOP 业务实现”的一致性。

## Run History
- 2026-03-02（feat/simple-config-raw-data-20260302）  
  - 已完成：配置简化增强，支持最简配置运行（缺失 `run/research/backtest` 时使用默认值）。  
  - 已完成：新增本地目录数据加载器，`data.path` 指向目录时自动合并 `*.parquet,*.csv`。  
  - 已完成：新增最简本地配置模板 `configs/minimal_local.yaml`，默认读取 `data/raw`。  
  - 已完成：`run_from_config`/README 增加最简示例与本地数据说明；补充 `adapter/synthetic` 概念说明。  
  - 已完成：新增测试覆盖目录读取与最简配置链路。  
  - 校验：`python3 -m ruff check .`、`python3 -m pytest -q`（93 passed）。  
  - 运行验证：`python3 apps/run_from_config.py --config configs/minimal_local.yaml --out outputs/research/factor/local_minimal_run_v2` 成功生成报告。  
  - 下一步：继续把配置向“研究笔记式”收敛（可选单因子快速模式、默认禁用非必要统计输出）。
