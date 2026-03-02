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
