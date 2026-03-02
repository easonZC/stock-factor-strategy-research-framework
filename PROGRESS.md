# Progress Log

## Purpose
- Track every complete Codex run in this repository.
- Keep each run auditable: what changed, how it was verified, and what comes next.

## Mandatory Per-Run Record
For every complete run, append one new entry at the top of `Run History` with:
- Date/time (America/Los_Angeles).
- Goal of the run.
- Files added/modified/deleted.
- Commands executed for validation.
- Git actions and result (`status`, `commit`, `push`).
- Next run direction.

## Strict Git Workflow
0. Scope Boundary
- Default operation scope is this repository path only:
  - `/home/oknotok/Projects/stock-factor-strategy-research-framework`
- Do not read/write Windows mounted paths (`/mnt/c`, `/mnt/d`, etc.) unless the user explicitly asks.

1. Sync Main
- `git switch main`
- `git fetch origin`
- `git pull --rebase origin main`

2. Create Branch
- `git switch -c <type>/<scope>-<yyyymmdd>`
- Never develop directly on `main`.

3. Implement
- Keep changes focused to one clear objective per run.
- Update `PROGRESS.md` in the same run.

4. Validate
- Run targeted checks/tests for changed scope.
- Ensure `git status -sb` only contains intended changes.

5. Commit
- `git add <files>`
- `git commit -m "<type>(<scope>): <summary>"`
- Commit message types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

6. Push Branch
- `git push -u origin <branch>`
- Never push `main` directly.
- Record commit hash and push result in `PROGRESS.md`.

7. Open PR
- Open pull request: `<branch> -> main`.
- Merge to `main` through PR only.
- Local guardrail: `.githooks/pre-push` blocks direct pushes to `main`.

## Run History

### Run 2026-03-01-004
- Time: 2026-03-01 (America/Los_Angeles)
- Goal: Start full SSF v2 refactor implementation for report-grade factor research and config-driven TS/CS workflows.
- Changes:
  - Created baseline backup tag: `legacy_before_refactor_20260301`.
  - Created working branch: `refactor/ssf-v2`.
  - Added `src/ssf/data/` package:
    - panel IO + sanitization (`read_panel`, `write_panel`, `PanelSanitizationConfig`)
    - synthetic generator (`generate_synthetic_panel`)
    - Sina adapter with auto schema mapping + explicit warnings (`prepare_sina_panel`)
    - universe filter (`apply_universe_filter`)
  - Upgraded strategy implementations to match workflow config args and added `FlexibleLongShortStrategy`.
  - Expanded model layer with reusable registry (`ridge/rf/mlp/lgbm`) and OOF training pipeline (`OOFSplitConfig`, `train_oof_model_factor`).
  - Updated `.gitignore` to keep hard constraints while allowing tracked source module `src/ssf/data/`.
  - Rewrote `README.md` into EN+ZH professional project guide with required commands and clear architecture.
  - Added architecture doc: `docs/architecture.md`.
  - Updated CI demo smoke check to enforce `>=6` plot artifacts.
- Validation commands:
  - `ruff check src scripts tests`
  - `pytest -q`
  - `python3 scripts/demo_factor_research.py --out outputs/factor_report_demo`
  - `python3 scripts/run_factor_research.py --panel data/panel_demo.parquet --factors momentum_20,volatility_20,liquidity_shock --horizons 1 5 10 20 --out outputs/factor_report`
  - `python3 scripts/prepare_data.py --adapter sina --data-dir /tmp/stock_sina_update --out data/panel_sina.parquet`
  - `python3 scripts/run_factor_research.py --panel data/panel_sina.parquet --out outputs/factor_report_sina`
  - `python3 scripts/run_from_config.py --config configs/cs_factor_demo.yaml --out outputs/cs_factor_demo`
  - `python3 scripts/run_from_config.py --config configs/ts_factor_demo.yaml --out outputs/ts_factor_demo`
- Validation summary:
  - Lint passed.
  - Tests passed: `6 passed`.
  - Synthetic, panel-input, Sina-adapter, and config-driven TS/CS report runs all completed and generated expected artifacts (`index.html`, `tables/summary.csv`, multiple plots).
- Git actions:
  - Local milestone commits completed:
    - `8f26bd2` (`chore(m1): add data module and harden gitignore boundaries`)
    - `a471653` (`feat(m2): align strategy interfaces and stabilize report entrypoint`)
    - `bee6ae1` (`feat(m3): add reusable model registry and oof model-factor training`)
    - `bfb8177` (`docs(m4): clarify ts/cs preprocessing defaults and leakage notes`)
  - Pending in this run: final `M5` docs/CI commit + push branch + PR update.
- Next run direction:
  - Finalize docs/CI polish commit (`M5`), push `refactor/ssf-v2`, and open/update PR against `main`.
  - Add deeper regression tests for report-table field completeness and Newey-West edge cases.

### Run 2026-03-01-003
- Time: 2026-03-01 (America/Los_Angeles)
- Goal: Verify `gh` availability/auth and confirm ability to create PR directly from CLI.
- Changes:
  - Updated `PROGRESS.md` with this run result.
- Validation commands:
  - `gh --version`
  - `gh auth status`
  - `gh pr status --repo easonZC/stock-factor-strategy-research-framework`
  - `gh pr list --repo easonZC/stock-factor-strategy-research-framework --head chore/branch-pr-workflow-20260301 --state all --limit 5`
  - `gh pr create --repo easonZC/stock-factor-strategy-research-framework --base main --head chore/branch-pr-workflow-20260301 ...`
- Validation summary:
  - `gh` is installed and authenticated as `easonZC`.
  - Token has `repo` scope and git protocol is `ssh`.
  - PR creation via `gh` succeeded.
- Git actions:
  - Working branch: `chore/branch-pr-workflow-20260301`.
  - PR created: `https://github.com/easonZC/stock-factor-strategy-research-framework/pull/1`
- Next run direction:
  - Keep creating branch-scoped commits and update this PR (or create new branch + new PR for a new objective).

### Run 2026-03-01-002
- Time: 2026-03-01 (America/Los_Angeles)
- Goal: Verify SSH push access and enforce branch + PR workflow (no direct push to `main`).
- Changes:
  - Added `.githooks/pre-push` to block direct pushes to `main`.
  - Updated strict git workflow to "new branch -> push branch -> open PR -> merge to main".
- Validation commands:
  - `git status -sb`
  - `git remote -v`
  - `ssh -T -o StrictHostKeyChecking=accept-new git@github.com`
  - `git config --get core.hooksPath`
  - `printf '... refs/heads/main ...' | .githooks/pre-push origin <remote>`
- Validation summary:
  - `origin` uses SSH URL (`git@github.com:...`).
  - SSH authentication to GitHub succeeded.
  - `core.hooksPath` is set to `.githooks`.
  - Hook simulation returns exit code `1` when target ref is `refs/heads/main`.
- Git actions:
  - Working branch for this run: `chore/branch-pr-workflow-20260301`.
  - Commit succeeded: `4e7afd8` (`docs(workflow): enforce branch and PR push policy`).
  - Push succeeded: `git push -u origin chore/branch-pr-workflow-20260301`.
  - PR URL (create): `https://github.com/easonZC/stock-factor-strategy-research-framework/pull/new/chore/branch-pr-workflow-20260301`
- Next run direction:
  - Keep using one branch per complete run and open a PR for merge.
  - Keep `PROGRESS.md` updated with branch name and push/PR status.

### Run 2026-03-01-001
- Time: 2026-03-01 (America/Los_Angeles)
- Goal: Create progress tracking, define strict git workflow, and verify git/repo access boundaries.
- Changes:
  - Added `PROGRESS.md`.
- Validation commands:
  - `git rev-parse --is-inside-work-tree`
  - `git status -sb`
  - `git rev-parse --abbrev-ref HEAD`
  - `git remote -v`
  - `git ls-remote --heads origin`
  - `ls -la /mnt`
- Validation summary:
  - Repository is a valid git work tree on branch `main`.
  - Remote `origin` is configured and reachable for read (`ls-remote` succeeded).
  - WSL can see mounted Windows drives under `/mnt` (for example `/mnt/c`, `/mnt/d`).
- Git actions:
  - Local commits succeeded:
    - `cc2e07b` (`docs(progress): add run log policy and initial entry`)
    - `5740a3e` (`docs(progress): record push auth blocker`)
    - `5ac7bfa` (`docs(progress): add repo-only scope boundary`)
  - Push attempts failed: `git push origin main` -> missing GitHub HTTPS credentials in this WSL session.
- Next run direction:
  - Continue appending one run entry per completed task.
  - Keep one objective per commit and include verification commands/results.
  - Configure authenticated push (SSH key or GitHub token) before requiring remote push on every run.
