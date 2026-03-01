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
  - Commit and push are executed on this branch (not on `main`).
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
