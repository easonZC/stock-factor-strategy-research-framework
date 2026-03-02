# Backup and Git Workflow

This repository now uses a branch + PR workflow for all Codex runs.

## Baseline backup before refactor
A legacy baseline tag is created before the major refactor:

```bash
DATE=$(date +%Y%m%d)
git tag legacy_before_refactor_${DATE}
git switch -c refactor/ssf-v2
```

In this workspace, the tag used is:
- `legacy_before_refactor_20260301`

And the main refactor branch is:
- `refactor/ssf-v2`

## Recommended per-run flow
```bash
git switch main
git fetch origin
git pull --rebase origin main
git switch -c <type>/<scope>-<yyyymmdd>
# implement + validate
git add <files>
git commit -m "<type>(<scope>): <summary>"
git push -u origin <branch>
# create PR: <branch> -> main
```

Direct pushes to `main` are blocked locally by `.githooks/pre-push`.
