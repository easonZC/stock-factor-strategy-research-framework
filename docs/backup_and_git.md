# Legacy Backup + Git Commands

This workspace was not a git repository at refactor start, so tag/branch could not be created directly.

Use the following commands after `git init`:

```bash
git init
git add .
git commit -m "chore: baseline before ssf-v2 refactor"

DATE=$(date +%Y%m%d)
git tag legacy_before_refactor_${DATE}
git checkout -b refactor/ssf-v2
```

A physical backup snapshot was created at:
- `backups/github_ready_backup_20260228_222225.zip`
