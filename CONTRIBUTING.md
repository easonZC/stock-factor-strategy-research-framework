# Contributing

This repository is meant to feel like a serious research stack, not a collection of ad-hoc scripts.

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

After setup, use the installed CLI:

```bash
factorlab --help
factorlab run --config examples/workflows/cs_factor.yaml --set data.path=data/raw
```

If you do not want an editable install, use:

```bash
python -m factorlab --help
```

## Public Surface

- `factorlab/`: all core library code
- `factorlab/cli/`: canonical CLI implementation
- `examples/workflows/`: canonical runnable configs
- `scripts/`: repo-local convenience entrypoint
- `apps/`: compatibility wrappers only
- `configs/`: compatibility config shims only

## Rules for Changes

- Add new product logic under `factorlab/`, not under `apps/`
- Add new runnable configs under `examples/workflows/`
- Keep `apps/` wrappers thin; they should only forward to package code
- Prefer extending existing helpers over creating parallel subsystems
- Fix root causes; avoid one-off script-only patches
- Keep code short, explicit, and auditable

## Validation

Run the full suite before handoff:

```bash
python3 -m pytest -q
```

For CLI-heavy changes, also sanity-check:

```bash
factorlab --help
python -m factorlab --help
```

## Output Policy

- Do not commit raw data or generated research outputs
- Treat `outputs/`, `artifacts/`, and local data directories as disposable runtime state

## Documentation Policy

When changing the public surface, update:

- `README.md`
- `docs/user_guide.md`
- `docs/architecture.md`

Keep the docs centered on the canonical CLI and canonical example configs.
