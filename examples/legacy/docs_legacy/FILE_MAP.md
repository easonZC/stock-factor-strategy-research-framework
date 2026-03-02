# FILE MAP

## A. Main Runtime
- `src_legacy_v1/program/prepare_data.py`: prepares base/factor datasets from raw daily data.
- `src_legacy_v1/program/execute_strategy.py`: executes strategy backtests and outputs performance reports.
- `src_legacy_v1/program/settings.py`: global runtime config (paths, strategy module, date window).
- `src_legacy_v1/program/runtime_utils.py`: shared core utility functions.
- `src_legacy_v1/program/performance.py`: performance metrics and plotting logic.
- `src_legacy_v1/program/finance_data.py`: optional finance-data transforms.
- `src_legacy_v1/program/pca_composite.py`: PCA composite-factor builder.

## B. Factor Research Source
- Directory: `src_legacy_v1/program/factors/`
- Purpose: factor definitions loaded dynamically by the runtime.
- Quick archive index: `../archive/legacy_factor_index.md`

## C. Strategy Research Source
- Directory: `src_legacy_v1/program/research_strategies/`
- Purpose: research/experimental strategy modules.

## D. Live Strategy Source
- Directory: `src_legacy_v1/program/live_strategies/`
- Purpose: production-style strategy modules.

## E. Factor Analytics Tooling
- `src_legacy_v1/analytics_tools/factor_research.py`: factor report entry script.
- `src_legacy_v1/analytics_tools/helpers/*.py`: analytics and plotting helpers.

## F. Experimental Mining Scripts
- Directory: `src_legacy_v1/factor_mining/`
- Purpose: independent factor-mining experiments.

## G. Visual Assets
- Directory: `assets/`
- Purpose: README display images.
