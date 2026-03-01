# FILE MAP

## A. Main Runtime
- `src/program/prepare_data.py`: prepares base/factor datasets from raw daily data.
- `src/program/execute_strategy.py`: executes strategy backtests and outputs performance reports.
- `src/program/settings.py`: global runtime config (paths, strategy module, date window).
- `src/program/runtime_utils.py`: shared core utility functions.
- `src/program/performance.py`: performance metrics and plotting logic.
- `src/program/finance_data.py`: optional finance-data transforms.
- `src/program/pca_composite.py`: PCA composite-factor builder.

## B. Factor Research Source
- Directory: `src/program/factors/`
- Purpose: factor definitions loaded dynamically by the runtime.

## C. Strategy Research Source
- Directory: `src/program/research_strategies/`
- Purpose: research/experimental strategy modules.

## D. Live Strategy Source
- Directory: `src/program/live_strategies/`
- Purpose: production-style strategy modules.

## E. Factor Analytics Tooling
- `src/analytics_tools/factor_research.py`: factor report entry script.
- `src/analytics_tools/helpers/*.py`: analytics and plotting helpers.

## F. Experimental Mining Scripts
- Directory: `src/factor_mining/`
- Purpose: independent factor-mining experiments.

## G. Visual Assets
- Directory: `assets/`
- Purpose: README display images.
