# Repo Map

## Core stack

- Django + DRF + Celery + Redis + PostgreSQL
- Exchange adapters centered on BingX / CCXT
- Main apps:
  - `core`: instruments, credentials, timestamps
  - `marketdata`: candles, funding, snapshots
  - `signals`: modules, sessions, allocator, regime, meta allocator
  - `execution`: orders, positions, exits, reports
  - `risk`: circuit breaker, risk events, dashboards, reports
  - `backtest`: replay engine and optimization helpers
  - `api`: admin/runtime APIs
  - `adapters`: exchange wrappers and retries

## High-value code paths

- `execution/tasks.py`: sizing, order placement, open-position management, report creation, cleanup logic
- `signals/tasks.py`: signal engine orchestration and SMC detection
- `signals/allocator.py`: module combination, thresholds, strong-trend solo, directional penalties
- `signals/multi_strategy.py`: module routing and runtime diagnostics
- `signals/modules/*.py`: `trend`, `meanrev`, `carry`, `grid`, `microvol`
- `signals/regime.py` and `signals/regime_mtf.py`: regime detection and MTF context
- `config/settings.py`: env parsing and beat/queue configuration

## Useful docs in repo

- `AGENTS.md`: primary project memory and non-obvious lessons
- `docs/AI_AUDIT_PROJECT_MAP.md`: one-file map for broad audits
- `docs/OPERATIONS_RUNBOOK.md`: deploy and runtime ops
- `docs/ENV_REFERENCE.md`: env keys and behavior
- `docs/CALIBRATION_CANONICAL.md`: calibration context
- `docs/TRADING_RULES.md`: behavior-level trading notes

## Useful commands

- `python manage.py perf_dashboard ...`
- `python manage.py monte_carlo ...`
- `python manage.py min_qty_risk_report --days 7`
- `python manage.py fit_regime ...`
- `python manage.py fit_garch ...`

## Analysis touchpoints

When a trade looks wrong, inspect:
- `OperationReport`
- `RiskEvent`
- relevant `Signal` rows around entry/exit
- latest candle parity on relevant symbols/timeframes
- close reason vs `mfe_r` / `mae_r`
