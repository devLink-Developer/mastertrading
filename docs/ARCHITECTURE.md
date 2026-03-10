# Architecture

Last update: 2026-03-01

## 1) Stack
- Django 5 + DRF
- Celery workers + Celery Beat
- Redis 7 (locks, cache, task broker backend)
- PostgreSQL 15
- Docker Compose for orchestration
- Exchange adapters via CCXT (BingX/KuCoin/Binance wrappers in `adapters/`)

## 2) Services (Compose)
- `web`: Django API/admin and management commands
- `worker`: execution and signal tasks
- `worker-data`: marketdata/ml side tasks
- `beat`: scheduled task triggers
- `market-data`: ingestion task runner
- `telegram-bot`: trade notifications
- `postgres`, `redis`: stateful dependencies

## 3) Django app responsibilities
- `core`: instruments, exchange credentials, base models
- `marketdata`: candles/funding/orderbook snapshots
- `signals`: modules (trend/meanrev/carry/smc), allocator, session policy
- `execution`: order lifecycle, position management, TP/SL/trailing, sync
- `risk`: risk events, dashboard commands, controls
- `backtest`: historical simulation engine
- `api`: DRF endpoints
- `adapters`: exchange wrappers and retries

## 4) Data model hotspots
- `marketdata.models.Candle`: `(instrument, timeframe, ts)` unique.
- `execution.models.Position`: open position snapshot used by runtime sync.
- `execution.models.OperationReport`: canonical closed-trade metrics table.
- `signals.models.Signal`: latest signal payload used by executor.

## 5) Runtime pipeline
1. Market data ingestion stores candles/funding.
2. `signals.tasks.run_signal_engine` generates module signals.
3. `signals.allocator` combines modules into `alloc_long/alloc_short/flat`.
4. `execution.tasks.execute_orders` validates entry gates and sends order.
5. Open position management handles breakeven/partial/trailing and killers.
6. `_sync_positions` reconciles exchange-vs-db and logs close reports.
7. Notifications sent (Telegram and admin visibility).

## 6) Timeframes and intent
- Entry/runtime loop: 1m/5m logic depending on module and guard.
- HTF context: 1h and 4h.
- Backtest high-fidelity mode: 1m.

## 7) Deployment shape used in production
- Main stack:
  - Repo path: `/opt/trading_bot`
  - Compose project: default (`trading_bot-*`)
  - Web port: `8008`
- Eudy stack:
  - Repo path: `/opt/trading_bot` (same git checkout)
  - Compose project: `trading_bot_eudy`
  - File: `docker-compose.eudy.yml`
  - Web host port: not exposed

## 8) Feature-flag-first pattern
- New behavior should be behind env flags in `config/settings.py`.
- Execution must keep deterministic fallback when optional models fail.
- Any AI/ML overlay should be fail-open or fail-safe by explicit flag.

## 9) Read this before editing core execution
- `execution/tasks.py`
- `signals/allocator.py`
- `signals/sessions.py`
- `config/settings.py`
- Existing audits under `docs/auditoria-codigo-*.md`
