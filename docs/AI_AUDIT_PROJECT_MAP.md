# AI Audit Project Map (MasterTrading)

Last update: 2026-03-01

Purpose
- Single-file map for AI/code-audit agents.
- Minimizes discovery time: where logic lives, what to verify, and in what order.
- Focused on production safety: risk, execution, allocator, regime, data integrity, and deployment.

Audit mode assumptions
- Use DB + Redis + exchange adapter behavior as a single system.
- Risk/leverage are non-negotiable constraints.
- Prefer deterministic, bounded, feature-flagged behavior.

Standalone guarantee
- This document is designed to be sufficient by itself.
- Assume the auditor has no access to other docs.
- If source code is also unavailable, audit against the contracts, formulas, defaults, and controls defined here.

---

## 0) Current Safety Baseline (Inline Contract)

Non-negotiable risk constraints
- Never increase `RISK_PER_TRADE_PCT` as part of calibration/audit recommendations.
- Never increase effective leverage caps.
- Never disable stop protection or drawdown controls.
- Any adaptive system must be bounded and fail-safe.

Current high-impact runtime defaults (production intent)
- `TRADING_ENABLED=true`
- `MAX_EFF_LEVERAGE=3.0` (hard cap)
- `MAX_EXPOSURE_PER_INSTRUMENT_PCT=0.25`
- `DAILY_DD_LIMIT=0.05`
- `WEEKLY_DD_LIMIT=0.10`
- `RISK_EVENT_DEDUP_SECONDS=300`
- `ALLOCATOR_MIN_MODULES_ACTIVE=2`
- `SIGNAL_FLIP_MIN_AGE_MINUTES=5` when enabled

Directional TP/SL and exits (contract)
- `TP_long = max(base_tp, ATR% * ATR_MULT_TP_LONG)`
- `TP_short = max(base_tp, ATR% * ATR_MULT_TP_SHORT)`
- `SL = max(base_sl, ATR% * ATR_MULT_SL, MIN_SL_PCT)`
- Adaptive trailing:
  - high vol => earlier activation
  - lock-in bounded by `[TRAILING_LOCKIN_MIN, TRAILING_LOCKIN_MAX]`

P3/P4 allocator contract
- P3: bounded meta-allocator weight/budget overlay with caps.
- P4 optional:
  - per-bucket (module) DD throttle/freeze,
  - per-bucket daily loss throttle/freeze,
  - optional strict isolation mode: unallocated risk is not redistributed.

---

## 1) Project Snapshot

Stack
- Django 5 + DRF
- Celery + Redis
- PostgreSQL
- Docker Compose
- CCXT adapters (KuCoin/BingX/Binance)

Primary domains
- `signals`: modules + allocator + meta-allocator + regime/garch helpers.
- `execution`: order lifecycle, sizing, exits, sync/reconciliation, AI entry gate.
- `risk`: drawdown baselines/events, dashboards, notifications, Monte Carlo.
- `marketdata`: candles/funding snapshots and collectors.
- `core`: credentials, account runtime context, AI provider config/context files.
- `backtest`: walk-forward engine and reports.

High-risk areas (audit first)
1. `execution/tasks.py`
2. `signals/allocator.py`
3. `signals/meta_allocator.py`
4. `execution/risk_policy.py`
5. `risk/drawdown_state.py`
6. `core/exchange_runtime.py`
7. `adapters/credentials.py` + exchange adapters

---

## 2) Runtime Dataflow (Canonical)

1. Market ingestion
- `marketdata/tasks.py` + `marketdata/management/commands/marketdata_loop.py`
- writes `marketdata.Candle`, `marketdata.FundingRate`.

2. Signal generation
- `signals/multi_strategy.py` module cycles (`trend`, `meanrev`, `carry`, optional `smc` in allocator input).
- emits `signals.Signal` for module-level and allocator-level signals.

3. Allocation
- `signals/allocator.py`: weighted net score, conflicts, min active modules, risk budget mix.
- `signals/meta_allocator.py`: bounded overlay (P3) + optional bucket isolation (P4).

4. Execution
- `execution/tasks.py::execute_orders`
- validates session/regime/macro/filter gates, computes risk/qty, opens/closes, manages trailing/partial/breakeven.
- reconciles exchange fills and logs `execution.OperationReport`.

5. Risk + monitoring
- `risk/drawdown_state.py` + `risk.models.RiskEvent`
- periodic dashboard and MC:
  - `risk/management/commands/perf_dashboard.py`
  - `risk/management/commands/monte_carlo.py`
  - `risk/tasks.py` nightly MC task.

6. Notifications
- `risk/notifications.py` + `risk/telegram_bot.py`.

---

## 3) Repository Map by Area

### 3.1 Core platform and config

Critical files
- `config/settings.py`: all env parsing, schedules, feature flags, safety defaults.
- `config/celery.py`: task routing and beat wiring.
- `docker-compose.yml`, `Dockerfile`: runtime topology and deploy behavior.
- `manage.py`: command entrypoint.

Audit focus
- Unsafe defaults or widening risk/leverage.
- Flag interactions (`MULTI_STRATEGY`, allocator/meta/P4, AI gate).
- Schedule duplication / task overlap.

### 3.2 Core app (`core/`)

Models and security
- `core/models.py`
  - `ExchangeCredential` (owner, alias, service, sandbox/live, leverage, ai_enabled).
  - `ApiProviderConfig`, `ApiContextFile`, `ApiTokenUsageLog`.
  - `AiFeedbackEvent` (structured event stream).
- `core/fields.py`, `core/crypto.py`: credential encryption path.

Runtime account selection
- `core/exchange_runtime.py`
  - resolves active account/service and returns `risk_namespace`.
  - key for multi-account isolation in risk and reporting.

AI context/runtime
- `core/api_runtime.py`: provider runtime, context compaction.
- `core/toon_validator.py` + `core/management/commands/validate_toon_context.py`: TOON gate.
- `core/ai_feedback.py`: JSONL + DB feedback stream.

Audit focus
- Secret exposure.
- Account alias/owner mapping consistency.
- AI config lookup precedence and fail-safe behavior.

### 3.3 Adapters (`adapters/`)

Files
- `adapters/credentials.py`: credential selection and precedence.
- `adapters/kucoin.py`, `adapters/bingx.py`, `adapters/binance.py`.

Audit focus
- Order parameter correctness (`reduceOnly`, leverage/margin mode).
- Retry behavior and exception normalization.
- Symbol normalization consistency.

### 3.4 Market data (`marketdata/`)

Files
- `marketdata/models.py`: candle/funding/orderbook models.
- `marketdata/tasks.py`: fetch/store cycle.
- `marketdata/management/commands/*`: backfill and loop commands.

Audit focus
- Timestamp integrity.
- Missing candle handling and stale-data behavior in execution.

### 3.5 Signals (`signals/`)

Core orchestration
- `signals/multi_strategy.py`
  - module engine cycle
  - allocator cycle
  - optional meta-allocator overlay injection

Allocator logic
- `signals/allocator.py`
  - module weighting and contribution math
  - confluence gates
  - risk budget mix and min multiplier floor

Meta allocator (P3/P4)
- `signals/meta_allocator.py`
  - module-attributed rolling stats
  - bounded weights/budgets
  - optional P4:
    - per-bucket DD/daily loss throttle and freeze
    - strict no-cross-subsidy budget mode

Modules
- `signals/modules/trend.py`
- `signals/modules/meanrev.py`
- `signals/modules/carry.py`
- `signals/tasks.py` (SMC + signal engine internals)
- `signals/modules/common.py`

Regime/vol models
- `signals/regime.py` (HMM helpers and cached regime risk mult)
- `signals/garch.py` (volatility forecast helpers)

Flags and policies
- `signals/feature_flags.py`
- `signals/direction_policy.py`
- `signals/sessions.py`

Audit focus
- Score sign handling and threshold semantics.
- Confluence/penalty interactions.
- Any risk amplification path.
- Determinism and bounded normalization.

### 3.6 Execution (`execution/`)

Core
- `execution/tasks.py`
  - `execute_orders`
  - open/close helpers, TP/SL, trailing, partial close, stale cleanup
  - exchange sync and close classification
  - risk events and notifications

Risk policy
- `execution/risk_policy.py`
  - `volatility_adjusted_risk`
  - max trades by ADX regime

AI and ML gate
- `execution/ai_entry_gate.py`
- `execution/ml_entry_filter.py`

Models
- `execution/models.py`
  - `Order`, `Position`, `OperationReport`, `BalanceSnapshot`
  - includes MFE/MAE/capture fields.

Audit focus
- Quantity rounding/min size constraints.
- SL/TP trigger math with fees.
- reconciliation race conditions, duplicate close logs.
- idempotency and lock safety.

### 3.7 Risk (`risk/`)

State and events
- `risk/models.py`
  - `RiskEvent`, `CircuitBreakerConfig`, `DrawdownBaseline`.
- `risk/drawdown_state.py`
  - DB source of truth + Redis cache for baseline state.

Analytics and controls
- `risk/report_controls.py`
- `risk/management/commands/perf_dashboard.py`
- `risk/management/commands/monte_carlo.py`
- `risk/tasks.py` (nightly MC)

Notifications
- `risk/notifications.py`, `risk/telegram_bot.py`

Audit focus
- DD baseline persistence across restart.
- event dedup/fingerprint behavior.
- MC assumptions/regime-awareness boundaries.

### 3.8 Backtest (`backtest/`)

Files
- `backtest/engine.py`
- `backtest/management/commands/backtest.py`
- additional compare/optimize commands.

Audit focus
- Isomorphism gaps vs live execution cadence.
- slippage/fees assumptions and close reason parity.

---

## 4) Data Model Map (Cross-App)

Core account/config
- `core.ExchangeCredential`
- `core.ApiProviderConfig`
- `core.ApiContextFile`
- `core.ApiTokenUsageLog`
- `core.AiFeedbackEvent`

Market and signals
- `marketdata.Candle`
- `marketdata.FundingRate`
- `signals.Signal`
- `signals.StrategyConfig`

Execution
- `execution.Order`
- `execution.Position`
- `execution.OperationReport`
- `execution.BalanceSnapshot`

Risk
- `risk.RiskEvent`
- `risk.CircuitBreakerConfig`
- `risk.DrawdownBaseline`

Backtest
- `backtest.BacktestRun`
- `backtest.BacktestTrade`

Audit DB checks (minimum)
1. `OperationReport.signal_id` linkage quality.
2. `Position.is_open` consistency vs exchange sync.
3. `DrawdownBaseline` uniqueness and continuity.
4. RiskEvent spam or missing dedup.

---

## 5) Feature-Flag Surface (Audit-Critical)

Core trading enablement
- `TRADING_ENABLED`
- `MODE`
- `MULTI_STRATEGY_ENABLED`
- `ALLOCATOR_ENABLED`

Risk and execution
- `RISK_PER_TRADE_PCT`
- `MAX_EFF_LEVERAGE`
- `MAX_EXPOSURE_PER_INSTRUMENT_PCT`
- `ATR_MULT_*`, `MIN_SL_PCT`
- `REGIME_*`, `BTC_*_BLOCK/HARDEN`

Allocator/meta
- `ALLOCATOR_*`
- `META_ALLOCATOR_*`
- P4 controls:
  - `META_ALLOCATOR_P4_ENABLED`
  - `META_ALLOCATOR_P4_STRICT_BUCKET_ISOLATION_ENABLED`
  - `META_ALLOCATOR_STRATEGY_DD_CAPS`
  - `META_ALLOCATOR_STRATEGY_DAILY_LOSS_CAPS`
  - `ALLOCATOR_BUDGET_MIX_MIN_MULT` (set `0.0` for strict isolation mode)

AI/ML
- `AI_ENTRY_GATE_*`
- `ML_ENTRY_FILTER_*`

Audit rule
- Any change set that touches >1 flag family must include explicit rollback plan.

---

## 6) Celery Tasks and Schedules (What Runs Automatically)

Signals/execution cadence
- signal and allocator cycles (via configured beat tasks in settings/celery wiring).
- execution loop task for live entries/exits.

Risk/analytics
- performance report tasks.
- nightly Monte Carlo (`MONTE_CARLO_NIGHTLY_*`).

Model refresh
- HMM/GARCH periodic tasks when enabled.

Audit focus
- duplicate schedules for same task.
- task lock correctness (`EXECUTION_LOCK_*` and module locks).

---

## 7) Tests Map (Where to Validate)

Signals
- `signals/tests.py`
- `signals/tests_dynamic_weights.py`
- `signals/tests_meta_allocator.py`
- `signals/tests_regime.py`
- `signals/tests_garch.py`

Execution/risk
- `execution/tests_tasks.py`
- `execution/tests_risk_policy.py`
- `risk/tests_drawdown_state.py`
- `risk/tests_monte_carlo_command.py`
- `risk/tests_report_controls.py`

Core/AI
- `core/tests_api_runtime.py`
- `core/tests_toon_validator.py`
- `execution/tests_ai_entry_gate.py`

Audit smoke command
```bash
python manage.py test signals execution risk core --noinput
```

---

## 8) Operational Map (Prod)

Server
- Host: `200.58.107.187`
- SSH port: `5344`
- User: `rortigoza`

Paths/stacks
- Main: `/opt/trading_bot` (`docker compose`)
- Eudy: same repo with compose project `trading_bot_eudy`

Deploy commands (inline, canonical)
1. Pull latest:
```bash
cd /opt/trading_bot
sudo -n git checkout main
sudo -n git pull --ff-only origin main
sudo -n git rev-parse --short HEAD
```

2. Deploy main:
```bash
cd /opt/trading_bot
sudo -n docker compose up -d --build web worker beat
sudo -n docker compose ps
```

3. Deploy eudy:
```bash
cd /opt/trading_bot
sudo -n docker compose -p trading_bot_eudy -f docker-compose.eudy.yml -f docker-compose.eudy.override.yml up -d --build web worker beat
sudo -n docker compose -p trading_bot_eudy -f docker-compose.eudy.yml -f docker-compose.eudy.override.yml ps
```

4. Quick health logs:
```bash
cd /opt/trading_bot
sudo -n docker compose logs --tail=120 worker
sudo -n docker compose -p trading_bot_eudy -f docker-compose.eudy.yml -f docker-compose.eudy.override.yml logs --tail=120 worker
```

Deployment invariants
1. `git pull --ff-only`.
2. build/restart `web`, `worker`, `beat` (main + eudy).
3. check `docker compose ps`.
4. verify logs for exceptions and task liveness.

---

## 9) AI Auditor Read Order (Optimized)

Single-file pass (required)
1. Read this file end-to-end.
2. Use sections 0, 2, 5, 8 and 10 as hard contract.
3. Do not assume behavior not described here.

Deep pass (code)
1. `signals/multi_strategy.py`
2. `signals/allocator.py`
3. `signals/meta_allocator.py`
4. `execution/tasks.py`
5. `execution/risk_policy.py`
6. `risk/drawdown_state.py`
7. `core/exchange_runtime.py`
8. `adapters/credentials.py` + active exchange adapter

---

## 10) Audit Checklist (Actionable)

Correctness
1. Are direction/score signs consistent from module signal to order side?
2. Is `risk_budget_pct` bounded by base risk and caps in all branches?
3. Do TP/SL and trailing calculations include fee-aware behavior as intended?

Risk isolation
1. Does `risk_namespace` isolate account/service/env correctly?
2. With P4 enabled, does a frozen bucket avoid risk reallocation in strict mode?
3. Are DD and daily loss bucket caps enforced without side effects?

Reliability
1. Is execution loop lock effective (no parallel opens)?
2. Are sync/reconciliation close reasons non-duplicative?
3. Are RiskEvents deduplicated correctly by fingerprint window?

Security
1. Any plaintext secrets in repo/logs/docs?
2. Encryption field path still applied for credentials?
3. `.env` and dumps excluded from git?

Observability
1. Are `OperationReport` fields sufficient (reason/sub-reason/mfe/mae/capture)?
2. Does perf dashboard expose symbol/direction/regime and capture stats?
3. Are nightly MC artifacts generated and reviewable?

If only this file is available
1. Validate proposed changes against Section 0 constraints before any optimization.
2. Reject recommendations that increase leverage or base risk.
3. Require bounded feature-flag rollout and rollback for every recommendation.
4. Require explicit checks for DD/event dedup and execution idempotency.

---

## 11) Machine-Readable Quick Index (for LLM parsing)

```yaml
project: mastertrading
critical_paths:
  - config/settings.py
  - signals/multi_strategy.py
  - signals/allocator.py
  - signals/meta_allocator.py
  - execution/tasks.py
  - execution/risk_policy.py
  - risk/drawdown_state.py
  - core/exchange_runtime.py
  - adapters/credentials.py
models:
  core: [ExchangeCredential, ApiProviderConfig, ApiContextFile, ApiTokenUsageLog, AiFeedbackEvent]
  marketdata: [Candle, FundingRate]
  signals: [Signal, StrategyConfig]
  execution: [Order, Position, OperationReport, BalanceSnapshot]
  risk: [RiskEvent, CircuitBreakerConfig, DrawdownBaseline]
flags:
  trading: [TRADING_ENABLED, MODE, MULTI_STRATEGY_ENABLED, ALLOCATOR_ENABLED]
  risk: [RISK_PER_TRADE_PCT, MAX_EFF_LEVERAGE, MAX_EXPOSURE_PER_INSTRUMENT_PCT]
  allocator: [ALLOCATOR_*, META_ALLOCATOR_*]
  p4: [META_ALLOCATOR_P4_ENABLED, META_ALLOCATOR_P4_STRICT_BUCKET_ISOLATION_ENABLED, META_ALLOCATOR_STRATEGY_DD_CAPS, META_ALLOCATOR_STRATEGY_DAILY_LOSS_CAPS]
deploy:
  host: 200.58.107.187
  port: 5344
  user: rortigoza
  path: /opt/trading_bot
  main_port: 8008
  eudy_port: 8010
```

---

## 12) Notes

- This map is intentionally redundant so an AI auditor can work from one file.
- If implementation scope changes (new models, new runtime flags, new task families), update this file in the same commit.
