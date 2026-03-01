# Trading Rules

Last update: 2026-03-01

This file summarizes live rules. Source of truth remains code:
- `signals/modules/*.py`
- `signals/tasks.py`
- `signals/allocator.py`
- `execution/tasks.py`
- `config/settings.py`

## 1) Signal modules

Trend (`signals/modules/trend.py`)
- Directional following with ADX and EMA context.
- Filters impulsive/late entries.

Mean Reversion (`signals/modules/meanrev.py`)
- Acts in lower-trend regimes.
- Uses z-score style deviation and bounce/impulse guards.

Carry (`signals/modules/carry.py`)
- Funding-based directional hint.
- Volatility cap avoids unstable setups.

SMC (`signals/tasks.py`)
- Multi-gate structure with confluence requirements.
- Uses HTF structure + trigger confirmation.

## 2) Allocator
- Combines active module outputs into net score.
- Supports static and dynamic module weights.
- Requires minimum active modules (`ALLOCATOR_MIN_MODULES_ACTIVE`).
- Produces `alloc_long`, `alloc_short`, or flat/no-trade.

## 3) Entry gates (execution)
- Trading enabled mode and exchange credential active.
- Session score threshold and optional dead-zone block.
- Spread and optional volume filters.
- Market regime checks (ADX/HTF constraints).
- Daily trade limits (adaptive by ADX).
- Per-instrument direction rules and cooldowns.
- Optional ML filter gate if enabled.

## 4) Sizing and leverage
- Fixed fractional risk-based sizing.
- Volatility-adjusted risk scaling.
- Per-instrument risk overrides and tiers.
- Effective leverage and margin buffer caps.
- Exchange minimum quantity/step constraints must be respected.

## 5) Exit logic
- Exchange SL/TP placement and reconciliation.
- Breakeven and trailing logic.
- Partial close at configured R multiple.
- Signal flip close (with optional min-age protection).
- Stale cleanup for old near-breakeven positions.
- HTF killer protections:
  - uptrend short killer
  - downtrend long killer

## 6) Sync and reporting
- `_sync_positions` detects exchange-closed positions.
- `close_sub_reason` classification used for analytics:
  - `exchange_stop`
  - `exchange_tp_limit`
  - `near_breakeven`
  - `bot_close_missed`
  - `unknown`
- `OperationReport` is the canonical closed-trade record.

## 7) Non-negotiable risk constraints
- Do not disable SL behavior in live.
- Do not increase multiple risk knobs in one deployment.
- Do not ship allocator threshold changes without backtest comparison.
- Keep dead-zone/session constraints explicit and traceable.

## 8) Current practical behavior observed
- Loss clusters have historically concentrated in `exchange_close/exchange_stop`.
- Directional bias can shift by regime/day; avoid global assumptions.
- Weekend behavior should be data-validated before global restrictions.

