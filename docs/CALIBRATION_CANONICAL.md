# Calibration Canonical (P0-P2)

Last update: 2026-03-01

Purpose
- Canonical baseline for the 2026-03 calibration cycle.
- Consolidates the approved incremental scope P0-P2 with rollout and rollback gates.

Source docs for this cycle
- `docs/META_ALLOCATOR_PSEUDOCODE_AND_MONTECARLO_2026.md`
- `docs/AUTOMATED_REGIME_AND_META_ALLOCATOR_2026.md`
- `docs/CALIBRATION_IMPLEMENTATION_AND_VALIDATION_2026.md`

## 1) Scope (Closed)

Included
- P0: drawdown baseline normalization (DB truth + Redis cache behavior) and risk-event dedup.
- P1: asymmetric TP by direction + BTC volatility hardening.
- P2: adaptive trailing, directional regime penalty, MFE/MAE capture metrics and dashboard support.

Excluded in this cycle
- Full meta-allocator capital rebalance and independent multi-strategy buckets (P3/P4).

## 2) Non-negotiable constraints

- `RISK_PER_TRADE_PCT` unchanged by this cycle.
- No leverage cap expansion.
- DB is source of truth for drawdown baselines.
- Rollout sequence is mandatory: `paper -> demo -> live`.

## 3) Runtime switches introduced

Execution/risk controls
- `ATR_MULT_TP_LONG`, `ATR_MULT_TP_SHORT`
- `BTC_VOL_RISK_HARDEN_ENABLED`
- `BTC_VOL_RISK_ATR_THRESHOLD`
- `BTC_VOL_RISK_MULT`
- `TRAILING_ADAPTIVE_ENABLED`
- `TRAILING_ACTIVATION_R_LOWVOL`
- `TRAILING_ACTIVATION_R_HIGHVOL`
- `TRAILING_ACTIVATION_ATR_THRESHOLD`
- `TRAILING_LOCKIN_MIN`
- `TRAILING_LOCKIN_MAX`
- `TRAILING_LOCKIN_SLOPE`
- `REGIME_DIRECTIONAL_PENALTY_ENABLED`
- `REGIME_BEAR_LONG_PENALTY`
- `REGIME_BULL_SHORT_PENALTY`
- `BTC_BEAR_LONG_BLOCK_ENABLED`
- `RISK_EVENT_DEDUP_SECONDS` (single dedup source for temporal bucketing)

## 4) Data model changes

- `risk.DrawdownBaseline`
  - Unique key: `(risk_namespace, period_type, period_key)`
  - Persists start equity and emitted DD state across restarts.
- `execution.OperationReport`
  - New nullable fields: `mfe_r`, `mae_r`, `mfe_capture_ratio`

## 5) Implementation map

P0
- `risk/drawdown_state.py`: deterministic API for baseline init/update/compute.
- `execution/tasks.py`:
  - daily/weekly DD checks consume drawdown state service.
  - DD events emit only when threshold breach and material DD delta.
  - `_create_risk_event` uses dedup fingerprint by namespace/kind/symbol/time-bucket.

P1
- `execution/tasks.py::_compute_tp_sl_prices`
  - Long TP uses `ATR_MULT_TP_LONG`; short TP uses `ATR_MULT_TP_SHORT`.
  - Backward compatibility: fallback to `ATR_MULT_TP`.
- `execution/risk_policy.py::volatility_adjusted_risk`
  - BTC ATR hardening multiplies effective risk only when enabled and ATR threshold is breached.

P2
- `execution/tasks.py::_check_trailing_stop`
  - Volatility-adaptive trailing activation threshold.
  - Dynamic lock-in clamp based on ATR.
  - Tracks `trail:max_fav`, `trail:max_adv`, `trail:sl_pct`.
- `execution/tasks.py::_compute_regime_adx_gate`
  - Adds per-symbol HTF directional bias (`bull|bear|neutral`) from 1h EMA20/EMA50 context.
- `execution/tasks.py::_attempt_entry_open`
  - Applies directional regime penalty to effective risk.
  - Optional BTC bear-long hard block.
- `execution/tasks.py::_log_operation`
  - Persists `mfe_r`, `mae_r`, `mfe_capture_ratio` and clears trailing state keys.
- `risk/management/commands/perf_dashboard.py`
  - Adds MFE capture stats (avg/p50/p75) and regime breakdown.
  - Adds symbol|direction|regime aggregated view.

## 6) Acceptance criteria (this cycle)

P0
- Restart does not reset daily/weekly baseline.
- No false extreme DD after cache miss.
- Repeated risk events are bounded by dedup window.

P1
- Risk caps/leverage unchanged.
- BTC high-volatility exposure reduced by config when enabled.
- Directional TP asymmetry active with fallback compatibility.

P2
- MFE/MAE metrics persisted in operation reports.
- Dashboard shows capture ratio and regime-aware buckets.
- No evidence of increased premature-stop spikes after rollout.

## 7) Validation protocol

Mandatory tests
- Unit tests for:
  - TP asymmetry
  - trailing adaptive behavior
  - regime directional penalty
  - BTC hardening gate
  - drawdown baseline persistence and event emission gating
- Integration checks:
  - restart simulation between DD cycles
  - repeated DD breaches do not spam events

Strategy validation
- Walk-forward 60/20/20, minimum 3 rolls.
- Segment by session and regime.
- Target thresholds:
  - PF improvement >= +0.10
  - MaxDD not worse than baseline
  - Sharpe improvement >= +0.20

## 8) Rollout runbook

1. Deploy P0 only.
2. Observe 48h:
   - DD baselines stable across worker/container restarts.
   - risk-event frequency stable.
3. Deploy P1.
4. Observe:
   - paper 5 days
   - demo 3 days
5. Deploy P2.
6. Observe:
   - paper 7 days
   - demo 7 days
7. Enable live gradual account-by-account.

## 9) Rollback runbook

Rollback triggers
- MaxDD worsens materially vs baseline.
- `mfe_capture_ratio` sustained < 0.45.
- Risk event flood or repeated order-send anomalies.

Steps
1. Disable new feature flags first (`TRAILING_ADAPTIVE_ENABLED=false`, `REGIME_DIRECTIONAL_PENALTY_ENABLED=false`, `BTC_VOL_RISK_HARDEN_ENABLED=false`).
2. Revert TP asymmetry by setting:
   - `ATR_MULT_TP_LONG=<legacy ATR_MULT_TP>`
   - `ATR_MULT_TP_SHORT=<legacy ATR_MULT_TP>`
3. Redeploy worker stack.
4. Validate:
   - orders open/close normal
   - DD events back to baseline frequency
   - dashboard metrics back to expected range

## 10) Monitoring checklist

Daily required metrics
- PF by direction, symbol, regime.
- `mfe_capture_ratio` avg and p50/p75.
- Close-reason distribution.
- BTC loss concentration.
- RiskEvent count and dedup effectiveness.

Alerting thresholds
- `mfe_capture_ratio < 0.45`
- BTC SL cluster exceeds configured threshold
- `order_send_error` spike or repeated DD breach alerts
