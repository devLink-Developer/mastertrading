# MASTERTRADING — AI Master Calibration Context Pack
Version: 2026-03
Purpose: Structured AI-readable quantitative calibration framework
Scope: Edge optimization without structural risk expansion

---

# SECTION 1 — HARD CONSTRAINTS (NON-NEGOTIABLE)

The system must NEVER:

- Increase RISK_PER_TRADE_PCT.
- Increase effective leverage.
- Reduce MIN_SIGNAL_SCORE globally without segmented validation.
- Modify multiple risk axes in one deploy.
- Optimize exclusively on one backtest window.
- Increase exposure caps.
- Remove regime or volatility guards.

All proposals must preserve structural risk discipline.

---

# SECTION 2 — CURRENT STRUCTURAL DIAGNOSIS

System architecture: stable and advanced.

Observed issues:

1. BTC concentrates large losses.
2. Shorts outperform longs in current regime.
3. TP/SL symmetric in asymmetric crypto market.
4. Trailing stop activates late (2.5R fixed).
5. Excessive daily_dd_limit events indicate baseline instability.
6. Long-only operation occurred in bearish macro.

Core issue:
Calibration misalignment, not architectural weakness.

---

# SECTION 3 — CALIBRATION OBJECTIVES

Improve:

- Profit Factor
- MFE capture ratio
- Regime alignment
- Equity smoothness
- BTC loss clustering

Without:

- Raising risk
- Lowering signal quality
- Increasing exposure

---

# SECTION 4 — APPROVED CALIBRATION IMPROVEMENTS

## 4.1 Directional TP Asymmetry

Implement:

ATR_MULT_TP_LONG = 1.6  
ATR_MULT_TP_SHORT = 2.2  
ATR_MULT_SL = 1.5  

Rationale:
Crypto falls faster than it rises.

Expected result:
Higher PF during bearish macro.

---

## 4.2 Volatility-Adaptive Trailing

If atr_pct high:
  trailing_activation_R = 1.5
Else:
  trailing_activation_R = 2.5

Dynamic lock-in:
Clamp between 0.4 and 0.7.

Goal:
Capture ≥ 60% of MFE.

---

## 4.3 Regime-Aware Direction Penalty

Enable both directions.

If regime == bear:
  reduce long score or allocator weight.

Optional:
Disable BTC longs in strong bearish ADX regime.

---

## 4.4 BTC Hardening

Reduce BTC-specific risk:

Static:
BTC risk < alt risk.

Or dynamic:
Reduce risk when volatility high.

BTC must never exceed 25% of total open risk.

---

## 4.5 Drawdown Baseline Normalization (Critical)

Persist daily baseline equity.
Prevent baseline reset on restart.
Throttle repeated identical DD events.

No calibration deploy before this fix.

---

## 4.6 Allocator Enhancement

New weight formula must consider:

- Win rate
- Expectancy (avg R)
- Regime factor
- Return volatility

Weights must be normalized.
Hard cap per module required.

---

# SECTION 5 — VALIDATION FRAMEWORK

Calibration accepted only if:

- PF improves ≥ 0.10.
- Max DD does not increase.
- Sharpe improves ≥ 0.20.
- Return variance stable or lower.
- Improvement consistent across regimes.

Mandatory:

- Walk-forward testing.
- OOS validation.
- Forward paper validation 2–4 weeks.

Reject calibration if forward PF drops >25% from backtest.

---

# SECTION 6 — EXIT EFFICIENCY METRIC

Define:

MFE_capture_ratio = realized_R / max_favorable_R

Target:
≥ 0.60

If < 0.50:
Trailing too late or TP too wide.

---

# SECTION 7 — CONFIDENCE BOOST CONTROL

Fractional Kelly must:

- Cap at 1.3x max.
- Activate only if:
  score > 0.9
  ML_prob > 0.75
  regime favorable

Never activate during:
- High volatility spikes
- Active drawdown
- Choppy regime

---

# SECTION 8 — DEPLOYMENT ORDER

Phase 0:
Fix drawdown baseline.

Phase 1:
TP asymmetry + BTC risk reduction.

Phase 2:
Adaptive trailing + regime penalty.

Phase 3:
Allocator enhancement + exit optimization.

Only one phase per deploy cycle.

---

# SECTION 9 — MONITORING REQUIREMENTS

Daily report must include:

- PF by direction
- PF by symbol
- PF by regime
- BTC exposure %
- MFE_capture_ratio
- Exit reason distribution
- Risk utilization %

Alert if:

- MFE_capture_ratio < 0.45
- BTC loss cluster occurs
- DD spikes abnormally
- Repeated RiskEvent spam

---

# SECTION 10 — PORTFOLIO EXTENSION RULE

Future strategies (scalp, meanrev) must:

- Have independent risk bucket.
- Independent DD limit.
- Separate reporting.
- Never share risk pool with swing.

---

# SECTION 11 — FAILURE CONDITIONS

Rollback immediately if:

- Worker instability occurs.
- Unexpected order rejections spike.
- Margin errors increase.
- 2 consecutive abnormal DD days.
- BTC cluster exceeds historical 95th percentile.

---

# SECTION 12 — STRATEGIC SUMMARY

The system does not require more complexity.

It requires:

- Asymmetry.
- Volatility adaptation.
- Regime awareness.
- BTC hardening.
- Clean drawdown logic.

Edge must come from calibration discipline, not parameter explosion.

---

END OF AI MASTER CONTEXT PACK