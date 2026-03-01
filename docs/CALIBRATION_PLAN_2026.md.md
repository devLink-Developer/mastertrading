# MASTERTRADING — Plan Integral de Calibración Cuantitativa
Version: 2026-03  
Scope: Edge Optimization Without Risk Expansion  
Status: Strategic Blueprint  
Audience: Humans + LLM Systems  

---

# 1. Purpose

This document defines a structured quantitative calibration plan for MasterTrading.

It does NOT propose:
- Architectural redesign
- Risk per trade increase
- Leverage expansion
- Curve fitting on short windows

It DOES propose:
- Structural bias correction
- Directional asymmetry
- Exit optimization
- BTC-specific hardening
- Drawdown normalization
- Regime-aware calibration

---

# 2. Current Diagnostic Summary

## 2.1 Observed Facts

- PnL negative concentrated in BTC longs
- Shorts outperform longs in current regime
- TP/SL symmetric in asymmetric market
- Trailing stop activates at fixed 2.5R
- Excessive daily_dd_limit events
- Risk scaling by ATR recently fixed
- Long-only mode previously active during bearish regime

## 2.2 Core Structural Issue

The system is robust architecturally, but calibration lacks:

- Directional asymmetry
- Volatility-adaptive exits
- BTC-specific risk hardening
- Regime-dependent aggression
- Stable drawdown baseline logic

---

# 3. Calibration Constraints

All changes must respect:

1. No increase in RISK_PER_TRADE_PCT
2. No increase in effective leverage
3. No simultaneous multi-axis risk expansion
4. One behavioral class change per deploy
5. Mandatory backtest + forward validation

---

# 4. Improvement 1 — Directional TP Asymmetry

## Problem

Crypto markets fall faster than they rise.  
Symmetric TP penalizes shorts.

## Proposal

Split TP multipliers:

ATR_MULT_TP_LONG = 1.6  
ATR_MULT_TP_SHORT = 2.2  

Maintain:

ATR_MULT_SL = 1.5  

## Rationale

- Shorts capture expansion
- Longs secure profit earlier
- Improves payoff asymmetry
- Enhances PF during bearish macro

## Validation Metrics

- PF per direction
- Expectancy per direction
- R-multiple distribution skew

---

# 5. Improvement 2 — Volatility-Adaptive Trailing

## Problem

Fixed 2.5R trailing activation is late in high volatility.

## Proposal

Volatility-conditioned activation:

if atr_pct > VOL_RISK_HIGH_ATR_PCT:
    trailing_activation_R = 1.5
else:
    trailing_activation_R = 2.5

Dynamic lock-in:

lock_in_pct = clamp(0.4 + atr_pct * k, 0.4, 0.7)

## Expected Effect

- Higher MFE capture
- Reduced profit giveback
- Lower equity volatility
- Improved Sharpe

---

# 6. Improvement 3 — Regime-Aware Directional Penalty

## Problem

System operated long-only in bearish macro.

## Proposal

SIGNAL_DIRECTION_MODE = both

With regime penalty:

if regime == bear:
    long_score *= 0.85

Alternative conservative guard:

Disable BTC longs when:
HTF_4H_trend == bear AND ADX > threshold

---

# 7. Improvement 4 — BTC Risk Hardening

## Problem

BTC concentrated large losses.

## Proposal A — Static Reduction

PER_INSTRUMENT_RISK:
{
  "BTCUSDT": 0.0012,
  "ETHUSDT": 0.0020,
  "SOLUSDT": 0.0020
}

## Proposal B — Volatility Adaptive

if instrument == BTC and atr_pct > threshold:
    effective_risk *= 0.75

---

# 8. Improvement 5 — Drawdown Baseline Normalization (P0)

## Problem

daily_dd_limit events show extreme values (~ -1.0).

Likely causes:
- Baseline reset
- Equity snapshot mismatch
- Redis namespace inconsistency

## Required Fixes

1. Persist daily baseline equity in DB
2. Do not compute DD from partial snapshots
3. Throttle repeated emissions
4. Use deterministic namespace key:
   dd_key = namespace + date

Emit only if:
abs(current_dd - last_emitted_dd) > 1%

---

# 9. Improvement 6 — Allocator Enhancement

## Problem

Dynamic weights consider win rate only.

## Proposal

Incorporate:

- Expectancy (avg R)
- Return volatility
- Regime factor

weight = winrate_adj * expectancy_adj * regime_factor

Normalize to prevent extreme overweighting.

---

# 10. Improvement 7 — Exit Efficiency Metric

Add new metric:

MFE_capture_ratio = realized_R / max_favorable_R

Target:
MFE_capture_ratio >= 0.60

If < 0.50:
- Trailing too late
- TP too wide
- Partial miscalibrated

---

# 11. Improvement 8 — Confidence Boost Control

Fractional Kelly constraints:

- Max multiplier: 1.3x in production
- Activate only if:
  score > 0.9
  ML_prob > 0.75
  regime favorable

Never activate when:
- High volatility regime
- Active drawdown
- Choppy market regime

---

# 12. Implementation Order

## Phase 1 (Critical)

- Fix DD baseline
- Apply TP asymmetry
- Reduce BTC risk

## Phase 2

- Adaptive trailing
- Regime directional penalty

## Phase 3

- Allocator enhancement
- MFE capture optimization

---

# 13. Target Metrics

| Metric | Current | Target |
|--------|---------|--------|
| PF | ~1.1 | 1.25+ |
| Win Rate | ~41% | 45–48% |
| Max DD | >30% | <20% |
| Sharpe | Low | >1.2 |
| MFE Capture | Unknown | >60% |

---

# 14. What Must NOT Be Done

- Do not increase base risk.
- Do not lower MIN_SIGNAL_SCORE globally.
- Do not enable scalping without deep 1m dataset.
- Do not change multiple risk knobs simultaneously.
- Do not optimize on single-window backtests.

---

# 15. Strategic View

The system does not need more complexity.

It needs:

- Asymmetry
- Volatility adaptation
- Regime awareness
- BTC-specific hardening
- Cleaner drawdown logic

---

# 16. Expected Result

If implemented correctly:

- Smoother equity curve
- Reduced BTC loss concentration
- Improved profit factor
- Lower structural drawdown
- Higher regime adaptability
- More stable forward performance

---

END OF DOCUMENT