# MASTERTRADING — Quantitative Model & Adaptive Regime Framework
Version: 2026-03
Type: Mathematical & System-Level Specification
Audience: Quant Developers + AI Systems

---

# 1. Purpose

This document formalizes:

- Mathematical structure of calibration improvements
- Regime-adaptive behavior model
- Exit efficiency framework
- Risk scaling equations
- Portfolio separation model
- Institutional-grade monitoring design

This is not conceptual. This is structural.

---

# 2. Mathematical Formalization

---

# 2.1 Risk Per Trade (Base Model)

Let:

E = equity  
r = base risk percent  
SL_abs = absolute stop distance  

Risk amount:

R_amt = E * r

Position size:

qty = R_amt / SL_abs

Constraint:

Effective leverage <= MAX_EFF_LEVERAGE

No calibration may modify r upward.

---

# 2.2 Volatility-Adjusted Risk

Let:

atr_pct = ATR / close  

Define scaling factor S_vol:

If atr_pct <= low_threshold:
    S_vol = 1.0
Else if atr_pct >= high_threshold:
    S_vol = VOL_RISK_MIN_SCALE
Else:
    Linear interpolation between 1.0 and VOL_RISK_MIN_SCALE

Effective risk:

r_eff = r * S_vol

BTC hardening:

If instrument == BTC:
    r_eff = r_eff * BTC_multiplier

---

# 2.3 Directional Asymmetry Model

Define:

TP_long = max(base_TP, atr_pct * ATR_MULT_TP_LONG)
TP_short = max(base_TP, atr_pct * ATR_MULT_TP_SHORT)

SL = max(base_SL, atr_pct * ATR_MULT_SL)

Expected payoff asymmetry:

E[R] = p_win * TP - (1 - p_win) * SL

Goal:

Increase E[R] without increasing SL.

---

# 2.4 MFE Capture Model

Define:

MFE = maximum favorable excursion  
R_realized = realized profit in R units  

MFE_capture_ratio = R_realized / MFE

Target:

Mean(MFE_capture_ratio) ≥ 0.60

If below:

Trailing activation too late OR
TP too wide OR
Partial close miscalibrated

---

# 2.5 Trailing Stop Adaptive Function

Define:

activation_R = f(atr_pct)

If atr_pct high:
    activation_R = 1.5
Else:
    activation_R = 2.5

Dynamic lock-in:

lock_in_pct = clamp(0.4 + k * atr_pct, 0.4, 0.7)

Trailing SL:

trail_SL = entry + sign * (MFE * lock_in_pct)

---

# 3. Regime Detection & Adaptive Logic

---

# 3.1 Regime Variables

Inputs:

- HTF trend (1H, 4H)
- ADX
- Volatility percentile
- HMM regime output (Trending/Choppy)

Define regime state:

Bull
Bear
Choppy
HighVol

---

# 3.2 Directional Weight Adjustment

If regime == Bear:
    long_weight *= 0.85

If regime == Bull:
    short_weight *= 0.85

If regime == Choppy:
    risk *= 0.80

If regime == HighVol:
    trailing_activation_R ↓
    risk ↓

---

# 3.3 Risk Throttle During Active Drawdown

If current_drawdown > 50% of daily_limit:
    r_eff *= 0.75

If > 75%:
    r_eff *= 0.50

Never increase risk during drawdown recovery.

---

# 4. Allocator Formalization

Current issue:
Win-rate-only weighting.

Improved weight:

Let:

W = win rate  
E = expectancy (avg R)  
V = inverse volatility of returns  
R = regime multiplier  

Weight:

weight = normalize(W * E * V * R)

Constraints:

- Hard cap per module
- Minimum modules active

---

# 5. Portfolio Risk Segmentation Model

Future-ready design:

Each strategy:

- Independent risk bucket
- Independent DD limit
- Independent reporting
- Separate equity tracking

Global exposure constraint:

Total risk across strategies <= global cap

No cross-subsidization of drawdown.

---

# 6. Statistical Validation Model

Each calibration version must compute:

PF = gross_profit / gross_loss

Sharpe ≈ mean(returns) / std(returns)

Ulcer Index:
sqrt(mean(drawdown^2))

Skewness:
Measure tail asymmetry

Risk of Ruin (Monte Carlo):
Simulate 10k equity paths.

Calibration accepted only if:

PF_new > PF_old  
Max_DD_new <= Max_DD_old  
Sharpe_new > Sharpe_old  
Ulcer_new <= Ulcer_old  

Across majority of regimes.

---

# 7. Institutional Monitoring Dashboard Design

Daily Quant Dashboard Must Include:

1. PF by direction
2. PF by symbol
3. PF by regime
4. MFE_capture_ratio
5. MAE distribution
6. Exit reason breakdown
7. BTC exposure %
8. Risk utilization %
9. Regime classification accuracy
10. Consecutive SL cluster detection

Red Flags:

- MFE_capture_ratio < 0.45
- BTC cluster > 95th percentile historical
- Sharpe collapses > 30%
- RiskEvent spike anomaly

---

# 8. Adaptive Automation Roadmap

Phase 1:
Manual calibration with monitoring.

Phase 2:
Semi-automatic regime-based parameter switching.

Phase 3:
Adaptive volatility-based trailing self-adjustment.

Phase 4:
Allocator meta-learning.

Never allow:
Self-modifying risk parameter.

---

# 9. Scalping Integration Framework (Future)

Scalp module must:

- Use 1m dataset > 3 months.
- Have slippage model.
- Independent DD limit.
- Independent risk bucket.
- OOS validation by session.

Scalp must not share risk with swing.

---

# 10. Strategic Conclusion

The system's strength is structural robustness.

Edge improvement must come from:

- Asymmetry
- Regime adaptation
- Volatility intelligence
- BTC-specific constraints
- Exit efficiency

Not from increasing risk.

The system should evolve toward:

Quant portfolio architecture  
Risk-segmented strategy allocation  
Adaptive regime response  
Institutional-grade monitoring  

---

END OF DOCUMENT