# MASTERTRADING — Automated Regime Engine & Meta-Allocator Framework
Version: 2026-03
Type: System-Level Quant Architecture
Audience: Quant Engineers + AI Systems

---

# 1. Purpose

This document defines:

- A fully formal Regime Engine
- An adaptive Meta-Allocator across strategies
- A Monte Carlo risk-of-ruin framework
- A capital allocation discipline model
- Institutional-grade drawdown containment logic

This transforms MasterTrading from calibrated strategy
into a portfolio-level adaptive quant system.

---

# 2. Automated Regime Engine

---

# 2.1 Regime Dimensions

Regime classification must not be binary.

Define regime as a vector:

R = (Trend, Volatility, Liquidity, Expansion)

Where:

Trend:
  +1 = Bull
  -1 = Bear
   0 = Neutral

Volatility:
  Low
  Normal
  High

Liquidity:
  Stable
  Thin

Expansion:
  Breakout
  Range

---

# 2.2 Inputs

Trend detection:
- EMA slope (1H, 4H)
- HTF structure
- HMM regime output

Volatility:
- ATR percentile (rolling 90-day)
- Realized volatility percentile

Liquidity:
- Orderbook spread percentile
- Funding stability

Expansion:
- True range percentile breakout detection

---

# 2.3 Regime Classification Function

Example:

If Trend == -1 AND Volatility == High:
    Regime = "Bear Expansion"

If Trend == +1 AND Volatility == Low:
    Regime = "Bull Compression"

If Trend == 0 AND Volatility == Low:
    Regime = "Range"

If Trend == 0 AND Volatility == High:
    Regime = "Choppy"

---

# 2.4 Regime-Driven Parameter Mapping

Each regime maps to:

- Risk multiplier
- TP multiplier
- Trailing activation
- Direction penalty
- Allocator bias

Example mapping:

Bear Expansion:
  risk *= 0.9
  long_penalty = strong
  trailing_activation_R = 1.5

Bull Compression:
  risk *= 1.0
  short_penalty = moderate
  TP_long slightly wider

Range:
  risk *= 0.8
  disable trend module bias

Choppy:
  risk *= 0.7
  tighten SL
  earlier trailing

---

# 3. Meta-Allocator Architecture

---

# 3.1 Strategy Layer Separation

Each strategy outputs:

- Expected R
- Confidence score
- Risk demand
- Regime suitability score

Strategies:

- SMC Swing
- Mean Reversion
- Future Scalp
- Carry Module

---

# 3.2 Meta-Allocator Objective

Maximize:

Expected Portfolio Return

Subject to:

- Risk budget constraint
- Drawdown tolerance
- Regime alignment

---

# 3.3 Capital Allocation Model

Let:

E_i = expectancy of strategy i  
σ_i = volatility of strategy i  
ρ_ij = correlation matrix  

Define risk-adjusted weight:

w_i = (E_i / σ_i) * RegimeFactor_i

Normalize weights under:

Σ w_i ≤ TotalRiskCap

If correlation high:
reduce overlapping exposure.

---

# 3.4 Risk Bucket Segmentation

Each strategy must have:

- Independent drawdown limit
- Independent daily loss limit
- Independent risk multiplier

If strategy DD exceeds limit:
reduce its allocation to near zero.

No cross-subsidizing.

---

# 4. Monte Carlo Risk-of-Ruin Framework

---

# 4.1 Purpose

Backtest is not enough.

We must simulate path dependency.

---

# 4.2 Inputs

- Trade return distribution (R multiples)
- Win rate
- Loss clustering frequency
- Volatility regime frequency

---

# 4.3 Simulation

Run 10,000 simulated equity paths:

Randomize:
- Trade sequence
- Regime transitions
- Volatility clustering

Compute:

- Max drawdown distribution
- Risk of ruin probability
- 95th percentile worst-case DD
- Capital survival probability over N trades

---

# 4.4 Acceptance Criteria

Strategy acceptable only if:

- Risk of ruin < 2%
- 95th percentile DD < acceptable threshold
- Recovery time < acceptable duration

---

# 5. Dynamic Risk Throttling

---

# 5.1 Drawdown-Based Throttle

If Portfolio_DD > 50% daily_limit:
  risk_global *= 0.8

If > 75%:
  risk_global *= 0.5

Never increase risk to "recover".

---

# 5.2 Volatility Shock Detection

If realized volatility > 95th percentile:

- Reduce risk 30%
- Tighten trailing
- Increase regime penalty sensitivity

---

# 6. Institutional Dashboard Extension

Add portfolio-level metrics:

- Strategy-level PF
- Cross-correlation heatmap
- Rolling Sharpe by strategy
- DD per strategy
- Risk contribution by strategy
- Monte Carlo survival curve
- Regime frequency breakdown

---

# 7. Failure Control Matrix

Immediate allocation freeze if:

- Strategy DD > defined limit
- Monte Carlo risk-of-ruin > 5%
- Correlation spike across modules
- Liquidity deterioration extreme

---

# 8. Evolution Roadmap

Stage 1:
Manual regime mapping.

Stage 2:
Semi-automatic parameter switching.

Stage 3:
Meta-allocator dynamic capital shifting.

Stage 4:
Adaptive reinforcement-style allocation (non-risk parameters only).

Never allow:
Self-modifying base risk parameter.

---

# 9. Strategic Outcome

After full implementation:

MasterTrading becomes:

- Multi-strategy adaptive portfolio
- Regime-aware capital allocator
- Risk-contained quant system
- Volatility-adaptive execution engine
- Institutional-level monitoring framework

This is no longer a bot.

It becomes a structured crypto quant engine.

---

END OF DOCUMENT