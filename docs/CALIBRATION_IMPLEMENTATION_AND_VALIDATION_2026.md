# MASTERTRADING — Calibration Implementation & Validation Framework
Version: 2026-03  
Type: Technical Execution & Statistical Validation Blueprint  
Audience: Quant Developers + LLM Systems  

---

# 1. Document Scope

This document complements the main calibration plan.

It defines:

- Technical implementation checklist
- Deployment safety sequencing
- Backtest protocol
- Forward validation protocol
- Statistical validation framework
- Monitoring metrics
- BTC-specific hardening model
- Portfolio-level extension logic
- LLM ingestion structure guidance

This document is execution-oriented.

---

# 2. Implementation Checklist (Deploy-Ready)

All changes must follow this order.

---

## Phase 0 — Safety Guard (Mandatory Before Any Edge Changes)

### 0.1 Fix Drawdown Baseline

Tasks:

- Persist daily equity baseline in DB.
- Compute DD only from stable baseline snapshot.
- Prevent baseline reset on container restart.
- Add dedup logic to RiskEvent emission.

Verification:

- Simulate container restart.
- Ensure daily DD does not jump to -100%.
- Ensure repeated cycles do not spam identical events.

No other calibration changes before this passes.

---

## Phase 1 — Low-Risk Edge Corrections

### 1.1 Implement TP Asymmetry

Modify `_compute_tp_sl_prices`:

Add:
- ATR_MULT_TP_LONG
- ATR_MULT_TP_SHORT

Validation:
- Compare PF by direction.
- Ensure no unintended SL distortion.

---

### 1.2 BTC Risk Hardening

Modify risk policy:

Option A (static):
PER_INSTRUMENT_RISK["BTCUSDT"] reduced.

Option B (adaptive):
If atr_pct high → reduce effective risk.

Validation:
- Compare BTC max loss cluster before/after.
- Check exposure caps respected.

---

## Phase 2 — Exit Optimization

### 2.1 Volatility-Adaptive Trailing

Modify:
- TRAILING_STOP_ACTIVATION_R logic.
- Dynamic lock-in percent.

Validation:
- MFE_capture_ratio improvement.
- Reduced giveback events.
- No early stop clustering.

---

### 2.2 Regime Direction Penalty

Modify allocator or signal score.

If regime == bear:
    reduce long score or weight.

Validation:
- Long trade count decreases in bearish regime.
- PF improves per regime segmentation.

---

## Phase 3 — Allocator Enhancement

Modify weight formula:

weight = f(winrate, expectancy, regime_factor, return_volatility)

Constraints:
- Normalize weights.
- Hard cap max module weight.

Validation:
- Compare module-level expectancy.
- Check no overweight concentration.

---

# 3. Backtest Protocol (Non-Overfitted)

All calibration changes must pass:

---

## 3.1 Walk-Forward Structure

Split data:

- Train window: 60%
- Validation window: 20%
- OOS window: 20%

Roll forward at least 3 cycles.

Never tune on OOS.

---

## 3.2 Regime Segmentation

Backtest separately on:

- Bull regime
- Bear regime
- High volatility
- Low volatility
- London session
- NY session
- Dead zone

Measure PF, Sharpe, DD per regime.

---

## 3.3 Stability Requirement

Calibration accepted only if:

- PF improves in >= 3 of 4 major regimes.
- Max DD does not increase.
- Variance of returns decreases OR remains stable.
- Expectancy improves or remains stable.

---

# 4. Forward Validation Protocol

Minimum forward test duration:

- 2–4 weeks paper mode.

Conditions:

- No parameter adjustment during test.
- Log MFE, MAE, exit reasons.
- Log regime classification.

If forward PF < backtest PF by > 25%:
Reject change.

---

# 5. Statistical Validation Framework

Each version must compute:

- Profit Factor
- Expectancy
- Std Dev of returns
- Max Drawdown
- Sharpe ratio (approx)
- Ulcer index
- Skewness of returns
- R-multiple distribution histogram

Additionally:

MFE_capture_ratio
MAE_distribution
Trade duration distribution

Reject calibration if:

- Distribution skew worsens materially.
- Tail losses increase.
- Return variance spikes.

---

# 6. BTC Hardened Model (Dedicated Section)

BTC requires separate calibration logic due to:

- Higher volatility clustering
- Liquidity profile
- News-driven spikes

Recommended constraints:

- Lower risk per trade
- Earlier trailing activation
- Slightly tighter SL floor
- Stronger regime filter

Optional:

Disable BTC longs when:
4H trend bearish AND ADX strong.

BTC must never exceed:
25% of total open risk.

---

# 7. Portfolio Strategy Separation Model

Long-term improvement:

Separate risk buckets per strategy:

- SMC Swing
- Mean Reversion
- Future Scalp module

Each strategy:

- Independent DD limit
- Independent risk multiplier
- Independent reporting

Never mix results in single equity curve internally.

---

# 8. Monitoring Dashboard Requirements

Add to daily report:

- PF by direction
- PF by symbol
- PF by regime
- MFE_capture_ratio
- Exit reason breakdown
- BTC exposure percent
- Avg R per strategy
- Risk utilization percent

Alert triggers:

- MFE_capture_ratio < 0.45
- 3 consecutive SL cluster BTC
- Regime misalignment detection

---

# 9. LLM Context Ingestion Structure

When used as context file:

Keep sections in this order:

1. Constraints
2. Current Calibration
3. Pending Improvements
4. Validation Rules
5. Safety Rules
6. Deployment Sequence

LLM must never propose:

- Risk increase
- Leverage increase
- Multi-axis simultaneous change

---

# 10. Failure Conditions

Abort deployment if:

- Worker crashes occur.
- Unexpected spike in rejected orders.
- Margin errors increase.
- RiskEvent spam resumes.
- Exchange reconciliation mismatches increase.

Rollback immediately if:

- 2 consecutive days of abnormal DD.
- BTC cluster loss > historical percentile 95%.

---

# 11. Quantitative Target Envelope

Acceptable improvement band:

PF: +0.10 minimum improvement  
Sharpe: +0.20 improvement  
Max DD: equal or lower  
Return variance: equal or lower  

Reject if:

Win rate improves but DD increases materially.

---

# 12. Scalping Readiness Gate (Future)

Before enabling scalp module:

- At least 3 months consistent 1m data.
- Realistic slippage model.
- Separate risk bucket.
- OOS validation across sessions.
- Independent DD limit.

Scalp must not share risk pool with swing.

---

# 13. Change Discipline Rules

For every deploy:

Record:

- Objective
- Files modified
- Risk expectation
- Backtest results
- Forward validation status
- Rollback commit

Never deploy:
- More than one risk-axis change simultaneously.

---

# 14. Expected Outcome

If executed correctly:

- Lower BTC loss concentration
- Higher MFE capture
- Improved directional asymmetry
- Reduced equity volatility
- Stable PF above 1.25
- Controlled max DD under 20%
- Higher regime adaptability

---

END OF DOCUMENT