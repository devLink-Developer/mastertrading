# MASTERTRADING — Meta-Allocator Pseudocode & Monte Carlo Risk Framework
Version: 2026-03
Type: Implementation Spec (Pseudo-code + Quant Validation)
Audience: Quant Engineers + AI Systems

---

## 0) Non-Negotiables

- NEVER increase base `RISK_PER_TRADE_PCT`.
- NEVER increase effective leverage caps.
- NEVER deploy multi-axis risk changes in one patch.
- Any adaptive logic must be:
  - deterministic given the same inputs,
  - bounded by strict caps,
  - fail-safe with conservative fallbacks.

---

## 1) Data Contracts (Strategy → Meta Allocator)

Each strategy `s` must emit a `StrategySignal` per instrument per cycle:

### 1.1 StrategySignal schema (minimum)

- `strategy_id`: string  (e.g., "smc_swing", "meanrev", "carry", "scalp_v1")
- `instrument`: string   (e.g., "BTCUSDT")
- `direction`: enum      ("long", "short", "flat")
- `score`: float         [0..1]  (internal confidence)
- `edge_estimate_r`: float  (expected return in R units, can be negative)
- `risk_demand`: float    [0..1]  (relative; 1 means wants full allocated risk budget for this strategy)
- `regime_fit`: float     [0..1]  (how suitable this strategy is for current regime)
- `constraints`:
  - `min_hold_minutes`: int
  - `max_hold_minutes`: int
  - `cooldown_minutes`: int
  - `symbol_blocked`: bool
- `meta`:
  - `timestamp_utc`: ISO
  - `features_hash`: string (optional for audit)
  - `reason`: short text (optional)

### 1.2 StrategyStats schema (rolling window, maintained by system)

Per `strategy_id` (optionally per instrument too):

- `win_rate`: float
- `expectancy_r`: float
- `stdev_r`: float
- `pf`: float
- `max_dd_pct`: float
- `loss_cluster_score`: float   (0..1)
- `mfe_capture_ratio`: float
- `correlation_to_portfolio`: float (optional)
- `sample_size`: int

---

## 2) Regime Engine (Deterministic)

### 2.1 Inputs

Per instrument:

- `atr_pct`: ATR / close
- `atr_pct_percentile_90d`: percentile of `atr_pct` within rolling 90d
- `trend_4h`: enum ("bull", "bear", "neutral")
- `adx_4h`: float
- `hmm_state`: enum ("trending", "choppy")
- `spread_pct_percentile`: percentile (liquidity proxy) (optional)
- `funding_abs`: abs(funding) (optional)

### 2.2 Regime states (example)

- `BULL_COMPRESSION`
- `BULL_EXPANSION`
- `BEAR_COMPRESSION`
- `BEAR_EXPANSION`
- `RANGE_LOWVOL`
- `CHOPPY_HIGHVOL`
- `LIQUIDITY_THIN`

### 2.3 Regime classification function

Pseudo-code:

function classify_regime(inputs):
vol = inputs.atr_pct_percentile_90d
trend = inputs.trend_4h
adx = inputs.adx_4h
hmm = inputs.hmm_state
liq_thin = (inputs.spread_pct_percentile >= 95) if available else false

if liq_thin:
    return LIQUIDITY_THIN

if hmm == "choppy" and vol >= 80:
    return CHOPPY_HIGHVOL

if trend == "bull":
    if vol >= 70 and adx >= 20:
        return BULL_EXPANSION
    else:
        return BULL_COMPRESSION

if trend == "bear":
    if vol >= 70 and adx >= 20:
        return BEAR_EXPANSION
    else:
        return BEAR_COMPRESSION

# trend neutral
if vol <= 35:
    return RANGE_LOWVOL
else:
    return CHOPPY_HIGHVOL

### 2.4 Regime parameter map (bounded)

Each regime returns a bounded parameter vector:

- `risk_mult`: float in [0.50, 1.00]
- `trailing_activation_r`: float in [1.2, 2.8]
- `long_penalty`: float in [0.00, 0.30]
- `short_penalty`: float in [0.00, 0.30]
- `tp_mult_long`: float in [1.2, 2.2]
- `tp_mult_short`: float in [1.6, 2.6]
- `strategy_fit_overrides`: dict(strategy_id → multiplier in [0.0, 1.2])

Example (illustrative):

- BEAR_EXPANSION:
  - risk_mult=0.85
  - trailing_activation_r=1.5
  - long_penalty=0.20
  - short_penalty=0.00
  - tp_mult_long=1.5
  - tp_mult_short=2.3
  - strategy_fit_overrides: {"meanrev":0.7, "smc_swing":1.0, "carry":1.0}

- RANGE_LOWVOL:
  - risk_mult=0.75
  - trailing_activation_r=2.2
  - long_penalty=0.10
  - short_penalty=0.10
  - tp_mult_long=1.4
  - tp_mult_short=1.8
  - strategy_fit_overrides: {"meanrev":1.1, "smc_swing":0.8}

---

## 3) Meta-Allocator (Strategy-Level + Instrument-Level)

### 3.1 Objective

Allocate risk budget across strategies (and optionally across instruments) to maximize expected return per unit risk, subject to:

- global risk cap,
- per-strategy drawdown caps,
- per-instrument exposure caps,
- correlation overlap control,
- regime suitability.

### 3.2 Risk Budget Definitions

- `R_global` = base risk budget per trade (fixed; non-negotiable)
- `R_regime` = `R_global * regime.risk_mult`
- `R_dd_throttle` = drawdown throttle multiplier (<= 1)
- `R_available` = `R_regime * R_dd_throttle`

### 3.3 Drawdown throttle (portfolio-level)


function dd_throttle(dd_pct, daily_dd_limit_pct):
# dd_pct is negative number (e.g., -0.06 for -6%)
x = abs(dd_pct) / daily_dd_limit_pct
if x >= 0.75: return 0.50
if x >= 0.50: return 0.80
return 1.00


### 3.4 Strategy eligibility gate

A strategy is eligible if:

- `stats.sample_size >= MIN_SAMPLE`  (e.g., 50 trades) OR allow with conservative prior
- `stats.max_dd_pct <= STRATEGY_DD_CAP`  (strategy-level)
- not in cooldown due to recent failures
- regime_fit >= MIN_REGIME_FIT (e.g., 0.40)
- emitted signal `direction != flat`
- instrument not blocked

### 3.5 Strategy weight formula (bounded and normalized)

Inputs per strategy `s`:

- `E_s`: expectancy_r (rolling)
- `V_s`: stdev_r (rolling)  (must be > 0)
- `W_s`: win_rate (rolling)
- `F_s`: regime_fit (signal)
- `C_s`: correlation penalty to portfolio / other strategies (optional)
- `L_s`: loss_cluster_score (0..1), higher means worse
- `P_s`: pf (rolling)

Define bounded components:

- `exp_factor = clamp((E_s - E_min) / (E_target - E_min), 0.0, 1.0)`
- `vol_factor = clamp(V_target / V_s, 0.5, 1.5)`
- `pf_factor = clamp(P_s / PF_target, 0.5, 1.5)`
- `fit_factor = clamp(F_s, 0.0, 1.2)`
- `cluster_penalty = clamp(1.0 - 0.5*L_s, 0.5, 1.0)`
- `corr_penalty = clamp(1.0 - C_s, 0.6, 1.0)`  (if C_s in [0..0.4] typical)

Raw weight:

`w_raw_s = exp_factor * vol_factor * pf_factor * fit_factor * cluster_penalty * corr_penalty`

Then apply hard caps:

`w_capped_s = min(w_raw_s, W_MAX)`  (e.g., 1.5)

Normalize:

`w_s = w_capped_s / sum(w_capped_all)`

Strategy risk allocation:

`R_s = R_available * w_s`

### 3.6 Instrument-level concentration guard

Given per-instrument exposure cap `EXPO_CAP`:

- compute total notional (or risk) for open positions + candidate
- ensure `instrument_risk_after <= EXPO_CAP * equity`

If violation:
- reduce `R_s` for that instrument
- or block new entry for that symbol

### 3.7 Direction penalty application

Adjust strategy's effective score if regime requires:

- `score_eff = score * (1 - long_penalty)` for longs
- `score_eff = score * (1 - short_penalty)` for shorts

If `score_eff < EXECUTION_MIN_SIGNAL_SCORE`: block entry.

### 3.8 Output: MetaDecision per instrument

For each instrument, choose the best eligible strategy (or blend if supported):

Option A (single winner):

- pick `s* = argmax(score_eff * w_s * edge_estimate_r)` subject to constraints
- allocate risk `R_s*`

Option B (blended positions) — only if execution supports multi-leg:
- allocate to top-k strategies with small k (<=2)
- ensure total risk <= R_available

Recommended default: Option A (simpler, safer).

---

## 4) Monte Carlo Risk-of-Ruin (Full Spec)

### 4.1 Why MC is required

Backtests are one realized path.
MC simulates path dependency and tail risk:

- loss clustering,
- regime transitions,
- correlation overlap,
- volatility shocks.

### 4.2 Inputs required (per strategy or aggregated portfolio)

- empirical distribution of trade returns in R units: `R_i`
- conditional distributions by regime: `R_i | regime`
- win rate by regime
- loss cluster model:
  - probability of consecutive losses
  - conditional loss magnitude escalation (optional)
- regime transition matrix `T` (Markov):
  - P(regime_j | regime_i)
- frequency distribution of regimes in historical sample
- correlation matrix between strategies (optional but recommended)

### 4.3 Return model options

Option 1 (bootstrap):
- sample `R` from historical trades (with regime conditioning)

Option 2 (parametric mixture):
- fit mixture of normals / t-distributions to R-multiples by regime

Prefer Option 1 initially (robust, fewer assumptions).

### 4.4 Regime transition model

Let regimes be `1..K`.

Transition matrix `T` where:
`T[i][j] = P(next_regime=j | current_regime=i)`

Simulated regime sequence:
- start from empirical prior distribution
- evolve by `T` each step

### 4.5 Volatility shock injection

Add shock states with probability `p_shock` (e.g., 1% per day) where:

- risk multiplier is reduced
- return distribution is more left-tailed
- slippage increases (optional)

### 4.6 Correlation-aware sampling (strategy portfolio)

If simulating multiple strategies simultaneously:

- sample latent market factor `Z ~ N(0,1)`
- for each strategy s:
  - `R_s = mu_s(regime) + beta_s*Z + eps_s`
  - `eps_s` sampled from residual distribution (bootstrap from residuals)

This approximates correlation without heavy copulas.

### 4.7 Equity path update

Equity evolves per trade:

`E_{t+1} = E_t * (1 + (R_trade * r_eff))`

Where:
- `R_trade` is in R multiples (e.g., +1.2R, -1.0R)
- `r_eff` is effective risk percent (bounded, includes regime + dd throttle)

Track:
- peak equity
- drawdown
- time-to-recovery

### 4.8 Ruin definition

Define ruin when any of:

- Equity drops below `E0 * (1 - RUIN_PCT)` (e.g., 50%)
- Or drawdown exceeds `MAX_DD_ACCEPTED` (e.g., 25%)
- Or margin constraints violated (optional model)

### 4.9 Monte Carlo algorithm (pseudo-code)


function monte_carlo(N_paths, N_trades, model):
results = []
for p in 1..N_paths:
E = E0
peak = E0
regime = sample_initial_regime()
max_dd = 0
ruined = false

    for t in 1..N_trades:
        # regime transition
        regime = sample_next_regime(regime, T)

        # apply shocks
        if rand() < p_shock:
            regime = SHOCK_REGIME

        # compute effective risk
        dd = (E - peak) / peak   # negative
        throttle = dd_throttle(dd, daily_dd_limit)
        r_eff = r_base * regime_risk_mult(regime) * throttle

        # sample return in R units
        R_trade = sample_return(regime, model)  # bootstrap or parametric

        # update equity
        E = E * (1 + R_trade * r_eff)

        if E > peak:
            peak = E
        dd_now = (E - peak) / peak
        max_dd = min(max_dd, dd_now)

        if abs(max_dd) >= MAX_DD_ACCEPTED or E <= E0*(1-RUIN_PCT):
            ruined = true
            break

    results.append({ "max_dd": max_dd, "final_E": E, "ruined": ruined })
return summarize(results)

### 4.10 Outputs to compute

- `P_ruin` = fraction ruined
- distribution of `max_dd`
- 95th percentile worst `max_dd`
- median final equity
- expected final equity (mean)
- recovery time distribution (optional)
- expected shortfall of DD (CVaR)

### 4.11 Acceptance criteria (portfolio-level)

Accept only if:

- `P_ruin <= 0.02` (2%)
- `DD_95pct <= DD_threshold` (e.g., 20%)
- expected equity growth positive
- improvement stable across regime sensitivity tests

---

## 5) Stress Test Matrix (Must Pass)

Run MC under:

1) Base regime frequencies  
2) Bear-heavy regime (increase bear states by +30%)  
3) High-vol shock frequency doubled  
4) Loss cluster frequency doubled  
5) Slippage increased (if modeled)  
6) Correlation spike (+0.2 on all off-diagonals)  

Calibration is rejected if it only works in Base.

---

## 6) Implementation Plan (Safe Sequencing)

1) Implement Regime Engine with deterministic mapping (no allocator change yet)
2) Add regime tagging to logs + reports
3) Implement DD baseline normalization (P0)
4) Add regime-based penalties (bounded)
5) Add strategy stats tracking
6) Implement Meta-Allocator in shadow mode (log decisions only)
7) Enable Meta-Allocator gating gradually:
   - start with allocation weight only (no blending)
   - keep conservative caps
8) Add Monte Carlo job to nightly pipeline (offline)
9) Promote to live only after forward validation

---

## 7) Logging & Audit Requirements (Mandatory)

Log every cycle:

- regime state per instrument
- strategy candidates with:
  - score, edge_estimate_r, regime_fit
- chosen strategy and allocated risk
- throttle state and dd snapshot
- reason codes for blocks
- post-trade outcomes:
  - realized R, MFE, MAE, duration

This is required to close the loop.

---

## 8) Fail-Safes

If Regime Engine fails:
- fallback regime = CHOPPY_HIGHVOL
- risk_mult = 0.70
- trailing_activation_r = 1.5
- apply conservative penalties

If Meta-Allocator fails:
- fallback to existing allocator behavior
- do NOT block execution unless explicit fail-safe is intended

---

END OF DOCUMENT
