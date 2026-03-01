# MASTERTRADING_AI_CONTEXT_V2026
FORMAT: TOON
MODE: TOKEN_OPTIMIZED
INTENT: CALIBRATION + REGIME + META_ALLOCATOR + RISK_DISCIPLINE

============================================================
SECTION: HARD_CONSTRAINTS
============================================================

NO:
- RISK_PER_TRADE increase
- LEVERAGE increase
- MULTI_AXIS_RISK_CHANGE per deploy
- GLOBAL_MIN_SIGNAL_SCORE decrease without regime segmentation
- EXPOSURE_CAP increase
- SELF_MODIFY_BASE_RISK
- RECOVERY_RISK_ESCALATION

ALWAYS:
- WALK_FORWARD_VALIDATION
- OOS_VALIDATION
- FORWARD_PAPER_MIN_2W
- MONTE_CARLO_CHECK
- LOG_REGIME_PER_CYCLE
- BOUNDED_MULTIPLIERS

============================================================
SECTION: CORE_OBJECTIVE
============================================================

OPTIMIZE:
- PROFIT_FACTOR
- EXPECTANCY_R
- MFE_CAPTURE_RATIO
- REGIME_ALIGNMENT
- BTC_LOSS_CLUSTER_REDUCTION
- EQUITY_SMOOTHNESS

WITHOUT:
- RISK_EXPANSION
- SIGNAL_QUALITY_DEGRADATION
- VOLATILITY_EXPOSURE_INCREASE

============================================================
SECTION: REGIME_ENGINE
============================================================

INPUTS:
- ATR_PCT
- ATR_PERCENTILE_90D
- TREND_4H
- ADX_4H
- HMM_STATE
- SPREAD_PERCENTILE (optional)

REGIME_CLASSIFICATION:

IF TREND=bear AND ATR_PCTL>=70 AND ADX>=20 → BEAR_EXPANSION
IF TREND=bear → BEAR_COMPRESSION
IF TREND=bull AND ATR_PCTL>=70 AND ADX>=20 → BULL_EXPANSION
IF TREND=bull → BULL_COMPRESSION
IF TREND=neutral AND ATR_PCTL<=35 → RANGE_LOWVOL
ELSE → CHOPPY_HIGHVOL

REGIME_PARAMS_BOUNDS:

risk_mult ∈ [0.50,1.00]
trailing_R ∈ [1.2,2.8]
long_penalty ∈ [0.00,0.30]
short_penalty ∈ [0.00,0.30]
tp_mult_long ∈ [1.2,2.2]
tp_mult_short ∈ [1.6,2.6]

FAILSAFE:
IF regime_engine_error → CHOPPY_HIGHVOL + risk_mult=0.70

============================================================
SECTION: RISK_MODEL
============================================================

BASE:
R_amt = equity * RISK_PER_TRADE_PCT

VOL_SCALING:
IF ATR_PCT <= LOW_THR → scale=1.0
IF ATR_PCT >= HIGH_THR → scale=VOL_RISK_MIN_SCALE
ELSE → linear_interpolation

BTC_HARDENING:
IF instrument=BTC → r_eff *= BTC_MULT ≤ 1.0

DD_THROTTLE:
x = |DD| / DAILY_DD_LIMIT
IF x>=0.75 → mult=0.50
IF x>=0.50 → mult=0.80
ELSE → mult=1.00

FINAL_RISK:
r_eff = base_r * vol_scale * regime_risk_mult * dd_mult

============================================================
SECTION: TP_SL_MODEL
============================================================

TP_LONG = max(base_tp, ATR_PCT * ATR_MULT_TP_LONG)
TP_SHORT = max(base_tp, ATR_PCT * ATR_MULT_TP_SHORT)
SL = max(base_sl, ATR_PCT * ATR_MULT_SL)

ASYMMETRY:
ATR_MULT_TP_LONG < ATR_MULT_TP_SHORT

============================================================
SECTION: TRAILING_MODEL
============================================================

IF ATR_PCT high → trailing_activation_R=1.5
ELSE → trailing_activation_R=2.5

lock_in_pct = clamp(0.4 + k*ATR_PCT, 0.4, 0.7)

MFE_CAPTURE_TARGET ≥ 0.60
ALERT IF < 0.45

============================================================
SECTION: META_ALLOCATOR
============================================================

INPUT_PER_STRATEGY:
- win_rate
- expectancy_R
- stdev_R
- pf
- regime_fit
- loss_cluster_score
- correlation_penalty

RAW_WEIGHT =
clamp(expectancy_norm,0,1) *
clamp(vol_inverse,0.5,1.5) *
clamp(pf_norm,0.5,1.5) *
regime_fit *
(1 - 0.5*loss_cluster_score) *
correlation_penalty

CAP:
weight ≤ W_MAX

NORMALIZE:
w_i = w_i / Σ w_i

ALLOCATED_RISK:
R_i = R_available * w_i

OPTION:
SINGLE_STRATEGY_WINNER (preferred)
NO_BLENDING unless validated

============================================================
SECTION: MONTE_CARLO
============================================================

N_PATHS ≥ 10,000
SIMULATE:
- regime_transitions (Markov)
- R_distribution_bootstrap_by_regime
- loss_clustering
- vol_shocks
- correlation_factor

RUIN:
IF equity <= E0*(1-RUIN_PCT) OR DD>=MAX_DD_ACCEPTED

ACCEPT_IF:
P_ruin ≤ 2%
DD_95pct ≤ threshold
Sharpe_new ≥ Sharpe_old
PF_new ≥ PF_old

RUN_STRESS:
- Bear-heavy regime
- Double vol shock
- Double loss cluster
- Correlation spike

============================================================
SECTION: PORTFOLIO_RULES
============================================================

Each strategy:
- Independent DD limit
- Independent risk bucket
- Independent reporting

NO:
Cross-subsidy of DD
Shared recovery risk escalation

============================================================
SECTION: MONITORING
============================================================

DAILY:
- PF_by_direction
- PF_by_symbol
- PF_by_regime
- BTC_exposure_pct
- MFE_capture_ratio
- Strategy_DD
- Risk_utilization_pct

RED_FLAGS:
- MFE_capture <0.45
- BTC_loss_cluster
- RiskEvent_spike
- Correlation_spike

============================================================
SECTION: DEPLOY_SEQUENCE
============================================================

1. Fix_DD_Baseline
2. Enable_Regime_Tagging
3. Apply_TP_Asymmetry
4. Apply_BTC_Hardening
5. Enable_Trailing_Adaptive
6. MetaAllocator_ShadowMode
7. MetaAllocator_Live_Bounded
8. MonteCarlo_Nightly

ONE_PHASE_PER_DEPLOY

============================================================
END_OF_TOON_CONTEXT
============================================================