# MASTERTRADING_AI_CORE_V2026
FORMAT: TOON
MODE: MINIMAL_CORE
INTENT: EXECUTION_RISK_REGIME_ALLOCATOR

============================================================
SECTION: HARD_CONSTRAINTS
============================================================

NO:
- BASE_RISK_INCREASE
- LEVERAGE_INCREASE
- MULTI_RISK_AXIS_CHANGE
- GLOBAL_SCORE_REDUCTION
- EXPOSURE_CAP_INCREASE
- RECOVERY_RISK_ESCALATION

ALWAYS:
- DD_THROTTLE
- REGIME_TAGGING
- BOUNDED_MULTIPLIERS
- MONTE_CARLO_VALIDATION
- ONE_PHASE_PER_DEPLOY

============================================================
SECTION: REGIME_ENGINE
============================================================

INPUTS:
- ATR_PCTL_90D
- TREND_4H
- ADX_4H
- HMM_STATE

IF TREND=bear AND ATR_PCTL_90D≥70 AND ADX≥20 → BEAR_EXP
IF TREND=bear → BEAR_COMP
IF TREND=bull AND ATR_PCTL_90D≥70 AND ADX≥20 → BULL_EXP
IF TREND=bull → BULL_COMP
IF ATR_PCTL_90D≤35 → RANGE
ELSE → CHOPPY

risk_mult ∈ [0.50,1.00]
long_penalty ∈ [0.00,0.30]
short_penalty ∈ [0.00,0.30]
trailing_R ∈ [1.2,2.8]

FAILSAFE:
REGIME_ERROR → CHOPPY + risk_mult=0.70

============================================================
SECTION: RISK_MODEL
============================================================

BASE:
R_amt = equity * RISK_PER_TRADE_PCT

VOL_SCALE:
ATR_LOW → scale=1.0
ATR_HIGH → scale=VOL_RISK_MIN_SCALE

DD_THROTTLE:
x=|DD|/DAILY_LIMIT
x≥0.75 → 0.50
x≥0.50 → 0.80
else → 1.00

BTC_HARDEN:
IF symbol=BTC → r_eff*=BTC_MULT≤1.0

FINAL:
r_eff = base_r * vol_scale * regime_mult * dd_mult

============================================================
SECTION: TP_SL
============================================================

TP_LONG = ATR * ATR_MULT_TP_LONG
TP_SHORT = ATR * ATR_MULT_TP_SHORT
SL = ATR * ATR_MULT_SL

CONSTRAINT:
ATR_MULT_TP_LONG < ATR_MULT_TP_SHORT

============================================================
SECTION: TRAILING
============================================================

IF ATR_PCTL_HIGH → trailing_R=1.5
ELSE → trailing_R=2.5

lock_in ∈ [0.40,0.70]

TARGET:
MFE_CAPTURE ≥ 0.60
ALERT IF <0.45

============================================================
SECTION: META_ALLOCATOR
============================================================

INPUT:
- expectancy_R
- stdev_R
- pf
- regime_fit
- loss_cluster
- corr_penalty

RAW_WEIGHT =
expectancy_norm *
vol_inverse *
pf_norm *
regime_fit *
(1 - 0.5*loss_cluster) *
corr_penalty

CAP:
weight≤W_MAX

NORMALIZE:
w_i = w_i / Σw_i

ALLOC:
R_i = R_available * w_i

PREFERRED:
SINGLE_STRATEGY_WINNER

============================================================
SECTION: MONTE_CARLO
============================================================

N_PATHS≥10000
SIMULATE:
- regime_transitions
- bootstrap_R_by_regime
- loss_clusters
- vol_shocks

RUIN:
equity≤E0*(1-RUIN_PCT) OR DD≥MAX_DD

ACCEPT_IF:
P_ruin≤2%
DD_95pct≤threshold
PF_new≥PF_old
Sharpe_new≥Sharpe_old

============================================================
SECTION: MONITORING
============================================================

DAILY:
- PF_by_direction
- PF_by_symbol
- PF_by_regime
- BTC_exposure_pct
- MFE_CAPTURE
- Strategy_DD
- Risk_utilization

RED_FLAGS:
- MFE_CAPTURE<0.45
- BTC_cluster
- DD_spike
- Correlation_spike

============================================================
END_OF_TOON_CONTEXT
============================================================