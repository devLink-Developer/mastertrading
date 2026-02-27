from __future__ import annotations

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "changeme-in-prod")
if not DEBUG and SECRET_KEY.strip() in {"", "changeme-in-prod", "change-me"}:
    warnings.warn(
        "Insecure SECRET_KEY detected with DEBUG=false. Set a strong SECRET_KEY in environment.",
        RuntimeWarning,
    )

USE_SQLITE = os.getenv("USE_SQLITE", "false").lower() == "true"
MODE = os.getenv("MODE", "paper").lower()
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "true").lower() == "true"
ORDER_SIZE_USDT = float(os.getenv("ORDER_SIZE_USDT", "1.0"))
MIN_EQUITY_USDT = float(os.getenv("MIN_EQUITY_USDT", "5.0"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.018"))   # 1.8% default TP floor (was 2% — tighter for earlier captures)
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.015"))     # 1.5% default SL floor (was 1.2% — wider to avoid noise stop-outs)
TP_SL_FEE_ADJUST_ENABLED = os.getenv("TP_SL_FEE_ADJUST_ENABLED", "true").lower() == "true"
TP_SL_ESTIMATED_ROUNDTRIP_FEE_PCT = max(
    0.0,
    float(os.getenv("TP_SL_ESTIMATED_ROUNDTRIP_FEE_PCT", "0.0010")),
)
TAKE_PROFIT_DYNAMIC_MULT = max(
    0.1,
    min(2.0, float(os.getenv("TAKE_PROFIT_DYNAMIC_MULT", "1.0"))),
)  # <1.0 = closer TP for earlier profit capture
TAKE_PROFIT_MIN_PCT = max(
    0.0,
    float(os.getenv("TAKE_PROFIT_MIN_PCT", "0.006")),
)  # hard floor after dynamic multipliers
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", "20"))      # 20 bps = 0.2%
SPREAD_DYNAMIC_BY_ATR_ENABLED = os.getenv("SPREAD_DYNAMIC_BY_ATR_ENABLED", "true").lower() == "true"
SPREAD_ATR_RELAX_FACTOR = float(os.getenv("SPREAD_ATR_RELAX_FACTOR", "0.12"))
MAX_DYNAMIC_SPREAD_BPS = float(os.getenv("MAX_DYNAMIC_SPREAD_BPS", "60"))
MAX_EFF_LEVERAGE = float(os.getenv("MAX_EFF_LEVERAGE", "2.0"))   # was 3.0 — reduce leverage to limit loss amplitude
ATR_MULT_TP = float(os.getenv("ATR_MULT_TP", "2.5"))           # 2.5x ATR TP (was 3x — more reachable, take profits earlier)
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "2.0"))           # 2.0x ATR SL (was 1.5x — give trades room to breathe)
MIN_ATR_FOR_ENTRY = float(os.getenv("MIN_ATR_FOR_ENTRY", "0.003"))  # Skip entries below this ATR% (ratio)
MIN_SL_PCT = float(os.getenv("MIN_SL_PCT", "0.012"))           # Absolute minimum SL of 1.2% regardless of ATR (was 0.8% — too tight for BTC)
ENTRY_VOLUME_FILTER_ENABLED = os.getenv("ENTRY_VOLUME_FILTER_ENABLED", "false").lower() == "true"
ENTRY_VOLUME_FILTER_TIMEFRAME = os.getenv("ENTRY_VOLUME_FILTER_TIMEFRAME", "5m").strip().lower()
if ENTRY_VOLUME_FILTER_TIMEFRAME not in {"1m", "5m", "15m", "1h", "4h"}:
    ENTRY_VOLUME_FILTER_TIMEFRAME = "5m"
ENTRY_VOLUME_FILTER_LOOKBACK = max(10, min(240, int(os.getenv("ENTRY_VOLUME_FILTER_LOOKBACK", "48"))))
ENTRY_VOLUME_FILTER_MIN_RATIO = max(0.0, float(os.getenv("ENTRY_VOLUME_FILTER_MIN_RATIO", "0.75")))
_ENTRY_VOLUME_FILTER_MIN_RATIO_BY_SESSION_RAW = os.getenv(
    "ENTRY_VOLUME_FILTER_MIN_RATIO_BY_SESSION",
    "{}",
)
ENTRY_VOLUME_FILTER_FAIL_OPEN = os.getenv("ENTRY_VOLUME_FILTER_FAIL_OPEN", "true").lower() == "true"
DAILY_DD_LIMIT = float(os.getenv("DAILY_DD_LIMIT", "0.05"))     # 5% daily max drawdown (was 100% disabled — now active)
_allowed_hosts_raw = os.getenv("ALLOWED_HOSTS", "127.0.0.1,localhost")
ALLOWED_HOSTS = [h.strip() for h in _allowed_hosts_raw.split(",") if h.strip()]
if not ALLOWED_HOSTS:
    ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
if not DEBUG and "*" in ALLOWED_HOSTS:
    warnings.warn(
        "ALLOWED_HOSTS contained '*' with DEBUG=false; falling back to localhost-only hosts.",
        RuntimeWarning,
    )
    ALLOWED_HOSTS = ["127.0.0.1", "localhost"]

# -- New: Risk-based position sizing --
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.0075"))  # 0.75% of equity risked per trade (was 1% — tighter risk per trade)
ORDER_MARGIN_BUFFER_PCT = float(os.getenv("ORDER_MARGIN_BUFFER_PCT", "0.03"))  # reserve margin for fees/slippage
ORDER_MARGIN_BUFFER_MAX_PCT = max(
    0.0,
    min(1.0, float(os.getenv("ORDER_MARGIN_BUFFER_MAX_PCT", "0.20"))),
)
MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "0.85"))  # Minimum signal score to open (was 0.80 — only take high-confidence signals)
EXECUTION_MIN_SIGNAL_SCORE = float(
    os.getenv("EXECUTION_MIN_SIGNAL_SCORE", str(MIN_SIGNAL_SCORE))
)  # extra safety gate at execution time
MAX_EXPOSURE_PER_INSTRUMENT_PCT = float(os.getenv("MAX_EXPOSURE_PER_INSTRUMENT_PCT", "0.25"))  # max 25% equity per instrument (was 33% — reduce concentration risk)
VOL_RISK_LOW_ATR_PCT = max(0.0, float(os.getenv("VOL_RISK_LOW_ATR_PCT", "0.008")))
VOL_RISK_HIGH_ATR_PCT = max(VOL_RISK_LOW_ATR_PCT + 1e-9, float(os.getenv("VOL_RISK_HIGH_ATR_PCT", "0.015")))
VOL_RISK_MIN_SCALE = max(0.0, min(1.0, float(os.getenv("VOL_RISK_MIN_SCALE", "0.6"))))

# -- New: Signal TTL --
SIGNAL_TTL_SECONDS = int(os.getenv("SIGNAL_TTL_SECONDS", "300"))  # 5 min, stale signals are ignored
SIGNAL_DEDUP_SECONDS = int(os.getenv("SIGNAL_DEDUP_SECONDS", "120"))  # prevent duplicate same-direction signals

# -- New: Trailing stop / partial close --
TRAILING_STOP_ENABLED = os.getenv("TRAILING_STOP_ENABLED", "true").lower() == "true"
TRAILING_STOP_ACTIVATION_R = float(os.getenv("TRAILING_STOP_ACTIVATION_R", "1.5"))  # activate trail after 1.5R (was 2.5 — start trailing sooner to protect gains)
# Once trailing is active, lock this fraction of the max favorable move as a dynamic SL.
# Example: 0.5 locks half of the max profit (aggressive profit-protection).
TRAILING_STOP_LOCK_IN_PCT = float(os.getenv("TRAILING_STOP_LOCK_IN_PCT", "0.6"))  # lock 60% of HWM (was 50% — more aggressive profit protection)
BREAKEVEN_STOP_ENABLED = os.getenv("BREAKEVEN_STOP_ENABLED", "true").lower() == "true"
BREAKEVEN_STOP_AT_R = float(os.getenv("BREAKEVEN_STOP_AT_R", "0.75"))  # move SL to entry after 0.75R in profit (was 1.0 — protect capital earlier)
BREAKEVEN_STOP_OFFSET_PCT = float(os.getenv("BREAKEVEN_STOP_OFFSET_PCT", "0.001"))  # 0.1% buffer above entry to cover fees/slippage (was 0 — losing on BE)
BREAKEVEN_WINDOW_MINUTES = int(os.getenv("BREAKEVEN_WINDOW_MINUTES", "0"))  # 0 = disabled (no time filter)
TRAILING_STATE_TTL_SECONDS = max(60, int(os.getenv("TRAILING_STATE_TTL_SECONDS", "172800")))
TRAILING_SL_MIN_MOVE_PCT = max(0.0, float(os.getenv("TRAILING_SL_MIN_MOVE_PCT", "0.0002")))
VOL_FAST_EXIT_ENABLED = os.getenv("VOL_FAST_EXIT_ENABLED", "false").lower() == "true"
VOL_FAST_EXIT_ATR_PCT = max(0.0, float(os.getenv("VOL_FAST_EXIT_ATR_PCT", "0.012")))
VOL_FAST_EXIT_TP_MULT = max(0.1, min(1.0, float(os.getenv("VOL_FAST_EXIT_TP_MULT", "0.75"))))
VOL_FAST_EXIT_MIN_TP_PCT = max(0.0, float(os.getenv("VOL_FAST_EXIT_MIN_TP_PCT", "0.006")))
VOL_FAST_EXIT_TRAIL_R_MULT = max(0.1, min(1.0, float(os.getenv("VOL_FAST_EXIT_TRAIL_R_MULT", "0.75"))))
VOL_FAST_EXIT_PARTIAL_R_MULT = max(0.1, min(1.0, float(os.getenv("VOL_FAST_EXIT_PARTIAL_R_MULT", "0.80"))))
SIGNAL_COOLDOWN_MINUTES = int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "1440"))  # 24h cooldown between trades per instrument
SIGNAL_COOLDOWN_AFTER_SL_MINUTES = int(os.getenv("SIGNAL_COOLDOWN_AFTER_SL_MINUTES", "360"))  # 6h cooldown after SL — NOTE: faster than 24h default, but SLOWER if .env sets SIGNAL_COOLDOWN_MINUTES < 360
PARTIAL_CLOSE_AT_R = float(os.getenv("PARTIAL_CLOSE_AT_R", "0.8"))  # close 50% at 0.8R (was 1.0 — secure partial profits earlier)
PARTIAL_CLOSE_PCT = float(os.getenv("PARTIAL_CLOSE_PCT", "0.5"))  # close 50% of position
PARTIAL_CLOSE_MIN_REMAINING_QTY = max(
    0.0,
    float(os.getenv("PARTIAL_CLOSE_MIN_REMAINING_QTY", "0.0")),
)
POSITION_QTY_EPSILON = max(0.0, float(os.getenv("POSITION_QTY_EPSILON", "1e-12")))
POSITION_OPENED_FALLBACK_MAX_HOURS = max(
    1,
    int(os.getenv("POSITION_OPENED_FALLBACK_MAX_HOURS", "72")),
)
EXCHANGE_CLOSE_CLASSIFY_STOP_SCALE = max(
    0.05,
    float(os.getenv("EXCHANGE_CLOSE_CLASSIFY_STOP_SCALE", "0.35")),
)
EXCHANGE_CLOSE_CLASSIFY_TP_SCALE = max(
    0.05,
    float(os.getenv("EXCHANGE_CLOSE_CLASSIFY_TP_SCALE", "0.35")),
)
EXCHANGE_CLOSE_CLASSIFY_MIN_BAND_PCT = max(
    0.0,
    float(os.getenv("EXCHANGE_CLOSE_CLASSIFY_MIN_BAND_PCT", "0.0015")),
)
EXCHANGE_CLOSE_CLASSIFY_BREAKEVEN_SCALE = max(
    0.0,
    float(os.getenv("EXCHANGE_CLOSE_CLASSIFY_BREAKEVEN_SCALE", "0.20")),
)
NEAR_BREAKEVEN_LOSS_TO_BE_PCT = max(
    0.0,
    float(os.getenv("NEAR_BREAKEVEN_LOSS_TO_BE_PCT", "0.0015")),
)
EXCHANGE_CLOSE_RECENT_BOT_CLOSE_MINUTES = max(
    1,
    int(os.getenv("EXCHANGE_CLOSE_RECENT_BOT_CLOSE_MINUTES", "5")),
)
EXCHANGE_CLOSE_DEDUP_MINUTES = max(
    1,
    int(os.getenv("EXCHANGE_CLOSE_DEDUP_MINUTES", "3")),
)
SL_RECONCILE_TOO_TIGHT_MULT = max(0.0, float(os.getenv("SL_RECONCILE_TOO_TIGHT_MULT", "0.80")))
SL_RECONCILE_TOO_WIDE_MULT = max(1.0, float(os.getenv("SL_RECONCILE_TOO_WIDE_MULT", "2.00")))
PYRAMIDING_ENABLED = os.getenv("PYRAMIDING_ENABLED", "true").lower() == "true"  # ENABLED: scale into winners (78-80% success rate per entry-signals skill)
PYRAMID_MAX_ADDS = int(os.getenv("PYRAMID_MAX_ADDS", "2"))
PYRAMID_ADD_AT_R = float(os.getenv("PYRAMID_ADD_AT_R", "0.8"))
PYRAMID_RISK_SCALE = float(os.getenv("PYRAMID_RISK_SCALE", "0.6"))
PYRAMID_MIN_MINUTES_BETWEEN_ADDS = int(os.getenv("PYRAMID_MIN_MINUTES_BETWEEN_ADDS", "3"))
ML_ENTRY_FILTER_ENABLED = os.getenv("ML_ENTRY_FILTER_ENABLED", "false").lower() == "true"
ML_ENTRY_FILTER_MODEL_PATH = os.getenv(
    "ML_ENTRY_FILTER_MODEL_PATH",
    str(BASE_DIR / "tmp" / "entry_filter_model.json"),
)
ML_ENTRY_FILTER_MODEL_DIR = os.getenv(
    "ML_ENTRY_FILTER_MODEL_DIR",
    str(BASE_DIR / "tmp" / "entry_filter_models"),
)
ML_ENTRY_FILTER_MIN_PROB = float(os.getenv("ML_ENTRY_FILTER_MIN_PROB", "0.52"))
ML_ENTRY_FILTER_FAIL_OPEN = os.getenv("ML_ENTRY_FILTER_FAIL_OPEN", "true").lower() == "true"
ML_ENTRY_FILTER_PER_SYMBOL_ENABLED = (
    os.getenv("ML_ENTRY_FILTER_PER_SYMBOL_ENABLED", "false").lower() == "true"
)
ML_ENTRY_FILTER_PER_STRATEGY_ENABLED = (
    os.getenv("ML_ENTRY_FILTER_PER_STRATEGY_ENABLED", "false").lower() == "true"
)
ML_ENTRY_FILTER_PER_SYMBOL_FALLBACK_GLOBAL = (
    os.getenv("ML_ENTRY_FILTER_PER_SYMBOL_FALLBACK_GLOBAL", "true").lower() == "true"
)
ML_ENTRY_FILTER_PER_STRATEGY_FALLBACK_GLOBAL = (
    os.getenv("ML_ENTRY_FILTER_PER_STRATEGY_FALLBACK_GLOBAL", "true").lower() == "true"
)
ML_ENTRY_FILTER_PER_SYMBOL_MIN_SAMPLES = int(
    os.getenv("ML_ENTRY_FILTER_PER_SYMBOL_MIN_SAMPLES", "50")
)
ML_ENTRY_FILTER_PER_STRATEGY_MIN_SAMPLES = int(
    os.getenv("ML_ENTRY_FILTER_PER_STRATEGY_MIN_SAMPLES", "80")
)
ML_ENTRY_FILTER_PER_SYMBOL_STRATEGY_MIN_SAMPLES = int(
    os.getenv("ML_ENTRY_FILTER_PER_SYMBOL_STRATEGY_MIN_SAMPLES", "40")
)
ML_ENTRY_FILTER_AUTO_TRAIN_ENABLED = (
    os.getenv("ML_ENTRY_FILTER_AUTO_TRAIN_ENABLED", "false").lower() == "true"
)
ML_ENTRY_FILTER_RETRAIN_ON_OPERATION_ENABLED = (
    os.getenv("ML_ENTRY_FILTER_RETRAIN_ON_OPERATION_ENABLED", "false").lower() == "true"
)
ML_ENTRY_FILTER_RETRAIN_ON_OPERATION_MIN_INTERVAL_SECONDS = int(
    os.getenv("ML_ENTRY_FILTER_RETRAIN_ON_OPERATION_MIN_INTERVAL_SECONDS", "0")
)
ML_ENTRY_FILTER_AUTO_APPLY_THRESHOLD = (
    os.getenv("ML_ENTRY_FILTER_AUTO_APPLY_THRESHOLD", "true").lower() == "true"
)
ML_ENTRY_FILTER_TRAIN_DAYS = int(os.getenv("ML_ENTRY_FILTER_TRAIN_DAYS", "21"))
ML_ENTRY_FILTER_TRAIN_SOURCE = os.getenv("ML_ENTRY_FILTER_TRAIN_SOURCE", "mixed").strip().lower()
if ML_ENTRY_FILTER_TRAIN_SOURCE not in {"live", "backtest", "mixed"}:
    ML_ENTRY_FILTER_TRAIN_SOURCE = "mixed"
ML_ENTRY_FILTER_TRAIN_BACKTEST_DAYS = int(os.getenv("ML_ENTRY_FILTER_TRAIN_BACKTEST_DAYS", "180"))
ML_ENTRY_FILTER_TRAIN_MIN_SAMPLES = int(os.getenv("ML_ENTRY_FILTER_TRAIN_MIN_SAMPLES", "120"))
ML_ENTRY_FILTER_TRAIN_EPOCHS = int(os.getenv("ML_ENTRY_FILTER_TRAIN_EPOCHS", "1200"))
ML_ENTRY_FILTER_TRAIN_LR = float(os.getenv("ML_ENTRY_FILTER_TRAIN_LR", "0.05"))
ML_ENTRY_FILTER_TRAIN_L2 = float(os.getenv("ML_ENTRY_FILTER_TRAIN_L2", "0.001"))
ML_ENTRY_FILTER_AUTO_TRAIN_HOUR = int(os.getenv("ML_ENTRY_FILTER_AUTO_TRAIN_HOUR", "0"))
ML_ENTRY_FILTER_AUTO_TRAIN_MINUTE = int(os.getenv("ML_ENTRY_FILTER_AUTO_TRAIN_MINUTE", "20"))
ML_TRAINING_QUEUE = os.getenv("ML_TRAINING_QUEUE", "ml")

EXECUTION_LOCK_ENABLED = os.getenv("EXECUTION_LOCK_ENABLED", "true").lower() == "true"
EXECUTION_LOCK_KEY = os.getenv("EXECUTION_LOCK_KEY", "lock:execute_orders")
EXECUTION_LOCK_TTL_SECONDS = int(os.getenv("EXECUTION_LOCK_TTL_SECONDS", "90"))

# -- New: Weekly drawdown --
WEEKLY_DD_LIMIT = float(os.getenv("WEEKLY_DD_LIMIT", "0.10"))  # 10%
CIRCUIT_BREAKER_CONSECUTIVE_LOSS_WINDOW_HOURS = max(
    0.0,
    float(os.getenv("CIRCUIT_BREAKER_CONSECUTIVE_LOSS_WINDOW_HOURS", "24")),
)

# -- Daily trade count limit (risk-management skill: 95% success rate) --
MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "6"))  # max new entries per day across all instruments
MAX_DAILY_TRADES_LOW_ADX = int(os.getenv("MAX_DAILY_TRADES_LOW_ADX", "3"))  # max trades when ADX < 20 (choppy market)
MAX_DAILY_TRADES_HIGH_ADX = int(os.getenv("MAX_DAILY_TRADES_HIGH_ADX", "10"))  # max trades when ADX > 25 (strong trend)
MAX_DAILY_TRADES_LOW_ADX_THRESHOLD = float(os.getenv("MAX_DAILY_TRADES_LOW_ADX_THRESHOLD", "20"))
MAX_DAILY_TRADES_HIGH_ADX_THRESHOLD = float(os.getenv("MAX_DAILY_TRADES_HIGH_ADX_THRESHOLD", "25"))
DAILY_TRADE_COUNT_TTL_SECONDS = max(60, int(os.getenv("DAILY_TRADE_COUNT_TTL_SECONDS", "90000")))

# -- Global market regime gate (BTC 1h ADX) --
# Block ALL new entries when BTC 1h ADX is below this threshold (choppy macro).
# Set to 0 to disable.
MARKET_REGIME_ADX_MIN = float(os.getenv("MARKET_REGIME_ADX_MIN", "0"))

# -- Signal flip min age gate --
# Prevent signal_flip close if position is younger than N minutes.
# Data: flips <5min have PnL +0.64% (147 trades), flips >=5min have +3.67% (23 trades).
# SL always respected regardless of age. Set to 0 to disable.
SIGNAL_FLIP_MIN_AGE_ENABLED = os.getenv("SIGNAL_FLIP_MIN_AGE_ENABLED", "false").lower() == "true"
SIGNAL_FLIP_MIN_AGE_MINUTES = float(os.getenv("SIGNAL_FLIP_MIN_AGE_MINUTES", "5"))

# -- Time-based stale position cleanup (risk skill: 82-88% success rate) --
STALE_POSITION_MAX_HOURS = int(os.getenv("STALE_POSITION_MAX_HOURS", "12"))  # close positions older than 12h if near breakeven
STALE_POSITION_PNL_BAND_PCT = float(os.getenv("STALE_POSITION_PNL_BAND_PCT", "0.005"))  # close if PnL between -0.5% and +0.5%
STALE_POSITION_ENABLED = os.getenv("STALE_POSITION_ENABLED", "true").lower() == "true"

# -- Short score penalty (entry-signals skill: shorts 25-35% vs longs 85-88%) --
SHORT_SCORE_PENALTY = float(os.getenv("SHORT_SCORE_PENALTY", "0.15"))  # penalize short signals to require much higher confluence

# -- Uptrend short killer (risk skill: 88% success rate, 675 samples) --
UPTREND_SHORT_KILLER_ENABLED = os.getenv("UPTREND_SHORT_KILLER_ENABLED", "true").lower() == "true"  # close shorts when HTF turns bullish
# -- Downtrend long killer (symmetry to protect longs when HTF flips bearish) --
DOWNTREND_LONG_KILLER_ENABLED = os.getenv("DOWNTREND_LONG_KILLER_ENABLED", "true").lower() == "true"

# -- New: Kill-switch thresholds --
MAX_CONSECUTIVE_ERRORS = int(os.getenv("MAX_CONSECUTIVE_ERRORS", "3"))
DATA_STALE_SECONDS = int(os.getenv("DATA_STALE_SECONDS", "300"))  # 5 min
RISK_EVENT_DEDUP_SECONDS = int(os.getenv("RISK_EVENT_DEDUP_SECONDS", "300"))

# -- New: Funding rate filter --
FUNDING_EXTREME_PERCENTILE = float(os.getenv("FUNDING_EXTREME_PERCENTILE", "0.001"))  # |rate| > 0.1% is extreme
FUNDING_FETCH_MIN_INTERVAL_SECONDS = int(
    os.getenv("FUNDING_FETCH_MIN_INTERVAL_SECONDS", "300")
)  # avoid over-polling funding endpoint
MARKETDATA_POLL_INTERVAL = int(os.getenv("MARKETDATA_POLL_INTERVAL", "15"))

# -- EMA confluence filter/scoring --
EMA_CONFLUENCE_ENABLED = os.getenv("EMA_CONFLUENCE_ENABLED", "true").lower() == "true"
_EMA_CONFLUENCE_PERIODS_RAW = os.getenv("EMA_CONFLUENCE_PERIODS", "20,50,200")
try:
    EMA_CONFLUENCE_PERIODS = sorted(
        {int(x.strip()) for x in _EMA_CONFLUENCE_PERIODS_RAW.split(",") if x.strip()}
    )
    if len(EMA_CONFLUENCE_PERIODS) < 2:
        EMA_CONFLUENCE_PERIODS = [20, 50, 200]
except Exception:
    EMA_CONFLUENCE_PERIODS = [20, 50, 200]
EMA_CONFLUENCE_BONUS = float(os.getenv("EMA_CONFLUENCE_BONUS", "0.08"))   # was 0.06 — stronger reward for EMA alignment
EMA_CONFLUENCE_PENALTY = float(os.getenv("EMA_CONFLUENCE_PENALTY", "0.12"))   # was 0.10 — stronger punishment for misalignment
EMA_CONFLUENCE_HARD_BLOCK = os.getenv("EMA_CONFLUENCE_HARD_BLOCK", "true").lower() == "true"  # ENABLED: block entries against EMA stack

# -- Session policy --
SESSION_POLICY_ENABLED = os.getenv("SESSION_POLICY_ENABLED", "true").lower() == "true"  # ENABLED: session-based score thresholds and dead zone blocking
SESSION_DEAD_ZONE_BLOCK = os.getenv("SESSION_DEAD_ZONE_BLOCK", "true").lower() == "true"
_SESSION_SCORE_MIN_RAW = os.getenv("SESSION_SCORE_MIN", "{}")
_SESSION_RISK_MULT_RAW = os.getenv("SESSION_RISK_MULTIPLIER", "{}")
_DIRECTION_MODE_RAW = os.getenv("SIGNAL_DIRECTION_MODE", "both")
SIGNAL_DIRECTION_MODE = _DIRECTION_MODE_RAW.strip().lower()
if SIGNAL_DIRECTION_MODE not in {"both", "long_only", "short_only", "disabled"}:
    SIGNAL_DIRECTION_MODE = "both"
MACRO_HIGH_IMPACT_FILTER_ENABLED = (
    os.getenv("MACRO_HIGH_IMPACT_FILTER_ENABLED", "false").lower() == "true"
)
MACRO_HIGH_IMPACT_BLOCK_ENTRIES = (
    os.getenv("MACRO_HIGH_IMPACT_BLOCK_ENTRIES", "false").lower() == "true"
)
MACRO_HIGH_IMPACT_RISK_MULTIPLIER = max(
    0.0,
    min(1.0, float(os.getenv("MACRO_HIGH_IMPACT_RISK_MULTIPLIER", "0.55"))),
)
_MACRO_HIGH_IMPACT_UTC_HOURS_RAW = os.getenv("MACRO_HIGH_IMPACT_UTC_HOURS", "13,14")
_MACRO_HIGH_IMPACT_WEEKDAYS_RAW = os.getenv("MACRO_HIGH_IMPACT_WEEKDAYS", "0,1,2,3,4")
_MACRO_HIGH_IMPACT_SESSIONS_RAW = os.getenv("MACRO_HIGH_IMPACT_SESSIONS", "ny,overlap")

# -- Per-instrument portfolio control (asymmetric) --
# Override risk_per_trade and cooldown per symbol. Keys are symbols.
import json as _json
_PER_INST_RISK_RAW = os.getenv("PER_INSTRUMENT_RISK", '{}')

# Risk tiers: hierarchical risk sizing (BTC < ETH < alts)
# Tier base_risk is fed into ATR scaling (not a flat override like PER_INSTRUMENT_RISK).
_INSTRUMENT_RISK_TIERS_RAW = os.getenv(
    "INSTRUMENT_RISK_TIERS",
    '{"base": 0.0020, "mid": 0.0025, "alt": 0.0015}',
)
_INSTRUMENT_TIER_MAP_RAW = os.getenv(
    "INSTRUMENT_TIER_MAP",
    '{"BTCUSDT": "base", "ETHUSDT": "base", "SOLUSDT": "mid", "XRPUSDT": "mid", "DOGEUSDT": "alt", "ADAUSDT": "alt", "LINKUSDT": "alt"}',
)
_PER_INST_COOLDOWN_RAW = os.getenv("PER_INSTRUMENT_COOLDOWN", '{}')
_PER_INST_SPREAD_RAW = os.getenv("PER_INSTRUMENT_MAX_SPREAD_BPS", '{}')
_PER_INST_DIRECTION_RAW = os.getenv("PER_INSTRUMENT_DIRECTION", '{}')
_ALLOCATOR_WEIGHTS_RAW = os.getenv(
    "ALLOCATOR_MODULE_WEIGHTS",
    '{"trend":0.45,"meanrev":0.35,"carry":0.20}',
)
_ALLOCATOR_RISK_BUDGETS_RAW = os.getenv(
    "ALLOCATOR_MODULE_RISK_BUDGETS",
    '{"trend":0.45,"meanrev":0.35,"carry":0.20}',
)
_MULTI_STRATEGY_UNIVERSE_RAW = os.getenv(
    "MULTI_STRATEGY_UNIVERSE",
    "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,LINKUSDT",
)
_ML_ENTRY_FILTER_PER_SYMBOLS_RAW = os.getenv(
    "ML_ENTRY_FILTER_PER_SYMBOLS",
    "BTCUSDT,ADAUSDT,DOGEUSDT",
)
_ML_ENTRY_FILTER_PER_STRATEGIES_RAW = os.getenv(
    "ML_ENTRY_FILTER_PER_STRATEGIES",
    "alloc_long,alloc_short",
)


def _parse_raw_mapping(raw: str) -> dict:
    """
    Parse mapping from JSON first; fallback to relaxed syntax:
    {BTCUSDT:disabled,ADAUSDT:long_only}
    """
    try:
        parsed = _json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    txt = (raw or "").strip()
    if not txt:
        return {}
    if txt.startswith("{") and txt.endswith("}"):
        txt = txt[1:-1]
    out: dict[str, str] = {}
    for piece in txt.split(","):
        item = piece.strip()
        if not item or ":" not in item:
            continue
        key, value = item.split(":", 1)
        key = key.strip().strip('"').strip("'")
        value = value.strip().strip('"').strip("'")
        if key:
            out[key] = value
    return out
try:
    PER_INSTRUMENT_RISK = {k: float(v) for k, v in _json.loads(_PER_INST_RISK_RAW).items()}
except Exception:
    PER_INSTRUMENT_RISK = {}
try:
    INSTRUMENT_RISK_TIERS = {k: float(v) for k, v in _json.loads(_INSTRUMENT_RISK_TIERS_RAW).items()}
except Exception:
    INSTRUMENT_RISK_TIERS = {"base": 0.0020, "mid": 0.0025, "alt": 0.0015}
try:
    INSTRUMENT_TIER_MAP = dict(_json.loads(_INSTRUMENT_TIER_MAP_RAW))
except Exception:
    INSTRUMENT_TIER_MAP = {}
INSTRUMENT_RISK_TIERS_ENABLED = os.getenv("INSTRUMENT_RISK_TIERS_ENABLED", "false").strip().lower() in ("true", "1", "yes")
try:
    PER_INSTRUMENT_COOLDOWN = {k: int(v) for k, v in _json.loads(_PER_INST_COOLDOWN_RAW).items()}
except Exception:
    PER_INSTRUMENT_COOLDOWN = {}
try:
    PER_INSTRUMENT_MAX_SPREAD_BPS = {k: float(v) for k, v in _json.loads(_PER_INST_SPREAD_RAW).items()}
except Exception:
    PER_INSTRUMENT_MAX_SPREAD_BPS = {}
try:
    PER_INSTRUMENT_DIRECTION = {
        str(k).upper(): (
            str(v).strip().lower()
            if str(v).strip().lower() in {"both", "long_only", "short_only", "disabled"}
            else "both"
        )
        for k, v in _parse_raw_mapping(_PER_INST_DIRECTION_RAW).items()
    }
except Exception:
    PER_INSTRUMENT_DIRECTION = {}
try:
    SESSION_SCORE_MIN = {k: float(v) for k, v in _json.loads(_SESSION_SCORE_MIN_RAW).items()}
except Exception:
    SESSION_SCORE_MIN = {}
try:
    SESSION_RISK_MULTIPLIER = {k: float(v) for k, v in _json.loads(_SESSION_RISK_MULT_RAW).items()}
except Exception:
    SESSION_RISK_MULTIPLIER = {}
try:
    _raw_volume_by_session = _json.loads(_ENTRY_VOLUME_FILTER_MIN_RATIO_BY_SESSION_RAW)
    if isinstance(_raw_volume_by_session, dict):
        ENTRY_VOLUME_FILTER_MIN_RATIO_BY_SESSION = {
            str(k).strip().lower(): max(0.0, float(v))
            for k, v in _raw_volume_by_session.items()
            if str(k).strip().lower() in {"asia", "london", "ny", "overlap", "dead"}
        }
    else:
        ENTRY_VOLUME_FILTER_MIN_RATIO_BY_SESSION = {}
except Exception:
    ENTRY_VOLUME_FILTER_MIN_RATIO_BY_SESSION = {}


def _parse_symbol_list(raw: str) -> list[str]:
    txt = (raw or "").strip()
    if not txt:
        return []
    if txt.startswith("["):
        try:
            parsed = _json.loads(txt)
            if isinstance(parsed, list):
                return [str(x).upper() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [x.strip().upper() for x in txt.split(",") if x.strip()]


def _parse_strategy_list(raw: str) -> list[str]:
    txt = (raw or "").strip()
    if not txt:
        return []
    values: list[str] = []
    if txt.startswith("["):
        try:
            parsed = _json.loads(txt)
            if isinstance(parsed, list):
                values = [str(x).strip().lower() for x in parsed if str(x).strip()]
        except Exception:
            values = []
    if not values:
        values = [x.strip().lower() for x in txt.split(",") if x.strip()]
    out: list[str] = []
    for val in values:
        cleaned = "".join(ch for ch in val if ch.isalnum() or ch == "_").strip("_")
        if cleaned:
            out.append(cleaned)
    # stable order + dedup
    seen = set()
    uniq: list[str] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def _parse_int_set(raw: str, *, min_val: int, max_val: int) -> set[int]:
    txt = (raw or "").strip()
    if not txt:
        return set()
    values: list[int] = []
    if txt.startswith("["):
        try:
            parsed = _json.loads(txt)
            if isinstance(parsed, list):
                values = [int(x) for x in parsed]
        except Exception:
            values = []
    if not values:
        for piece in txt.split(","):
            p = piece.strip()
            if not p:
                continue
            try:
                values.append(int(p))
            except Exception:
                continue
    return {v for v in values if min_val <= v <= max_val}


def _parse_session_set(raw: str) -> set[str]:
    allowed = {"asia", "london", "ny", "overlap", "dead"}
    txt = (raw or "").strip()
    if not txt:
        return set()
    values: list[str] = []
    if txt.startswith("["):
        try:
            parsed = _json.loads(txt)
            if isinstance(parsed, list):
                values = [str(x).strip().lower() for x in parsed if str(x).strip()]
        except Exception:
            values = []
    if not values:
        values = [x.strip().lower() for x in txt.split(",") if x.strip()]
    return {v for v in values if v in allowed}


# -- Multi-strategy portfolio controls --
MULTI_STRATEGY_ENABLED = os.getenv("MULTI_STRATEGY_ENABLED", "false").lower() == "true"
MODULE_TREND_ENABLED = os.getenv("MODULE_TREND_ENABLED", "true").lower() == "true"
MODULE_MEANREV_ENABLED = os.getenv("MODULE_MEANREV_ENABLED", "true").lower() == "true"
MODULE_CARRY_ENABLED = os.getenv("MODULE_CARRY_ENABLED", "true").lower() == "true"
MODULE_SIGNAL_TTL_SECONDS = int(os.getenv("MODULE_SIGNAL_TTL_SECONDS", "120"))
MODULE_LOOKBACK_BARS = int(os.getenv("MODULE_LOOKBACK_BARS", "240"))
MODULE_ADX_TREND_MIN = float(os.getenv("MODULE_ADX_TREND_MIN", "20.0"))
MODULE_TREND_HTF_ADX_MIN = float(os.getenv("MODULE_TREND_HTF_ADX_MIN", "0.0"))
MODULE_ADX_RANGE_MAX = float(os.getenv("MODULE_ADX_RANGE_MAX", "18.0"))
MODULE_MEANREV_Z_ENTRY = float(os.getenv("MODULE_MEANREV_Z_ENTRY", "1.2"))
MODULE_CARRY_FUNDING_MULT = float(os.getenv("MODULE_CARRY_FUNDING_MULT", "1.8"))
MODULE_SYMBOL_WARMUP_BARS = int(os.getenv("MODULE_SYMBOL_WARMUP_BARS", "300"))
MODULE_IMPULSE_FILTER_ENABLED = os.getenv("MODULE_IMPULSE_FILTER_ENABLED", "true").lower() == "true"
MODULE_IMPULSE_LOOKBACK = max(8, min(120, int(os.getenv("MODULE_IMPULSE_LOOKBACK", "20"))))
MODULE_IMPULSE_BODY_MULT = max(1.0, float(os.getenv("MODULE_IMPULSE_BODY_MULT", "2.2")))
MODULE_IMPULSE_MIN_BODY_PCT = max(0.0, float(os.getenv("MODULE_IMPULSE_MIN_BODY_PCT", "0.006")))
MODULE_IMPULSE_MAX_EMA20_DIST_PCT = max(
    0.0,
    float(os.getenv("MODULE_IMPULSE_MAX_EMA20_DIST_PCT", "0.008")),
)
MODULE_MEANREV_IMPULSE_BLOCK_ENABLED = (
    os.getenv("MODULE_MEANREV_IMPULSE_BLOCK_ENABLED", "true").lower() == "true"
)
MODULE_CARRY_MAX_ATR_PCT = max(0.0, float(os.getenv("MODULE_CARRY_MAX_ATR_PCT", "0.020")))

# SMC anti-chase gate (avoid entering right after displacement candles).
SMC_IMPULSE_CHASE_FILTER_ENABLED = (
    os.getenv("SMC_IMPULSE_CHASE_FILTER_ENABLED", "true").lower() == "true"
)
SMC_IMPULSE_LOOKBACK = max(8, min(120, int(os.getenv("SMC_IMPULSE_LOOKBACK", "20"))))
SMC_IMPULSE_BODY_MULT = max(1.0, float(os.getenv("SMC_IMPULSE_BODY_MULT", "2.0")))
SMC_IMPULSE_MIN_BODY_PCT = max(0.0, float(os.getenv("SMC_IMPULSE_MIN_BODY_PCT", "0.006")))
SMC_IMPULSE_MAX_EMA20_DIST_PCT = max(0.0, float(os.getenv("SMC_IMPULSE_MAX_EMA20_DIST_PCT", "0.009")))
SMC_SESSION_FILTER_ENABLED = os.getenv("SMC_SESSION_FILTER_ENABLED", "true").lower() == "true"
SMC_ALLOWED_SESSIONS = _parse_session_set(os.getenv("SMC_ALLOWED_SESSIONS", "london,ny,overlap"))
# ADX minimum on HTF for SMC entries — blocks entries in choppy/weak markets
SMC_ADX_MIN = max(0.0, float(os.getenv("SMC_ADX_MIN", "18.0")))

ALLOCATOR_ENABLED = os.getenv("ALLOCATOR_ENABLED", "true").lower() == "true"
ALLOCATOR_INCLUDE_SMC = os.getenv("ALLOCATOR_INCLUDE_SMC", "false").lower() == "true"
ALLOCATOR_NET_THRESHOLD = float(os.getenv("ALLOCATOR_NET_THRESHOLD", "0.20"))
ALLOCATOR_LONG_SCORE_PENALTY = max(0.0, min(1.0, float(os.getenv("ALLOCATOR_LONG_SCORE_PENALTY", "1.0"))))
ALLOCATOR_CONFLICT_POLICY = os.getenv("ALLOCATOR_CONFLICT_POLICY", "net_score").strip().lower()
if ALLOCATOR_CONFLICT_POLICY not in {"net_score"}:
    ALLOCATOR_CONFLICT_POLICY = "net_score"
ALLOCATOR_MIN_MODULES_ACTIVE = int(os.getenv("ALLOCATOR_MIN_MODULES_ACTIVE", "2"))
ALLOCATOR_STRONG_TREND_SOLO_ENABLED = (
    os.getenv("ALLOCATOR_STRONG_TREND_SOLO_ENABLED", "true").lower() == "true"
)
ALLOCATOR_STRONG_TREND_ADX_MIN = max(
    0.0, float(os.getenv("ALLOCATOR_STRONG_TREND_ADX_MIN", "25.0"))
)
ALLOCATOR_STRONG_TREND_CONFIDENCE_MIN = max(
    0.0,
    min(
        1.0, float(os.getenv("ALLOCATOR_STRONG_TREND_CONFIDENCE_MIN", "0.80"))
    ),
)
ALLOCATOR_CARRY_CONTRA_TREND_DAMPEN_ENABLED = (
    os.getenv("ALLOCATOR_CARRY_CONTRA_TREND_DAMPEN_ENABLED", "true").lower()
    == "true"
)
ALLOCATOR_CARRY_CONTRA_TREND_DAMPEN_MULT = max(
    0.0,
    min(
        1.0, float(os.getenv("ALLOCATOR_CARRY_CONTRA_TREND_DAMPEN_MULT", "0.50"))
    ),
)
ALLOCATOR_SMC_CONFLUENCE_BOOST_ENABLED = (
    os.getenv("ALLOCATOR_SMC_CONFLUENCE_BOOST_ENABLED", "true").lower() == "true"
)
ALLOCATOR_SMC_CONFLUENCE_WEIGHT_MULT = max(
    1.0,
    float(os.getenv("ALLOCATOR_SMC_CONFLUENCE_WEIGHT_MULT", "1.25")),
)
ALLOCATOR_SMC_NON_CONFLUENCE_WEIGHT_MULT = max(
    0.0,
    min(1.0, float(os.getenv("ALLOCATOR_SMC_NON_CONFLUENCE_WEIGHT_MULT", "0.85"))),
)
ALLOCATOR_SMC_CONFLUENCE_MIN_SCORE = max(
    0.0,
    min(1.0, float(os.getenv("ALLOCATOR_SMC_CONFLUENCE_MIN_SCORE", "0.80"))),
)
ALLOCATOR_SMC_ALLOWED_SESSIONS = _parse_session_set(
    os.getenv("ALLOCATOR_SMC_ALLOWED_SESSIONS", "london,ny,overlap")
)
if not ALLOCATOR_SMC_ALLOWED_SESSIONS:
    ALLOCATOR_SMC_ALLOWED_SESSIONS = set(SMC_ALLOWED_SESSIONS)

# --- Dynamic (Bayesian) allocator weights ---
ALLOCATOR_DYNAMIC_WEIGHTS_ENABLED = (
    os.getenv("ALLOCATOR_DYNAMIC_WEIGHTS_ENABLED", "false").lower() == "true"
)
ALLOCATOR_DYNAMIC_WINDOW_DAYS = int(os.getenv("ALLOCATOR_DYNAMIC_WINDOW_DAYS", "7"))
ALLOCATOR_DYNAMIC_ALPHA_PRIOR = float(os.getenv("ALLOCATOR_DYNAMIC_ALPHA_PRIOR", "2.0"))
ALLOCATOR_DYNAMIC_BETA_PRIOR = float(os.getenv("ALLOCATOR_DYNAMIC_BETA_PRIOR", "2.0"))
ALLOCATOR_DYNAMIC_MIN_MULT = float(os.getenv("ALLOCATOR_DYNAMIC_MIN_MULT", "0.5"))
ALLOCATOR_DYNAMIC_MAX_MULT = float(os.getenv("ALLOCATOR_DYNAMIC_MAX_MULT", "2.0"))
ALLOCATOR_DYNAMIC_MIN_TRADES = int(os.getenv("ALLOCATOR_DYNAMIC_MIN_TRADES", "10"))

# --- HMM Regime Detection ---
HMM_REGIME_ENABLED = os.getenv("HMM_REGIME_ENABLED", "false").lower() == "true"
HMM_REGIME_N_STATES = int(os.getenv("HMM_REGIME_N_STATES", "2"))
HMM_REGIME_LOOKBACK_BARS = int(os.getenv("HMM_REGIME_LOOKBACK_BARS", "500"))
HMM_REGIME_TIMEFRAME = os.getenv("HMM_REGIME_TIMEFRAME", "1h").strip()
HMM_REGIME_REFIT_HOURS = int(os.getenv("HMM_REGIME_REFIT_HOURS", "6"))
HMM_REGIME_CACHE_TTL_HOURS = int(os.getenv("HMM_REGIME_CACHE_TTL_HOURS", "12"))
HMM_REGIME_PER_SYMBOL = os.getenv("HMM_REGIME_PER_SYMBOL", "false").lower() == "true"
HMM_REGIME_TRENDING_RISK_MULT = float(os.getenv("HMM_REGIME_TRENDING_RISK_MULT", "1.0"))
HMM_REGIME_CHOPPY_RISK_MULT = float(os.getenv("HMM_REGIME_CHOPPY_RISK_MULT", "0.7"))
HMM_REGIME_LABEL_HYSTERESIS_VOL = max(
    0.0,
    float(os.getenv("HMM_REGIME_LABEL_HYSTERESIS_VOL", "0.0005")),
)
HMM_REGIME_LABEL_MEMORY_TTL_HOURS = max(
    1,
    int(os.getenv("HMM_REGIME_LABEL_MEMORY_TTL_HOURS", "168")),
)

# --- GARCH Volatility Forecasting ---
GARCH_ENABLED = os.getenv("GARCH_ENABLED", "false").lower() == "true"
GARCH_LOOKBACK_BARS = int(os.getenv("GARCH_LOOKBACK_BARS", "500"))
GARCH_TIMEFRAME = os.getenv("GARCH_TIMEFRAME", "1h").strip()
GARCH_REFIT_HOURS = int(os.getenv("GARCH_REFIT_HOURS", "6"))
GARCH_CACHE_TTL_HOURS = int(os.getenv("GARCH_CACHE_TTL_HOURS", "12"))
GARCH_BLEND_WEIGHT = float(os.getenv("GARCH_BLEND_WEIGHT", "0.6"))
GARCH_MAX_PERSISTENCE = max(0.0, float(os.getenv("GARCH_MAX_PERSISTENCE", "1.0")))
GARCH_BLEND_VOL_FLOOR_PCT = max(0.0, float(os.getenv("GARCH_BLEND_VOL_FLOOR_PCT", "0.003")))
GARCH_BLEND_FLOOR_ATR_RATIO = max(0.0, float(os.getenv("GARCH_BLEND_FLOOR_ATR_RATIO", "0.5")))

try:
    ALLOCATOR_MODULE_WEIGHTS = {
        str(k).strip().lower(): float(v)
        for k, v in _json.loads(_ALLOCATOR_WEIGHTS_RAW).items()
    }
except Exception:
    ALLOCATOR_MODULE_WEIGHTS = {"trend": 0.45, "meanrev": 0.30, "carry": 0.15, "smc": 0.10}
try:
    ALLOCATOR_MODULE_RISK_BUDGETS = {
        str(k).strip().lower(): float(v)
        for k, v in _json.loads(_ALLOCATOR_RISK_BUDGETS_RAW).items()
    }
except Exception:
    ALLOCATOR_MODULE_RISK_BUDGETS = {"trend": 0.45, "meanrev": 0.30, "carry": 0.15, "smc": 0.10}

MULTI_STRATEGY_UNIVERSE = _parse_symbol_list(_MULTI_STRATEGY_UNIVERSE_RAW)
ML_ENTRY_FILTER_PER_SYMBOLS = _parse_symbol_list(_ML_ENTRY_FILTER_PER_SYMBOLS_RAW)
ML_ENTRY_FILTER_PER_STRATEGIES = _parse_strategy_list(_ML_ENTRY_FILTER_PER_STRATEGIES_RAW)
SHADOW_TRADING_ENABLED = os.getenv("SHADOW_TRADING_ENABLED", "false").lower() == "true"
LIVE_GRADUAL_ENABLED = os.getenv("LIVE_GRADUAL_ENABLED", "true").lower() == "true"
LIVE_GRADUAL_MAX_MODULES = int(os.getenv("LIVE_GRADUAL_MAX_MODULES", "3"))
LIVE_GRADUAL_MAX_SYMBOLS_PER_MODULE = int(
    os.getenv("LIVE_GRADUAL_MAX_SYMBOLS_PER_MODULE", "7")
)
_LIVE_GRADUAL_MODULE_PRIORITY_RAW = os.getenv("LIVE_GRADUAL_MODULE_PRIORITY", "{}")
try:
    LIVE_GRADUAL_MODULE_PRIORITY = {
        str(k).strip().lower(): float(v)
        for k, v in _json.loads(_LIVE_GRADUAL_MODULE_PRIORITY_RAW).items()
    }
except Exception:
    LIVE_GRADUAL_MODULE_PRIORITY = {}
MACRO_HIGH_IMPACT_UTC_HOURS = _parse_int_set(
    _MACRO_HIGH_IMPACT_UTC_HOURS_RAW,
    min_val=0,
    max_val=23,
)
MACRO_HIGH_IMPACT_WEEKDAYS = _parse_int_set(
    _MACRO_HIGH_IMPACT_WEEKDAYS_RAW,
    min_val=0,
    max_val=6,
)
MACRO_HIGH_IMPACT_SESSIONS = _parse_session_set(_MACRO_HIGH_IMPACT_SESSIONS_RAW)

# -- New: Telegram alerts --
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
PERFORMANCE_REPORT_ENABLED = os.getenv("PERFORMANCE_REPORT_ENABLED", "true").lower() == "true"
PERFORMANCE_REPORT_WINDOW_MINUTES = int(os.getenv("PERFORMANCE_REPORT_WINDOW_MINUTES", "180"))
PERFORMANCE_REPORT_BEAT_ENABLED = os.getenv("PERFORMANCE_REPORT_BEAT_ENABLED", "true").lower() == "true"
PERFORMANCE_REPORT_BEAT_MODE = os.getenv("PERFORMANCE_REPORT_BEAT_MODE", "interval").strip().lower()
if PERFORMANCE_REPORT_BEAT_MODE not in {"interval", "daily"}:
    PERFORMANCE_REPORT_BEAT_MODE = "interval"
_PERF_BEAT_RAW = int(os.getenv("PERFORMANCE_REPORT_BEAT_MINUTES", "180"))
PERFORMANCE_REPORT_BEAT_MINUTES = max(1, min(1440, _PERF_BEAT_RAW))
_PERF_BEAT_HOUR_RAW = int(os.getenv("PERFORMANCE_REPORT_BEAT_HOUR", "0"))
_PERF_BEAT_MINUTE_RAW = int(os.getenv("PERFORMANCE_REPORT_BEAT_MINUTE", "0"))
PERFORMANCE_REPORT_BEAT_HOUR = max(0, min(23, _PERF_BEAT_HOUR_RAW))
PERFORMANCE_REPORT_BEAT_MINUTE = max(0, min(59, _PERF_BEAT_MINUTE_RAW))

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "rest_framework.authtoken",
    "core",
    "marketdata",
    "signals",
    "execution",
    "risk",
    "backtest",
    "api",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

if USE_SQLITE:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("POSTGRES_DB", "mastertrading"),
            "USER": os.getenv("POSTGRES_USER", "mastertrading"),
            "PASSWORD": os.getenv("POSTGRES_PASSWORD", "mastertrading"),
            "HOST": os.getenv("POSTGRES_HOST", "localhost"),
            "PORT": os.getenv("POSTGRES_PORT", "5434" if DEBUG else "5432"),
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = os.getenv("DEFAULT_TIMEZONE", "UTC")
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.TokenAuthentication",
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        (
            "rest_framework.permissions.IsAuthenticatedOrReadOnly"
            if os.getenv("API_PUBLIC_READ_ENABLED", "false").lower() == "true"
            else "rest_framework.permissions.IsAuthenticated"
        ),
    ],
}

REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "").strip()
_default_redis_url = "redis://localhost:6379/0"
if REDIS_PASSWORD:
    _default_redis_url = f"redis://:{REDIS_PASSWORD}@localhost:6379/0"

CELERY_BROKER_URL = os.getenv("REDIS_URL", _default_redis_url)
CELERY_RESULT_BACKEND = os.getenv("REDIS_URL", _default_redis_url)
CELERY_TIMEZONE = TIME_ZONE
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_DEFAULT_QUEUE = os.getenv("CELERY_TASK_DEFAULT_QUEUE", "celery")
CELERY_DLQ_REDIS_KEY = os.getenv("CELERY_DLQ_REDIS_KEY", "celery:dlq")
CELERY_DLQ_MAXLEN = max(100, int(os.getenv("CELERY_DLQ_MAXLEN", "2000")))
CELERY_NOTIFY_ON_FAILURE = os.getenv("CELERY_NOTIFY_ON_FAILURE", "true").lower() == "true"

# Queue routing: trading tasks get priority over market data
CELERY_TASK_ROUTES = {
    "signals.tasks.run_signal_engine": {"queue": "trading"},
    "signals.tasks.run_trend_engine": {"queue": "trading"},
    "signals.tasks.run_meanrev_engine": {"queue": "trading"},
    "signals.tasks.run_carry_engine": {"queue": "trading"},
    "signals.tasks.run_portfolio_allocator": {"queue": "trading"},
    "execution.tasks.execute_orders": {"queue": "trading"},
    "execution.tasks.retrain_entry_filter_model": {"queue": ML_TRAINING_QUEUE},
    "execution.tasks._log_operation": {"queue": "trading"},
    "risk.tasks.send_performance_report": {"queue": "trading"},
    "marketdata.tasks.fetch_ohlcv_and_funding": {"queue": "marketdata"},
    "marketdata.tasks.fetch_instrument_data": {"queue": "marketdata"},
}

from celery.schedules import crontab

CELERY_BEAT_SCHEDULE = {
    "run-signal-engine": {
        "task": "signals.tasks.run_signal_engine",
        "schedule": crontab(),  # every minute
    },
    "run-trend-engine": {
        "task": "signals.tasks.run_trend_engine",
        "schedule": crontab(),  # every minute
    },
    "run-meanrev-engine": {
        "task": "signals.tasks.run_meanrev_engine",
        "schedule": crontab(),  # every minute
    },
    "run-carry-engine": {
        "task": "signals.tasks.run_carry_engine",
        "schedule": crontab(),  # every minute
    },
    "run-portfolio-allocator": {
        "task": "signals.tasks.run_portfolio_allocator",
        "schedule": crontab(),  # every minute
    },
    "execute-orders": {
        "task": "execution.tasks.execute_orders",
        "schedule": crontab(),  # every minute
    },
    "fetch-ohlcv-and-funding": {
        "task": "marketdata.tasks.fetch_ohlcv_and_funding",
        "schedule": MARKETDATA_POLL_INTERVAL,
    },
}

if PERFORMANCE_REPORT_ENABLED:
    # Runtime scheduling is controlled in risk.tasks.send_performance_report
    # so Telegram can change cadence (interval/daily/on-off) without restarting beat.
    CELERY_BEAT_SCHEDULE["send-performance-report"] = {
        "task": "risk.tasks.send_performance_report",
        "schedule": crontab(),  # every minute
    }

if ML_ENTRY_FILTER_AUTO_TRAIN_ENABLED:
    CELERY_BEAT_SCHEDULE["retrain-entry-filter-model"] = {
        "task": "execution.tasks.retrain_entry_filter_model",
        "schedule": crontab(
            hour=max(0, min(23, ML_ENTRY_FILTER_AUTO_TRAIN_HOUR)),
            minute=max(0, min(59, ML_ENTRY_FILTER_AUTO_TRAIN_MINUTE)),
        ),
    }

if HMM_REGIME_ENABLED:
    CELERY_BEAT_SCHEDULE["run-regime-detection"] = {
        "task": "signals.tasks.run_regime_detection",
        "schedule": crontab(minute=0, hour=f"*/{HMM_REGIME_REFIT_HOURS}"),
    }

if GARCH_ENABLED:
    CELERY_BEAT_SCHEDULE["run-garch-forecast"] = {
        "task": "signals.tasks.run_garch_forecast",
        "schedule": crontab(minute=15, hour=f"*/{GARCH_REFIT_HOURS}"),
    }

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": '{"timestamp":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}',
        },
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "json"},
    },
    "root": {"handlers": ["console"], "level": "INFO"},
}
EXCHANGE = os.getenv("EXCHANGE", "kucoin")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
KUCOIN_SANDBOX = os.getenv("KUCOIN_SANDBOX", "true").lower() == "true"
DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
