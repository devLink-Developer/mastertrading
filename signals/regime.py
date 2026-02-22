"""
HMM-based market regime detection.

Fits a 2-state Gaussian HMM on 1 h candle features (log-return,
realised volatility, ADX) to classify the current market environment
as *trending* or *choppy*.

The module exposes three public helpers consumed by the allocator
and by a periodic Celery task:

    fit_and_predict(symbol)   → retrain + predict + cache in Redis
    get_cached_regime(symbol) → read the latest cached regime state
    regime_risk_mult(symbol)  → convenience: cached risk multiplier (default 1.0)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from django.conf import settings
from django.utils import timezone as dj_tz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering (pure numpy/pandas, no I/O)
# ---------------------------------------------------------------------------

def _compute_features(df: pd.DataFrame, vol_window: int = 24) -> Optional[np.ndarray]:
    """Build feature matrix from a 1 h OHLCV DataFrame.

    Features (per bar):
        0: log-return             log(close_t / close_{t-1})
        1: realised volatility    rolling std of log-return (vol_window bars)
        2: ADX(14)                classic Wilder ADX, normalised to [0, 1]

    Returns an (N, 3) numpy array or None if insufficient data.
    """
    if df is None or len(df) < vol_window + 30:
        return None

    closes = df["close"].astype(float).values
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values

    # 1) Log-returns
    log_ret = np.diff(np.log(closes))

    # 2) Realised volatility (rolling std)
    real_vol = pd.Series(log_ret).rolling(vol_window).std().values

    # 3) ADX(14)
    adx_period = 14
    adx_arr = _adx_series(highs, lows, closes, adx_period)

    # Trim to common length (shortest is real_vol because of rolling)
    min_len = min(len(log_ret), len(real_vol), len(adx_arr))
    log_ret = log_ret[-min_len:]
    real_vol = real_vol[-min_len:]
    adx_arr = adx_arr[-min_len:]

    # Drop leading NaNs from rolling windows
    mask = ~(np.isnan(real_vol) | np.isnan(adx_arr) | np.isnan(log_ret))
    features = np.column_stack([log_ret[mask], real_vol[mask], adx_arr[mask] / 100.0])

    if len(features) < 50:
        return None
    return features


def _adx_series(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                period: int = 14) -> np.ndarray:
    """Vectorised ADX computation returning an array aligned to closes[1:]."""
    n = len(closes)
    if n < period * 2 + 1:
        return np.full(n - 1, np.nan)

    plus_dm = np.zeros(n - 1)
    minus_dm = np.zeros(n - 1)
    tr = np.zeros(n - 1)

    for i in range(1, n):
        hi_diff = highs[i] - highs[i - 1]
        lo_diff = lows[i - 1] - lows[i]
        plus_dm[i - 1] = max(hi_diff, 0) if hi_diff > lo_diff else 0
        minus_dm[i - 1] = max(lo_diff, 0) if lo_diff > hi_diff else 0
        tr[i - 1] = max(highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1]))

    # Wilder smoothing
    def _wilder_smooth(arr: np.ndarray, p: int) -> np.ndarray:
        out = np.full_like(arr, np.nan)
        out[p - 1] = np.mean(arr[:p])
        for j in range(p, len(arr)):
            out[j] = (out[j - 1] * (p - 1) + arr[j]) / p
        return out

    sm_tr = _wilder_smooth(tr, period)
    sm_plus = _wilder_smooth(plus_dm, period)
    sm_minus = _wilder_smooth(minus_dm, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = np.where(sm_tr > 0, sm_plus / sm_tr * 100, 0)
        minus_di = np.where(sm_tr > 0, sm_minus / sm_tr * 100, 0)
        dx = np.where(
            (plus_di + minus_di) > 0,
            np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100,
            0,
        )

    adx = _wilder_smooth(dx, period)
    return adx


# ---------------------------------------------------------------------------
# HMM fitting & prediction
# ---------------------------------------------------------------------------

def _fit_hmm(features: np.ndarray, n_states: int = 2,
             n_iter: int = 100, random_state: int = 42):
    """Fit a GaussianHMM and return (model, state_sequence)."""
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )
    model.fit(features)
    states = model.predict(features)
    return model, states


def _label_states(model, n_states: int = 2) -> dict[int, dict]:
    """Assign semantic labels based on emission means.

    The state with **higher mean realised-vol** (feature index 1)
    is labelled *choppy*; the other is *trending*.
    """
    means = model.means_  # shape (n_states, n_features)
    vol_means = means[:, 1]  # feature 1 = realised vol
    choppy_idx = int(np.argmax(vol_means))

    labels: dict[int, dict] = {}
    for s in range(n_states):
        if s == choppy_idx:
            labels[s] = {
                "name": "choppy",
                "risk_mult": float(getattr(
                    settings, "HMM_REGIME_CHOPPY_RISK_MULT", 0.7)),
            }
        else:
            labels[s] = {
                "name": "trending",
                "risk_mult": float(getattr(
                    settings, "HMM_REGIME_TRENDING_RISK_MULT", 1.0)),
            }
    return labels


# ---------------------------------------------------------------------------
# Public API — fit, cache & read
# ---------------------------------------------------------------------------

def fit_and_predict(symbol: str = "BTCUSDT") -> Optional[dict]:
    """Fit HMM on recent 1 h candles, predict current regime, cache in Redis.

    Returns dict with keys:
        state, name, risk_mult, confidence, mean_vol, mean_adx, ts
    or None on failure.
    """
    from marketdata.models import Candle
    from core.models import Instrument

    try:
        inst = Instrument.objects.get(symbol=symbol, enabled=True)
    except Instrument.DoesNotExist:
        logger.warning("regime: instrument %s not found", symbol)
        return None

    lookback = int(getattr(settings, "HMM_REGIME_LOOKBACK_BARS", 500))
    n_states = int(getattr(settings, "HMM_REGIME_N_STATES", 2))
    tf = getattr(settings, "HMM_REGIME_TIMEFRAME", "1h")

    qs = (
        Candle.objects.filter(instrument=inst, timeframe=tf)
        .order_by("-ts")
        .values("ts", "open", "high", "low", "close", "volume")[:lookback]
    )
    if not qs:
        logger.warning("regime: no %s candles for %s", tf, symbol)
        return None

    rows = list(qs)
    rows.reverse()  # chronological order
    df = pd.DataFrame(rows)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    features = _compute_features(df)
    if features is None:
        logger.warning("regime: insufficient features for %s (%d bars)", symbol, len(df))
        return None

    try:
        model, states = _fit_hmm(features, n_states=n_states)
    except Exception as exc:
        logger.error("regime: HMM fit failed for %s: %s", symbol, exc)
        return None

    labels = _label_states(model, n_states)
    current_state = int(states[-1])
    label = labels[current_state]

    # Confidence: posterior probability of current state at last observation
    posteriors = model.predict_proba(features)
    confidence = float(posteriors[-1, current_state])

    # Summary stats for the current regime
    mean_vol = float(model.means_[current_state, 1])
    mean_adx = float(model.means_[current_state, 2] * 100)

    result = {
        "state": current_state,
        "name": label["name"],
        "risk_mult": label["risk_mult"],
        "confidence": round(confidence, 4),
        "mean_vol": round(mean_vol, 6),
        "mean_adx": round(mean_adx, 2),
        "n_bars": len(features),
        "ts": dj_tz.now().isoformat(),
    }

    # Cache in Redis
    _cache_regime(symbol, result)
    logger.info("regime: %s → state=%d (%s) conf=%.2f risk_mult=%.2f vol=%.6f adx=%.1f",
                symbol, current_state, label["name"], confidence,
                label["risk_mult"], mean_vol, mean_adx)
    return result


def fit_and_predict_all() -> dict[str, dict]:
    """Fit + predict regime for all active instruments.

    Returns {symbol: regime_dict}.
    """
    from core.models import Instrument

    results: dict[str, dict] = {}
    symbols = list(
        Instrument.objects.filter(enabled=True)
        .values_list("symbol", flat=True)
    )
    # Use BTC as the primary reference; optionally fit per-symbol
    fit_per_symbol = bool(getattr(settings, "HMM_REGIME_PER_SYMBOL", False))

    if fit_per_symbol:
        for sym in symbols:
            r = fit_and_predict(sym)
            if r:
                results[sym] = r
    else:
        # BTC-only regime: apply same regime to all symbols
        btc_result = fit_and_predict("BTCUSDT")
        if btc_result:
            for sym in symbols:
                results[sym] = btc_result
                _cache_regime(sym, btc_result)

    return results


def get_cached_regime(symbol: str) -> Optional[dict]:
    """Read the cached regime state from Redis."""
    import redis as _redis

    try:
        r = _redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
        raw = r.get(f"regime:{symbol}")
        if raw:
            return json.loads(raw)
    except Exception as exc:
        logger.debug("regime: Redis read failed for %s: %s", symbol, exc)
    return None


def regime_risk_mult(symbol: str) -> float:
    """Return the regime-based risk multiplier (default 1.0 if unavailable)."""
    if not getattr(settings, "HMM_REGIME_ENABLED", False):
        return 1.0
    cached = get_cached_regime(symbol)
    if cached:
        return float(cached.get("risk_mult", 1.0))
    return 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_regime(symbol: str, data: dict, ttl_hours: int | None = None) -> None:
    """Write regime state to Redis with TTL."""
    import redis as _redis

    if ttl_hours is None:
        ttl_hours = int(getattr(settings, "HMM_REGIME_CACHE_TTL_HOURS", 12))
    try:
        r = _redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
        r.setex(f"regime:{symbol}", ttl_hours * 3600, json.dumps(data))
    except Exception as exc:
        logger.warning("regime: Redis write failed for %s: %s", symbol, exc)


# ---------------------------------------------------------------------------
# Backtest helper (no Redis, no DB — works with DataFrames)
# ---------------------------------------------------------------------------

def predict_regime_from_df(df_1h: pd.DataFrame, n_states: int = 2) -> Optional[dict]:
    """Predict regime from a 1 h candle DataFrame (for backtest use).

    Returns the same dict structure as fit_and_predict but without caching.
    """
    features = _compute_features(df_1h)
    if features is None:
        return None

    try:
        model, states = _fit_hmm(features, n_states=n_states)
    except Exception:
        return None

    labels = _label_states(model, n_states)
    current_state = int(states[-1])
    label = labels[current_state]

    posteriors = model.predict_proba(features)
    confidence = float(posteriors[-1, current_state])

    return {
        "state": current_state,
        "name": label["name"],
        "risk_mult": label["risk_mult"],
        "confidence": round(confidence, 4),
        "mean_vol": round(float(model.means_[current_state, 1]), 6),
        "mean_adx": round(float(model.means_[current_state, 2] * 100), 2),
        "n_bars": len(features),
    }
