"""
GARCH(1,1) volatility forecasting module.

Fits a GARCH(1,1) model on 1 h log-returns to produce a forward-looking
conditional volatility estimate (σ_{t+1}).  This replaces/augments the
backward-looking ATR for position sizing and TP/SL placement.

Public API:
    fit_and_forecast(symbol)        → fit + 1-step forecast + cache in Redis
    fit_and_forecast_all()          → all active instruments
    get_cached_forecast(symbol)     → read cached forecast from Redis
    garch_vol_pct(symbol)           → convenience: cached hourly vol as fraction
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
# Core GARCH fitting
# ---------------------------------------------------------------------------

def _fit_garch(returns: np.ndarray, p: int = 1, q: int = 1) -> Optional[dict]:
    """Fit a GARCH(p,q) model on a return series (rescaled ×100 for numerics).

    Returns dict with:
        cond_vol   — 1-step-ahead conditional volatility (in original scale)
        omega, alpha, beta — GARCH parameters
        log_likelihood
        aic, bic
    or None on failure.
    """
    from arch import arch_model

    # arch library works better with returns × 100
    rescaled = returns * 100.0

    try:
        model = arch_model(
            rescaled,
            vol="Garch",
            p=p,
            q=q,
            mean="Zero",          # zero-mean for short-horizon crypto
            dist="studentst",     # fat tails typical in crypto
            rescale=False,
        )
        result = model.fit(disp="off", show_warning=False)
    except Exception as exc:
        logger.error("garch: fit failed: %s", exc)
        return None

    # 1-step-ahead forecast
    try:
        fcast = result.forecast(horizon=1)
        # variance forecast (already in rescaled² units)
        var_forecast = float(fcast.variance.iloc[-1, 0])
        vol_forecast = np.sqrt(var_forecast) / 100.0  # back to original scale
    except Exception as exc:
        logger.error("garch: forecast failed: %s", exc)
        return None

    params = result.params
    return {
        "cond_vol": float(vol_forecast),
        "omega": float(params.get("omega", 0)) / 10000.0,   # back to original scale
        "alpha": float(params.get("alpha[1]", 0)),
        "beta": float(params.get("beta[1]", 0)),
        "persistence": float(params.get("alpha[1]", 0)) + float(params.get("beta[1]", 0)),
        "log_likelihood": float(result.loglikelihood),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "n_obs": int(result.nobs),
    }


# ---------------------------------------------------------------------------
# Public API — fit, forecast, cache
# ---------------------------------------------------------------------------

def fit_and_forecast(symbol: str = "BTCUSDT") -> Optional[dict]:
    """Fit GARCH(1,1) on recent 1h candles, forecast 1-step vol, cache in Redis.

    Returns dict with keys:
        cond_vol, cond_vol_annualized, omega, alpha, beta, persistence,
        aic, bic, n_obs, ts
    or None on failure.
    """
    from marketdata.models import Candle
    from core.models import Instrument

    try:
        inst = Instrument.objects.get(symbol=symbol, enabled=True)
    except Instrument.DoesNotExist:
        logger.warning("garch: instrument %s not found", symbol)
        return None

    lookback = int(getattr(settings, "GARCH_LOOKBACK_BARS", 500))
    tf = getattr(settings, "GARCH_TIMEFRAME", "1h")

    qs = (
        Candle.objects.filter(instrument=inst, timeframe=tf)
        .order_by("-ts")
        .values("ts", "close")[:lookback]
    )
    if not qs:
        logger.warning("garch: no %s candles for %s", tf, symbol)
        return None

    rows = list(qs)
    rows.reverse()  # chronological
    closes = np.array([float(r["close"]) for r in rows])

    if len(closes) < 60:
        logger.warning("garch: insufficient data for %s (%d bars)", symbol, len(closes))
        return None

    # Log-returns
    log_ret = np.diff(np.log(closes))

    fit_result = _fit_garch(log_ret)
    if fit_result is None:
        return None

    # Annualize: hourly vol × sqrt(24*365) ≈ × sqrt(8760)
    ann_vol = fit_result["cond_vol"] * np.sqrt(8760)

    result = {
        "cond_vol": round(fit_result["cond_vol"], 8),
        "cond_vol_annualized": round(ann_vol, 4),
        "omega": round(fit_result["omega"], 10),
        "alpha": round(fit_result["alpha"], 6),
        "beta": round(fit_result["beta"], 6),
        "persistence": round(fit_result["persistence"], 6),
        "aic": round(fit_result["aic"], 2),
        "bic": round(fit_result["bic"], 2),
        "n_obs": fit_result["n_obs"],
        "ts": dj_tz.now().isoformat(),
    }

    _cache_forecast(symbol, result)
    logger.info(
        "garch: %s → cond_vol=%.6f ann_vol=%.2f%% alpha=%.4f beta=%.4f persist=%.4f",
        symbol, result["cond_vol"], ann_vol * 100,
        result["alpha"], result["beta"], result["persistence"],
    )
    return result


def fit_and_forecast_all() -> dict[str, dict]:
    """Fit + forecast GARCH for all active instruments.

    Returns {symbol: forecast_dict}.
    """
    from core.models import Instrument

    symbols = list(
        Instrument.objects.filter(enabled=True).values_list("symbol", flat=True)
    )
    results: dict[str, dict] = {}
    for sym in symbols:
        r = fit_and_forecast(sym)
        if r:
            results[sym] = r
    return results


def get_cached_forecast(symbol: str) -> Optional[dict]:
    """Read the cached GARCH forecast from Redis."""
    import redis as _redis

    try:
        r = _redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
        raw = r.get(f"garch:{symbol}")
        if raw:
            return json.loads(raw)
    except Exception as exc:
        logger.debug("garch: Redis read failed for %s: %s", symbol, exc)
    return None


def garch_vol_pct(symbol: str) -> Optional[float]:
    """Return cached GARCH conditional vol as a fraction (e.g. 0.0064 = 0.64%).

    Returns None if GARCH is disabled or no cached value.
    """
    if not getattr(settings, "GARCH_ENABLED", False):
        return None
    cached = get_cached_forecast(symbol)
    if cached:
        return float(cached.get("cond_vol", 0)) or None
    return None


# ---------------------------------------------------------------------------
# Blended volatility: ATR + GARCH
# ---------------------------------------------------------------------------

def blended_vol(symbol: str, atr_pct: float | None) -> float | None:
    """Blend ATR (backward-looking) with GARCH forecast (forward-looking).

    blend = w × garch + (1-w) × atr

    Falls back to pure ATR if GARCH unavailable, or pure GARCH if ATR unavailable.
    Returns None only if both are unavailable.
    """
    garch = garch_vol_pct(symbol)
    w = float(getattr(settings, "GARCH_BLEND_WEIGHT", 0.6))
    w = max(0.0, min(1.0, w))

    if garch is not None and atr_pct is not None and atr_pct > 0:
        return w * garch + (1.0 - w) * atr_pct
    if garch is not None:
        return garch
    if atr_pct is not None and atr_pct > 0:
        return atr_pct
    return None


# ---------------------------------------------------------------------------
# Backtest helper (no Redis, no DB — works with DataFrames)
# ---------------------------------------------------------------------------

def forecast_vol_from_df(df: pd.DataFrame) -> Optional[float]:
    """Forecast 1-step-ahead conditional vol from a 1h OHLCV DataFrame.

    Returns conditional vol as a fraction (e.g. 0.0064), or None.
    """
    if df is None or len(df) < 60:
        return None

    closes = df["close"].astype(float).values
    log_ret = np.diff(np.log(closes))

    fit_result = _fit_garch(log_ret)
    if fit_result is None:
        return None
    return fit_result["cond_vol"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_forecast(symbol: str, data: dict, ttl_hours: int | None = None) -> None:
    """Write GARCH forecast to Redis with TTL."""
    import redis as _redis

    if ttl_hours is None:
        ttl_hours = int(getattr(settings, "GARCH_CACHE_TTL_HOURS", 12))
    try:
        r = _redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
        r.setex(f"garch:{symbol}", ttl_hours * 3600, json.dumps(data))
    except Exception as exc:
        logger.warning("garch: Redis write failed for %s: %s", symbol, exc)
