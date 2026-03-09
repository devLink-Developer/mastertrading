from __future__ import annotations

from typing import Any

import pandas as pd
from django.conf import settings

from core.models import Instrument
from signals.modules.common import compute_adx, latest_candles

REGIME_STATES = {
    "bull_confirmed",
    "bull_weak",
    "range",
    "transition",
    "bear_weak",
    "bear_confirmed",
}

RECOMMENDED_BIASES = {
    "long_bias",
    "short_bias",
    "tactical_long",
    "tactical_short",
    "balanced",
}


def _empty_snapshot() -> dict[str, Any]:
    return {
        "monthly_regime": "transition",
        "weekly_regime": "transition",
        "daily_regime": "transition",
        "confidence": 0.0,
    }


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy().sort_index()
    resampled = pd.DataFrame(
        {
            "open": work["open"].resample(rule).first(),
            "high": work["high"].resample(rule).max(),
            "low": work["low"].resample(rule).min(),
            "close": work["close"].resample(rule).last(),
            "volume": work["volume"].resample(rule).sum(),
        }
    ).dropna()
    return resampled


def _classify_regime_from_df(
    df: pd.DataFrame,
    *,
    fast_span: int,
    slow_span: int,
    slope_bars: int,
    adx_period: int,
    range_gap_pct: float,
    strong_adx: float,
) -> dict[str, Any]:
    if df.empty or len(df) < max(slow_span + slope_bars + 2, adx_period * 2 + 1):
        return {"state": "transition", "confidence": 0.0, "adx": None}

    work = df.copy().sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        work[col] = work[col].astype(float)
    closes = work["close"].astype(float)
    ema_fast = closes.ewm(span=fast_span, adjust=False).mean()
    ema_slow = closes.ewm(span=slow_span, adjust=False).mean()
    last = float(closes.iloc[-1])
    fast_now = float(ema_fast.iloc[-1])
    slow_now = float(ema_slow.iloc[-1])
    fast_prev = float(ema_fast.iloc[-1 - slope_bars]) if len(ema_fast) > slope_bars else float(ema_fast.iloc[0])
    gap_pct = ((fast_now - slow_now) / slow_now) if slow_now else 0.0
    slope_pct = ((fast_now - fast_prev) / abs(fast_prev)) if fast_prev else 0.0
    price_vs_fast = ((last - fast_now) / fast_now) if fast_now else 0.0

    adx = compute_adx(work, period=adx_period)
    adx_val = float(adx) if adx is not None else 0.0

    bull_stack = fast_now > slow_now and last >= fast_now and slope_pct > 0
    bear_stack = fast_now < slow_now and last <= fast_now and slope_pct < 0
    is_range = abs(gap_pct) <= range_gap_pct and adx_val < max(18.0, strong_adx - 4.0)

    if bull_stack and adx_val >= strong_adx:
        state = "bull_confirmed"
    elif bear_stack and adx_val >= strong_adx:
        state = "bear_confirmed"
    elif bull_stack:
        state = "bull_weak"
    elif bear_stack:
        state = "bear_weak"
    elif is_range:
        state = "range"
    else:
        state = "transition"

    confidence = min(
        1.0,
        max(
            0.0,
            abs(gap_pct) * 8.0 + abs(slope_pct) * 6.0 + max(0.0, adx_val - 12.0) / 25.0 + abs(price_vs_fast) * 4.0,
        ),
    )
    return {
        "state": state,
        "confidence": round(confidence, 4),
        "adx": round(adx_val, 4),
        "gap_pct": round(gap_pct, 6),
        "slope_pct": round(slope_pct, 6),
    }


def consolidate_lead_state(
    monthly_regime: str,
    weekly_regime: str,
    daily_regime: str,
) -> str:
    monthly = str(monthly_regime or "transition").strip().lower()
    weekly = str(weekly_regime or "transition").strip().lower()
    daily = str(daily_regime or "transition").strip().lower()
    bear_states = {"bear_confirmed", "bear_weak"}
    bull_states = {"bull_confirmed", "bull_weak"}

    if monthly == "bear_confirmed" and weekly in bear_states:
        return "bear_confirmed"
    if monthly == "bull_confirmed" and weekly in bull_states:
        return "bull_confirmed"
    if monthly in bear_states and weekly in bear_states:
        return "bear_weak"
    if monthly in bull_states and weekly in bull_states:
        return "bull_weak"
    if weekly == "range" and daily == "range":
        return "range"
    if monthly == "range" and weekly == "range":
        return "range"
    return "transition"


def recommended_bias(
    monthly_regime: str,
    weekly_regime: str,
    daily_regime: str,
    lead_state: str | None = None,
) -> str:
    monthly = str(monthly_regime or "transition").strip().lower()
    weekly = str(weekly_regime or "transition").strip().lower()
    daily = str(daily_regime or "transition").strip().lower()
    lead = str(lead_state or consolidate_lead_state(monthly, weekly, daily)).strip().lower()

    if lead in {"bear_confirmed", "bear_weak"}:
        if daily in {"bull_confirmed", "bull_weak", "transition"}:
            return "tactical_long"
        return "short_bias"
    if lead in {"bull_confirmed", "bull_weak"}:
        if daily in {"bear_confirmed", "bear_weak", "transition"}:
            return "tactical_short"
        return "long_bias"
    if monthly == "range" or weekly == "range":
        return "balanced"
    if daily in {"bull_confirmed", "bull_weak"}:
        return "long_bias"
    if daily in {"bear_confirmed", "bear_weak"}:
        return "short_bias"
    return "balanced"


def build_symbol_regime_snapshot(
    instrument: Instrument,
    lookback: int | None = None,
) -> dict[str, Any]:
    d1_lookback = max(120, int(lookback or getattr(settings, "MTF_REGIME_D1_LOOKBACK", 400) or 400))
    df_d1 = latest_candles(instrument, "1d", lookback=d1_lookback)
    if df_d1.empty:
        return _empty_snapshot()

    df_weekly = _resample_ohlcv(df_d1, "W")
    df_monthly = _resample_ohlcv(df_d1, "ME")

    daily = _classify_regime_from_df(
        df_d1.tail(max(90, min(d1_lookback, 180))),
        fast_span=20,
        slow_span=50,
        slope_bars=5,
        adx_period=14,
        range_gap_pct=0.01,
        strong_adx=float(getattr(settings, "MTF_REGIME_DAILY_STRONG_ADX", 22.0) or 22.0),
    )
    weekly = _classify_regime_from_df(
        df_weekly.tail(52),
        fast_span=4,
        slow_span=12,
        slope_bars=2,
        adx_period=3,
        range_gap_pct=0.015,
        strong_adx=float(getattr(settings, "MTF_REGIME_WEEKLY_STRONG_ADX", 20.0) or 20.0),
    )
    monthly = _classify_regime_from_df(
        df_monthly.tail(24),
        fast_span=3,
        slow_span=6,
        slope_bars=1,
        adx_period=2,
        range_gap_pct=0.02,
        strong_adx=float(getattr(settings, "MTF_REGIME_MONTHLY_STRONG_ADX", 18.0) or 18.0),
    )

    return {
        "monthly_regime": monthly["state"],
        "weekly_regime": weekly["state"],
        "daily_regime": daily["state"],
        "confidence": round(
            (
                float(monthly.get("confidence", 0.0) or 0.0)
                + float(weekly.get("confidence", 0.0) or 0.0)
                + float(daily.get("confidence", 0.0) or 0.0)
            ) / 3.0,
            4,
        ),
        "monthly_meta": monthly,
        "weekly_meta": weekly,
        "daily_meta": daily,
    }
