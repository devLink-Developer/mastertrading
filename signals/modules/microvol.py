from __future__ import annotations

import math

import pandas as pd
from django.conf import settings

from .common import (
    compute_adx,
    compute_atr_pct,
    impulse_bar_state,
    is_impulse_bar,
    normalize_score,
)


def detect(
    df_ltf: pd.DataFrame,
    df_htf: pd.DataFrame,
    _funding_rates: list[float],
    session: str,
    symbol: str = "",
) -> dict | None:
    if df_ltf.empty or df_htf.empty or len(df_ltf) < 90 or len(df_htf) < 60:
        return None

    symbol_norm = str(symbol or "").strip().upper()
    allowed_symbols = set(getattr(settings, "MODULE_MICROVOL_ALLOWED_SYMBOLS", set()) or set())
    if allowed_symbols and symbol_norm not in allowed_symbols:
        return None

    session_norm = str(session or "").strip().lower()
    allowed_sessions = set(getattr(settings, "MODULE_MICROVOL_ALLOWED_SESSIONS", set()) or set())
    if allowed_sessions and session_norm not in allowed_sessions:
        return None

    closes_htf = df_htf["close"].astype(float)
    highs_ltf = df_ltf["high"].astype(float)
    lows_ltf = df_ltf["low"].astype(float)
    closes_ltf = df_ltf["close"].astype(float)
    volumes_ltf = df_ltf["volume"].astype(float)
    if closes_htf.empty or closes_ltf.empty:
        return None

    adx_htf = compute_adx(df_htf, period=14)
    if adx_htf is None:
        return None
    adx_min = max(0.0, float(getattr(settings, "MODULE_MICROVOL_HTF_ADX_MIN", 18.0)))
    adx_max = max(adx_min, float(getattr(settings, "MODULE_MICROVOL_HTF_ADX_MAX", 55.0)))
    if float(adx_htf) < adx_min or float(adx_htf) > adx_max:
        return None

    ema20_htf = float(closes_htf.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50_htf = float(closes_htf.ewm(span=50, adjust=False).mean().iloc[-1])
    last_htf = float(closes_htf.iloc[-1])
    htf_pullback_tol = max(
        0.0,
        min(0.02, float(getattr(settings, "MODULE_MICROVOL_HTF_PULLBACK_TOLERANCE_PCT", 0.003))),
    )
    htf_direction = ""
    if ema20_htf > ema50_htf and last_htf >= ema20_htf * (1.0 - htf_pullback_tol):
        htf_direction = "long"
    elif ema20_htf < ema50_htf and last_htf <= ema20_htf * (1.0 + htf_pullback_tol):
        htf_direction = "short"
    if not htf_direction:
        return None

    atr_pct = compute_atr_pct(df_ltf, period=14)
    if atr_pct is None:
        return None
    atr_min = max(0.0, float(getattr(settings, "MODULE_MICROVOL_ATR_MIN_PCT", 0.0025)))
    atr_max = max(atr_min, float(getattr(settings, "MODULE_MICROVOL_ATR_MAX_PCT", 0.0250)))
    if float(atr_pct) < atr_min or float(atr_pct) > atr_max:
        return None

    state = impulse_bar_state(
        df_ltf,
        lookback=int(getattr(settings, "MODULE_MICROVOL_IMPULSE_LOOKBACK", 20)),
    )
    if not state:
        return None
    impulse, impulse_threshold = is_impulse_bar(
        state,
        body_mult=float(getattr(settings, "MODULE_MICROVOL_IMPULSE_BODY_MULT", 1.8)),
        min_body_pct=float(getattr(settings, "MODULE_MICROVOL_IMPULSE_MIN_BODY_PCT", 0.0035)),
    )
    candle_direction = str(state.get("candle_direction") or "").strip().lower()
    if not impulse or candle_direction != htf_direction:
        return None

    ema20_ltf = float(closes_ltf.ewm(span=20, adjust=False).mean().iloc[-1])
    last = float(closes_ltf.iloc[-1])
    if last <= 0:
        return None
    ema20_dist_pct = abs(last - ema20_ltf) / last if last > 0 else 0.0
    max_ema20_dist = max(0.0, float(getattr(settings, "MODULE_MICROVOL_MAX_EMA20_DIST_PCT", 0.010)))
    if max_ema20_dist > 0 and ema20_dist_pct > max_ema20_dist:
        return None

    breakout_lookback = max(8, int(getattr(settings, "MODULE_MICROVOL_BREAKOUT_LOOKBACK", 20)))
    if len(highs_ltf) < breakout_lookback + 2 or len(lows_ltf) < breakout_lookback + 2:
        return None
    prior_high = float(highs_ltf.iloc[-(breakout_lookback + 1):-1].max())
    prior_low = float(lows_ltf.iloc[-(breakout_lookback + 1):-1].min())
    breakout_buffer = max(0.0, float(getattr(settings, "MODULE_MICROVOL_BREAKOUT_BUFFER_PCT", 0.0010)))
    breakout_up = last >= prior_high * (1.0 + breakout_buffer)
    breakout_down = last <= prior_low * (1.0 - breakout_buffer)

    direction = ""
    if htf_direction == "long" and breakout_up:
        direction = "long"
    elif htf_direction == "short" and breakout_down:
        direction = "short"
    if not direction:
        return None

    vol_lookback = max(10, int(getattr(settings, "MODULE_MICROVOL_VOLUME_LOOKBACK", 30)))
    if len(volumes_ltf) < vol_lookback + 2:
        return None
    avg_vol = float(volumes_ltf.iloc[-(vol_lookback + 1):-1].mean())
    last_vol = float(volumes_ltf.iloc[-1])
    if not math.isfinite(avg_vol) or avg_vol <= 0:
        return None
    volume_ratio = last_vol / avg_vol
    min_volume_ratio = max(0.0, float(getattr(settings, "MODULE_MICROVOL_MIN_VOLUME_RATIO", 1.6)))
    if volume_ratio < min_volume_ratio:
        return None

    close_loc = float(state.get("close_location", 0.5) or 0.5)
    if direction == "long" and close_loc < 0.60:
        return None
    if direction == "short" and close_loc > 0.40:
        return None

    breakout_ref = prior_high if direction == "long" else prior_low
    breakout_pct = abs(last - breakout_ref) / max(1e-12, breakout_ref)
    breakout_norm = min(1.0, breakout_pct / max(0.0005, breakout_buffer * 2.0 if breakout_buffer > 0 else 0.002))
    volume_norm = min(1.0, max(0.0, (volume_ratio - min_volume_ratio) / max(0.25, min_volume_ratio)))
    atr_norm = min(1.0, max(0.0, (float(atr_pct) - atr_min) / max(1e-9, atr_max - atr_min)))
    adx_norm = min(1.0, max(0.0, (float(adx_htf) - adx_min) / max(1e-9, adx_max - adx_min)))
    close_norm = close_loc if direction == "long" else (1.0 - close_loc)
    raw = (
        0.30 * breakout_norm
        + 0.25 * volume_norm
        + 0.20 * atr_norm
        + 0.15 * adx_norm
        + 0.10 * close_norm
    )
    confidence = normalize_score(max(0.05, min(1.0, raw)))
    min_conf = max(0.0, min(1.0, float(getattr(settings, "MODULE_MICROVOL_MIN_CONFIDENCE", 0.55))))
    if confidence < min_conf:
        return None

    return {
        "direction": direction,
        "raw_score": confidence,
        "confidence": confidence,
        "reasons": {
            "session": session_norm,
            "atr_pct": round(float(atr_pct) * 100, 4),
            "adx_htf": round(float(adx_htf), 4),
            "htf_direction": htf_direction,
            "htf_pullback_tolerance_pct": round(float(htf_pullback_tol) * 100, 4),
            "ema20_htf": round(float(ema20_htf), 6),
            "ema50_htf": round(float(ema50_htf), 6),
            "ema20_ltf": round(float(ema20_ltf), 6),
            "ema20_dist_pct": round(float(ema20_dist_pct) * 100, 4),
            "prior_high": round(float(prior_high), 6),
            "prior_low": round(float(prior_low), 6),
            "breakout_pct": round(float(breakout_pct) * 100, 4),
            "breakout_buffer_pct": round(float(breakout_buffer) * 100, 4),
            "volume_ratio": round(float(volume_ratio), 4),
            "close_location": round(float(close_loc), 4),
            "impulse_threshold_pct": round(float(impulse_threshold) * 100, 4),
            "active_modules": ["microvol"],
        },
    }
