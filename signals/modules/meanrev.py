from __future__ import annotations

import numpy as np
import pandas as pd
from django.conf import settings

from .common import (
    bounce_pct,
    compute_adx,
    impulse_bar_state,
    is_impulse_bar,
    normalize_score,
)


def detect(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, _funding_rates: list[float], session: str) -> dict | None:
    if df_ltf.empty or df_htf.empty or len(df_ltf) < 120 or len(df_htf) < 80:
        return None

    adx_htf = compute_adx(df_htf, period=14)
    if adx_htf is None:
        return None
    range_max = float(getattr(settings, "MODULE_ADX_RANGE_MAX", 18.0))
    if adx_htf > range_max:
        return None

    closes = df_ltf["close"].astype(float)
    ema = closes.ewm(span=20, adjust=False).mean()
    dev = closes - ema
    window = min(60, len(dev))
    if window < 30:
        return None
    mean = float(dev.tail(window).mean())
    std = float(dev.tail(window).std(ddof=0))
    if std <= 0:
        return None
    z = float((dev.iloc[-1] - mean) / std)

    z_entry = float(getattr(settings, "MODULE_MEANREV_Z_ENTRY", 1.2))
    direction = ""
    if z >= z_entry:
        direction = "short"
    elif z <= -z_entry:
        direction = "long"
    if not direction:
        return None

    impulse_details: dict = {}
    if bool(getattr(settings, "MODULE_MEANREV_IMPULSE_BLOCK_ENABLED", True)):
        state = impulse_bar_state(
            df_ltf,
            lookback=int(getattr(settings, "MODULE_IMPULSE_LOOKBACK", 20)),
        )
        if state:
            impulse, impulse_threshold = is_impulse_bar(
                state,
                body_mult=float(getattr(settings, "MODULE_IMPULSE_BODY_MULT", 2.2)),
                min_body_pct=float(getattr(settings, "MODULE_IMPULSE_MIN_BODY_PCT", 0.006)),
            )
            candle_direction = str(state.get("candle_direction") or "")
            if impulse and candle_direction in {"long", "short"} and candle_direction != direction:
                # Avoid immediate knife-catching right after displacement candles.
                return None
            impulse_details = {
                "impulse": bool(impulse),
                "impulse_threshold_pct": round(impulse_threshold * 100, 4),
                "body_pct": round(float(state.get("body_pct", 0.0) or 0.0) * 100, 4),
                "candle_direction": candle_direction,
            }

    # --- Bounce / counter-momentum filter ---
    bounce_lookback = int(getattr(settings, "MODULE_BOUNCE_LOOKBACK", 30))
    bounce_block_pct = float(getattr(settings, "MODULE_BOUNCE_BLOCK_PCT", 0.50))
    bounce_info = bounce_pct(df_ltf, lookback=bounce_lookback)
    if bounce_info:
        if direction == "short" and bounce_info.get("bounce_from_low_pct", 0) >= bounce_block_pct:
            return None
        if direction == "long" and bounce_info.get("bounce_from_high_pct", 0) >= bounce_block_pct:
            return None

    raw = normalize_score(min(1.0, abs(z) / max(2.5, z_entry + 0.8)))
    reasons = {
        "session": session,
        "adx_htf": round(float(adx_htf), 4),
        "zscore": round(z, 4),
        "z_entry": round(z_entry, 4),
        "ema20": round(float(ema.iloc[-1]), 6),
        "last": round(float(closes.iloc[-1]), 6),
    }
    if impulse_details:
        reasons["impulse_guard"] = impulse_details
    if bounce_info:
        reasons["bounce_guard"] = bounce_info

    return {
        "direction": direction,
        "raw_score": raw,
        "confidence": raw,
        "reasons": reasons,
    }
