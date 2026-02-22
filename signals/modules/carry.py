from __future__ import annotations

import numpy as np
import pandas as pd
from django.conf import settings

from .common import compute_atr_pct, normalize_score


def detect(df_ltf: pd.DataFrame, _df_htf: pd.DataFrame, funding_rates: list[float], session: str) -> dict | None:
    if df_ltf.empty or len(df_ltf) < 80:
        return None
    if not funding_rates or len(funding_rates) < 6:
        return None

    current = float(funding_rates[-1])
    avg_recent = float(np.mean(funding_rates[-6:]))
    std_recent = float(np.std(funding_rates[-12:])) if len(funding_rates) >= 12 else 0.0

    base_thr = float(getattr(settings, "FUNDING_EXTREME_PERCENTILE", 0.001))
    mult = float(getattr(settings, "MODULE_CARRY_FUNDING_MULT", 1.8))
    threshold = max(base_thr * mult, 1e-6)

    direction = ""
    if current >= threshold:
        direction = "short"
    elif current <= -threshold:
        direction = "long"
    if not direction:
        return None

    atr_pct = compute_atr_pct(df_ltf, period=14) or 0.0
    max_atr_pct = float(getattr(settings, "MODULE_CARRY_MAX_ATR_PCT", 0.020))
    if max_atr_pct > 0 and atr_pct >= max_atr_pct:
        # Funding edges degrade in extreme volatility spikes.
        return None
    vol_penalty = min(0.30, atr_pct * 8.0)
    signal_strength = min(1.0, abs(current) / threshold)
    mean_reversion_hint = 0.1 if (direction == "short" and current > avg_recent) or (direction == "long" and current < avg_recent) else 0.0
    raw = max(0.05, signal_strength - vol_penalty + mean_reversion_hint)
    confidence = normalize_score(raw)

    return {
        "direction": direction,
        "raw_score": confidence,
        "confidence": confidence,
        "reasons": {
            "session": session,
            "funding_current": round(current, 8),
            "funding_avg_6": round(avg_recent, 8),
            "funding_std_12": round(std_recent, 8),
            "funding_threshold": round(threshold, 8),
            "atr_pct": round(float(atr_pct) * 100, 4),
        },
    }
