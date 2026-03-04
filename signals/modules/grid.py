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


def _regime_gate(symbol: str) -> tuple[bool, dict]:
    require_choppy = bool(getattr(settings, "MODULE_GRID_REQUIRE_CHOPPY_REGIME", True))
    hmm_enabled = bool(getattr(settings, "HMM_REGIME_ENABLED", False))
    fail_open = bool(getattr(settings, "MODULE_GRID_REGIME_FAIL_OPEN", True))
    if not require_choppy or not hmm_enabled:
        return True, {"enabled": False, "status": "bypass"}
    if not symbol:
        return fail_open, {"enabled": True, "status": "no_symbol", "fail_open": fail_open}

    try:
        from signals.regime import get_cached_regime

        cached = get_cached_regime(symbol)
    except Exception:
        cached = None
    if not isinstance(cached, dict):
        return fail_open, {"enabled": True, "status": "cache_miss", "fail_open": fail_open}

    name = str(cached.get("name", "")).strip().lower()
    ok = name == "choppy"
    return ok, {
        "enabled": True,
        "status": "ok" if ok else "blocked",
        "name": name or "unknown",
        "confidence": round(float(cached.get("confidence", 0.0) or 0.0), 4),
    }


def detect(
    df_ltf: pd.DataFrame,
    df_htf: pd.DataFrame,
    _funding_rates: list[float],
    session: str,
    symbol: str = "",
) -> dict | None:
    if df_ltf.empty or df_htf.empty or len(df_ltf) < 140 or len(df_htf) < 100:
        return None

    symbol_norm = str(symbol or "").strip().upper()
    allowed_symbols = set(getattr(settings, "MODULE_GRID_ALLOWED_SYMBOLS", set()) or set())
    if allowed_symbols and symbol_norm not in allowed_symbols:
        return None

    allowed_sessions = set(getattr(settings, "MODULE_GRID_ALLOWED_SESSIONS", set()) or set())
    if allowed_sessions and str(session).strip().lower() not in allowed_sessions:
        return None

    adx_htf = compute_adx(df_htf, period=14)
    if adx_htf is None:
        return None
    adx_min = max(0.0, float(getattr(settings, "MODULE_GRID_ADX_MIN", 8.0)))
    adx_max = max(adx_min, float(getattr(settings, "MODULE_GRID_ADX_MAX", 22.0)))
    if float(adx_htf) < adx_min or float(adx_htf) > adx_max:
        return None

    closes_htf = df_htf["close"].astype(float)
    if len(closes_htf) < 60:
        return None
    ema20 = float(closes_htf.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(closes_htf.ewm(span=50, adjust=False).mean().iloc[-1])
    if ema20 <= 0 or ema50 <= 0:
        return None
    ema_gap_pct = abs(ema20 - ema50) / ema50
    ema_gap_max = max(0.0, float(getattr(settings, "MODULE_GRID_EMA_GAP_MAX_PCT", 0.012)))
    if ema_gap_max > 0 and ema_gap_pct > ema_gap_max:
        return None

    atr_pct = compute_atr_pct(df_ltf, period=14)
    if atr_pct is None:
        return None
    atr_min = max(0.0, float(getattr(settings, "MODULE_GRID_ATR_MIN_PCT", 0.006)))
    atr_max = max(atr_min, float(getattr(settings, "MODULE_GRID_ATR_MAX_PCT", 0.03)))
    if float(atr_pct) < atr_min or float(atr_pct) > atr_max:
        return None

    closes = df_ltf["close"].astype(float)
    bb_period = max(10, int(getattr(settings, "MODULE_GRID_BB_PERIOD", 20)))
    bb_std = max(0.5, float(getattr(settings, "MODULE_GRID_BB_STD", 2.0)))
    if len(closes) < bb_period + 10:
        return None
    bb_mean = float(closes.rolling(bb_period).mean().iloc[-1])
    bb_sigma = float(closes.rolling(bb_period).std(ddof=0).iloc[-1])
    if not math.isfinite(bb_mean) or not math.isfinite(bb_sigma) or bb_sigma <= 0:
        return None

    last = float(closes.iloc[-1])
    if last <= 0:
        return None
    z = (last - bb_mean) / bb_sigma
    z_entry = max(0.5, float(getattr(settings, "MODULE_GRID_Z_ENTRY", 1.2)))

    lookback = max(20, int(getattr(settings, "MODULE_GRID_RANGE_LOOKBACK", 96)))
    window = closes.tail(lookback)
    if len(window) < max(20, lookback // 2):
        return None
    range_high = float(window.max())
    range_low = float(window.min())
    if range_high <= range_low:
        return None
    range_width = range_high - range_low
    range_width_pct = range_width / last if last > 0 else 0.0
    min_range_width = max(0.0, float(getattr(settings, "MODULE_GRID_MIN_RANGE_WIDTH_PCT", 0.012)))
    if min_range_width > 0 and range_width_pct < min_range_width:
        return None

    edge_buffer = max(0.0, float(getattr(settings, "MODULE_GRID_RANGE_EDGE_BUFFER_PCT", 0.002)))
    near_top = last >= (range_high * (1.0 - edge_buffer))
    near_bottom = last <= (range_low * (1.0 + edge_buffer))

    direction = ""
    if z >= z_entry and near_top:
        direction = "short"
    elif z <= -z_entry and near_bottom:
        direction = "long"
    if not direction:
        return None

    impulse_details: dict = {}
    if bool(getattr(settings, "MODULE_GRID_IMPULSE_BLOCK_ENABLED", True)):
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
                return None
            impulse_details = {
                "impulse": bool(impulse),
                "impulse_threshold_pct": round(float(impulse_threshold) * 100, 4),
                "body_pct": round(float(state.get("body_pct", 0.0) or 0.0) * 100, 4),
                "candle_direction": candle_direction,
            }

    regime_ok, regime_meta = _regime_gate(symbol)
    if not regime_ok:
        return None

    range_mid = (range_high + range_low) * 0.5
    half_range = max(1e-12, range_width * 0.5)
    edge_norm = min(1.0, abs(last - range_mid) / half_range)
    z_norm = min(1.0, abs(z) / max(2.2, z_entry + 0.8))
    vol_norm = min(1.0, max(0.0, (float(atr_pct) - atr_min) / max(1e-9, atr_max - atr_min)))
    raw = max(0.05, min(1.0, 0.45 * z_norm + 0.35 * edge_norm + 0.20 * vol_norm))
    confidence = normalize_score(raw)

    min_conf = max(0.0, min(1.0, float(getattr(settings, "MODULE_GRID_MIN_CONFIDENCE", 0.40))))
    if confidence < min_conf:
        return None

    reasons = {
        "session": session,
        "adx_htf": round(float(adx_htf), 4),
        "atr_pct": round(float(atr_pct) * 100, 4),
        "zscore": round(float(z), 4),
        "z_entry": round(float(z_entry), 4),
        "bb_mean": round(float(bb_mean), 6),
        "bb_std": round(float(bb_sigma), 6),
        "range_low": round(float(range_low), 6),
        "range_high": round(float(range_high), 6),
        "range_width_pct": round(float(range_width_pct) * 100, 4),
        "near_top": bool(near_top),
        "near_bottom": bool(near_bottom),
        "edge_buffer_pct": round(float(edge_buffer) * 100, 4),
        "ema20_htf": round(float(ema20), 6),
        "ema50_htf": round(float(ema50), 6),
        "ema_gap_pct": round(float(ema_gap_pct) * 100, 4),
        "regime_gate": regime_meta,
    }
    if impulse_details:
        reasons["impulse_guard"] = impulse_details

    return {
        "direction": direction,
        "raw_score": confidence,
        "confidence": confidence,
        "reasons": reasons,
    }
