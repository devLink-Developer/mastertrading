from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MTF range helpers
# ---------------------------------------------------------------------------

def _fetch_daily_candles(symbol: str, lookback: int = 60) -> pd.DataFrame:
    """Fetch 1d candles from DB for a given symbol.  Returns empty DF on any error."""
    try:
        from core.models import Instrument
        from .common import latest_candles

        inst = Instrument.objects.filter(symbol=symbol, enabled=True).first()
        if inst is None:
            return pd.DataFrame()
        return latest_candles(inst, "1d", lookback=lookback)
    except Exception:
        return pd.DataFrame()


def _compute_mtf_ranges(
    df_d1: pd.DataFrame,
    last_price: float,
) -> dict | None:
    """Compute monthly (30d) and weekly (7d) high/low from daily candles.

    Returns dict with range levels or None if insufficient data.
    """
    if df_d1.empty or len(df_d1) < 7:
        return None

    highs = df_d1["high"].astype(float)
    lows = df_d1["low"].astype(float)

    # Weekly range (last 7 daily candles)
    w_hi = float(highs.tail(7).max())
    w_lo = float(lows.tail(7).min())

    # Monthly range (last 30 daily candles, or all available if < 30)
    m_bars = min(30, len(df_d1))
    m_hi = float(highs.tail(m_bars).max())
    m_lo = float(lows.tail(m_bars).min())

    if m_hi <= m_lo or w_hi <= w_lo or last_price <= 0:
        return None

    # Composite range: intersection gives tighter, more meaningful levels
    # Use weekly as primary, monthly as context
    range_high = w_hi
    range_low = w_lo
    range_width = range_high - range_low
    range_width_pct = range_width / last_price

    # Position within the range: 0.0 = at low, 1.0 = at high
    range_position = (last_price - range_low) / range_width if range_width > 0 else 0.5

    return {
        "weekly_high": w_hi,
        "weekly_low": w_lo,
        "monthly_high": m_hi,
        "monthly_low": m_lo,
        "range_high": range_high,
        "range_low": range_low,
        "range_width_pct": range_width_pct,
        "range_position": range_position,
        "daily_bars": len(df_d1),
    }


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

    # --- HTF gates (ADX range, EMA gap) ---
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

    # --- LTF volatility gate ---
    atr_pct = compute_atr_pct(df_ltf, period=14)
    if atr_pct is None:
        return None
    atr_min = max(0.0, float(getattr(settings, "MODULE_GRID_ATR_MIN_PCT", 0.006)))
    atr_max = max(atr_min, float(getattr(settings, "MODULE_GRID_ATR_MAX_PCT", 0.03)))
    if float(atr_pct) < atr_min or float(atr_pct) > atr_max:
        return None

    closes = df_ltf["close"].astype(float)
    last = float(closes.iloc[-1])
    if last <= 0:
        return None

    # --- MTF range (primary driver: 1d candles) ---
    use_mtf = bool(getattr(settings, "MODULE_GRID_MTF_RANGE_ENABLED", True))
    mtf = None
    if use_mtf and symbol_norm:
        d1_lookback = max(7, int(getattr(settings, "MODULE_GRID_D1_LOOKBACK", 45)))
        df_d1 = _fetch_daily_candles(symbol_norm, lookback=d1_lookback)
        mtf = _compute_mtf_ranges(df_d1, last)

    # --- LTF Bollinger as secondary confirmation ---
    bb_period = max(10, int(getattr(settings, "MODULE_GRID_BB_PERIOD", 20)))
    bb_std = max(0.5, float(getattr(settings, "MODULE_GRID_BB_STD", 2.0)))
    if len(closes) < bb_period + 10:
        return None
    bb_mean = float(closes.rolling(bb_period).mean().iloc[-1])
    bb_sigma = float(closes.rolling(bb_period).std(ddof=0).iloc[-1])
    if not math.isfinite(bb_mean) or not math.isfinite(bb_sigma) or bb_sigma <= 0:
        return None
    z = (last - bb_mean) / bb_sigma
    z_entry = max(0.5, float(getattr(settings, "MODULE_GRID_Z_ENTRY", 1.2)))

    # --- Decide direction ---
    direction = ""
    buy_zone_pct = max(0.0, min(0.5, float(getattr(settings, "MODULE_GRID_BUY_ZONE_PCT", 0.15))))
    sell_zone_pct = max(0.0, min(0.5, float(getattr(settings, "MODULE_GRID_SELL_ZONE_PCT", 0.15))))

    if mtf is not None:
        # MTF-driven: position within weekly range
        pos = mtf["range_position"]  # 0.0 = at low, 1.0 = at high
        if pos <= buy_zone_pct and z <= -z_entry * 0.5:
            direction = "long"
        elif pos >= (1.0 - sell_zone_pct) and z >= z_entry * 0.5:
            direction = "short"
    else:
        # Fallback: original Bollinger + LTF range logic
        lookback = max(20, int(getattr(settings, "MODULE_GRID_RANGE_LOOKBACK", 96)))
        window = closes.tail(lookback)
        if len(window) < max(20, lookback // 2):
            return None
        range_high = float(window.max())
        range_low = float(window.min())
        if range_high <= range_low:
            return None
        range_width_pct = (range_high - range_low) / last
        min_range_width = max(0.0, float(getattr(settings, "MODULE_GRID_MIN_RANGE_WIDTH_PCT", 0.012)))
        if min_range_width > 0 and range_width_pct < min_range_width:
            return None
        edge_buffer = max(0.0, float(getattr(settings, "MODULE_GRID_RANGE_EDGE_BUFFER_PCT", 0.002)))
        near_top = last >= (range_high * (1.0 - edge_buffer))
        near_bottom = last <= (range_low * (1.0 + edge_buffer))
        if z >= z_entry and near_top:
            direction = "short"
        elif z <= -z_entry and near_bottom:
            direction = "long"

    if not direction:
        return None

    # --- Min range width gate (MTF mode) ---
    if mtf is not None:
        min_range_width = max(0.0, float(getattr(settings, "MODULE_GRID_MIN_RANGE_WIDTH_PCT", 0.012)))
        if min_range_width > 0 and mtf["range_width_pct"] < min_range_width:
            return None

    # --- Impulse guard ---
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

    # --- Regime gate ---
    regime_ok, regime_meta = _regime_gate(symbol)
    if not regime_ok:
        return None

    # --- Score ---
    z_norm = min(1.0, abs(z) / max(2.2, z_entry + 0.8))
    vol_norm = min(1.0, max(0.0, (float(atr_pct) - atr_min) / max(1e-9, atr_max - atr_min)))

    if mtf is not None:
        pos = mtf["range_position"]
        # How deep into buy/sell zone: 1.0 = at range extreme, 0.0 = at zone boundary
        if direction == "long":
            edge_depth = max(0.0, min(1.0, 1.0 - (pos / max(1e-9, buy_zone_pct))))
        else:
            edge_depth = max(0.0, min(1.0, (pos - (1.0 - sell_zone_pct)) / max(1e-9, sell_zone_pct)))
        raw = max(0.05, min(1.0, 0.50 * edge_depth + 0.30 * z_norm + 0.20 * vol_norm))
    else:
        # Fallback scoring (original)
        lookback = max(20, int(getattr(settings, "MODULE_GRID_RANGE_LOOKBACK", 96)))
        window = closes.tail(lookback)
        r_hi = float(window.max())
        r_lo = float(window.min())
        r_mid = (r_hi + r_lo) * 0.5
        half_r = max(1e-12, (r_hi - r_lo) * 0.5)
        edge_norm = min(1.0, abs(last - r_mid) / half_r)
        raw = max(0.05, min(1.0, 0.45 * z_norm + 0.35 * edge_norm + 0.20 * vol_norm))

    confidence = normalize_score(raw)
    min_conf = max(0.0, min(1.0, float(getattr(settings, "MODULE_GRID_MIN_CONFIDENCE", 0.40))))
    if confidence < min_conf:
        return None

    # --- Build SL/TP hints from structural levels ---
    sl_price_hint = 0.0
    tp_price_hint = 0.0
    sl_buffer_pct = max(0.0, float(getattr(settings, "MODULE_GRID_SL_BUFFER_PCT", 0.003)))
    tp_buffer_pct = max(0.0, float(getattr(settings, "MODULE_GRID_TP_BUFFER_PCT", 0.002)))

    if mtf is not None:
        if direction == "long":
            # SL just below weekly low; TP toward weekly high
            sl_price_hint = mtf["range_low"] * (1.0 - sl_buffer_pct)
            tp_price_hint = mtf["range_high"] * (1.0 - tp_buffer_pct)
        else:
            # SL just above weekly high; TP toward weekly low
            sl_price_hint = mtf["range_high"] * (1.0 + sl_buffer_pct)
            tp_price_hint = mtf["range_low"] * (1.0 + tp_buffer_pct)

    # --- Reasons ---
    reasons: dict = {
        "session": session,
        "adx_htf": round(float(adx_htf), 4),
        "atr_pct": round(float(atr_pct) * 100, 4),
        "zscore": round(float(z), 4),
        "z_entry": round(float(z_entry), 4),
        "bb_mean": round(float(bb_mean), 6),
        "bb_std": round(float(bb_sigma), 6),
        "ema20_htf": round(float(ema20), 6),
        "ema50_htf": round(float(ema50), 6),
        "ema_gap_pct": round(float(ema_gap_pct) * 100, 4),
        "regime_gate": regime_meta,
        "mtf_enabled": use_mtf,
    }
    if mtf is not None:
        reasons["mtf"] = {
            "weekly_high": round(mtf["weekly_high"], 6),
            "weekly_low": round(mtf["weekly_low"], 6),
            "monthly_high": round(mtf["monthly_high"], 6),
            "monthly_low": round(mtf["monthly_low"], 6),
            "range_position": round(mtf["range_position"], 4),
            "range_width_pct": round(mtf["range_width_pct"] * 100, 4),
            "daily_bars": mtf["daily_bars"],
        }
        reasons["sl_price_hint"] = round(sl_price_hint, 6)
        reasons["tp_price_hint"] = round(tp_price_hint, 6)
    else:
        # Fallback range info
        lookback = max(20, int(getattr(settings, "MODULE_GRID_RANGE_LOOKBACK", 96)))
        window = closes.tail(lookback)
        reasons["range_low"] = round(float(window.min()), 6)
        reasons["range_high"] = round(float(window.max()), 6)
        r_width = float(window.max()) - float(window.min())
        reasons["range_width_pct"] = round(r_width / last * 100, 4) if last > 0 else 0

    if impulse_details:
        reasons["impulse_guard"] = impulse_details

    result: dict = {
        "direction": direction,
        "raw_score": confidence,
        "confidence": confidence,
        "reasons": reasons,
    }
    # Attach SL/TP hints at payload top level for execution to pick up
    if sl_price_hint > 0:
        result["sl_price_hint"] = round(sl_price_hint, 8)
    if tp_price_hint > 0:
        result["tp_price_hint"] = round(tp_price_hint, 8)

    return result
