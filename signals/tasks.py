from __future__ import annotations

import logging
from typing import Tuple, Optional, List
from datetime import timedelta

import numpy as np
import pandas as pd
from celery import shared_task
from django.conf import settings
from django.utils import timezone as dj_tz

from core.models import Instrument
from marketdata.models import Candle, FundingRate
from signals.sessions import (
    get_current_session,
    get_session_score_min,
    is_dead_session,
)
from signals.direction_policy import (
    get_direction_mode,
    is_direction_allowed,
)
from .models import Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _latest_candles(instrument: Instrument, tf: str, lookback: int = 300) -> pd.DataFrame:
    qs = (
        Candle.objects.filter(instrument=instrument, timeframe=tf)
        .order_by("-ts")[:lookback]
        .values("ts", "open", "high", "low", "close", "volume")
    )
    if not qs:
        return pd.DataFrame()
    df = pd.DataFrame(list(qs)).sort_values("ts")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df.set_index("ts", inplace=True)
    return df


def _latest_funding(instrument: Instrument, lookback: int = 100) -> List[float]:
    """Return recent funding rates as a list of floats (newest last)."""
    qs = (
        FundingRate.objects.filter(instrument=instrument)
        .order_by("-ts")[:lookback]
        .values_list("rate", flat=True)
    )
    return [float(r) for r in reversed(list(qs))]


# ---------------------------------------------------------------------------
# Volatility / regime helpers  (used by regime filter + backtest engine)
# ---------------------------------------------------------------------------

def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """ATR as a percentage of last close price.  Works on any OHLCV DataFrame."""
    if len(df) < period + 1:
        return None
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    trs = []
    for i in range(1, len(df)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    atr = float(np.mean(trs[-period:]))
    last_close = closes[-1]
    if last_close == 0:
        return None
    return atr / last_close * 100  # return as percentage


def compute_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Classic ADX (Welles Wilder).  Returns value 0-100 or None."""
    if len(df) < period * 2 + 1:
        return None
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)

    plus_dm = np.zeros(len(df))
    minus_dm = np.zeros(len(df))
    tr = np.zeros(len(df))

    for i in range(1, len(df)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    # Wilder smoothing
    atr_arr = np.zeros(len(df))
    plus_di_arr = np.zeros(len(df))
    minus_di_arr = np.zeros(len(df))

    atr_arr[period] = np.sum(tr[1:period + 1])
    s_plus = np.sum(plus_dm[1:period + 1])
    s_minus = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, len(df)):
        atr_arr[i] = atr_arr[i - 1] - atr_arr[i - 1] / period + tr[i]
        s_plus = s_plus - s_plus / period + plus_dm[i]
        s_minus = s_minus - s_minus / period + minus_dm[i]
        if atr_arr[i] > 0:
            plus_di_arr[i] = 100 * s_plus / atr_arr[i]
            minus_di_arr[i] = 100 * s_minus / atr_arr[i]

    dx = np.zeros(len(df))
    for i in range(period + 1, len(df)):
        denom = plus_di_arr[i] + minus_di_arr[i]
        if denom > 0:
            dx[i] = 100 * abs(plus_di_arr[i] - minus_di_arr[i]) / denom

    # ADX = smoothed DX over period
    start = period * 2
    if start >= len(df):
        return None
    adx = float(np.mean(dx[start - period + 1:start + 1]))
    for i in range(start + 1, len(df)):
        adx = (adx * (period - 1) + dx[i]) / period
    return round(adx, 2)


def check_regime_filter(df: pd.DataFrame, config=None) -> Tuple[bool, dict]:
    """
    Check whether the current market regime allows trading.

    Args:
        df: OHLCV DataFrame for the configured timeframe.
        config: RegimeFilterConfig instance (or None to load from DB).

    Returns:
        (allowed, details) — allowed=False means market is sideways / low-vol.
    """
    if config is None:
        from risk.models import RegimeFilterConfig
        try:
            config = RegimeFilterConfig.get()
        except Exception:
            return True, {"regime_filter": "db_unavailable"}

    if not config.enabled:
        return True, {"regime_filter": "disabled"}

    if df.empty or len(df) < 30:
        return True, {"regime_filter": "insufficient_data"}

    if config.filter_type == "atr_pct":
        atr_pct = compute_atr_pct(df, period=config.atr_period)
        if atr_pct is None:
            return True, {"regime_filter": "atr_calc_failed"}
        allowed = atr_pct >= config.atr_min_pct
        return allowed, {
            "regime_filter": "atr_pct",
            "atr_pct": round(atr_pct, 4),
            "threshold": config.atr_min_pct,
            "allowed": allowed,
        }

    elif config.filter_type == "adx":
        adx_val = compute_adx(df, period=config.adx_period)
        if adx_val is None:
            return True, {"regime_filter": "adx_calc_failed"}
        allowed = adx_val >= config.adx_min
        return allowed, {
            "regime_filter": "adx",
            "adx": adx_val,
            "threshold": config.adx_min,
            "allowed": allowed,
        }

    return True, {"regime_filter": "unknown_type"}


# ---------------------------------------------------------------------------
# Market structure helpers
# ---------------------------------------------------------------------------

def _swing_points(df: pd.DataFrame, period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Detect swing highs and swing lows using rolling window."""
    highs = df["high"]
    lows = df["low"]
    window = period * 2 + 1
    swing_high = (highs == highs.rolling(window=window, center=True).max())
    swing_low = (lows == lows.rolling(window=window, center=True).min())
    return swing_high.fillna(False), swing_low.fillna(False)


def _get_swing_levels(df: pd.DataFrame, period: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return DataFrames of swing highs and swing lows with their prices and timestamps."""
    swing_high_mask, swing_low_mask = _swing_points(df, period)
    sw_highs = df.loc[swing_high_mask, ["high"]].copy()
    sw_highs.columns = ["price"]
    sw_lows = df.loc[swing_low_mask, ["low"]].copy()
    sw_lows.columns = ["price"]
    return sw_highs, sw_lows


def _trend_from_swings(df: pd.DataFrame, period: int = 3) -> str:
    """Determine trend from swing structure + EMA confirmation.

    Uses deeper swing analysis (5 points) combined with EMA 20/50 cross
    to avoid labeling a macro downtrend as 'range' due to a minor bounce.
    """
    sw_highs, sw_lows = _get_swing_levels(df, period)
    highs = sw_highs["price"].tail(5).values
    lows = sw_lows["price"].tail(5).values

    # Swing-based trend (check majority of recent swings)
    swing_trend = "range"
    if len(highs) >= 3 and len(lows) >= 3:
        hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
        lh_count = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i - 1])
        hl_count = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i - 1])
        ll_count = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i - 1])
        if hh_count > lh_count and hl_count > ll_count:
            swing_trend = "bull"
        elif lh_count > hh_count and ll_count > hl_count:
            swing_trend = "bear"
        # Asymmetric: lower highs dominate even if lows are mixed → still bear
        elif lh_count >= 2 and lh_count > hh_count:
            swing_trend = "bear"
        elif hh_count >= 2 and hh_count > lh_count:
            swing_trend = "bull"

    # EMA confirmation: EMA20 vs EMA50 cross on HTF
    ema_trend = "range"
    if len(df) >= 50:
        closes = df["close"].values
        ema20 = pd.Series(closes).ewm(span=20).mean().values
        ema50 = pd.Series(closes).ewm(span=50).mean().values
        ema_diff_pct = (ema20[-1] - ema50[-1]) / ema50[-1] * 100 if ema50[-1] else 0
        price_vs_ema20 = (closes[-1] - ema20[-1]) / ema20[-1] * 100 if ema20[-1] else 0
        if ema_diff_pct > 0.5 and price_vs_ema20 > -1:
            ema_trend = "bull"
        elif ema_diff_pct < -0.5 and price_vs_ema20 < 1:
            ema_trend = "bear"

    # Combine: if both agree → clear trend; if EMA says bear but swings say range → bear
    if swing_trend == ema_trend:
        return swing_trend
    if ema_trend == "bear" and swing_trend == "range":
        return "bear"  # EMA downtrend overrides ambiguous swing range
    if ema_trend == "bull" and swing_trend == "range":
        return "bull"  # EMA uptrend overrides ambiguous swing range
    # Conflicting signals → range
    return "range"


# ---------------------------------------------------------------------------
# CHoCH vs BOS detection
# ---------------------------------------------------------------------------

def _detect_structure_break(df: pd.DataFrame, period: int = 2) -> Tuple[Optional[str], dict]:
    """
    Detect the most recent structure break:
    - BOS (Break of Structure): continuation break (HH in uptrend, LL in downtrend)
    - CHoCH (Change of Character): reversal break (HH after downtrend, LL after uptrend)

    Returns (break_type, details) where break_type is 'bos_bull', 'bos_bear', 'choch_bull', 'choch_bear', or None.
    """
    sw_highs, sw_lows = _get_swing_levels(df, period)
    if len(sw_highs) < 3 or len(sw_lows) < 3:
        return None, {}

    highs = sw_highs["price"].tail(4).values
    lows = sw_lows["price"].tail(4).values
    last_close = df["close"].iloc[-1]

    # Prior trend from the earlier swings (2nd/3rd-to-last)
    if len(highs) >= 3 and len(lows) >= 3:
        prior_hh = highs[-3] < highs[-2]  # was making higher highs
        prior_ll = lows[-3] > lows[-2]    # was making lower lows
        prior_trend_bull = prior_hh and (lows[-3] <= lows[-2])
        prior_trend_bear = prior_ll and (highs[-3] >= highs[-2])
    else:
        prior_trend_bull = False
        prior_trend_bear = False

    # Current break: does last close break the most recent swing?
    last_swing_high = highs[-1]
    last_swing_low = lows[-1]

    if last_close > last_swing_high:
        if prior_trend_bear or (not prior_trend_bull):
            return "choch_bull", {"break_level": float(last_swing_high), "type": "CHoCH", "direction": "bull"}
        else:
            return "bos_bull", {"break_level": float(last_swing_high), "type": "BOS", "direction": "bull"}

    if last_close < last_swing_low:
        if prior_trend_bull or (not prior_trend_bear):
            return "choch_bear", {"break_level": float(last_swing_low), "type": "CHoCH", "direction": "bear"}
        else:
            return "bos_bear", {"break_level": float(last_swing_low), "type": "BOS", "direction": "bear"}

    return None, {}


# ---------------------------------------------------------------------------
# Liquidity sweep detection
# ---------------------------------------------------------------------------

def _detect_liquidity_sweep(df: pd.DataFrame, period: int = 3, lookback_bars: int = 20) -> Tuple[Optional[str], dict]:
    """
    Detect liquidity sweep: price breaks a swing level but quickly reverses (rejection).
    - Sweep low: wick below swing low but close above it (bullish sweep)
    - Sweep high: wick above swing high but close below it (bearish sweep)

    Returns ('sweep_low', details) or ('sweep_high', details) or (None, {}).
    """
    sw_highs, sw_lows = _get_swing_levels(df, period)

    if len(sw_highs) < 2 or len(sw_lows) < 2:
        return None, {}

    recent = df.tail(lookback_bars)
    if len(recent) < 3:
        return None, {}

    # Check last few candles for sweep pattern
    for i in range(-1, max(-4, -len(recent)), -1):
        candle_low = recent["low"].iloc[i]
        candle_high = recent["high"].iloc[i]
        candle_close = recent["close"].iloc[i]
        candle_open = recent["open"].iloc[i]

        # Sweep low (bullish): wick went below a swing low but candle closed above it
        for _, sw_row in sw_lows.tail(5).iterrows():
            sw_level = sw_row["price"]
            if candle_low < sw_level and candle_close > sw_level:
                # Rejection: close back above the level, with a wick
                wick_ratio = (candle_close - candle_low) / (candle_high - candle_low + 1e-10)
                if wick_ratio > 0.5:  # strong rejection
                    return "sweep_low", {
                        "sweep_level": float(sw_level),
                        "wick_low": float(candle_low),
                        "close": float(candle_close),
                        "wick_ratio": round(float(wick_ratio), 3),
                    }

        # Sweep high (bearish): wick went above a swing high but candle closed below it
        for _, sw_row in sw_highs.tail(5).iterrows():
            sw_level = sw_row["price"]
            if candle_high > sw_level and candle_close < sw_level:
                wick_ratio = (candle_high - candle_close) / (candle_high - candle_low + 1e-10)
                if wick_ratio > 0.5:
                    return "sweep_high", {
                        "sweep_level": float(sw_level),
                        "wick_high": float(candle_high),
                        "close": float(candle_close),
                        "wick_ratio": round(float(wick_ratio), 3),
                    }

    return None, {}


# ---------------------------------------------------------------------------
# FVG (Fair Value Gap) detection
# ---------------------------------------------------------------------------

def _detect_fvg(df: pd.DataFrame, lookback: int = 10) -> Tuple[Optional[str], dict]:
    """
    Detect Fair Value Gaps (3-candle imbalance pattern).
    Bull FVG: candle[i] low > candle[i-2] high (gap up)
    Bear FVG: candle[i] high < candle[i-2] low (gap down)
    """
    if len(df) < 5:
        return None, {}
    recent = df.tail(lookback)
    last_close_price = recent["close"].iloc[-1] if len(recent) > 0 else 0
    # Minimum FVG size: 0.05% of the current price to filter out noise
    min_gap_pct = 0.0005
    min_gap = last_close_price * min_gap_pct if last_close_price else 0

    # Scan from newest to oldest
    for i in range(len(recent) - 1, 1, -1):
        h_prev2 = recent["high"].iloc[i - 2]
        l_prev2 = recent["low"].iloc[i - 2]
        h_cur = recent["high"].iloc[i]
        l_cur = recent["low"].iloc[i]
        mid_vol = recent["volume"].iloc[i - 1]

        if l_cur > h_prev2:
            gap_size = float(l_cur - h_prev2)
            if gap_size < min_gap:
                continue  # too small, noise
            return "bull", {
                "gap_top": float(l_cur),
                "gap_bottom": float(h_prev2),
                "gap_size": gap_size,
                "gap_pct": round(gap_size / last_close_price * 100, 4) if last_close_price else 0,
                "displacement_volume": float(mid_vol),
            }
        if h_cur < l_prev2:
            gap_size = float(l_prev2 - h_cur)
            if gap_size < min_gap:
                continue  # too small, noise
            return "bear", {
                "gap_top": float(l_prev2),
                "gap_bottom": float(h_cur),
                "gap_size": gap_size,
                "gap_pct": round(gap_size / last_close_price * 100, 4) if last_close_price else 0,
                "displacement_volume": float(mid_vol),
            }
    return None, {}


# ---------------------------------------------------------------------------
# Order Block detection (simplified)
# ---------------------------------------------------------------------------

def _detect_order_block(df: pd.DataFrame, lookback: int = 20) -> Tuple[Optional[str], dict]:
    """
    Detect order blocks: last contrary candle before an impulsive move.
    - Bull OB: last bearish candle before a strong bullish impulse
    - Bear OB: last bullish candle before a strong bearish impulse
    """
    if len(df) < lookback:
        return None, {}

    recent = df.tail(lookback)
    closes = recent["close"].values
    opens = recent["open"].values
    highs = recent["high"].values
    lows = recent["low"].values

    # Calculate average body size for threshold
    bodies = np.abs(closes - opens)
    avg_body = np.mean(bodies) if len(bodies) > 0 else 0
    impulse_threshold = avg_body * 2.0  # impulsive = 2x average body

    # Scan from recent to older
    for i in range(len(recent) - 1, 2, -1):
        body_cur = abs(closes[i] - opens[i])
        if body_cur < impulse_threshold:
            continue

        # Bullish impulse (close > open, large body)
        if closes[i] > opens[i]:
            # Look for last bearish candle before this impulse
            for j in range(i - 1, max(i - 5, -1), -1):
                if closes[j] < opens[j]:  # bearish candle
                    return "bull", {
                        "ob_high": float(highs[j]),
                        "ob_low": float(lows[j]),
                        "ob_index": int(j),
                        "impulse_size": float(body_cur),
                    }

        # Bearish impulse
        if closes[i] < opens[i]:
            for j in range(i - 1, max(i - 5, -1), -1):
                if closes[j] > opens[j]:  # bullish candle
                    return "bear", {
                        "ob_high": float(highs[j]),
                        "ob_low": float(lows[j]),
                        "ob_index": int(j),
                        "impulse_size": float(body_cur),
                    }

    return None, {}


# ---------------------------------------------------------------------------
# Funding rate filter
# ---------------------------------------------------------------------------

def _funding_filter(rates: List[float], direction: str) -> Tuple[bool, dict]:
    """
    Check if funding rate is adverse for the intended direction.
    Returns (allowed, details).
    - For longs: block if funding is extremely positive (crowded longs, high carry cost)
    - For shorts: block if funding is extremely negative (crowded shorts, squeeze risk)
    """
    if not rates:
        return True, {"funding": "no_data"}

    threshold = settings.FUNDING_EXTREME_PERCENTILE
    current_rate = rates[-1]

    details = {
        "current_funding": round(current_rate, 6),
        "threshold": threshold,
    }

    if direction == "long" and current_rate > threshold:
        details["blocked_reason"] = "funding_too_positive_for_long"
        return False, details

    if direction == "short" and current_rate < -threshold:
        details["blocked_reason"] = "funding_too_negative_for_short"
        return False, details

    # Bonus: funding normalizing toward entry direction is favorable
    if len(rates) >= 3:
        avg_recent = sum(rates[-3:]) / 3
        if direction == "long" and avg_recent < 0 and current_rate > avg_recent:
            details["bonus"] = "funding_normalizing_bullish"
        elif direction == "short" and avg_recent > 0 and current_rate < avg_recent:
            details["bonus"] = "funding_normalizing_bearish"

    return True, details


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

def _compute_score(conditions: dict) -> float:
    """
    Compute a weighted signal score from 0 to 1 based on fulfilled conditions.
    Higher score = higher confidence.
    """
    weights = {
        "htf_trend_aligned": 0.20,    # HTF trend supports direction
        "structure_break": 0.20,       # CHoCH or BOS detected
        "liquidity_sweep": 0.20,       # Sweep + rejection (core SMC)
        "confirmation_candle": 0.05,   # Last candle confirms direction
        "fvg_aligned": 0.10,           # FVG in same direction
        "order_block": 0.10,           # OB detected
        "funding_ok": 0.10,            # Funding not adverse
        "choch_bonus": 0.05,           # CHoCH (reversal) gets bonus vs BOS
        "htf_adx_strong": 0.05,        # ADX > 25 on HTF = strong trend bonus
    }

    score = 0.0
    for key, weight in weights.items():
        if conditions.get(key, False):
            score += weight

    return round(min(score, 1.0), 3)


def _smc_impulse_bar_state(df_ltf: pd.DataFrame, lookback: int = 20) -> dict | None:
    bars = max(8, int(lookback or 20))
    if df_ltf.empty or len(df_ltf) < bars + 2:
        return None

    work = df_ltf.tail(bars + 1).copy()
    opens = work["open"].astype(float).values
    highs = work["high"].astype(float).values
    lows = work["low"].astype(float).values
    closes = work["close"].astype(float).values
    if len(closes) < 2:
        return None

    last_open = float(opens[-1])
    last_close = float(closes[-1])
    last_high = float(highs[-1])
    last_low = float(lows[-1])
    if last_close <= 0:
        return None

    body_pct = abs(last_close - last_open) / last_close
    recent_bodies = []
    for i in range(max(0, len(closes) - 1 - bars), len(closes) - 1):
        c = float(closes[i])
        if c <= 0:
            continue
        recent_bodies.append(abs(float(closes[i]) - float(opens[i])) / c)
    avg_body_pct = float(np.mean(recent_bodies)) if recent_bodies else 0.0

    ema20 = float(pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1])
    ema20_dist_pct = abs(last_close - ema20) / last_close if last_close > 0 else 0.0
    candle_direction = "flat"
    if last_close > last_open:
        candle_direction = "long"
    elif last_close < last_open:
        candle_direction = "short"

    range_size = max(0.0, last_high - last_low)
    if range_size > 0:
        close_loc = (last_close - last_low) / range_size
    else:
        close_loc = 0.5

    return {
        "body_pct": float(body_pct),
        "avg_body_pct": float(avg_body_pct),
        "ema20_dist_pct": float(ema20_dist_pct),
        "candle_direction": candle_direction,
        "close_location": float(close_loc),
    }


def _smc_is_impulse_bar(
    state: dict | None,
    *,
    body_mult: float = 2.0,
    min_body_pct: float = 0.006,
) -> tuple[bool, float]:
    if not isinstance(state, dict):
        return False, max(0.0, float(min_body_pct or 0.0))
    body = max(0.0, float(state.get("body_pct", 0.0) or 0.0))
    avg = max(0.0, float(state.get("avg_body_pct", 0.0) or 0.0))
    mult = max(1.0, float(body_mult or 1.0))
    floor = max(0.0, float(min_body_pct or 0.0))
    threshold = max(floor, avg * mult)
    return body >= threshold, threshold


def _ema_confluence_state(
    df_htf: pd.DataFrame,
    direction: str,
    periods: List[int],
) -> tuple[bool | None, dict]:
    """
    Evaluate EMA stack confluence for the given direction.
    Returns (aligned|None, details). None means not enough data.
    """
    clean_periods = sorted({int(p) for p in periods if int(p) > 0})
    if len(clean_periods) < 2:
        clean_periods = [20, 50, 200]
    if df_htf.empty:
        return None, {"enabled": True, "status": "no_data", "periods": clean_periods}

    need = max(clean_periods) + 1
    if len(df_htf) < need:
        return None, {
            "enabled": True,
            "status": "insufficient_data",
            "periods": clean_periods,
            "needed_bars": need,
            "have_bars": len(df_htf),
        }

    closes = df_htf["close"].astype(float)
    last_price = float(closes.iloc[-1])

    ema_map: dict[int, float] = {}
    for p in clean_periods:
        ema_map[p] = float(closes.ewm(span=p, adjust=False).mean().iloc[-1])

    ordered_emas = [ema_map[p] for p in clean_periods]
    fast_ema = ordered_emas[0]
    if direction == "long":
        stack_ok = all(ordered_emas[i] > ordered_emas[i + 1] for i in range(len(ordered_emas) - 1))
        price_ok = last_price > fast_ema
    else:
        stack_ok = all(ordered_emas[i] < ordered_emas[i + 1] for i in range(len(ordered_emas) - 1))
        price_ok = last_price < fast_ema
    aligned = bool(stack_ok and price_ok)

    return aligned, {
        "enabled": True,
        "status": "ok",
        "periods": clean_periods,
        "ema_values": {str(k): round(v, 6) for k, v in ema_map.items()},
        "last_price": round(last_price, 6),
        "stack_ok": stack_ok,
        "price_ok": price_ok,
        "aligned": aligned,
    }


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def _detect_signal(
    df_ltf: pd.DataFrame,
    df_htf: pd.DataFrame,
    funding_rates: List[float],
    min_score: float | None = None,
    session: str | None = None,
    df_htf_secondary: pd.DataFrame | None = None,
) -> Tuple[bool, str, dict, float]:
    """
    Enhanced SMC signal detection:
    1. HTF trend context (4h/1h) — requires dual-TF confirmation when available
    2. Liquidity sweep on LTF (core SMC entry)
    3. CHoCH or BOS confirmation
    4. FVG alignment
    5. Order Block zone
    6. Funding rate filter

    Returns (triggered, direction, explain_payload, score).
    """
    if df_ltf.empty or len(df_ltf) < 30 or df_htf.empty or len(df_htf) < 30:
        return False, "", {}, 0.0

    explain: dict = {}
    conditions: dict = {}
    active_session = str(session or get_current_session()).strip().lower()
    explain["session"] = active_session

    if bool(getattr(settings, "SMC_SESSION_FILTER_ENABLED", True)):
        allowed_sessions = set(getattr(settings, "SMC_ALLOWED_SESSIONS", set()) or set())
        if allowed_sessions and active_session not in allowed_sessions:
            explain["allowed_sessions"] = sorted(allowed_sessions)
            explain["result"] = (
                f"blocked_by_session_window ({active_session} not in {sorted(allowed_sessions)})"
            )
            return False, "", explain, 0.0

    # 1. HTF trend
    htf_trend = _trend_from_swings(df_htf, period=3)
    explain["htf_trend"] = htf_trend

    # 1b. Dual-HTF confirmation: if secondary HTF (1h) is available, require agreement.
    # This prevents entering when 4h says bull but 1h is still bearish (or vice versa).
    htf_secondary_trend = None
    if df_htf_secondary is not None and not df_htf_secondary.empty and len(df_htf_secondary) >= 30:
        htf_secondary_trend = _trend_from_swings(df_htf_secondary, period=3)
        explain["htf_secondary_trend"] = htf_secondary_trend
        if htf_trend != htf_secondary_trend:
            # Allow if one is trending and other is range, but block if conflicting
            if htf_trend == "range" or htf_secondary_trend == "range":
                # Use the trending one as primary signal (range = neutral)
                explain["htf_dual_note"] = "one_range_allow"
            else:
                # Conflicting trends (bull vs bear) → block signal
                explain["result"] = f"htf_dual_conflict ({htf_trend} vs {htf_secondary_trend})"
                return False, "", explain, 0.0

    # 2. LTF structure break (CHoCH / BOS)
    break_type, break_details = _detect_structure_break(df_ltf, period=2)
    explain["structure_break"] = break_details if break_type else "none"

    # 3. Liquidity sweep
    sweep_type, sweep_details = _detect_liquidity_sweep(df_ltf, period=3)
    explain["liquidity_sweep"] = sweep_details if sweep_type else "none"

    # 4. FVG
    fvg_dir, fvg_details = _detect_fvg(df_ltf)
    explain["fvg"] = fvg_details if fvg_dir else "none"

    # 5. Order Block
    ob_dir, ob_details = _detect_order_block(df_ltf)
    explain["order_block"] = ob_details if ob_dir else "none"

    # Determine direction
    direction = ""

    # 5b. Confirmation candle: last closed candle confirms direction (scoring only)
    last_close = df_ltf["close"].iloc[-1]
    prev_close = df_ltf["close"].iloc[-2] if len(df_ltf) >= 2 else last_close
    candle_bullish = last_close > prev_close
    candle_bearish = last_close < prev_close

    # --- v7: Confluent gate = sweep AND structure_break AND HTF alignment ---
    # Require BOTH a liquidity sweep AND a structure break in the same direction
    # plus HTF trend must not be counter to the direction.
    # This dramatically reduces false signals vs the OR gate of v4-v6.

    # Step 1: Determine candidate direction from sweep + CHoCH confluence
    # v8: Require CHoCH specifically (reversal signal), NOT BOS (continuation)
    # CHoCH is much rarer and indicates a true structural shift
    long_sweep = sweep_type == "sweep_low"
    long_choch = break_type == "choch_bull"
    short_sweep = sweep_type == "sweep_high"
    short_choch = break_type == "choch_bear"

    long_confluence = long_sweep and long_choch
    short_confluence = short_sweep and short_choch

    if long_confluence:
        direction = "long"
    elif short_confluence:
        direction = "short"
    else:
        explain["result"] = "no_confluence"
        return False, "", explain, 0.0

    # Step 2: HTF trend must SUPPORT direction (hard gate)
    # v9: Block 'range' as well — only trade when HTF confirms direction
    # Range markets produce too many false SMC signals
    if direction == "long" and htf_trend != "bull":
        explain["result"] = f"htf_not_aligned ({htf_trend} != bull)"
        return False, "", explain, 0.0
    if direction == "short" and htf_trend != "bear":
        explain["result"] = f"htf_not_aligned ({htf_trend} != bear)"
        return False, "", explain, 0.0

    # Step 2b: EMA hard gate — block entries that are strongly against the macro trend
    # Even if HTF is 'range', if price is well below EMA50 don't go long (and vice versa)
    if len(df_htf) >= 50:
        htf_closes = df_htf["close"].values
        htf_ema50 = float(pd.Series(htf_closes).ewm(span=50).mean().values[-1])
        if htf_ema50 > 0:
            price_vs_ema50 = (htf_closes[-1] - htf_ema50) / htf_ema50
            if direction == "long" and price_vs_ema50 < -0.03:
                explain["result"] = f"ema50_counter_trend (price {price_vs_ema50:+.2%} vs EMA50)"
                return False, "", explain, 0.0
            if direction == "short" and price_vs_ema50 > 0.03:
                explain["result"] = f"ema50_counter_trend (price {price_vs_ema50:+.2%} vs EMA50)"
                return False, "", explain, 0.0

    # Step 2c: ADX trend strength gate — block entries in choppy/weak-trending markets.
    # Requires ADX > 18 on HTF (moderate trending), blocking noisy price action.
    smc_adx_min = float(getattr(settings, "SMC_ADX_MIN", 18.0))
    if smc_adx_min > 0:
        htf_adx = compute_adx(df_htf, period=14)
        explain["htf_adx"] = round(htf_adx, 2) if htf_adx is not None else None
        if htf_adx is not None and htf_adx < smc_adx_min:
            explain["result"] = f"htf_adx_too_low ({htf_adx:.1f} < {smc_adx_min})"
            return False, "", explain, 0.0

    # Step 3: Confirmation candle required (hard gate)
    if direction == "long" and not candle_bullish:
        explain["result"] = "no_confirmation_candle"
        return False, "", explain, 0.0
    if direction == "short" and not candle_bearish:
        explain["result"] = "no_confirmation_candle"
        return False, "", explain, 0.0

    # Step 3b: Anti-chase guard. After displacement candles, wait pullback/retest.
    if bool(getattr(settings, "SMC_IMPULSE_CHASE_FILTER_ENABLED", True)):
        impulse_state = _smc_impulse_bar_state(
            df_ltf,
            lookback=int(getattr(settings, "SMC_IMPULSE_LOOKBACK", 20)),
        )
        if impulse_state:
            impulse, impulse_threshold = _smc_is_impulse_bar(
                impulse_state,
                body_mult=float(getattr(settings, "SMC_IMPULSE_BODY_MULT", 2.0)),
                min_body_pct=float(getattr(settings, "SMC_IMPULSE_MIN_BODY_PCT", 0.006)),
            )
            max_ema_dist = float(getattr(settings, "SMC_IMPULSE_MAX_EMA20_DIST_PCT", 0.009))
            explain["impulse_guard"] = {
                "impulse": bool(impulse),
                "impulse_threshold_pct": round(impulse_threshold * 100, 4),
                "body_pct": round(float(impulse_state.get("body_pct", 0.0) or 0.0) * 100, 4),
                "ema20_dist_pct": round(float(impulse_state.get("ema20_dist_pct", 0.0) or 0.0) * 100, 4),
                "close_location": round(float(impulse_state.get("close_location", 0.5) or 0.5), 4),
                "candle_direction": str(impulse_state.get("candle_direction") or "flat"),
                "max_ema20_dist_pct": round(max_ema_dist * 100, 4),
            }
            if (
                impulse
                and str(impulse_state.get("candle_direction") or "") == direction
                and float(impulse_state.get("ema20_dist_pct", 0.0) or 0.0) >= max_ema_dist
            ):
                explain["result"] = "blocked_by_impulse_chase"
                return False, "", explain, 0.0

    # 6. Funding filter
    funding_ok, funding_details = _funding_filter(funding_rates, direction)
    explain["funding"] = funding_details
    if not funding_ok:
        explain["result"] = "blocked_by_funding"
        return False, "", explain, 0.0

    # 7. EMA confluence (multi-period stack)
    ema_confluence_enabled = bool(getattr(settings, "EMA_CONFLUENCE_ENABLED", True))
    ema_aligned: bool | None = None
    if ema_confluence_enabled:
        ema_periods = list(getattr(settings, "EMA_CONFLUENCE_PERIODS", [20, 50, 200]))
        ema_aligned, ema_details = _ema_confluence_state(df_htf, direction, ema_periods)
        explain["ema_confluence"] = ema_details
        if (
            ema_aligned is False
            and bool(getattr(settings, "EMA_CONFLUENCE_HARD_BLOCK", False))
        ):
            explain["result"] = "blocked_by_ema_confluence"
            return False, "", explain, 0.0
    else:
        explain["ema_confluence"] = {"enabled": False, "status": "disabled"}

    # Build conditions dict for scoring
    conditions["htf_trend_aligned"] = (
        (direction == "long" and htf_trend in ("bull", "range"))
        or (direction == "short" and htf_trend in ("bear", "range"))
    )
    conditions["structure_break"] = (
        (direction == "long" and break_type in ("choch_bull", "bos_bull"))
        or (direction == "short" and break_type in ("choch_bear", "bos_bear"))
    )
    conditions["liquidity_sweep"] = (
        (direction == "long" and sweep_type == "sweep_low")
        or (direction == "short" and sweep_type == "sweep_high")
    )
    conditions["fvg_aligned"] = (fvg_dir == "bull" and direction == "long") or (fvg_dir == "bear" and direction == "short")
    conditions["order_block"] = (ob_dir == "bull" and direction == "long") or (ob_dir == "bear" and direction == "short")
    conditions["confirmation_candle"] = (
        (direction == "long" and candle_bullish)
        or (direction == "short" and candle_bearish)
    )
    conditions["funding_ok"] = funding_ok
    conditions["choch_bonus"] = break_type in ("choch_bull", "choch_bear")
    # ADX > 25 on HTF = strong trend, earns scoring bonus
    htf_adx_val = explain.get("htf_adx")
    conditions["htf_adx_strong"] = bool(htf_adx_val is not None and htf_adx_val > 25)
    if ema_confluence_enabled and ema_aligned is not None:
        conditions["ema_confluence"] = ema_aligned

    score = _compute_score(conditions)
    if ema_confluence_enabled and ema_aligned is not None:
        ema_bonus = float(getattr(settings, "EMA_CONFLUENCE_BONUS", 0.06))
        ema_penalty = float(getattr(settings, "EMA_CONFLUENCE_PENALTY", 0.10))
        if ema_aligned:
            score = min(1.0, score + ema_bonus)
            explain["ema_score_adjustment"] = {"type": "bonus", "value": ema_bonus}
        else:
            score = max(0.0, score - ema_penalty)
            explain["ema_score_adjustment"] = {"type": "penalty", "value": ema_penalty}

    # Short score penalty: entry-signals skill data shows shorts succeed only
    # 25-35% vs 85-88% for longs.  Penalise short signals so they need much
    # higher confluence to pass the minimum score threshold.
    short_penalty = float(getattr(settings, "SHORT_SCORE_PENALTY", 0.0))
    if direction == "short" and short_penalty > 0:
        score = max(0.0, score - short_penalty)
        explain["short_score_penalty"] = short_penalty

    score = round(score, 3)
    explain["conditions"] = conditions
    explain["score"] = score
    explain["direction"] = direction
    explain["result"] = "signal_emitted"

    # Minimum score threshold to emit signal
    effective_min_score = float(
        min_score if min_score is not None else getattr(settings, "MIN_SIGNAL_SCORE", 0.80)
    )
    explain["session_score_min"] = effective_min_score
    if score < effective_min_score:
        explain["result"] = (
            f"score_too_low_for_{active_session} ({score} < {effective_min_score})"
        )
        return False, "", explain, score

    return True, direction, explain, score


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@shared_task(
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
    acks_late=True,
)
def run_signal_engine():
    now = dj_tz.now()
    emitted = 0
    dedup_seconds = int(getattr(settings, "SIGNAL_DEDUP_SECONDS", 120))
    session_policy_enabled = bool(getattr(settings, "SESSION_POLICY_ENABLED", False))
    session_dead_zone_block = bool(getattr(settings, "SESSION_DEAD_ZONE_BLOCK", True))
    current_session = get_current_session()
    session_min_score = float(getattr(settings, "MIN_SIGNAL_SCORE", 0.80))
    if session_policy_enabled:
        session_min_score = get_session_score_min(
            current_session,
            getattr(settings, "SESSION_SCORE_MIN", {}),
        )
        if session_dead_zone_block and is_dead_session(current_session):
            logger.info(
                "Session policy blocked signal emission: session=%s dead_zone=true",
                current_session,
            )
            return "signals_emitted=0"

    # Load regime filter config once (singleton)
    from risk.models import RegimeFilterConfig
    try:
        regime_cfg = RegimeFilterConfig.get()
    except Exception:
        regime_cfg = None

    for inst in Instrument.objects.filter(enabled=True):
        df_ltf = _latest_candles(inst, "5m", lookback=240)
        df_htf_1h = _latest_candles(inst, "1h", lookback=240)

        # Also try 4h for better HTF context
        df_htf_4h = _latest_candles(inst, "4h", lookback=120)

        # Primary HTF = 4h if available, else 1h
        # Secondary HTF = 1h (for dual-TF confirmation)
        if not df_htf_4h.empty and len(df_htf_4h) >= 30:
            df_htf = df_htf_4h
            df_htf_secondary = df_htf_1h if (not df_htf_1h.empty and len(df_htf_1h) >= 30) else None
        else:
            df_htf = df_htf_1h
            df_htf_secondary = None

        # ── Regime filter: skip instrument if market is sideways ──
        if regime_cfg and regime_cfg.enabled:
            # Pick the timeframe configured in the filter
            if regime_cfg.atr_timeframe in ("4h",) and not df_htf_4h.empty:
                df_regime = df_htf_4h
            elif regime_cfg.atr_timeframe in ("1h",) and not df_htf.empty:
                df_regime = df_htf
            else:
                df_regime = df_ltf  # fallback to LTF
            regime_ok, regime_details = check_regime_filter(df_regime, regime_cfg)
            if not regime_ok:
                logger.info(
                    "Regime filter blocked %s: %s", inst.symbol, regime_details
                )
                continue

        funding_rates = _latest_funding(inst, lookback=50)

        ok, direction, reason, score = _detect_signal(
            df_ltf,
            df_htf,
            funding_rates,
            min_score=session_min_score if session_policy_enabled else None,
            session=current_session if session_policy_enabled else None,
            df_htf_secondary=df_htf_secondary,
        )
        if ok:
            if not is_direction_allowed(direction, inst.symbol):
                logger.info(
                    "Direction policy blocked signal: %s %s mode=%s",
                    inst.symbol,
                    direction,
                    get_direction_mode(inst.symbol),
                )
                continue
            strategy_name = f"smc_{direction}"
            if dedup_seconds > 0:
                dedup_from = now - timedelta(seconds=dedup_seconds)
                if Signal.objects.filter(
                    instrument=inst,
                    strategy=strategy_name,
                    ts__gte=dedup_from,
                ).exists():
                    logger.info(
                        "Signal dedup skipped: %s %s within %ss",
                        inst.symbol,
                        direction,
                        dedup_seconds,
                    )
                    continue
            # Sanitize reason dict: convert numpy types to native Python for JSON
            import json as _json

            def _sanitize(obj):
                if isinstance(obj, dict):
                    return {k: _sanitize(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_sanitize(v) for v in obj]
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                return obj

            safe_reason = _sanitize(reason)
            Signal.objects.create(
                strategy=strategy_name,
                instrument=inst,
                ts=now,
                payload_json={
                    "timeframe_ltf": "5m",
                    "timeframe_htf": "4h" if (not df_htf_4h.empty and len(df_htf_4h) >= 30) else "1h",
                    "reason": safe_reason,
                },
                score=float(score),
            )
            emitted += 1
            logger.info(
                "Signal emitted: %s %s score=%.3f session=%s min_score=%.3f conditions=%s",
                inst.symbol,
                direction,
                score,
                current_session if session_policy_enabled else get_current_session(),
                session_min_score if session_policy_enabled else float(getattr(settings, "MIN_SIGNAL_SCORE", 0.80)),
                reason.get("conditions", {}),
            )
    return f"signals_emitted={emitted}"


@shared_task
def run_trend_engine():
    from .multi_strategy import run_module_engine

    return run_module_engine("trend")


@shared_task
def run_meanrev_engine():
    from .multi_strategy import run_module_engine

    return run_module_engine("meanrev")


@shared_task
def run_carry_engine():
    from .multi_strategy import run_module_engine

    return run_module_engine("carry")


@shared_task
def run_portfolio_allocator():
    from .multi_strategy import run_allocator_cycle

    return run_allocator_cycle()


@shared_task
def run_regime_detection():
    """Periodic HMM regime refit (scheduled by beat, e.g. every 6h)."""
    from django.conf import settings as s

    if not getattr(s, "HMM_REGIME_ENABLED", False):
        return "hmm_regime_disabled"
    from .regime import fit_and_predict_all

    results = fit_and_predict_all()
    return {sym: r.get("name") for sym, r in results.items()} if results else "no_results"


@shared_task
def run_garch_forecast():
    """Periodic GARCH refit (scheduled by beat, e.g. every 6h)."""
    from django.conf import settings as s

    if not getattr(s, "GARCH_ENABLED", False):
        return "garch_disabled"
    from .garch import fit_and_forecast_all

    results = fit_and_forecast_all()
    return {sym: round(r.get("cond_vol", 0), 6) for sym, r in results.items()} if results else "no_results"
