from __future__ import annotations

from datetime import timedelta
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import redis
from django.conf import settings
from django.utils import timezone as dj_tz

from core.models import Instrument
from marketdata.models import Candle, FundingRate
from signals.models import Signal


def redis_client():
    try:
        return redis.from_url(settings.CELERY_BROKER_URL)
    except Exception:
        return None


def acquire_task_lock(name: str, ttl_seconds: int = 55) -> bool:
    client = redis_client()
    if client is None:
        return True
    key = f"lock:signals:{name}"
    try:
        return bool(client.set(key, "1", nx=True, ex=ttl_seconds))
    except Exception:
        return True


def normalize_score(val: float) -> float:
    try:
        num = float(val)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, round(num, 4)))


def direction_to_sign(direction: str) -> int:
    direct = (direction or "").strip().lower()
    if direct == "long":
        return 1
    if direct == "short":
        return -1
    return 0


def sign_to_direction(sign: float, threshold: float = 0.0) -> str:
    if sign > threshold:
        return "long"
    if sign < -threshold:
        return "short"
    return "flat"


def strategy_module(strategy: str) -> tuple[str, str]:
    """
    Convert strategy name into (module, direction).
    Supported formats:
      - mod_trend_long
      - alloc_short
      - smc_long
    """
    strategy_name = (strategy or "").strip().lower()
    if strategy_name.startswith("alloc_"):
        return "allocator", strategy_name.replace("alloc_", "", 1)
    if strategy_name.startswith("smc_"):
        return "smc", strategy_name.replace("smc_", "", 1)
    if not strategy_name.startswith("mod_"):
        return "", ""
    parts = strategy_name.split("_")
    if len(parts) < 3:
        return "", ""
    return parts[1], parts[2]


def get_multi_universe_instruments() -> list[Instrument]:
    qs = Instrument.objects.filter(enabled=True)
    universe = list(getattr(settings, "MULTI_STRATEGY_UNIVERSE", []) or [])
    if universe:
        qs = qs.filter(symbol__in=universe)
    return list(qs.order_by("symbol"))


def latest_candles(instrument: Instrument, tf: str, lookback: int = 300) -> pd.DataFrame:
    rows = list(
        Candle.objects.filter(instrument=instrument, timeframe=tf)
        .order_by("-ts")[:lookback]
        .values("ts", "open", "high", "low", "close", "volume")
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("ts")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df.set_index("ts", inplace=True)
    return df


def latest_funding_rates(instrument: Instrument, lookback: int = 80) -> list[float]:
    qs = (
        FundingRate.objects.filter(instrument=instrument)
        .order_by("-ts")[:lookback]
        .values_list("rate", flat=True)
    )
    return [float(x) for x in reversed(list(qs))]


def compute_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
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
    atr_arr = np.zeros(len(df))
    plus_di_arr = np.zeros(len(df))
    minus_di_arr = np.zeros(len(df))
    atr_arr[period] = np.sum(tr[1 : period + 1])
    s_plus = np.sum(plus_dm[1 : period + 1])
    s_minus = np.sum(minus_dm[1 : period + 1])
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
    start = period * 2
    if start >= len(df):
        return None
    adx = float(np.mean(dx[start - period + 1 : start + 1]))
    for i in range(start + 1, len(df)):
        adx = (adx * (period - 1) + dx[i]) / period
    return round(adx, 4)


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if len(df) < period + 1:
        return None
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
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
    if last_close <= 0:
        return None
    return atr / last_close


def impulse_bar_state(df: pd.DataFrame, lookback: int = 20) -> dict | None:
    """
    Snapshot of the latest candle relative to recent body behavior.
    Returns None when there is not enough data.
    """
    bars = max(8, int(lookback or 20))
    if df.empty or len(df) < bars + 2:
        return None

    work = df.tail(bars + 1).copy()
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


def is_impulse_bar(state: dict | None, body_mult: float = 2.2, min_body_pct: float = 0.006) -> tuple[bool, float]:
    """
    Determine if latest candle body is an impulse vs recent average.
    Returns (is_impulse, dynamic_threshold).
    """
    if not isinstance(state, dict):
        return False, max(0.0, float(min_body_pct or 0.0))
    body = max(0.0, float(state.get("body_pct", 0.0) or 0.0))
    avg = max(0.0, float(state.get("avg_body_pct", 0.0) or 0.0))
    mult = max(1.0, float(body_mult or 1.0))
    floor = max(0.0, float(min_body_pct or 0.0))
    threshold = max(floor, avg * mult)
    return body >= threshold, threshold


def bounce_pct(df: pd.DataFrame, lookback: int = 30) -> dict:
    """Measure how far price has bounced from its recent extreme.

    Returns a dict with:
      - bounce_from_low_pct:  % price has risen from   N-bar low  (positive = bouncing up)
      - bounce_from_high_pct: % price has fallen from  N-bar high (positive = falling)
      - last: current close
      - low_N / high_N: the extreme values
    Useful to avoid shorting into a bounce or longing into a dump.
    """
    if df.empty or len(df) < lookback:
        return {}
    closes = df["close"].astype(float)
    window = closes.tail(lookback)
    last = float(window.iloc[-1])
    low_n = float(window.min())
    high_n = float(window.max())
    if low_n <= 0 or high_n <= 0:
        return {}
    return {
        "bounce_from_low_pct": round((last - low_n) / low_n * 100, 4),
        "bounce_from_high_pct": round((high_n - last) / high_n * 100, 4),
        "last": round(last, 6),
        "low_N": round(low_n, 6),
        "high_N": round(high_n, 6),
    }


def has_warmup_data(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, min_bars: int) -> bool:
    return len(df_ltf) >= min_bars and len(df_htf) >= max(60, int(min_bars * 0.5))


def emit_signal(
    instrument: Instrument,
    strategy: str,
    module: str,
    direction: str,
    raw_score: float,
    confidence: float,
    reasons: dict | None = None,
    *,
    net_score: float = 0.0,
    risk_budget_pct: float = 0.0,
    symbol_state: str = "open",
    dedup_seconds: int | None = None,
    ts=None,
) -> bool:
    now = ts or dj_tz.now()
    dedup = int(dedup_seconds or getattr(settings, "MODULE_SIGNAL_TTL_SECONDS", 120))
    if dedup > 0:
        recent_from = now - timedelta(seconds=dedup)
        if Signal.objects.filter(
            instrument=instrument,
            strategy=strategy,
            ts__gte=recent_from,
        ).exists():
            return False

    payload = {
        "module": module,
        "direction": direction,
        "raw_score": normalize_score(raw_score),
        "net_score": round(float(net_score or 0.0), 6),
        "confidence": normalize_score(confidence),
        "risk_budget_pct": round(float(risk_budget_pct or 0.0), 6),
        "symbol_state": symbol_state,
        "reasons": reasons or {},
    }
    Signal.objects.create(
        strategy=strategy,
        instrument=instrument,
        ts=now,
        payload_json=payload,
        score=float(payload["confidence"]),
    )
    return True


def now_utc():
    return dj_tz.now()
