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


def _kalman_smooth(prices: np.ndarray) -> np.ndarray | None:
    """Scalar Kalman filter on price series.

    State:  x_t  (estimated level)
    Measurement:  z_t = x_t + v_t
    Transition:  x_t = x_t-1 + w_t

    R (measurement noise) estimated from price returns variance.
    Q (process noise) = R * kalman_q_ratio.

    Returns array of filtered state estimates, same length as *prices*.
    """
    n = len(prices)
    if n < 20:
        return None
    p = prices.astype(np.float64)
    rets = np.diff(p) / p[:-1]
    R = float(np.var(rets)) * p[-1] ** 2
    if R <= 0:
        return None
    q_ratio = float(getattr(settings, "MODULE_TREND_KALMAN_Q_RATIO", 0.01) or 0.01)
    Q = R * max(1e-6, q_ratio)

    x = p[0]
    P = R
    out = np.empty(n, dtype=np.float64)
    out[0] = x
    for i in range(1, n):
        # Predict
        x_pred = x
        P_pred = P + Q
        # Update
        K = P_pred / (P_pred + R)
        x = x_pred + K * (p[i] - x_pred)
        P = (1.0 - K) * P_pred
        out[i] = x
    return out


def detect(
    df_ltf: pd.DataFrame,
    df_htf: pd.DataFrame,
    _funding_rates: list[float],
    session: str,
    symbol: str = "",
) -> dict | None:
    if df_ltf.empty or df_htf.empty or len(df_ltf) < 80 or len(df_htf) < 80:
        return None

    adx = compute_adx(df_ltf, period=14)
    if adx is None:
        return None
    trend_min = float(getattr(settings, "MODULE_ADX_TREND_MIN", 20.0))
    if adx < trend_min:
        return None

    # --- HTF ADX confirmation gate ---
    htf_adx_min = float(getattr(settings, "MODULE_TREND_HTF_ADX_MIN", 0.0))
    adx_htf = compute_adx(df_htf, period=14)
    if htf_adx_min > 0 and (adx_htf is None or adx_htf < htf_adx_min):
        return None

    closes = df_htf["close"].astype(float)
    ema20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
    last = float(closes.iloc[-1])
    if ema20 <= 0 or ema50 <= 0 or last <= 0:
        return None

    # --- Optional Kalman-filtered level ---
    kalman_level = None
    kalman_enabled = bool(getattr(settings, "MODULE_TREND_KALMAN_ENABLED", True))
    if kalman_enabled and len(closes) >= 50:
        kf = _kalman_smooth(closes.values)
        if kf is not None:
            kalman_level = float(kf[-1])

    pullback_tol = max(
        0.0,
        float(getattr(settings, "MODULE_TREND_EMA20_PULLBACK_TOLERANCE_PCT", 0.003) or 0.003),
    )
    direction = ""
    if ema20 > ema50 and last >= (ema20 * (1.0 - pullback_tol)):
        direction = "long"
    elif ema20 < ema50 and last <= (ema20 * (1.0 + pullback_tol)):
        direction = "short"
    if not direction:
        # Kalman fallback: if EMA direction unclear but Kalman slope is decisive
        if kalman_enabled and kalman_level is not None and len(closes) >= 50:
            kf_arr = _kalman_smooth(closes.values)
            if kf_arr is not None and len(kf_arr) >= 5:
                slope = float(kf_arr[-1] - kf_arr[-5]) / max(1e-12, float(kf_arr[-5]))
                slope_min = float(getattr(settings, "MODULE_TREND_KALMAN_SLOPE_MIN", 0.003) or 0.003)
                if slope > slope_min and last >= ema50:
                    direction = "long"
                elif slope < -slope_min and last <= ema50:
                    direction = "short"
    if not direction:
        return None

    impulse_details: dict = {}
    if bool(getattr(settings, "MODULE_IMPULSE_FILTER_ENABLED", True)):
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
            max_ema_dist = float(getattr(settings, "MODULE_IMPULSE_MAX_EMA20_DIST_PCT", 0.008))
            if (
                impulse
                and state.get("candle_direction") == direction
                and float(state.get("ema20_dist_pct", 0.0) or 0.0) >= max_ema_dist
            ):
                # Avoid trend entries right after displacement candles (wait pullback/retest).
                return None
            impulse_details = {
                "impulse": bool(impulse),
                "impulse_threshold_pct": round(impulse_threshold * 100, 4),
                "body_pct": round(float(state.get("body_pct", 0.0) or 0.0) * 100, 4),
                "ema20_dist_pct": round(float(state.get("ema20_dist_pct", 0.0) or 0.0) * 100, 4),
            }

    # --- Bounce / counter-momentum filter ---
    # Block shorts when price has bounced strongly off low (and longs off high).
    bounce_lookback = int(getattr(settings, "MODULE_BOUNCE_LOOKBACK", 30))
    bounce_block_pct = float(getattr(settings, "MODULE_BOUNCE_BLOCK_PCT", 0.50))
    bounce_info = bounce_pct(df_ltf, lookback=bounce_lookback)
    if bounce_info:
        if direction == "short" and bounce_info.get("bounce_from_low_pct", 0) >= bounce_block_pct:
            return None  # price bouncing up too hard to short
        if direction == "long" and bounce_info.get("bounce_from_high_pct", 0) >= bounce_block_pct:
            return None  # price dumping too hard to long

    ema_gap = abs(ema20 - ema50) / ema50
    raw = 0.30 + min(0.40, ema_gap * 20.0) + min(0.30, max(0.0, adx - trend_min) / 60.0)

    # --- Kalman alignment boost ---
    kalman_boost = 0.0
    if kalman_enabled and kalman_level is not None:
        kl_above_ema50 = kalman_level > ema50
        if (direction == "long" and kl_above_ema50) or (direction == "short" and not kl_above_ema50):
            kalman_boost = 0.05
            raw = min(1.0, raw + kalman_boost)

    # --- Volume confirmation ---
    vol_ratio = None
    vol_boost = 0.0
    vol_penalty = 0.0
    if bool(getattr(settings, "MODULE_TREND_VOLUME_CONFIRM_ENABLED", True)):
        try:
            vols = df_ltf["volume"].astype(float)
            if len(vols) >= 20 and vols.iloc[-1] > 0:
                vol_sma = float(vols.tail(20).mean())
                if vol_sma > 0:
                    vol_ratio = float(vols.iloc[-1] / vol_sma)
                    vol_min = float(getattr(settings, "MODULE_TREND_VOLUME_MIN_RATIO", 0.8))
                    vol_strong = float(getattr(settings, "MODULE_TREND_VOLUME_STRONG_RATIO", 1.5))
                    if vol_ratio < vol_min:
                        vol_penalty = min(0.10, (vol_min - vol_ratio) * 0.15)
                        raw = max(0.0, raw - vol_penalty)
                    elif vol_ratio >= vol_strong:
                        vol_boost = min(0.08, (vol_ratio - vol_strong) * 0.06)
                        raw = min(1.0, raw + vol_boost)
        except Exception:
            pass

    confidence = normalize_score(raw)
    reasons = {
        "session": session,
        "adx_ltf": round(float(adx), 4),
        "adx_htf": round(float(adx_htf), 4) if adx_htf is not None else None,
        "ema20": round(ema20, 6),
        "ema50": round(ema50, 6),
        "last": round(last, 6),
        "ema_gap_pct": round(ema_gap * 100, 4),
        "ema20_pullback_tolerance_pct": round(pullback_tol * 100, 4),
    }
    if kalman_level is not None:
        reasons["kalman_level"] = round(kalman_level, 6)
        if kalman_boost > 0:
            reasons["kalman_boost"] = round(kalman_boost, 4)
    if vol_ratio is not None:
        reasons["volume_ratio"] = round(vol_ratio, 4)
        if vol_boost > 0:
            reasons["volume_boost"] = round(vol_boost, 4)
        if vol_penalty > 0:
            reasons["volume_penalty"] = round(vol_penalty, 4)
    if impulse_details:
        reasons["impulse_guard"] = impulse_details
    if bounce_info:
        reasons["bounce_guard"] = bounce_info

    return {
        "direction": direction,
        "raw_score": confidence,
        "confidence": confidence,
        "reasons": reasons,
    }
