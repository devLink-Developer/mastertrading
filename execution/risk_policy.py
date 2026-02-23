from __future__ import annotations

from django.conf import settings


def max_daily_trades_for_adx(htf_adx: float | None) -> int:
    """
    Regime-adaptive trade throttling using configurable ADX thresholds.
    """
    low_adx_limit = int(getattr(settings, "MAX_DAILY_TRADES_LOW_ADX", 3))
    normal_limit = int(getattr(settings, "MAX_DAILY_TRADES", 6))
    high_adx_limit = int(getattr(settings, "MAX_DAILY_TRADES_HIGH_ADX", 10))
    low_adx_threshold = float(getattr(settings, "MAX_DAILY_TRADES_LOW_ADX_THRESHOLD", 20.0) or 20.0)
    high_adx_threshold = float(getattr(settings, "MAX_DAILY_TRADES_HIGH_ADX_THRESHOLD", 25.0) or 25.0)

    if htf_adx is None:
        return normal_limit
    if htf_adx < low_adx_threshold:
        return low_adx_limit
    if htf_adx > high_adx_threshold:
        return high_adx_limit
    return normal_limit


def volatility_adjusted_risk(symbol: str, atr_pct: float | None, base_risk: float) -> float:
    """
    Shared risk sizing policy used by live execution and backtest.
    """
    # 1) Per-instrument cap: do not allow symbol config to raise allocator/base risk.
    per_inst = getattr(settings, "PER_INSTRUMENT_RISK", {})
    if symbol in per_inst:
        try:
            per_symbol_risk = float(per_inst[symbol])
        except Exception:
            per_symbol_risk = float(base_risk)
        return max(0.0, min(per_symbol_risk, float(base_risk)))

    # 2) Risk tiers: use tier-specific base risk, then ATR scaling.
    effective_base = float(base_risk)
    if getattr(settings, "INSTRUMENT_RISK_TIERS_ENABLED", False):
        tier_map = getattr(settings, "INSTRUMENT_TIER_MAP", {})
        tiers = getattr(settings, "INSTRUMENT_RISK_TIERS", {})
        tier_name = tier_map.get(symbol, "")
        if tier_name and tier_name in tiers:
            try:
                effective_base = float(tiers[tier_name])
            except Exception:
                effective_base = float(base_risk)

    if atr_pct is None or atr_pct <= 0:
        return effective_base

    low_vol = max(0.0, float(getattr(settings, "VOL_RISK_LOW_ATR_PCT", 0.008) or 0.008))
    high_vol = max(low_vol + 1e-9, float(getattr(settings, "VOL_RISK_HIGH_ATR_PCT", 0.015) or 0.015))
    min_scale = max(0.0, min(1.0, float(getattr(settings, "VOL_RISK_MIN_SCALE", 0.6) or 0.6)))

    if atr_pct <= low_vol:
        return effective_base
    if atr_pct >= high_vol:
        return effective_base * min_scale

    ratio = (atr_pct - low_vol) / (high_vol - low_vol)
    scale = 1.0 - ratio * (1.0 - min_scale)
    return effective_base * scale
