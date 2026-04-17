from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
import time

import numpy as np
from django.conf import settings
from django.utils import timezone as dj_tz

from execution.models import OperationReport
from signals.allocator import MODULE_ORDER, normalize_weight_map
from signals.models import Signal

logger = logging.getLogger(__name__)
_OVERLAY_CACHE: dict[str, Any] = {"ts": 0.0, "payload": None}


@dataclass
class ModuleMetrics:
    n: int = 0
    win_rate: float = 0.5
    expectancy: float = 0.0
    stdev: float = 0.0
    profit_factor: float = 1.0
    loss_cluster: float = 0.0
    regime_fit: float = 1.0
    corr_penalty: float = 1.0
    max_dd_pct: float = 0.0
    today_pnl_pct: float = 0.0
    dd_throttle_mult: float = 1.0
    daily_loss_throttle_mult: float = 1.0
    bucket_freeze: bool = False
    sample_mult: float = 1.0
    data_readiness: float = 1.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _normalize_map(raw: dict[str, float], fallback: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, _safe_float(v)) for v in raw.values())
    if total <= 0:
        return dict(fallback)
    return {k: max(0.0, _safe_float(raw.get(k, 0.0))) / total for k in MODULE_ORDER}


def _cap_normalized_weights(weights: dict[str, float], max_weight: float) -> dict[str, float]:
    """
    Enforce a real post-normalization max share per module.

    Without this second pass, clamping raw weights before normalization still allows one
    module to dominate near 100% when the rest collapse toward zero.
    """
    cap = _clamp(max_weight, 0.0, 1.0)
    normalized = _normalize_map(weights, weights)
    if cap >= 1.0:
        return normalized

    positive_modules = [m for m in MODULE_ORDER if float(normalized.get(m, 0.0)) > 1e-12]
    if not positive_modules:
        return normalized
    if len(positive_modules) * cap < 1.0 - 1e-9:
        return normalized

    out = {m: 0.0 for m in MODULE_ORDER}
    remaining = set(positive_modules)
    remaining_total = 1.0

    while remaining:
        base_sum = sum(float(normalized[m]) for m in remaining)
        if base_sum <= 1e-12 or remaining_total <= 1e-12:
            break
        proposed = {
            m: remaining_total * (float(normalized[m]) / base_sum)
            for m in remaining
        }
        overflow = [m for m, val in proposed.items() if val > cap + 1e-12]
        if not overflow:
            for m, val in proposed.items():
                out[m] = float(val)
            remaining.clear()
            break
        for module in overflow:
            out[module] = cap
            remaining.remove(module)
            remaining_total = max(0.0, remaining_total - cap)

    total = sum(out.values())
    if total <= 0:
        return normalized
    return {m: float(out.get(m, 0.0)) / total for m in MODULE_ORDER}


def _strategy_to_module(strategy_name: str) -> str:
    st = str(strategy_name or "").strip().lower()
    if st.startswith("mod_"):
        parts = st.split("_")
        if len(parts) >= 3:
            mod = parts[1]
            return mod if mod in MODULE_ORDER else ""
    if st.startswith("smc_"):
        return "smc"
    if st.startswith("alloc_"):
        return "allocator"
    return ""


def _signal_module_shares(sig: Signal | None) -> dict[str, float]:
    if sig is None:
        return {}
    payload = sig.payload_json or {}
    reasons = payload.get("reasons") if isinstance(payload, dict) else {}
    rows = reasons.get("module_contributions") if isinstance(reasons, dict) else None

    shares: dict[str, float] = {}
    if isinstance(rows, list):
        tmp: list[tuple[str, float]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            mod = str(row.get("module") or "").strip().lower()
            if mod not in MODULE_ORDER:
                continue
            contrib = abs(_safe_float(row.get("contribution"), 0.0))
            if contrib <= 0:
                contrib = abs(_safe_float(row.get("confidence"), 0.0))
            if contrib <= 0:
                contrib = 1.0
            tmp.append((mod, contrib))
        if tmp:
            total = sum(v for _, v in tmp) or 1.0
            for mod, val in tmp:
                shares[mod] = shares.get(mod, 0.0) + (val / total)
            return shares

    inferred = _strategy_to_module(sig.strategy)
    if inferred in MODULE_ORDER:
        return {inferred: 1.0}
    return {}


def _loss_cluster_score(returns: list[float]) -> float:
    n = len(returns)
    if n <= 0:
        return 0.0
    streak = 0
    max_streak = 0
    for r in returns:
        if r <= 0:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0
    return _clamp(max_streak / max(1, n), 0.0, 1.0)


def _compute_regime_fit(
    *,
    module: str,
    returns_trending: dict[str, list[float]],
    returns_non_trending: dict[str, list[float]],
    min_regime_trades: int = 5,
) -> float:
    """Score how well *module* performs in the current HMM regime.

    Uses the daily_regime field from OperationReport to partition historical
    trades into trending (bull/bear_confirmed) and non-trending (transition,
    bear_weak, etc.).  Then looks up the current HMM state via Redis cache
    and returns a multiplier in [0.5, 1.5]:
        >1.0 → module performs better in the current regime
        <1.0 → module performs worse in the current regime
         1.0 → insufficient data or regime unavailable
    """
    trending_arr = returns_trending.get(module, [])
    non_trending_arr = returns_non_trending.get(module, [])
    if len(trending_arr) < min_regime_trades and len(non_trending_arr) < min_regime_trades:
        return 1.0

    # Determine current dominant regime from HMM cache (most symbols share BTC regime)
    current_label = ""
    if bool(getattr(settings, "HMM_REGIME_ENABLED", False)):
        try:
            from signals.regime import get_cached_regime
            cached = get_cached_regime("BTCUSDT")
            if cached:
                current_label = str(cached.get("label", "") or "").strip().lower()
        except Exception:
            pass

    if current_label not in ("trending", "choppy"):
        return 1.0

    # Compute mean return for each regime partition
    mean_trending = float(np.mean(trending_arr)) if len(trending_arr) >= min_regime_trades else None
    mean_non_trending = float(np.mean(non_trending_arr)) if len(non_trending_arr) >= min_regime_trades else None

    if current_label == "trending":
        current_mean = mean_trending
        other_mean = mean_non_trending
    else:
        current_mean = mean_non_trending
        other_mean = mean_trending

    if current_mean is None:
        return 1.0

    # Reward modules that do better in the current regime than in the other
    if other_mean is not None and other_mean != 0:
        ratio = current_mean / abs(other_mean) if abs(other_mean) > 1e-9 else 1.0
        # ratio > 1 → better in current regime, ratio < 1 → worse
        fit = 0.5 + 0.5 * _clamp(ratio, 0.0, 2.0)
    elif current_mean > 0:
        fit = 1.2  # positive in current regime, no comparison data → mild boost
    elif current_mean < 0:
        fit = 0.8  # negative in current regime → mild penalty
    else:
        fit = 1.0

    return _clamp(fit, 0.5, 1.5)


def _max_drawdown_pct(returns: list[float]) -> float:
    if not returns:
        return 0.0
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= max(1e-9, 1.0 + float(r))
        if equity > peak:
            peak = equity
        dd = (equity - peak) / max(1e-9, peak)
        if dd < max_dd:
            max_dd = dd
    return abs(float(max_dd))


def _throttle_from_ratio(ratio: float, *, mult_at_50: float, mult_at_75: float) -> tuple[float, bool]:
    x = max(0.0, float(ratio))
    if x >= 1.0:
        return 0.0, True
    if x >= 0.75:
        return _clamp(mult_at_75, 0.0, 1.0), False
    if x >= 0.50:
        return _clamp(mult_at_50, 0.0, 1.0), False
    return 1.0, False


def _compute_correlation_penalties(
    *,
    module_day_returns: dict[str, dict[str, float]],
    strength: float,
    min_points: int,
    min_penalty: float,
) -> dict[str, float]:
    penalties = {m: 1.0 for m in MODULE_ORDER}
    all_days = sorted(
        {
            day
            for day_map in module_day_returns.values()
            for day in day_map.keys()
        }
    )
    if len(all_days) < max(3, min_points):
        return penalties

    arrays: dict[str, np.ndarray] = {}
    for module in MODULE_ORDER:
        day_map = module_day_returns.get(module, {})
        arr = np.array([_safe_float(day_map.get(day, 0.0), 0.0) for day in all_days], dtype=float)
        arrays[module] = arr

    for module in MODULE_ORDER:
        arr_m = arrays[module]
        if len(arr_m) < min_points or np.std(arr_m) <= 1e-12:
            penalties[module] = 1.0
            continue
        pos_corrs: list[float] = []
        for other in MODULE_ORDER:
            if other == module:
                continue
            arr_o = arrays[other]
            if len(arr_o) < min_points or np.std(arr_o) <= 1e-12:
                continue
            corr = float(np.corrcoef(arr_m, arr_o)[0, 1])
            if np.isnan(corr):
                continue
            if corr > 0:
                pos_corrs.append(corr)
        mean_pos = float(np.mean(pos_corrs)) if pos_corrs else 0.0
        penalties[module] = _clamp(1.0 - (strength * mean_pos), min_penalty, 1.0)
    return penalties


def compute_meta_weights_from_metrics(
    *,
    base_weights: dict[str, float],
    metrics: dict[str, ModuleMetrics],
    weight_cap: float,
    loss_cluster_penalty: float,
    pf_target: float,
    single_winner_enabled: bool,
    single_winner_min_weight: float,
    min_base_weight_share_by_module: dict[str, float] | None = None,
    p4_enabled: bool = False,
    p4_min_sample: int = 50,
) -> tuple[dict[str, float], dict[str, Any]]:
    base = normalize_weight_map(base_weights, base_weights)
    min_base_share_map = {
        str(k).strip().lower(): _clamp(_safe_float(v, 0.0), 0.0, 1.0)
        for k, v in (min_base_weight_share_by_module or {}).items()
    }
    n_by_mod = {m: int(metrics.get(m, ModuleMetrics()).n) for m in MODULE_ORDER}
    exp_by_mod = {m: float(metrics.get(m, ModuleMetrics()).expectancy) for m in MODULE_ORDER}
    std_by_mod = {m: max(1e-9, float(metrics.get(m, ModuleMetrics()).stdev)) for m in MODULE_ORDER}

    ready_modules = [
        m
        for m in MODULE_ORDER
        if n_by_mod[m] > 0 and float(metrics.get(m, ModuleMetrics()).data_readiness) >= 1.0
    ]
    if not ready_modules:
        ready_modules = [m for m in MODULE_ORDER if n_by_mod[m] > 0]

    exp_vals = [exp_by_mod[m] for m in ready_modules]
    exp_min = min(exp_vals) if exp_vals else 0.0
    exp_max = max(exp_vals) if exp_vals else 0.0
    std_vals = [std_by_mod[m] for m in ready_modules]
    std_med = float(np.median(std_vals)) if std_vals else 1.0

    raw: dict[str, float] = {}
    diagnostics: dict[str, dict[str, float | int | bool]] = {}

    for module in MODULE_ORDER:
        mm = metrics.get(module, ModuleMetrics())
        base_w = float(base.get(module, 0.0))
        min_base_share = float(min_base_share_map.get(module, 0.0))
        if base_w <= 0:
            raw[module] = 0.0
            diagnostics[module] = {"n": mm.n, "raw": 0.0}
            continue

        data_readiness = _clamp(float(mm.data_readiness), 0.0, 1.0)
        if mm.n <= 0:
            raw[module] = 0.0 if p4_enabled else max(0.0, base_w)
            diagnostics[module] = {
                "n": mm.n,
                "base": round(base_w, 6),
                "data_readiness": round(data_readiness, 6),
                "raw": round(raw[module], 6),
                "reason": "no_data_zero_p4" if p4_enabled else "no_data_keep_base",
            }
            continue

        if exp_max > exp_min:
            exp_norm = _clamp((mm.expectancy - exp_min) / (exp_max - exp_min), 0.0, 1.0)
        else:
            exp_norm = 0.5
        exp_norm = max(0.05, exp_norm)

        vol_inverse = _clamp(std_med / max(1e-9, mm.stdev), 0.5, 1.5)
        pf_norm = _clamp(mm.profit_factor / max(0.2, pf_target), 0.5, 1.5)
        cluster_factor = _clamp(1.0 - (loss_cluster_penalty * mm.loss_cluster), 0.30, 1.0)
        regime_fit = _clamp(mm.regime_fit, 0.5, 1.5)
        corr_pen = _clamp(mm.corr_penalty, 0.4, 1.0)
        sample_mult = _clamp(mm.sample_mult, 0.0, 1.0)

        strategy_guard = 1.0
        if p4_enabled:
            if mm.n < max(1, int(p4_min_sample)):
                strategy_guard *= sample_mult
            strategy_guard *= _clamp(mm.dd_throttle_mult, 0.0, 1.0)
            strategy_guard *= _clamp(mm.daily_loss_throttle_mult, 0.0, 1.0)
            if bool(mm.bucket_freeze):
                strategy_guard = 0.0

        computed_raw = (
            base_w
            * exp_norm
            * vol_inverse
            * pf_norm
            * cluster_factor
            * regime_fit
            * corr_pen
        )
        # Keep the optimizer-provided base share until enough live evidence exists,
        # then progressively hand control to the meta overlay as sample size matures.
        blended_raw = (base_w * (1.0 - data_readiness)) + (computed_raw * data_readiness)
        cap_value = max(0.05, weight_cap)
        raw_w = min(blended_raw * strategy_guard, cap_value)
        floor_raw = 0.0
        if strategy_guard > 0.0 and min_base_share > 0.0:
            floor_raw = min(base_w * min_base_share * strategy_guard, cap_value)
            raw_w = max(raw_w, floor_raw)
        raw[module] = max(0.0, raw_w)
        diagnostics[module] = {
            "n": mm.n,
            "base": round(base_w, 6),
            "exp": round(mm.expectancy, 6),
            "std": round(mm.stdev, 6),
            "pf": round(mm.profit_factor, 6),
            "loss_cluster": round(mm.loss_cluster, 6),
            "corr_pen": round(mm.corr_penalty, 6),
            "max_dd_pct": round(mm.max_dd_pct, 6),
            "today_pnl_pct": round(mm.today_pnl_pct, 6),
            "dd_throttle_mult": round(mm.dd_throttle_mult, 6),
            "daily_loss_throttle_mult": round(mm.daily_loss_throttle_mult, 6),
            "sample_mult": round(mm.sample_mult, 6),
            "data_readiness": round(data_readiness, 6),
            "min_base_share": round(min_base_share, 6),
            "bucket_freeze": bool(mm.bucket_freeze),
            "strategy_guard": round(strategy_guard, 6),
            "raw_computed": round(computed_raw, 6),
            "raw_blended": round(blended_raw, 6),
            "raw_floor": round(floor_raw, 6),
            "raw": round(raw_w, 6),
        }

    raw_total = sum(max(0.0, float(raw.get(m, 0.0))) for m in MODULE_ORDER)
    if raw_total <= 0 and p4_enabled:
        normalized = {m: 0.0 for m in MODULE_ORDER}
    else:
        normalized = _cap_normalized_weights(_normalize_map(raw, base), weight_cap)

    winner = ""
    if single_winner_enabled:
        ordered = sorted(normalized.items(), key=lambda kv: kv[1], reverse=True)
        top_mod, top_w = ordered[0]
        if top_w >= max(0.0, float(single_winner_min_weight)):
            winner = top_mod
            normalized = {m: (1.0 if m == top_mod else 0.0) for m in MODULE_ORDER}

    diag = {
        "winner": winner,
        "weights": {m: round(normalized[m], 6) for m in MODULE_ORDER},
        "module_metrics": diagnostics,
    }
    return normalized, diag


def _risk_budgets_from_weights(
    *,
    weights: dict[str, float],
    fallback_budgets: dict[str, float],
    bucket_caps: dict[str, float],
    strict_isolation: bool = False,
    max_total_budget: float = 1.0,
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for module in MODULE_ORDER:
        cap = _clamp(_safe_float(bucket_caps.get(module, 1.0), 1.0), 0.01, 1.0)
        raw[module] = max(0.0, float(weights.get(module, 0.0))) * cap
    if not strict_isolation:
        return _normalize_map(raw, fallback_budgets)

    total = sum(max(0.0, _safe_float(v, 0.0)) for v in raw.values())
    max_total = _clamp(_safe_float(max_total_budget, 1.0), 0.10, 1.0)
    if total > max_total and total > 1e-12:
        scale = max_total / total
        raw = {k: float(v) * scale for k, v in raw.items()}

    return {m: max(0.0, float(raw.get(m, 0.0))) for m in MODULE_ORDER}


def _collect_module_metrics(
    *,
    lookback_days: int,
    min_trades: int,
) -> tuple[dict[str, ModuleMetrics], dict[str, Any]]:
    cutoff = dj_tz.now() - timedelta(days=max(1, int(lookback_days)))
    reports = list(
        OperationReport.objects.filter(closed_at__gte=cutoff)
        .order_by("closed_at")
        .values("pnl_pct", "closed_at", "signal_id", "daily_regime")
    )
    if not reports:
        return {}, {"trade_count": 0, "reason": "no_reports"}

    signal_ids: set[int] = set()
    for r in reports:
        sid = str(r.get("signal_id") or "").strip()
        if sid.isdigit():
            signal_ids.add(int(sid))
    signal_map: dict[int, Signal] = {}
    if signal_ids:
        for sig in Signal.objects.filter(id__in=signal_ids).only("id", "strategy", "payload_json"):
            signal_map[int(sig.id)] = sig

    module_returns: dict[str, list[float]] = {m: [] for m in MODULE_ORDER}
    module_day_returns: dict[str, dict[str, float]] = {m: defaultdict(float) for m in MODULE_ORDER}
    # Regime-partitioned returns for regime_fit computation
    _TRENDING_REGIMES = {"bull_confirmed", "bear_confirmed"}
    module_returns_trending: dict[str, list[float]] = {m: [] for m in MODULE_ORDER}
    module_returns_non_trending: dict[str, list[float]] = {m: [] for m in MODULE_ORDER}
    today_key = dj_tz.now().date().isoformat()
    module_today_pnl: dict[str, float] = {m: 0.0 for m in MODULE_ORDER}
    total_attrib = 0

    for rep in reports:
        ret = _safe_float(rep.get("pnl_pct"), 0.0)
        sid = str(rep.get("signal_id") or "").strip()
        sig = signal_map.get(int(sid)) if sid.isdigit() else None
        shares = _signal_module_shares(sig)
        if not shares:
            continue
        day_key = rep["closed_at"].date().isoformat() if rep.get("closed_at") else ""
        daily_regime = str(rep.get("daily_regime") or "").strip().lower()
        is_trending_trade = daily_regime in _TRENDING_REGIMES
        for module, share in shares.items():
            attrib = ret * _clamp(share, 0.0, 1.0)
            module_returns[module].append(attrib)
            if is_trending_trade:
                module_returns_trending[module].append(attrib)
            elif daily_regime:  # only count if regime was actually recorded
                module_returns_non_trending[module].append(attrib)
            if day_key:
                module_day_returns[module][day_key] += attrib
                if day_key == today_key:
                    module_today_pnl[module] = module_today_pnl.get(module, 0.0) + attrib
            total_attrib += 1

    strength = _clamp(float(getattr(settings, "META_ALLOCATOR_CORR_PENALTY_STRENGTH", 0.5) or 0.5), 0.0, 2.0)
    min_points = max(3, int(getattr(settings, "META_ALLOCATOR_CORR_MIN_POINTS", 8) or 8))
    min_pen = _clamp(float(getattr(settings, "META_ALLOCATOR_CORR_MIN_PENALTY", 0.60) or 0.60), 0.10, 1.0)
    corr_penalties = _compute_correlation_penalties(
        module_day_returns=module_day_returns,
        strength=strength,
        min_points=min_points,
        min_penalty=min_pen,
    )
    p4_enabled = bool(getattr(settings, "META_ALLOCATOR_P4_ENABLED", False))
    p4_min_sample = max(1, int(getattr(settings, "META_ALLOCATOR_P4_MIN_SAMPLE", 50) or 50))
    p4_dd_mult_50 = _clamp(
        float(getattr(settings, "META_ALLOCATOR_P4_DD_THROTTLE_AT_50", 0.80) or 0.80),
        0.0,
        1.0,
    )
    p4_dd_mult_75 = _clamp(
        float(getattr(settings, "META_ALLOCATOR_P4_DD_THROTTLE_AT_75", 0.50) or 0.50),
        0.0,
        1.0,
    )
    dd_caps = dict(getattr(settings, "META_ALLOCATOR_STRATEGY_DD_CAPS", {}) or {})
    daily_caps = dict(getattr(settings, "META_ALLOCATOR_STRATEGY_DAILY_LOSS_CAPS", {}) or {})

    out: dict[str, ModuleMetrics] = {}
    for module in MODULE_ORDER:
        arr = module_returns.get(module, [])
        data_readiness = _clamp(len(arr) / max(1.0, float(min_trades)), 0.0, 1.0)
        if not arr:
            out[module] = ModuleMetrics(
                n=0,
                corr_penalty=float(corr_penalties.get(module, 1.0)),
                data_readiness=data_readiness,
            )
            continue
        wins = [x for x in arr if x > 0]
        losses = [x for x in arr if x < 0]
        pos_sum = sum(wins)
        neg_sum = abs(sum(losses))
        pf = pos_sum / neg_sum if neg_sum > 1e-12 else (2.0 if pos_sum > 0 else 1.0)
        stdev = float(np.std(arr)) if len(arr) > 1 else abs(arr[0]) if arr else 0.0
        max_dd_pct = _max_drawdown_pct(arr)
        today_pnl_pct = float(module_today_pnl.get(module, 0.0))
        dd_throttle_mult = 1.0
        daily_loss_throttle_mult = 1.0
        dd_freeze = False
        daily_freeze = False
        sample_mult = 1.0

        if p4_enabled:
            dd_cap = max(1e-9, _safe_float(dd_caps.get(module, 0.10), 0.10))
            dd_ratio = max_dd_pct / dd_cap
            dd_throttle_mult, dd_freeze = _throttle_from_ratio(
                dd_ratio,
                mult_at_50=p4_dd_mult_50,
                mult_at_75=p4_dd_mult_75,
            )

            daily_cap = max(1e-9, _safe_float(daily_caps.get(module, 0.03), 0.03))
            daily_loss_pct = abs(min(0.0, today_pnl_pct))
            daily_ratio = daily_loss_pct / daily_cap
            daily_loss_throttle_mult, daily_freeze = _throttle_from_ratio(
                daily_ratio,
                mult_at_50=p4_dd_mult_50,
                mult_at_75=p4_dd_mult_75,
            )

            if len(arr) < p4_min_sample:
                sample_mult = _clamp(len(arr) / max(1.0, float(p4_min_sample)), 0.20, 1.0)

        out[module] = ModuleMetrics(
            n=len(arr),
            win_rate=(len(wins) / len(arr)) if arr else 0.5,
            expectancy=float(np.mean(arr)) if arr else 0.0,
            stdev=stdev,
            profit_factor=pf,
            loss_cluster=_loss_cluster_score(arr),
            regime_fit=_compute_regime_fit(
                module=module,
                returns_trending=module_returns_trending,
                returns_non_trending=module_returns_non_trending,
                min_regime_trades=max(3, int(getattr(settings, "META_ALLOCATOR_REGIME_FIT_MIN_TRADES", 5) or 5)),
            ),
            corr_penalty=float(corr_penalties.get(module, 1.0)),
            max_dd_pct=max_dd_pct,
            today_pnl_pct=today_pnl_pct,
            dd_throttle_mult=dd_throttle_mult,
            daily_loss_throttle_mult=daily_loss_throttle_mult,
            bucket_freeze=bool(dd_freeze or daily_freeze),
            sample_mult=sample_mult,
            data_readiness=data_readiness,
        )

    return out, {
        "trade_count": len(reports),
        "attributed": total_attrib,
        "p4_enabled": p4_enabled,
        "today_key": today_key,
    }


def compute_meta_allocator_overlay(
    *,
    base_weights: dict[str, float],
    base_risk_budgets: dict[str, float],
) -> dict[str, Any]:
    enabled = bool(getattr(settings, "META_ALLOCATOR_ENABLED", False))
    if not enabled:
        return {
            "enabled": False,
            "weights": normalize_weight_map(base_weights, base_weights),
            "risk_budgets": normalize_weight_map(base_risk_budgets, base_risk_budgets),
            "diag": {"enabled": False, "reason": "disabled"},
        }

    cache_seconds = max(30, int(getattr(settings, "META_ALLOCATOR_CACHE_SECONDS", 300) or 300))
    now_ts = time.time()
    cached = _OVERLAY_CACHE.get("payload")
    if cached and (now_ts - float(_OVERLAY_CACHE.get("ts", 0.0))) < cache_seconds:
        return dict(cached)

    lookback_days = max(3, int(getattr(settings, "META_ALLOCATOR_LOOKBACK_DAYS", 21) or 21))
    min_trades = max(3, int(getattr(settings, "META_ALLOCATOR_MIN_TRADES", 12) or 12))
    metrics, meta = _collect_module_metrics(
        lookback_days=lookback_days,
        min_trades=min_trades,
    )
    if not metrics:
        logger.info("meta_allocator fallback: insufficient module metrics")
        payload = {
            "enabled": True,
            "weights": normalize_weight_map(base_weights, base_weights),
            "risk_budgets": normalize_weight_map(base_risk_budgets, base_risk_budgets),
            "diag": {
                "enabled": True,
                "reason": "insufficient_metrics",
                **meta,
            },
        }
        _OVERLAY_CACHE["ts"] = now_ts
        _OVERLAY_CACHE["payload"] = dict(payload)
        return payload

    weight_cap = _clamp(float(getattr(settings, "META_ALLOCATOR_WEIGHT_CAP", 0.65) or 0.65), 0.10, 1.0)
    loss_cluster_penalty = _clamp(
        float(getattr(settings, "META_ALLOCATOR_LOSS_CLUSTER_PENALTY", 0.50) or 0.50),
        0.0,
        1.5,
    )
    pf_target = max(0.2, float(getattr(settings, "META_ALLOCATOR_PF_TARGET", 1.20) or 1.20))
    single_winner_enabled = bool(getattr(settings, "META_ALLOCATOR_SINGLE_WINNER_ENABLED", False))
    single_winner_min_weight = _clamp(
        float(getattr(settings, "META_ALLOCATOR_SINGLE_WINNER_MIN_WEIGHT", 0.42) or 0.42),
        0.0,
        1.0,
    )
    p4_enabled = bool(getattr(settings, "META_ALLOCATOR_P4_ENABLED", False))
    p4_min_sample = max(1, int(getattr(settings, "META_ALLOCATOR_P4_MIN_SAMPLE", 50) or 50))
    p4_strict_isolation = bool(
        getattr(settings, "META_ALLOCATOR_P4_STRICT_BUCKET_ISOLATION_ENABLED", False)
    )
    p4_max_total_budget = _clamp(
        float(getattr(settings, "META_ALLOCATOR_P4_MAX_TOTAL_RISK_BUDGET", 1.0) or 1.0),
        0.10,
        1.0,
    )

    weights, diag = compute_meta_weights_from_metrics(
        base_weights=normalize_weight_map(base_weights, base_weights),
        metrics=metrics,
        weight_cap=weight_cap,
        loss_cluster_penalty=loss_cluster_penalty,
        pf_target=pf_target,
        single_winner_enabled=single_winner_enabled,
        single_winner_min_weight=single_winner_min_weight,
        min_base_weight_share_by_module=dict(
            getattr(settings, "META_ALLOCATOR_MIN_BASE_WEIGHT_SHARE_BY_MODULE", {}) or {}
        ),
        p4_enabled=p4_enabled,
        p4_min_sample=p4_min_sample,
    )

    bucket_caps = dict(getattr(settings, "META_ALLOCATOR_BUCKET_CAPS", {}) or {})
    risk_budgets = _risk_budgets_from_weights(
        weights=weights,
        fallback_budgets=normalize_weight_map(base_risk_budgets, base_risk_budgets),
        bucket_caps=bucket_caps,
        strict_isolation=(p4_enabled and p4_strict_isolation),
        max_total_budget=p4_max_total_budget,
    )

    diag_out = {
        "enabled": True,
        "lookback_days": lookback_days,
        "min_trades": min_trades,
        "p4_enabled": p4_enabled,
        "p4_strict_bucket_isolation": bool(p4_enabled and p4_strict_isolation),
        "risk_budget_total": round(sum(float(risk_budgets.get(m, 0.0)) for m in MODULE_ORDER), 6),
        "summary": {
            "winner": str(diag.get("winner") or ""),
            "weights": diag.get("weights", {}),
        },
        "stats": meta,
        "module_metrics": diag.get("module_metrics", {}),
    }

    payload = {
        "enabled": True,
        "weights": weights,
        "risk_budgets": risk_budgets,
        "diag": diag_out,
    }
    _OVERLAY_CACHE["ts"] = now_ts
    _OVERLAY_CACHE["payload"] = dict(payload)
    return payload
