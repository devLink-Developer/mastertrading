from __future__ import annotations

import logging
from datetime import timedelta
from typing import Iterable

from django.conf import settings

from .modules.common import direction_to_sign, normalize_score, sign_to_direction


MODULE_ORDER = ("trend", "meanrev", "carry", "smc")

logger = logging.getLogger(__name__)


def normalize_weight_map(raw_map: dict | None, fallback: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    source = raw_map or {}
    for module in MODULE_ORDER:
        try:
            out[module] = max(0.0, float(source.get(module, fallback.get(module, 0.0))))
        except Exception:
            out[module] = max(0.0, float(fallback.get(module, 0.0)))
    total = sum(out.values())
    if total <= 0:
        return {k: v for k, v in fallback.items() if k in MODULE_ORDER}
    return {k: (v / total) for k, v in out.items()}


def default_weight_map() -> dict[str, float]:
    include_smc = bool(getattr(settings, "ALLOCATOR_INCLUDE_SMC", False))
    fallback = (
        {"trend": 0.30, "meanrev": 0.25, "carry": 0.15, "smc": 0.30}
        if include_smc
        else {"trend": 0.45, "meanrev": 0.35, "carry": 0.20, "smc": 0.0}
    )
    return normalize_weight_map(getattr(settings, "ALLOCATOR_MODULE_WEIGHTS", {}), fallback)


def default_risk_budget_map() -> dict[str, float]:
    include_smc = bool(getattr(settings, "ALLOCATOR_INCLUDE_SMC", False))
    fallback = (
        {"trend": 0.30, "meanrev": 0.25, "carry": 0.15, "smc": 0.30}
        if include_smc
        else {"trend": 0.45, "meanrev": 0.35, "carry": 0.20, "smc": 0.0}
    )
    return normalize_weight_map(getattr(settings, "ALLOCATOR_MODULE_RISK_BUDGETS", {}), fallback)


# ---------------------------------------------------------------------------
# Bayesian rolling dynamic weights
# ---------------------------------------------------------------------------

def _module_rolling_stats(days: int = 7) -> dict[str, dict]:
    """Compute per-module win/loss counts from recent OperationReports.

    Returns {module_name: {"wins": int, "losses": int, "n": int, "pnl": float}}.
    Trades are attributed to ALL modules that were active during that trade
    (from the allocator's module_contributions in the Signal payload).
    """
    from execution.models import OperationReport
    from signals.models import Signal
    from django.utils import timezone as dj_tz

    cutoff = dj_tz.now() - timedelta(days=days)
    reports = (
        OperationReport.objects.filter(closed_at__gte=cutoff)
        .only("signal_id", "pnl_abs", "outcome")
    )

    # Collect signal IDs
    sig_ids: set[int] = set()
    report_list: list[dict] = []
    for r in reports:
        pnl = float(r.pnl_abs)
        win = pnl > 0
        entry = {"signal_id": None, "win": win, "pnl": pnl}
        if r.signal_id:
            try:
                entry["signal_id"] = int(r.signal_id)
                sig_ids.add(entry["signal_id"])
            except (ValueError, TypeError):
                pass
        report_list.append(entry)

    # Fetch signals to extract module contributions
    sig_modules: dict[int, list[str]] = {}
    if sig_ids:
        for sig in Signal.objects.filter(id__in=sig_ids).only("id", "payload_json"):
            if not sig.payload_json:
                continue
            reasons = sig.payload_json.get("reasons", {})
            contribs = reasons.get("module_contributions", [])
            mods = []
            for c in contribs:
                name = str(c.get("module", "")).strip().lower()
                if name in MODULE_ORDER:
                    mods.append(name)
            if mods:
                sig_modules[sig.id] = mods

    # Aggregate per module
    stats: dict[str, dict] = {m: {"wins": 0, "losses": 0, "n": 0, "pnl": 0.0} for m in MODULE_ORDER}
    for r in report_list:
        sid = r["signal_id"]
        modules = sig_modules.get(sid, []) if sid else []
        if not modules:
            continue
        for mod in modules:
            if mod not in stats:
                continue
            stats[mod]["n"] += 1
            stats[mod]["pnl"] += r["pnl"]
            if r["win"]:
                stats[mod]["wins"] += 1
            else:
                stats[mod]["losses"] += 1

    return stats


def dynamic_weight_map(
    base_weights: dict[str, float] | None = None,
    days: int | None = None,
    alpha_prior: float | None = None,
    beta_prior: float | None = None,
    min_mult: float | None = None,
    max_mult: float | None = None,
    min_trades: int | None = None,
) -> dict[str, float]:
    """Compute allocator weights adjusted by recent module performance.

    Uses Bayesian Beta-Binomial model:
      - Prior: Beta(alpha_prior, beta_prior) — weakly informative
      - Posterior mean: (alpha + wins) / (alpha + beta + wins + losses)
      - Multiplier: posterior_mean / 0.5  (above 50% → boost, below → dampen)
      - Clamped to [min_mult, max_mult] of base weight

    Returns normalized weight map (sums to 1.0).
    Falls back to static base_weights if not enough data or on error.
    """
    if base_weights is None:
        base_weights = default_weight_map()

    _days = days if days is not None else int(getattr(settings, "ALLOCATOR_DYNAMIC_WINDOW_DAYS", 7))
    _alpha = alpha_prior if alpha_prior is not None else float(getattr(settings, "ALLOCATOR_DYNAMIC_ALPHA_PRIOR", 2.0))
    _beta = beta_prior if beta_prior is not None else float(getattr(settings, "ALLOCATOR_DYNAMIC_BETA_PRIOR", 2.0))
    _min_mult = min_mult if min_mult is not None else float(getattr(settings, "ALLOCATOR_DYNAMIC_MIN_MULT", 0.5))
    _max_mult = max_mult if max_mult is not None else float(getattr(settings, "ALLOCATOR_DYNAMIC_MAX_MULT", 2.0))
    _min_trades = min_trades if min_trades is not None else int(getattr(settings, "ALLOCATOR_DYNAMIC_MIN_TRADES", 10))

    try:
        stats = _module_rolling_stats(days=_days)
    except Exception as exc:
        logger.warning("dynamic_weight_map: failed to load stats: %s", exc)
        return dict(base_weights)

    adjusted: dict[str, float] = {}
    adjustments_log: dict[str, dict] = {}

    for module in MODULE_ORDER:
        base_w = float(base_weights.get(module, 0.0))
        ms = stats.get(module, {"wins": 0, "losses": 0, "n": 0})
        wins = ms["wins"]
        losses = ms["losses"]
        n = ms["n"]

        if n < _min_trades or base_w <= 0:
            # Not enough data — keep base weight unchanged
            adjusted[module] = base_w
            adjustments_log[module] = {
                "base": round(base_w, 4), "mult": 1.0,
                "posterior": 0.5, "n": n, "reason": "insufficient_data",
            }
            continue

        posterior_mean = (_alpha + wins) / (_alpha + _beta + wins + losses)
        mult = posterior_mean / 0.5  # center at 1.0
        mult = max(_min_mult, min(_max_mult, mult))
        new_w = base_w * mult

        adjusted[module] = new_w
        adjustments_log[module] = {
            "base": round(base_w, 4),
            "mult": round(mult, 4),
            "posterior": round(posterior_mean, 4),
            "wins": wins,
            "losses": losses,
            "n": n,
            "wr": round(wins / n, 4) if n > 0 else 0.0,
        }

    # Normalize to sum = 1.0
    total = sum(adjusted.values())
    if total <= 0:
        logger.warning("dynamic_weight_map: all weights zero, falling back to base")
        return dict(base_weights)

    result = {k: v / total for k, v in adjusted.items()}

    logger.info(
        "dynamic_weight_map: %s",
        {m: f"{result[m]:.3f} (×{adjustments_log[m]['mult']:.2f})" for m in MODULE_ORDER},
    )
    return result


def resolve_symbol_allocation(
    module_signals: Iterable[dict],
    *,
    threshold: float,
    base_risk_pct: float,
    session_risk_mult: float,
    weights: dict[str, float],
    risk_budgets: dict[str, float],
    min_active_modules: int | None = None,
) -> dict:
    net_score = 0.0
    abs_capacity = 0.0
    module_contributions: list[dict] = []

    for sig in module_signals:
        module = str(sig.get("module", "")).strip().lower()
        if module not in MODULE_ORDER:
            continue
        sign = direction_to_sign(str(sig.get("direction", "")))
        if sign == 0:
            continue
        confidence = normalize_score(float(sig.get("confidence", 0.0)))
        weight_base = float(weights.get(module, 0.0))
        weight_mult = 1.0
        if (
            module == "smc"
            and bool(getattr(settings, "ALLOCATOR_SMC_CONFLUENCE_BOOST_ENABLED", True))
        ):
            if bool(sig.get("smc_confluence", False)):
                weight_mult = float(
                    getattr(settings, "ALLOCATOR_SMC_CONFLUENCE_WEIGHT_MULT", 1.25)
                )
            else:
                weight_mult = float(
                    getattr(settings, "ALLOCATOR_SMC_NON_CONFLUENCE_WEIGHT_MULT", 0.85)
                )
        weight_mult = max(0.0, float(weight_mult))
        weight = weight_base * weight_mult
        contribution = weight * confidence * sign
        net_score += contribution
        abs_capacity += abs(weight * confidence)
        module_contributions.append(
            {
                "module": module,
                "direction": "long" if sign > 0 else "short",
                "confidence": round(confidence, 4),
                "weight": round(weight, 4),
                "weight_base": round(weight_base, 4),
                "weight_mult": round(weight_mult, 4),
                "smc_confluence": bool(sig.get("smc_confluence", False)),
                "contribution": round(contribution, 6),
            }
        )

    required_modules = max(
        1,
        int(
            min_active_modules
            if min_active_modules is not None
            else getattr(settings, "ALLOCATOR_MIN_MODULES_ACTIVE", 2)
        ),
    )
    if len(module_contributions) < required_modules:
        return {
            "direction": "flat",
            "raw_score": 0.0,
            "net_score": 0.0,
            "confidence": 0.0,
            "risk_budget_pct": 0.0,
            "symbol_state": "blocked",
            "reasons": {
                "threshold": round(float(threshold), 6),
                "module_contributions": module_contributions,
                "active_module_count": len(module_contributions),
                "required_modules": required_modules,
                "abs_capacity": round(abs_capacity, 6),
                "base_risk_pct": round(float(base_risk_pct), 6),
                "session_risk_mult": round(float(session_risk_mult), 6),
                "budget_mix": 0.0,
            },
        }

    # --- Direction-aware score penalty (longs penalized in bear/range regimes) ---
    long_penalty = float(getattr(settings, "ALLOCATOR_LONG_SCORE_PENALTY", 1.0))
    if net_score > 0 and long_penalty < 1.0:
        net_score *= long_penalty

    direction = sign_to_direction(net_score, threshold=threshold)
    allocator_confidence = 0.0
    if abs_capacity > 0:
        allocator_confidence = min(1.0, abs(net_score) / abs_capacity)
    allocator_confidence = normalize_score(allocator_confidence)

    net_sign = direction_to_sign(direction)
    budget_mix = 0.0
    if net_sign != 0:
        for row in module_contributions:
            row_sign = 1 if row["direction"] == "long" else -1
            if row_sign == net_sign:
                budget_mix += float(risk_budgets.get(row["module"], 0.0)) * float(row["confidence"])
    budget_mix = max(0.0, min(1.0, budget_mix))

    risk_budget_pct = 0.0
    if direction in {"long", "short"}:
        risk_budget_pct = max(0.0, base_risk_pct) * allocator_confidence * max(0.0, session_risk_mult)
        if budget_mix > 0:
            risk_budget_pct *= max(0.30, min(1.0, budget_mix))

    symbol_state = "open" if direction in {"long", "short"} else "skip"
    return {
        "direction": direction,
        "raw_score": round(abs(net_score), 6),
        "net_score": round(net_score, 6),
        "confidence": allocator_confidence,
        "risk_budget_pct": round(risk_budget_pct, 6),
        "symbol_state": symbol_state,
        "reasons": {
            "threshold": round(float(threshold), 6),
            "module_contributions": module_contributions,
            "active_module_count": len(module_contributions),
            "abs_capacity": round(abs_capacity, 6),
            "base_risk_pct": round(float(base_risk_pct), 6),
            "session_risk_mult": round(float(session_risk_mult), 6),
            "budget_mix": round(float(budget_mix), 6),
        },
    }
