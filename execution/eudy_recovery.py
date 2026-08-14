from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

from django.conf import settings
from django.utils import timezone as dj_tz
from django.utils.dateparse import parse_datetime

from .models import OperationReport


@dataclass(frozen=True)
class EudyEdgeDecision:
    allowed: bool
    risk_mult: float
    status: str
    reason: str
    sample_size: int = 0
    profit_factor: float = 0.0
    expectancy_pct: float = 0.0


def _normalize(value: object) -> str:
    return str(value or "").strip().lower()


def is_eudy_recovery_account(account_alias: str) -> bool:
    if not bool(getattr(settings, "EUDY_RECOVERY_ENABLED", False)):
        return False
    aliases = {
        _normalize(value)
        for value in getattr(settings, "EUDY_RECOVERY_ACCOUNT_ALIASES", {"eudy"})
        if _normalize(value)
    }
    return _normalize(account_alias) in aliases


def _profit_factor(pnls: list[float]) -> float:
    gross_win = sum(value for value in pnls if value > 0)
    gross_loss = abs(sum(value for value in pnls if value < 0))
    if gross_loss <= 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def classify_eudy_edge_sample(
    pnl_pct_values: Iterable[float],
    *,
    min_trades: int,
    min_profit_factor: float,
    min_expectancy_pct: float,
    exploration_risk_mult: float,
    context: str,
) -> EudyEdgeDecision:
    """Classify one regime+direction cohort using net, fee-aware returns."""
    pnls = [float(value or 0.0) for value in pnl_pct_values]
    sample_size = len(pnls)
    exploration_mult = max(0.0, min(float(exploration_risk_mult), 1.0))

    if sample_size < max(1, int(min_trades)):
        return EudyEdgeDecision(
            allowed=exploration_mult > 0,
            risk_mult=exploration_mult,
            status="explore",
            reason=(
                f"eudy_edge_explore:{context}:"
                f"n={sample_size}<{max(1, int(min_trades))}:mult={exploration_mult:.2f}"
            ),
            sample_size=sample_size,
        )

    profit_factor = _profit_factor(pnls)
    expectancy_pct = sum(pnls) / sample_size
    if profit_factor >= float(min_profit_factor) and expectancy_pct >= float(min_expectancy_pct):
        return EudyEdgeDecision(
            allowed=True,
            risk_mult=1.0,
            status="validated",
            reason=(
                f"eudy_edge_validated:{context}:n={sample_size}:"
                f"pf={profit_factor:.3f}:expect={expectancy_pct * 100:.3f}%"
            ),
            sample_size=sample_size,
            profit_factor=profit_factor,
            expectancy_pct=expectancy_pct,
        )

    return EudyEdgeDecision(
        allowed=False,
        risk_mult=0.0,
        status="blocked",
        reason=(
            f"eudy_edge_negative:{context}:n={sample_size}:"
            f"pf={profit_factor:.3f},required={float(min_profit_factor):.3f}:"
            f"expect={expectancy_pct * 100:.3f}%,required={float(min_expectancy_pct) * 100:.3f}%"
        ),
        sample_size=sample_size,
        profit_factor=profit_factor,
        expectancy_pct=expectancy_pct,
    )


def evaluate_eudy_edge_guard(
    *,
    account_alias: str,
    side: str,
    daily_regime: str,
    btc_lead_state: str,
    recommended_bias: str,
) -> EudyEdgeDecision:
    """Evaluate Eudy's current regime+side cohort from recent live outcomes.

    This is deliberately account-wide rather than symbol-specific. Existing
    symbol and symbol-side health guards remain responsible for local failures,
    while this guard prevents a broad market regime from consuming risk.
    """
    if not is_eudy_recovery_account(account_alias):
        return EudyEdgeDecision(True, 1.0, "bypass", "eudy_edge_bypass")

    side_text = _normalize(side)
    day_text = _normalize(daily_regime)
    lead_text = _normalize(btc_lead_state)
    bias_text = _normalize(recommended_bias)
    context = f"day={day_text or 'unknown'},lead={lead_text or 'unknown'},bias={bias_text or 'unknown'},side={side_text or 'unknown'}"

    min_trades = max(1, int(getattr(settings, "EUDY_EDGE_GUARD_MIN_TRADES", 12)))
    min_profit_factor = max(
        0.0,
        float(getattr(settings, "EUDY_EDGE_GUARD_MIN_PROFIT_FACTOR", 1.15)),
    )
    min_expectancy_pct = float(
        getattr(settings, "EUDY_EDGE_GUARD_MIN_EXPECTANCY_PCT", 0.0)
    )
    exploration_mult = max(
        0.0,
        min(float(getattr(settings, "EUDY_EDGE_GUARD_EXPLORATION_RISK_MULT", 0.5)), 1.0),
    )

    # Missing regime state is treated as an unvalidated context, never as full risk.
    if not all((side_text, day_text, lead_text, bias_text)):
        return classify_eudy_edge_sample(
            [],
            min_trades=min_trades,
            min_profit_factor=min_profit_factor,
            min_expectancy_pct=min_expectancy_pct,
            exploration_risk_mult=exploration_mult,
            context=context,
        )

    lookback_days = max(1, int(getattr(settings, "EUDY_EDGE_GUARD_LOOKBACK_DAYS", 120)))
    max_trades = max(min_trades, int(getattr(settings, "EUDY_EDGE_GUARD_MAX_TRADES", 60)))
    cutoff = dj_tz.now() - timedelta(days=lookback_days)
    reset_raw = str(getattr(settings, "EUDY_EDGE_GUARD_RESET_AT", "") or "").strip()
    reset_at = parse_datetime(reset_raw) if reset_raw else None
    if reset_at is not None:
        if dj_tz.is_naive(reset_at):
            reset_at = dj_tz.make_aware(reset_at, dj_tz.get_current_timezone())
        cutoff = max(cutoff, reset_at)

    try:
        reports = list(
            OperationReport.objects.filter(
                mode=str(getattr(settings, "MODE", "live") or "live"),
                side=side_text,
                daily_regime__iexact=day_text,
                btc_lead_state__iexact=lead_text,
                recommended_bias__iexact=bias_text,
                closed_at__gte=cutoff,
            )
            .order_by("-closed_at")
            .values_list("pnl_pct", flat=True)[:max_trades]
        )
    except Exception as exc:
        # A telemetry/database failure must not restore full risk. Keep learning
        # at reduced size so a transient outage does not freeze the account.
        return EudyEdgeDecision(
            allowed=exploration_mult > 0,
            risk_mult=exploration_mult,
            status="telemetry_error",
            reason=f"eudy_edge_telemetry_error:{type(exc).__name__}:mult={exploration_mult:.2f}",
        )

    return classify_eudy_edge_sample(
        reports,
        min_trades=min_trades,
        min_profit_factor=min_profit_factor,
        min_expectancy_pct=min_expectancy_pct,
        exploration_risk_mult=exploration_mult,
        context=context,
    )
