from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz

from execution.models import OperationReport
from signals.sessions import get_current_session


def _to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _normalize(value: str | None, default: str = "") -> str:
    return str(value or default).strip().lower()


def _trade_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [_to_float(row["pnl_abs"]) for row in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    trade_count = len(rows)
    win_count = len(wins)
    loss_count = len(losses)
    profit_factor = (
        round(gross_profit / gross_loss, 3)
        if gross_loss > 0
        else ("inf" if gross_profit > 0 else 0.0)
    )
    return {
        "trade_count": trade_count,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": round((win_count / trade_count) * 100, 2) if trade_count else 0.0,
        "total_pnl": round(sum(pnls), 4),
        "profit_factor": profit_factor,
        "avg_pnl": round(sum(pnls) / trade_count, 4) if trade_count else 0.0,
    }


@dataclass(frozen=True)
class _Variant:
    name: str
    description: str
    predicate: Callable[[dict[str, Any]], bool]


def _variants() -> list[_Variant]:
    return [
        _Variant(
            name="block_ny_open_buy_balanced_transition",
            description=(
                "Bloquea longs en ny_open cuando btc_lead_state=transition "
                "y recommended_bias=balanced"
            ),
            predicate=lambda row: (
                row["session"] == "ny_open"
                and row["side"] == "buy"
                and row["btc_lead_state"] == "transition"
                and row["recommended_bias"] == "balanced"
            ),
        ),
        _Variant(
            name="block_ny_open_buy_balanced",
            description="Bloquea longs en ny_open cuando recommended_bias=balanced",
            predicate=lambda row: (
                row["session"] == "ny_open"
                and row["side"] == "buy"
                and row["recommended_bias"] == "balanced"
            ),
        ),
        _Variant(
            name="block_ny_open_buy_weak_long_context",
            description=(
                "Bloquea longs en ny_open cuando el contexto es long debil: "
                "recommended_bias in {balanced,tactical_long} y "
                "btc_lead_state in {transition,bear_weak,bear_confirmed}"
            ),
            predicate=lambda row: (
                row["session"] == "ny_open"
                and row["side"] == "buy"
                and row["recommended_bias"] in {"balanced", "tactical_long"}
                and row["btc_lead_state"] in {"transition", "bear_weak", "bear_confirmed"}
            ),
        ),
    ]


def build_ny_open_buy_context_audit(*, days: int = 30, symbol: str = "") -> dict[str, Any]:
    cutoff = dj_tz.now() - timedelta(days=max(1, int(days)))
    qs = (
        OperationReport.objects.filter(closed_at__gte=cutoff)
        .select_related("instrument")
        .order_by("closed_at")
    )
    if symbol:
        qs = qs.filter(instrument__symbol=str(symbol).strip().upper())

    rows: list[dict[str, Any]] = []
    for op in qs:
        entry_dt = getattr(op, "opened_at", None) or op.closed_at
        rows.append(
            {
                "id": op.id,
                "symbol": op.instrument.symbol,
                "session": get_current_session(entry_dt),
                "side": _normalize(op.side),
                "opened_at": entry_dt.isoformat() if entry_dt else "",
                "closed_at": op.closed_at.isoformat() if op.closed_at else "",
                "pnl_abs": _to_float(op.pnl_abs),
                "pnl_pct": _to_float(op.pnl_pct),
                "outcome": _normalize(op.outcome),
                "reason": _normalize(op.reason),
                "close_sub_reason": _normalize(op.close_sub_reason),
                "monthly_regime": _normalize(op.monthly_regime, "unknown"),
                "weekly_regime": _normalize(op.weekly_regime, "unknown"),
                "daily_regime": _normalize(op.daily_regime, "unknown"),
                "btc_lead_state": _normalize(op.btc_lead_state, "unknown"),
                "recommended_bias": _normalize(op.recommended_bias, "unknown"),
            }
        )

    baseline = _trade_metrics(rows)
    result: dict[str, Any] = {
        "days": int(days),
        "cutoff": cutoff.isoformat(),
        "symbol_filter": str(symbol or "").strip().upper(),
        "baseline": baseline,
        "variants": [],
    }

    for variant in _variants():
        removed = [row for row in rows if variant.predicate(row)]
        kept = [row for row in rows if not variant.predicate(row)]
        removed_metrics = _trade_metrics(removed)
        kept_metrics = _trade_metrics(kept)
        result["variants"].append(
            {
                "name": variant.name,
                "description": variant.description,
                "baseline": baseline,
                "after_block": kept_metrics,
                "removed": removed_metrics,
                "delta_total_pnl": round(
                    _to_float(kept_metrics["total_pnl"]) - _to_float(baseline["total_pnl"]),
                    4,
                ),
                "affected_rows": removed,
            }
        )

    return result


class Command(BaseCommand):
    help = (
        "Replay sobre OperationReport reales para evaluar variantes de bloqueo "
        "de longs en ny_open segun contexto MTF (bias/lead_state)."
    )

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--symbol", default="")
        parser.add_argument("--json", default="")

    def handle(self, *args, **options):
        report = build_ny_open_buy_context_audit(
            days=int(options["days"]),
            symbol=str(options.get("symbol") or ""),
        )

        if options.get("json"):
            with open(options["json"], "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
            self.stdout.write(self.style.SUCCESS(f"Saved -> {options['json']}"))
            return

        self.stdout.write("== NY Open Buy Context Audit ==")
        self.stdout.write(
            "Baseline: trades={trade_count} pnl={total_pnl} pf={profit_factor} wr={win_rate}%".format(
                **report["baseline"]
            )
        )
        for variant in report["variants"]:
            self.stdout.write("")
            self.stdout.write(f"[{variant['name']}] {variant['description']}")
            self.stdout.write(
                "after: trades={trade_count} pnl={total_pnl} pf={profit_factor} wr={win_rate}%".format(
                    **variant["after_block"]
                )
            )
            self.stdout.write(
                "removed: trades={trade_count} pnl={total_pnl} wins={wins} losses={losses}".format(
                    **variant["removed"]
                )
            )
            self.stdout.write(f"delta_total_pnl={variant['delta_total_pnl']}")
            for row in variant["affected_rows"][:10]:
                self.stdout.write(
                    "  - {symbol} {opened_at} pnl={pnl_abs} lead={btc_lead_state} "
                    "bias={recommended_bias} monthly={monthly_regime} weekly={weekly_regime} "
                    "daily={daily_regime} reason={reason}:{close_sub_reason}".format(**row)
                )
