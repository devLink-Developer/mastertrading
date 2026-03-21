from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz

from execution.models import OperationReport
from execution.tasks import _compute_tp_sl_prices
from marketdata.models import Candle
from signals.sessions import SESSION_WINDOWS, get_current_session


def _f(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _norm(value: str | None, default: str = "") -> str:
    return str(value or default).strip().lower()


def _session_bounds(ts: datetime) -> tuple[str, datetime, datetime]:
    current = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    session_name = get_current_session(current)
    minute_of_day = current.hour * 60 + current.minute
    for name, start, end in SESSION_WINDOWS:
        if name != session_name:
            continue
        start_min = start[0] * 60 + start[1]
        end_min = end[0] * 60 + end[1]
        start_dt = current.replace(hour=start[0], minute=start[1], second=0, microsecond=0)
        end_dt = current.replace(hour=end[0], minute=end[1], second=0, microsecond=0)
        if start_min < end_min:
            return session_name, start_dt, end_dt
        if minute_of_day < end_min:
            start_dt = start_dt - timedelta(days=1)
        else:
            end_dt = end_dt + timedelta(days=1)
        return session_name, start_dt, end_dt
    fallback = current.replace(hour=0, minute=0, second=0, microsecond=0)
    return session_name, fallback, fallback + timedelta(days=1)


def _atr_pct_at_ts(op: OperationReport, period: int = 14, tf: str = "5m") -> float | None:
    anchor = getattr(op, "opened_at", None) or getattr(op, "closed_at", None)
    if anchor is None:
        return None
    rows = list(
        Candle.objects.filter(
            instrument=op.instrument,
            timeframe=tf,
            ts__lte=anchor,
        )
        .order_by("-ts")[: period + 1]
        .values("high", "low", "close")
    )
    if len(rows) < period + 1:
        return None
    rows.reverse()
    prev_close = _f(rows[0]["close"])
    trs: list[float] = []
    for row in rows[1:]:
        high = _f(row["high"])
        low = _f(row["low"])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        prev_close = _f(row["close"])
    last_close = _f(rows[-1]["close"])
    if last_close <= 0:
        return None
    return (sum(trs[-period:]) / period) / last_close


def _previous_day_levels(op: OperationReport) -> tuple[float | None, float | None]:
    anchor = getattr(op, "opened_at", None) or getattr(op, "closed_at", None)
    if anchor is None:
        return None, None
    current = anchor if anchor.tzinfo else anchor.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    day_start = current.replace(hour=0, minute=0, second=0, microsecond=0)
    row = (
        Candle.objects.filter(
            instrument=op.instrument,
            timeframe="1d",
            ts__lt=day_start,
        )
        .order_by("-ts")
        .values("high", "low")
        .first()
    )
    if not row:
        return None, None
    return _f(row["high"]), _f(row["low"])


def _post_entry_micro_followthrough(op: OperationReport, minutes: int = 2) -> tuple[float, float]:
    entry_ts = getattr(op, "opened_at", None)
    entry_price = _f(op.entry_price)
    if entry_ts is None or entry_price <= 0:
        return 0.0, 0.0
    candles = list(
        Candle.objects.filter(
            instrument=op.instrument,
            timeframe="1m",
            ts__gt=entry_ts,
            ts__lte=entry_ts + timedelta(minutes=max(1, int(minutes))),
        )
        .order_by("ts")
        .values("high", "low")
    )
    side = _norm(op.side)
    best = 0.0
    worst = 0.0
    for row in candles:
        high = _f(row["high"])
        low = _f(row["low"])
        if side == "buy":
            best = max(best, (high - entry_price) / entry_price)
            worst = min(worst, (low - entry_price) / entry_price)
        else:
            best = max(best, (entry_price - low) / entry_price)
            worst = min(worst, (entry_price - high) / entry_price)
    return round(best, 6), round(worst, 6)


def _structure_metrics(op: OperationReport) -> dict[str, Any]:
    entry_ts = getattr(op, "opened_at", None) or getattr(op, "closed_at", None)
    entry_price = _f(op.entry_price)
    side = _norm(op.side)
    if entry_ts is None or entry_price <= 0:
        return {}

    session_name, session_start, _session_end = _session_bounds(entry_ts)
    candles = list(
        Candle.objects.filter(
            instrument=op.instrument,
            timeframe="1m",
            ts__gte=session_start,
            ts__lte=entry_ts,
        )
        .order_by("ts")
        .values("high", "low", "close")
    )
    if not candles:
        return {"session": session_name}

    session_high = max(_f(row["high"]) for row in candles)
    session_low = min(_f(row["low"]) for row in candles)
    session_range_abs = max(0.0, session_high - session_low)
    session_range_pct = (session_range_abs / entry_price) if entry_price > 0 else 0.0
    range_denom = session_range_abs if session_range_abs > 0 else None

    if side == "buy":
        progress = ((entry_price - session_low) / range_denom) if range_denom else 0.5
        entry_extreme_gap_pct = max(0.0, (session_high - entry_price) / entry_price)
    else:
        progress = ((session_high - entry_price) / range_denom) if range_denom else 0.5
        entry_extreme_gap_pct = max(0.0, (entry_price - session_low) / entry_price)

    atr_pct = _atr_pct_at_ts(op)
    tp_price, _sl_price, tp_pct, _sl_pct = _compute_tp_sl_prices(
        side,
        entry_price,
        atr_pct,
        recommended_bias=getattr(op, "recommended_bias", "") or "",
    )

    if side == "buy":
        tp_extension_abs = max(0.0, _f(tp_price) - session_high)
    else:
        tp_extension_abs = max(0.0, session_low - _f(tp_price))

    tp_extension_pct = (tp_extension_abs / entry_price) if entry_price > 0 else 0.0
    tp_extension_vs_session_range = (tp_extension_abs / session_range_abs) if session_range_abs > 0 else 0.0

    prev_day_high, prev_day_low = _previous_day_levels(op)
    if side == "buy":
        prev_day_barrier = bool(
            prev_day_high is not None and entry_price < prev_day_high < _f(tp_price)
        )
    else:
        prev_day_barrier = bool(
            prev_day_low is not None and _f(tp_price) < prev_day_low < entry_price
        )

    follow_2m_best, follow_2m_worst = _post_entry_micro_followthrough(op, minutes=2)

    return {
        "session": session_name,
        "session_start": session_start.isoformat(),
        "session_high_so_far": round(session_high, 8),
        "session_low_so_far": round(session_low, 8),
        "session_range_pct": round(session_range_pct, 6),
        "session_progress": round(progress, 6),
        "entry_extreme_gap_pct": round(entry_extreme_gap_pct, 6),
        "atr_pct_at_entry": round(_f(atr_pct), 6),
        "tp_price_reconstructed": round(_f(tp_price), 8),
        "tp_pct_reconstructed": round(_f(tp_pct), 6),
        "tp_extension_pct": round(tp_extension_pct, 6),
        "tp_extension_vs_session_range": round(tp_extension_vs_session_range, 6),
        "prev_day_high": round(_f(prev_day_high), 8) if prev_day_high is not None else None,
        "prev_day_low": round(_f(prev_day_low), 8) if prev_day_low is not None else None,
        "prev_day_barrier_between_entry_and_tp": prev_day_barrier,
        "post_entry_2m_best_pct": follow_2m_best,
        "post_entry_2m_worst_pct": follow_2m_worst,
    }


def _trade_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [_f(row["pnl_abs"]) for row in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    count = len(rows)
    return {
        "trade_count": count,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round((len(wins) / count) * 100, 2) if count else 0.0,
        "total_pnl": round(sum(pnls), 4),
        "profit_factor": (
            round(gross_profit / gross_loss, 3)
            if gross_loss > 0
            else ("inf" if gross_profit > 0 else 0.0)
        ),
        "avg_pnl": round(sum(pnls) / count, 4) if count else 0.0,
        "avg_session_progress": round(sum(_f(row.get("session_progress")) for row in rows) / count, 6) if count else 0.0,
        "avg_tp_extension_pct": round(sum(_f(row.get("tp_extension_pct")) for row in rows) / count, 6) if count else 0.0,
        "avg_tp_extension_vs_session_range": round(
            sum(_f(row.get("tp_extension_vs_session_range")) for row in rows) / count, 6
        ) if count else 0.0,
        "avg_post_entry_2m_best_pct": round(
            sum(_f(row.get("post_entry_2m_best_pct")) for row in rows) / count, 6
        ) if count else 0.0,
        "avg_post_entry_2m_worst_pct": round(
            sum(_f(row.get("post_entry_2m_worst_pct")) for row in rows) / count, 6
        ) if count else 0.0,
    }


@dataclass(frozen=True)
class _Variant:
    name: str
    description: str
    predicate: Callable[[dict[str, Any]], bool]


def _variants() -> list[_Variant]:
    return [
        _Variant(
            name="block_chasing_top_needing_extension",
            description=(
                "Bloquea longs que entran arriba del 85% del rango de sesion "
                "y cuyo TP exige mas del 25% del rango ya desarrollado"
            ),
            predicate=lambda row: (
                row["side"] == "buy"
                and row["session"] == "ny_open"
                and _f(row.get("session_progress")) >= 0.85
                and _f(row.get("tp_extension_vs_session_range")) >= 0.25
            ),
        ),
        _Variant(
            name="block_chasing_top_with_prevday_barrier",
            description=(
                "Bloquea longs en ny_open que persiguen extremo de sesion "
                "y ademas tienen previous-day high entre entry y TP"
            ),
            predicate=lambda row: (
                row["side"] == "buy"
                and row["session"] == "ny_open"
                and _f(row.get("session_progress")) >= 0.85
                and bool(row.get("prev_day_barrier_between_entry_and_tp"))
            ),
        ),
        _Variant(
            name="block_extreme_chase",
            description=(
                "Bloquea longs en ny_open muy pegados al high de sesion "
                "cuando el TP todavia requiere breakout de al menos 0.20%"
            ),
            predicate=lambda row: (
                row["side"] == "buy"
                and row["session"] == "ny_open"
                and _f(row.get("entry_extreme_gap_pct")) <= 0.0015
                and _f(row.get("tp_extension_pct")) >= 0.0020
            ),
        ),
    ]


def build_session_structure_audit(
    *,
    days: int = 14,
    symbol: str = "",
    sessions: set[str] | None = None,
    side: str = "buy",
) -> dict[str, Any]:
    cutoff = dj_tz.now() - timedelta(days=max(1, int(days)))
    qs = (
        OperationReport.objects.filter(closed_at__gte=cutoff)
        .select_related("instrument")
        .order_by("closed_at")
    )
    if symbol:
        qs = qs.filter(instrument__symbol=symbol.upper())
    if side:
        qs = qs.filter(side=side)

    rows: list[dict[str, Any]] = []
    session_filter = {str(x).strip().lower() for x in (sessions or set()) if str(x).strip()}

    for op in qs:
        base = {
            "id": op.id,
            "symbol": op.instrument.symbol,
            "side": _norm(op.side),
            "opened_at": (getattr(op, "opened_at", None) or op.closed_at).isoformat(),
            "closed_at": op.closed_at.isoformat() if op.closed_at else "",
            "pnl_abs": _f(op.pnl_abs),
            "pnl_pct": _f(op.pnl_pct),
            "outcome": _norm(op.outcome),
            "reason": _norm(op.reason),
            "close_sub_reason": _norm(op.close_sub_reason),
            "recommended_bias": _norm(op.recommended_bias, "unknown"),
            "btc_lead_state": _norm(op.btc_lead_state, "unknown"),
            "monthly_regime": _norm(op.monthly_regime, "unknown"),
            "weekly_regime": _norm(op.weekly_regime, "unknown"),
            "daily_regime": _norm(op.daily_regime, "unknown"),
        }
        metrics = _structure_metrics(op)
        row = {**base, **metrics}
        if session_filter and row.get("session") not in session_filter:
            continue
        rows.append(row)

    baseline = _trade_metrics(rows)
    wins = [row for row in rows if _f(row["pnl_abs"]) > 0]
    losses = [row for row in rows if _f(row["pnl_abs"]) < 0]

    report = {
        "window_days": int(days),
        "symbol_filter": str(symbol or "").strip().upper(),
        "sessions": sorted(session_filter),
        "side": side,
        "baseline": baseline,
        "wins": _trade_metrics(wins),
        "losses": _trade_metrics(losses),
        "variants": [],
    }

    for variant in _variants():
        removed = [row for row in rows if variant.predicate(row)]
        kept = [row for row in rows if not variant.predicate(row)]
        report["variants"].append(
            {
                "name": variant.name,
                "description": variant.description,
                "after_block": _trade_metrics(kept),
                "removed": _trade_metrics(removed),
                "delta_total_pnl": round(
                    _f(_trade_metrics(kept)["total_pnl"]) - _f(baseline["total_pnl"]),
                    4,
                ),
                "affected_rows": removed,
            }
        )

    return report


class Command(BaseCommand):
    help = (
        "Audita trades reales contra estructura de sesion y variantes de gate "
        "para chase/reachability del TP."
    )

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=14)
        parser.add_argument("--symbol", default="")
        parser.add_argument("--sessions", default="ny_open,london")
        parser.add_argument("--side", default="buy")
        parser.add_argument("--json", default="")

    def handle(self, *args, **options):
        sessions = {x.strip().lower() for x in str(options["sessions"] or "").split(",") if x.strip()}
        report = build_session_structure_audit(
            days=int(options["days"]),
            symbol=str(options.get("symbol") or ""),
            sessions=sessions,
            side=str(options.get("side") or "buy").strip().lower(),
        )

        if options.get("json"):
            path = Path(options["json"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(f"Saved -> {path}"))
            return

        self.stdout.write("== Session Structure Audit ==")
        self.stdout.write(
            "Baseline: trades={trade_count} pnl={total_pnl} pf={profit_factor} wr={win_rate}%".format(
                **report["baseline"]
            )
        )
        self.stdout.write(
            "Wins: trades={trade_count} avg_progress={avg_session_progress} avg_tp_ext={avg_tp_extension_pct} "
            "avg_tp_ext_vs_range={avg_tp_extension_vs_session_range} "
            "post2m_best={avg_post_entry_2m_best_pct} post2m_worst={avg_post_entry_2m_worst_pct}".format(
                **report["wins"]
            )
        )
        self.stdout.write(
            "Losses: trades={trade_count} avg_progress={avg_session_progress} avg_tp_ext={avg_tp_extension_pct} "
            "avg_tp_ext_vs_range={avg_tp_extension_vs_session_range} "
            "post2m_best={avg_post_entry_2m_best_pct} post2m_worst={avg_post_entry_2m_worst_pct}".format(
                **report["losses"]
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
                    "  - {symbol} {opened_at} pnl={pnl_abs} session={session} progress={session_progress} "
                    "tp_ext={tp_extension_pct} tp_ext_vs_range={tp_extension_vs_session_range} "
                    "barrier={prev_day_barrier_between_entry_and_tp} bias={recommended_bias} "
                    "lead={btc_lead_state}".format(**row)
                )
