import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz

from execution.models import OperationReport
from marketdata.models import Candle
from signals.sessions import get_current_session


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _directional_pct(side: str, entry_price: float, exit_price: float) -> float:
    if entry_price <= 0 or exit_price <= 0:
        return 0.0
    sign = 1 if str(side or "").strip().lower() == "buy" else -1
    return ((exit_price - entry_price) / entry_price) * sign


def _loss_replay_metrics(op: OperationReport, post_minutes: int) -> dict[str, Any]:
    closed_at = getattr(op, "closed_at", None)
    entry_price = _f(op.entry_price)
    if closed_at is None or entry_price <= 0:
        return {
            "post_minutes": int(post_minutes),
            "candles": 0,
            "recovered_to_entry": False,
            "minutes_to_recovery": None,
            "best_after_close_pct": 0.0,
            "worst_after_close_pct": 0.0,
            "close_after_window_pct": 0.0,
        }

    candles = list(
        Candle.objects.filter(
            instrument=op.instrument,
            timeframe="1m",
            ts__gt=closed_at,
            ts__lte=closed_at + timedelta(minutes=max(1, int(post_minutes or 1))),
        ).order_by("ts")
    )
    if not candles:
        return {
            "post_minutes": int(post_minutes),
            "candles": 0,
            "recovered_to_entry": False,
            "minutes_to_recovery": None,
            "best_after_close_pct": 0.0,
            "worst_after_close_pct": 0.0,
            "close_after_window_pct": 0.0,
        }

    side = str(op.side or "").strip().lower()
    recovered = False
    recovery_min = None
    best_pct = 0.0
    worst_pct = 0.0
    last_close_pct = 0.0

    for candle in candles:
        high = _f(candle.high)
        low = _f(candle.low)
        close = _f(candle.close)
        if side == "buy":
            favorable_pct = max(0.0, (high - entry_price) / entry_price)
            adverse_pct = min(0.0, (low - entry_price) / entry_price)
            if not recovered and high >= entry_price:
                recovered = True
                recovery_min = round((candle.ts - closed_at).total_seconds() / 60.0, 2)
            last_close_pct = (close - entry_price) / entry_price
        else:
            favorable_pct = max(0.0, (entry_price - low) / entry_price)
            adverse_pct = min(0.0, (entry_price - high) / entry_price)
            if not recovered and low <= entry_price:
                recovered = True
                recovery_min = round((candle.ts - closed_at).total_seconds() / 60.0, 2)
            last_close_pct = (entry_price - close) / entry_price
        best_pct = max(best_pct, favorable_pct)
        worst_pct = min(worst_pct, adverse_pct)

    return {
        "post_minutes": int(post_minutes),
        "candles": len(candles),
        "recovered_to_entry": recovered,
        "minutes_to_recovery": recovery_min,
        "best_after_close_pct": round(best_pct, 6),
        "worst_after_close_pct": round(worst_pct, 6),
        "close_after_window_pct": round(last_close_pct, 6),
    }


def _loss_category(op: OperationReport, replay: dict[str, Any]) -> str:
    reason = str(op.reason or "").strip().lower()
    sub_reason = str(op.close_sub_reason or "").strip().lower()
    recovered = bool(replay.get("recovered_to_entry"))
    close_after_window_pct = _f(replay.get("close_after_window_pct"))

    if reason == "exchange_close" and sub_reason == "unknown":
        return "bug_candidate"
    if reason in {"uptrend_short_kill", "downtrend_long_kill"}:
        if recovered and close_after_window_pct > 0:
            return "timing_candidate"
        return "real_loss_bad_entry"
    if reason == "exchange_close" and sub_reason in {"exchange_stop", "likely_liquidation"}:
        if recovered and close_after_window_pct < 0:
            return "real_loss_stop_late_recovery"
        return "real_loss_stop"
    if reason == "sl":
        return "real_loss_stop"
    return "real_loss_bad_entry"


def build_recent_loss_audit(*, days: int, post_minutes: int, symbol: str = "", reason: str = "") -> dict[str, Any]:
    cutoff = dj_tz.now() - timedelta(days=max(1, int(days or 1)))
    qs = (
        OperationReport.objects.filter(
            closed_at__gte=cutoff,
            outcome=OperationReport.Outcome.LOSS,
        )
        .select_related("instrument")
        .order_by("-closed_at")
    )
    if symbol:
        qs = qs.filter(instrument__symbol=symbol.upper())
    if reason:
        qs = qs.filter(reason=reason)

    rows: list[dict[str, Any]] = []
    by_category: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "pnl_abs": 0.0})
    by_symbol: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "pnl_abs": 0.0})
    by_session: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "pnl_abs": 0.0})
    alt_hold_summary = {
        "actual_pnl_abs": 0.0,
        "alt_close_after_window_pnl_abs": 0.0,
        "delta_vs_actual": 0.0,
    }

    for op in qs:
        replay = _loss_replay_metrics(op, post_minutes)
        category = _loss_category(op, replay)
        pnl_abs = _f(op.pnl_abs)
        session_name = get_current_session(getattr(op, "opened_at", None) or op.closed_at)
        close_after_window_pct = _f(replay["close_after_window_pct"])
        alt_pnl_abs = close_after_window_pct * _f(op.notional_usdt)

        row = {
            "id": op.id,
            "symbol": op.instrument.symbol,
            "side": op.side,
            "session": session_name,
            "opened_at": op.opened_at.isoformat() if op.opened_at else None,
            "closed_at": op.closed_at.isoformat() if op.closed_at else None,
            "reason": op.reason,
            "close_sub_reason": op.close_sub_reason,
            "entry_price": _f(op.entry_price),
            "exit_price": _f(op.exit_price),
            "pnl_abs": pnl_abs,
            "pnl_pct": _f(op.pnl_pct),
            "mfe_r": _f(op.mfe_r),
            "mae_r": _f(op.mae_r),
            "category": category,
            **replay,
            "alt_close_after_window_pnl_abs": round(alt_pnl_abs, 8),
        }
        rows.append(row)

        for bucket in (by_category[category], by_symbol[op.instrument.symbol], by_session[session_name]):
            bucket["count"] += 1
            bucket["pnl_abs"] += pnl_abs

        alt_hold_summary["actual_pnl_abs"] += pnl_abs
        alt_hold_summary["alt_close_after_window_pnl_abs"] += alt_pnl_abs

    alt_hold_summary["delta_vs_actual"] = (
        alt_hold_summary["alt_close_after_window_pnl_abs"] - alt_hold_summary["actual_pnl_abs"]
    )

    def _sorted_summary(source: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows_out = []
        for key, vals in source.items():
            rows_out.append(
                {
                    "bucket": key,
                    "count": int(vals["count"]),
                    "pnl_abs": round(_f(vals["pnl_abs"]), 8),
                }
            )
        rows_out.sort(key=lambda item: (item["pnl_abs"], -item["count"]))
        return rows_out

    return {
        "window_days": int(days),
        "post_minutes": int(post_minutes),
        "total_losses": len(rows),
        "alt_hold_summary": {k: round(_f(v), 8) for k, v in alt_hold_summary.items()},
        "by_category": _sorted_summary(by_category),
        "by_symbol": _sorted_summary(by_symbol),
        "by_session": _sorted_summary(by_session),
        "rows": rows,
    }


class Command(BaseCommand):
    help = "Audit recent losing trades and replay what happened after the close."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=5)
        parser.add_argument("--post-minutes", type=int, default=60)
        parser.add_argument("--symbol", type=str, default="")
        parser.add_argument("--reason", type=str, default="")
        parser.add_argument("--json", type=str, default="")

    def handle(self, *args, **opts):
        report = build_recent_loss_audit(
            days=opts["days"],
            post_minutes=opts["post_minutes"],
            symbol=opts["symbol"],
            reason=opts["reason"],
        )
        if report["total_losses"] <= 0:
            self.stdout.write(self.style.WARNING("No recent losses found."))
            return

        self.stdout.write(
            self.style.NOTICE(
                f"\nRecent Loss Audit | window={report['window_days']}d | replay={report['post_minutes']}m"
            )
        )
        self.stdout.write(f"  Total losses: {report['total_losses']}")
        alt = report["alt_hold_summary"]
        self.stdout.write(
            "  Alt-hold summary: "
            f"actual={alt['actual_pnl_abs']:.4f} "
            f"alt={alt['alt_close_after_window_pnl_abs']:.4f} "
            f"delta={alt['delta_vs_actual']:+.4f}"
        )

        for label in ("by_category", "by_symbol", "by_session"):
            self.stdout.write(f"\n  {label}:")
            for row in report[label]:
                self.stdout.write(
                    f"    {row['bucket']:<24} n={row['count']:<3} pnl={row['pnl_abs']:+.4f}"
                )

        if opts["json"]:
            path = Path(opts["json"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(f"\nJSON written to {path}"))
