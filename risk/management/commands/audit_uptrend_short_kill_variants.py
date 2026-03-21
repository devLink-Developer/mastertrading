import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz

from execution.models import OperationReport
from execution.tasks import _compute_tp_sl_prices
from marketdata.models import Candle


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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


def _directional_pct(side: str, entry_price: float, exit_price: float) -> float:
    if entry_price <= 0 or exit_price <= 0:
        return 0.0
    sign = 1 if str(side or "").strip().lower() == "buy" else -1
    return ((exit_price - entry_price) / entry_price) * sign


def _replay_variant(op: OperationReport, grace_minutes: int, stop_price: float) -> dict[str, Any]:
    closed_at = getattr(op, "closed_at", None)
    entry_price = _f(op.entry_price)
    side = str(op.side or "").strip().lower()
    if closed_at is None or entry_price <= 0 or grace_minutes <= 0:
        return {
            "grace_minutes": int(grace_minutes),
            "candles": 0,
            "status": "no_data",
            "exit_price": 0.0,
            "alt_pnl_pct": 0.0,
            "alt_pnl_abs": 0.0,
            "minutes_to_exit": None,
        }

    candles = list(
        Candle.objects.filter(
            instrument=op.instrument,
            timeframe="1m",
            ts__gt=closed_at,
            ts__lte=closed_at + timedelta(minutes=grace_minutes),
        ).order_by("ts")
    )
    if not candles:
        return {
            "grace_minutes": int(grace_minutes),
            "candles": 0,
            "status": "no_data",
            "exit_price": 0.0,
            "alt_pnl_pct": 0.0,
            "alt_pnl_abs": 0.0,
            "minutes_to_exit": None,
        }

    status = "window_close"
    exit_price = _f(candles[-1].close)
    minutes_to_exit = round((candles[-1].ts - closed_at).total_seconds() / 60.0, 2)
    stop_price_val = max(0.0, _f(stop_price))

    for candle in candles:
        high = _f(candle.high)
        low = _f(candle.low)
        if side == "sell" and stop_price_val > 0 and high >= stop_price_val:
            status = "stop_breach"
            exit_price = stop_price_val
            minutes_to_exit = round((candle.ts - closed_at).total_seconds() / 60.0, 2)
            break
        if side == "buy" and stop_price_val > 0 and low <= stop_price_val:
            status = "stop_breach"
            exit_price = stop_price_val
            minutes_to_exit = round((candle.ts - closed_at).total_seconds() / 60.0, 2)
            break

    alt_pnl_pct = _directional_pct(side, entry_price, exit_price)
    alt_pnl_abs = alt_pnl_pct * _f(op.notional_usdt)
    return {
        "grace_minutes": int(grace_minutes),
        "candles": len(candles),
        "status": status,
        "exit_price": round(exit_price, 8),
        "alt_pnl_pct": round(alt_pnl_pct, 6),
        "alt_pnl_abs": round(alt_pnl_abs, 8),
        "minutes_to_exit": minutes_to_exit,
    }


def build_uptrend_short_kill_variant_report(*, days: int, grace_windows: list[int]) -> dict[str, Any]:
    cutoff = dj_tz.now() - timedelta(days=max(1, int(days or 1)))
    ops = list(
        OperationReport.objects.filter(
            closed_at__gte=cutoff,
            reason="uptrend_short_kill",
            outcome=OperationReport.Outcome.LOSS,
            side="sell",
        )
        .select_related("instrument")
        .order_by("-closed_at")
    )

    rows: list[dict[str, Any]] = []
    summary_by_grace: list[dict[str, Any]] = []
    for op in ops:
        atr_pct = _atr_pct_at_ts(op)
        _, stop_price, _, _ = _compute_tp_sl_prices("sell", _f(op.entry_price), atr_pct)
        variants = []
        for grace in grace_windows:
            variants.append(_replay_variant(op, int(grace), stop_price))

        rows.append(
            {
                "id": op.id,
                "symbol": op.instrument.symbol,
                "opened_at": op.opened_at.isoformat() if op.opened_at else None,
                "closed_at": op.closed_at.isoformat() if op.closed_at else None,
                "entry_price": _f(op.entry_price),
                "actual_exit_price": _f(op.exit_price),
                "actual_pnl_abs": _f(op.pnl_abs),
                "actual_pnl_pct": _f(op.pnl_pct),
                "atr_pct_at_entry": round(_f(atr_pct), 6),
                "reconstructed_stop_price": round(_f(stop_price), 8),
                "variants": variants,
            }
        )

    for grace in grace_windows:
        alt_sum = 0.0
        actual_sum = 0.0
        improved = 0
        worsened = 0
        stop_breaches = 0
        usable = 0
        for row in rows:
            variant = next((v for v in row["variants"] if int(v["grace_minutes"]) == int(grace)), None)
            if not variant or variant["status"] == "no_data":
                continue
            usable += 1
            actual = _f(row["actual_pnl_abs"])
            alt = _f(variant["alt_pnl_abs"])
            actual_sum += actual
            alt_sum += alt
            if variant["status"] == "stop_breach":
                stop_breaches += 1
            if alt > actual:
                improved += 1
            elif alt < actual:
                worsened += 1
        summary_by_grace.append(
            {
                "grace_minutes": int(grace),
                "trades": usable,
                "actual_pnl_abs": round(actual_sum, 8),
                "alt_pnl_abs": round(alt_sum, 8),
                "delta_vs_actual": round(alt_sum - actual_sum, 8),
                "improved": improved,
                "worsened": worsened,
                "stop_breaches": stop_breaches,
            }
        )

    return {
        "window_days": int(days),
        "grace_windows": [int(g) for g in grace_windows],
        "trades": len(rows),
        "summary_by_grace": summary_by_grace,
        "rows": rows,
    }


class Command(BaseCommand):
    help = "Replay recent uptrend_short_kill losses with stop-aware grace windows."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=5)
        parser.add_argument("--grace", type=str, default="15,30,45,60")
        parser.add_argument("--json", type=str, default="")

    def handle(self, *args, **opts):
        grace_windows = [
            max(1, int(chunk.strip()))
            for chunk in str(opts["grace"] or "15,30,45,60").split(",")
            if str(chunk).strip()
        ]
        report = build_uptrend_short_kill_variant_report(
            days=opts["days"],
            grace_windows=grace_windows,
        )
        if report["trades"] <= 0:
            self.stdout.write(self.style.WARNING("No recent uptrend_short_kill losses found."))
            return

        self.stdout.write(
            self.style.NOTICE(
                f"\nUptrend Short Kill Variants | window={report['window_days']}d | trades={report['trades']}"
            )
        )
        for row in report["summary_by_grace"]:
            self.stdout.write(
                "  "
                f"grace={row['grace_minutes']:>3}m "
                f"actual={row['actual_pnl_abs']:+.4f} "
                f"alt={row['alt_pnl_abs']:+.4f} "
                f"delta={row['delta_vs_actual']:+.4f} "
                f"improved={row['improved']} "
                f"worsened={row['worsened']} "
                f"stop_breaches={row['stop_breaches']}"
            )

        if opts["json"]:
            path = Path(opts["json"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(f"\nJSON written to {path}"))
