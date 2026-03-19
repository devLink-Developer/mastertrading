from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz

from core.models import Instrument
from execution.models import BalanceSnapshot
from execution.tasks import (
    _actual_stop_risk_amount,
    _compute_stop_distance,
    _min_qty_risk_guard,
    _risk_based_qty,
    _volatility_adjusted_risk,
)
from marketdata.models import Candle
from risk.models import RiskEvent


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class Command(BaseCommand):
    help = "Report minimum-lot forced risk vs target risk for enabled instruments."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=7)
        parser.add_argument("--symbol", type=str, default="")
        parser.add_argument("--json", type=str, default="")

    def handle(self, *args, **opts):
        rows = self._build_rows(days=opts["days"], symbol=opts["symbol"])
        if not rows:
            self.stdout.write(self.style.WARNING("No instruments with usable price data."))
            return

        self.stdout.write(
            self.style.NOTICE(
                f"\nMin-Qty Risk Report | equity={rows[0]['equity']:.4f} | "
                f"window={opts['days']}d"
                f"{(' | ' + opts['symbol']) if opts['symbol'] else ''}"
            )
        )
        self.stdout.write(
            "  "
            f"{'Symbol':<10} {'Px':>9} {'Lot':>8} {'Target$':>9} {'RiskQty':>9} "
            f"{'MinNot$':>9} {'Actual$':>9} {'xRisk':>7} {'Now':>6} {'Events':>6}"
        )
        self.stdout.write("  " + "-" * 96)
        for row in rows:
            self.stdout.write(
                "  "
                f"{row['symbol']:<10} "
                f"{row['price']:>9.4f} "
                f"{row['min_qty']:>8.4f} "
                f"{row['target_risk_amount']:>9.4f} "
                f"{row['risk_qty']:>9.4f} "
                f"{row['min_notional']:>9.4f} "
                f"{row['actual_risk_amount']:>9.4f} "
                f"{row['risk_multiplier']:>7.2f} "
                f"{('block' if row['blocked_now'] else 'ok'):>6} "
                f"{row['event_count']:>6}"
            )

        if opts["json"]:
            path = Path(opts["json"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(f"\nJSON written to {path}"))

    def _build_rows(self, *, days: int, symbol: str = "") -> list[dict[str, Any]]:
        balance = BalanceSnapshot.objects.order_by("-created_at").first()
        equity = _f(balance.equity_usdt if balance else 0.0)
        if equity <= 0:
            return []

        cutoff = dj_tz.now() - timedelta(days=max(1, int(days or 1)))
        instruments = Instrument.objects.filter(enabled=True).order_by("symbol")
        if symbol:
            instruments = instruments.filter(symbol=symbol.upper())

        rows: list[dict[str, Any]] = []
        for inst in instruments:
            price = self._latest_price(inst)
            if price <= 0:
                continue
            atr_pct = self._atr_pct_for_report(inst)
            base_risk_pct = _f(settings.PER_INSTRUMENT_RISK.get(inst.symbol, settings.RISK_PER_TRADE_PCT))
            effective_risk_pct = _volatility_adjusted_risk(inst.symbol, atr_pct, base_risk_pct)
            target_risk_amount = max(0.0, equity * effective_risk_pct)
            stop_distance_pct = _compute_stop_distance(inst, "buy", price)
            if not stop_distance_pct or stop_distance_pct <= 0:
                stop_distance_pct = max(
                    _f(getattr(settings, "STOP_LOSS_PCT", 0.0)),
                    _f(getattr(settings, "MIN_SL_PCT", 0.0)),
                )
            min_qty = _f(inst.lot_size or 0.0)
            if min_qty <= 0 or stop_distance_pct <= 0:
                continue
            risk_qty = _risk_based_qty(
                equity,
                price,
                stop_distance_pct,
                1.0,
                leverage=1.0,
                risk_pct=effective_risk_pct,
            )
            qty = max(risk_qty, min_qty)
            blocked_now, actual_risk_amount, risk_multiplier = _min_qty_risk_guard(
                qty=qty,
                risk_qty=risk_qty,
                min_qty=min_qty,
                entry_price=price,
                stop_distance_pct=stop_distance_pct,
                contract_size=1.0,
                target_risk_amount=target_risk_amount,
            )
            if actual_risk_amount <= 0:
                actual_risk_amount = _actual_stop_risk_amount(
                    qty=qty,
                    entry_price=price,
                    stop_distance_pct=stop_distance_pct,
                    contract_size=1.0,
                )
            if risk_multiplier <= 0 and target_risk_amount > 0:
                risk_multiplier = actual_risk_amount / target_risk_amount

            event_count = RiskEvent.objects.filter(
                kind="min_qty_risk_guard",
                instrument=inst,
                ts__gte=cutoff,
            ).count()
            rows.append(
                {
                    "symbol": inst.symbol,
                    "equity": equity,
                    "price": round(price, 8),
                    "atr_pct": round(atr_pct, 6),
                    "stop_distance_pct": round(stop_distance_pct, 6),
                    "base_risk_pct": round(base_risk_pct, 6),
                    "effective_risk_pct": round(effective_risk_pct, 6),
                    "target_risk_amount": round(target_risk_amount, 8),
                    "risk_qty": round(risk_qty, 8),
                    "min_qty": round(min_qty, 8),
                    "min_notional": round(min_qty * price, 8),
                    "actual_risk_amount": round(actual_risk_amount, 8),
                    "risk_multiplier": round(risk_multiplier, 4),
                    "blocked_now": bool(blocked_now),
                    "event_count": int(event_count),
                }
            )
        rows.sort(key=lambda row: (row["risk_multiplier"], row["event_count"]), reverse=True)
        return rows

    def _latest_price(self, inst: Instrument) -> float:
        for tf in ("1m", "5m", "15m", "1h"):
            candle = Candle.objects.filter(instrument=inst, timeframe=tf).order_by("-ts").first()
            if candle is not None and _f(candle.close) > 0:
                return _f(candle.close)
        return 0.0

    def _atr_pct_for_report(self, inst: Instrument, period: int = 14) -> float:
        candles = list(
            Candle.objects.filter(instrument=inst, timeframe="5m")
            .order_by("-ts")[: max(period + 1, 20)]
        )
        if len(candles) < period + 1:
            return 0.0
        candles.reverse()
        trs: list[float] = []
        prev_close = None
        for candle in candles:
            high = _f(candle.high)
            low = _f(candle.low)
            close = _f(candle.close)
            if prev_close is None:
                tr = high - low
            else:
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
            prev_close = close
        if len(trs) < period:
            return 0.0
        atr = sum(trs[-period:]) / period
        last_close = _f(candles[-1].close)
        if last_close <= 0:
            return 0.0
        return atr / last_close
