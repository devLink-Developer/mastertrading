from __future__ import annotations

from decimal import Decimal

from django.core.management.base import BaseCommand

from adapters.bingx import BingXFuturesAdapter
from core.models import Instrument
from execution.tasks import _is_tick_size_mode, _to_float


def _price_tick(market: dict | None, *, precision_mode=None) -> float:
    if not isinstance(market, dict):
        return 0.0
    try:
        precision = market.get("precision") or {}
        price_raw = precision.get("price")
        price_val = _to_float(price_raw)
        if price_val <= 0:
            return 0.0
        if _is_tick_size_mode(precision_mode):
            return price_val
        if float(price_val).is_integer():
            return 10 ** (-int(price_val))
        return price_val
    except Exception:
        return 0.0


def _amount_step(market: dict | None, *, precision_mode=None) -> float:
    if not isinstance(market, dict):
        return 0.0
    try:
        precision = market.get("precision") or {}
        amount_raw = precision.get("amount")
        amount_val = _to_float(amount_raw)
        if amount_val < 0:
            return 0.0
        if _is_tick_size_mode(precision_mode):
            return amount_val if amount_val > 0 else 0.0
        if float(amount_val).is_integer():
            return 10 ** (-int(amount_val))
        return amount_val if amount_val > 0 else 0.0
    except Exception:
        return 0.0


class Command(BaseCommand):
    help = "Sync tick_size and lot_size for enabled instruments from BingX market metadata."

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbols",
            default="",
            help="Comma-separated symbols to sync. Defaults to all enabled instruments.",
        )

    def handle(self, *args, **options):
        symbols_raw = str(options.get("symbols") or "")
        requested = {
            s.strip().upper()
            for s in symbols_raw.split(",")
            if s.strip()
        }

        qs = Instrument.objects.filter(enabled=True).order_by("symbol")
        if requested:
            qs = qs.filter(symbol__in=requested)

        adapter = BingXFuturesAdapter.from_config()
        precision_mode = getattr(adapter.client, "precisionMode", None)
        updated = 0

        for inst in qs:
            mapped = adapter._map_symbol(inst.symbol)
            market = adapter.client.market(mapped)

            limits = market.get("limits") or {}
            amount_limits = limits.get("amount") or {}
            lot_size = max(
                _to_float(amount_limits.get("min")),
                _amount_step(market, precision_mode=precision_mode),
            )
            tick_size = _price_tick(market, precision_mode=precision_mode)

            changed = []
            if lot_size > 0 and _to_float(inst.lot_size) != lot_size:
                inst.lot_size = Decimal(str(lot_size))
                changed.append("lot_size")
            if tick_size > 0 and _to_float(inst.tick_size) != tick_size:
                inst.tick_size = Decimal(str(tick_size))
                changed.append("tick_size")

            if changed:
                inst.save(update_fields=changed)
                updated += 1

            self.stdout.write(
                f"{inst.symbol}: lot_size={lot_size} tick_size={tick_size} changed={','.join(changed) or 'no'}"
            )

        self.stdout.write(self.style.SUCCESS(f"Sync complete. Updated: {updated}"))
