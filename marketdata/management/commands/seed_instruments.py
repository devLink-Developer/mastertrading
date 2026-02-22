from django.core.management.base import BaseCommand

from core.models import Instrument

DEFAULT = [
    {"symbol": "BTCUSDT", "base": "BTC", "quote": "USDT"},
    {"symbol": "ETHUSDT", "base": "ETH", "quote": "USDT"},
    {"symbol": "SOLUSDT", "base": "SOL", "quote": "USDT"},
    {"symbol": "XRPUSDT", "base": "XRP", "quote": "USDT"},
    {"symbol": "DOGEUSDT", "base": "DOGE", "quote": "USDT"},
    {"symbol": "ADAUSDT", "base": "ADA", "quote": "USDT"},
    {"symbol": "LINKUSDT", "base": "LINK", "quote": "USDT"},
]


class Command(BaseCommand):
    help = "Seed default top instruments for Binance Futures"

    def handle(self, *args, **options):
        created = 0
        for item in DEFAULT:
            inst, is_created = Instrument.objects.get_or_create(
                symbol=item["symbol"],
                defaults={
                    "exchange": "binance",
                    "base": item["base"],
                    "quote": item["quote"],
                    "kind": Instrument.InstrumentKind.PERP,
                    "enabled": True,
                },
            )
            if is_created:
                created += 1
        self.stdout.write(self.style.SUCCESS(f"Seed complete. Created: {created}"))
