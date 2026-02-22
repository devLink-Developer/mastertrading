from django.test import TestCase

from .models import Instrument


class InstrumentModelTest(TestCase):
    def test_str(self):
        inst = Instrument.objects.create(
            symbol="BTCUSDT",
            exchange="binance",
            base="BTC",
            quote="USDT",
            kind=Instrument.InstrumentKind.PERP,
        )
        self.assertIn("BTCUSDT", str(inst))
