from datetime import datetime, timezone

from django.test import TestCase

from core.models import Instrument
from .models import Candle


class CandleModelTest(TestCase):
    def test_unique_per_instrument_timeframe_timestamp(self):
        inst = Instrument.objects.create(
            symbol="ETHUSDT", exchange="binance", base="ETH", quote="USDT"
        )
        Candle.objects.create(
            instrument=inst,
            timeframe=Candle.Timeframe.M1,
            ts=datetime.now(timezone.utc),
            open=100,
            high=110,
            low=90,
            close=105,
            volume=10,
        )
        self.assertEqual(Candle.objects.count(), 1)
