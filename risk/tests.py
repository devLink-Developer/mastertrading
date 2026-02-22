from django.test import TestCase

from core.models import Instrument
from .models import RiskEvent


class RiskEventTest(TestCase):
    def test_create_risk_event(self):
        inst = Instrument.objects.create(
            symbol="BCHUSDT", exchange="binance", base="BCH", quote="USDT"
        )
        evt = RiskEvent.objects.create(kind="kill-switch", instrument=inst)
        self.assertEqual(evt.kind, "kill-switch")
