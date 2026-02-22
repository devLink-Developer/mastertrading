from datetime import datetime, timezone

from django.test import TestCase

from core.models import Instrument
from .models import Order


class OrderModelTest(TestCase):
    def test_create_order(self):
        inst = Instrument.objects.create(
            symbol="BTCUSDT", exchange="binance", base="BTC", quote="USDT"
        )
        order = Order.objects.create(
            instrument=inst,
            side=Order.OrderSide.BUY,
            type=Order.OrderType.MARKET,
            qty=1,
            status=Order.OrderStatus.NEW,
            correlation_id="test-123",
        )
        self.assertEqual(order.instrument.symbol, "BTCUSDT")
