from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from django.test import TestCase, override_settings

from core.models import Instrument
from execution.models import BalanceSnapshot
from marketdata.models import Candle
from risk.tasks import send_min_qty_risk_report


class MinQtyRiskReportTaskTest(TestCase):
    def setUp(self):
        self.inst = Instrument.objects.create(
            symbol="SOLUSDT",
            exchange="bingx",
            base="SOL",
            quote="USDT",
            enabled=True,
            lot_size=1.0,
            tick_size=0.001,
        )
        base_ts = datetime(2026, 3, 19, 0, 0, tzinfo=timezone.utc)
        price = 90.0
        for idx in range(30):
            Candle.objects.create(
                instrument=self.inst,
                timeframe="5m",
                ts=base_ts + timedelta(minutes=5 * idx),
                open=price + (idx * 0.05),
                high=price + (idx * 0.05) + 0.20,
                low=price + (idx * 0.05) - 0.20,
                close=price + (idx * 0.05) + 0.02,
                volume=1000 + idx,
            )
        Candle.objects.create(
            instrument=self.inst,
            timeframe="1m",
            ts=base_ts + timedelta(hours=3),
            open=90.24,
            high=90.35,
            low=90.10,
            close=90.24,
            volume=1500,
        )
        BalanceSnapshot.objects.create(
            equity_usdt=57.93,
            free_usdt=57.93,
            notional_usdt=0,
            eff_leverage=0,
        )

    @override_settings(
        TELEGRAM_ENABLED=True,
        MIN_QTY_RISK_REPORT_ENABLED=True,
        MIN_QTY_RISK_REPORT_DAYS=7,
        RISK_PER_TRADE_PCT=0.003,
        PER_INSTRUMENT_RISK={"SOLUSDT": 0.002},
        STOP_LOSS_PCT=0.007,
        MIN_SL_PCT=0.012,
        ATR_MULT_SL=1.5,
        MIN_QTY_RISK_GUARD_ENABLED=True,
        MIN_QTY_RISK_MULTIPLIER_MAX=3.0,
        MIN_QTY_DYNAMIC_ALLOWLIST_WATCH_MULTIPLIER=2.0,
        MIN_QTY_DYNAMIC_ALLOWLIST_BLOCK_MULTIPLIER=3.0,
    )
    @patch("risk.tasks.send_telegram", return_value=True)
    def test_send_min_qty_risk_report_sends_blocked_summary(self, send_mock):
        result = send_min_qty_risk_report()
        self.assertEqual(result, "min_qty_risk_report:sent=1")
        self.assertEqual(send_mock.call_count, 1)
        message = send_mock.call_args[0][0]
        self.assertIn("Min-Qty Risk Daily", message)
        self.assertIn("Blocked (1)", message)
        self.assertIn("SOLUSDT", message)
