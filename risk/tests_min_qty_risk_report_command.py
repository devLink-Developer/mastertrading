from datetime import datetime, timedelta, timezone

from django.test import TestCase, override_settings

from core.models import Instrument
from execution.models import BalanceSnapshot
from marketdata.models import Candle
from risk.management.commands.min_qty_risk_report import Command
from risk.models import RiskEvent


class MinQtyRiskReportCommandTest(TestCase):
    def setUp(self):
        self.command = Command()
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
        RiskEvent.objects.create(
            instrument=self.inst,
            kind="min_qty_risk_guard",
            severity=RiskEvent.Severity.WARN,
            details_json={"risk_mult": 8.4},
        )

    @override_settings(
        RISK_PER_TRADE_PCT=0.003,
        PER_INSTRUMENT_RISK={"SOLUSDT": 0.002},
        STOP_LOSS_PCT=0.007,
        MIN_SL_PCT=0.012,
        ATR_MULT_SL=1.5,
        MIN_QTY_RISK_GUARD_ENABLED=True,
        MIN_QTY_RISK_MULTIPLIER_MAX=3.0,
    )
    def test_build_rows_reports_symbol_and_event_count(self):
        rows = self.command._build_rows(days=7, symbol="SOLUSDT")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["symbol"], "SOLUSDT")
        self.assertEqual(row["event_count"], 1)
        self.assertGreater(row["min_notional"], 90.0)
        self.assertGreater(row["risk_multiplier"], 3.0)
        self.assertTrue(row["blocked_now"])
