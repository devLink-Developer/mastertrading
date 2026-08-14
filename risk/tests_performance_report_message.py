from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

from django.test import TestCase

from core.models import Instrument
from execution.models import BalanceSnapshot, OperationReport, Position
from risk.tasks import _build_performance_report
from signals.models import Signal


class PerformanceReportMessageTest(TestCase):
    def setUp(self):
        self.now = datetime(2026, 8, 14, 19, 0, tzinfo=timezone.utc)
        self.inst = Instrument.objects.create(
            symbol="LINKUSDT",
            exchange="bingx",
            base="LINK",
            quote="USDT",
        )

    def _operation(self, pnl_abs: str, outcome: str) -> None:
        OperationReport.objects.create(
            instrument=self.inst,
            side="sell",
            qty=Decimal("1"),
            entry_price=Decimal("10"),
            exit_price=Decimal("9"),
            pnl_abs=Decimal(pnl_abs),
            pnl_pct=Decimal("0.01") if Decimal(pnl_abs) > 0 else Decimal("-0.01"),
            outcome=outcome,
            reason="tp" if outcome == OperationReport.Outcome.WIN else "sl",
            mode="live",
            closed_at=self.now - timedelta(minutes=10),
        )

    @patch("risk.tasks.get_runtime_exchange_context")
    @patch("risk.tasks.dj_tz.now")
    def test_report_prioritizes_account_results_and_plain_language(self, now_mock, runtime_mock):
        now_mock.return_value = self.now
        runtime_mock.return_value = {
            "service": "bingx",
            "sandbox": False,
            "primary_asset": "USDT",
            "account_alias": "eudy",
        }

        baseline = BalanceSnapshot.objects.create(
            equity_usdt=Decimal("10.00"),
            free_usdt=Decimal("10.00"),
            eff_leverage=Decimal("0"),
        )
        latest = BalanceSnapshot.objects.create(
            equity_usdt=Decimal("10.20"),
            free_usdt=Decimal("8.20"),
            eff_leverage=Decimal("0.50"),
        )
        BalanceSnapshot.objects.filter(pk=baseline.pk).update(
            created_at=self.now - timedelta(minutes=61)
        )
        BalanceSnapshot.objects.filter(pk=latest.pk).update(
            created_at=self.now - timedelta(minutes=1)
        )

        self._operation("0.30", OperationReport.Outcome.WIN)
        self._operation("-0.10", OperationReport.Outcome.LOSS)
        Position.objects.create(
            instrument=self.inst,
            qty=Decimal("2"),
            avg_price=Decimal("10"),
            unrealized_pnl=Decimal("0.05"),
            pnl_pct=Decimal("0.0025"),
            side=Position.Side.SHORT,
            is_open=True,
            mode="live",
        )
        for strategy in ("alloc_long", "alloc_short", "alloc_flat"):
            Signal.objects.create(
                strategy=strategy,
                instrument=self.inst,
                ts=self.now - timedelta(minutes=5),
                score=0.5,
                payload_json={},
            )

        message = _build_performance_report(window_minutes=60)

        self.assertIn("Resumen Eudy \u00B7 \u00FAltimos 60 min", message)
        self.assertIn("Variaci\u00F3n de equity: +0.2000 USDT (+2.00%)", message)
        self.assertIn("Operaciones cerradas: 2 (1 ganadas \u00B7 1 perdidas", message)
        self.assertIn("Acierto: 50.0%", message)
        self.assertIn("Posiciones abiertas (1)", message)
        self.assertIn("LINKUSDT \u00B7 SHORT", message)
        self.assertIn("Decisiones: 1 compra \u00B7 1 venta \u00B7 1 sin entrada", message)
        self.assertNotIn("mod trend=", message)
        self.assertNotIn("alloc long=", message)
