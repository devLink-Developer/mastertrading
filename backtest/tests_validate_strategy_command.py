from datetime import timedelta
from io import StringIO

from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone

from core.models import Instrument
from execution.models import OperationReport
from signals.models import Signal


class ValidateStrategyCommandTest(TestCase):
    def test_live_reports_load_symbol_and_signal_strategy(self):
        instrument = Instrument.objects.create(
            symbol="BTCUSDT",
            exchange="bingx",
            base="BTC",
            quote="USDT",
            enabled=True,
        )
        now = timezone.now()
        signal = Signal.objects.create(
            instrument=instrument,
            strategy="alloc_long",
            ts=now - timedelta(minutes=90),
            score=0.7,
            payload_json={},
        )
        for idx, pnl_pct in enumerate([0.01, -0.004, 0.006, -0.003, 0.008, -0.002]):
            opened_at = now - timedelta(minutes=120 - idx * 10)
            OperationReport.objects.create(
                instrument=instrument,
                side="buy",
                qty=1,
                entry_price=100,
                exit_price=100 + pnl_pct * 100,
                pnl_abs=pnl_pct,
                pnl_pct=pnl_pct,
                notional_usdt=100,
                margin_used_usdt=20,
                fee_usdt=0,
                leverage=5,
                equity_before=100,
                equity_after=100 + pnl_pct,
                mode="live",
                opened_at=opened_at,
                closed_at=opened_at + timedelta(minutes=5),
                outcome=OperationReport.Outcome.WIN if pnl_pct > 0 else OperationReport.Outcome.LOSS,
                reason="tp" if pnl_pct > 0 else "flat_signal_timeout",
                signal_id=str(signal.id),
            )

        out = StringIO()
        call_command(
            "validate_strategy",
            "--days",
            "30",
            "--folds",
            "2",
            "--embargo-days",
            "0",
            stdout=out,
        )

        text = out.getvalue()
        self.assertIn("Loaded 6 trades", text)
        self.assertIn("PURGED K-FOLD", text)
