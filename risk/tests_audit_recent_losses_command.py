from datetime import datetime, timedelta, timezone

from django.test import TestCase
from django.utils import timezone as dj_tz

from core.models import Instrument
from execution.models import OperationReport
from marketdata.models import Candle
from risk.management.commands.audit_recent_losses import build_recent_loss_audit


class AuditRecentLossesCommandTest(TestCase):
    def setUp(self):
        self.inst = Instrument.objects.create(
            symbol="ETHUSDT",
            exchange="bingx",
            base="ETH",
            quote="USDT",
            enabled=True,
        )

    def _seed_post_close_candles(self, *, start: datetime, closes: list[float], highs: list[float], lows: list[float]) -> None:
        for idx, close in enumerate(closes):
            Candle.objects.create(
                instrument=self.inst,
                timeframe="1m",
                ts=start + timedelta(minutes=idx + 1),
                open=close,
                high=highs[idx],
                low=lows[idx],
                close=close,
                volume=100 + idx,
            )

    def test_build_recent_loss_audit_flags_timing_and_bug_candidates(self):
        now = dj_tz.now().replace(microsecond=0)

        op_timing = OperationReport.objects.create(
            instrument=self.inst,
            side="sell",
            qty=1,
            entry_price=100,
            exit_price=101,
            pnl_abs=-1,
            pnl_pct=-0.01,
            notional_usdt=100,
            margin_used_usdt=20,
            fee_usdt=0,
            leverage=5,
            equity_before=1000,
            equity_after=999,
            mode="demo",
            opened_at=now - timedelta(minutes=20),
            closed_at=now,
            outcome=OperationReport.Outcome.LOSS,
            reason="uptrend_short_kill",
            close_sub_reason="",
            signal_id="1",
            correlation_id="1-ETHUSDT",
        )
        self._seed_post_close_candles(
            start=op_timing.closed_at,
            closes=[100.8, 100.4, 99.8],
            highs=[101.1, 100.6, 100.0],
            lows=[100.2, 99.9, 99.4],
        )

        op_bug = OperationReport.objects.create(
            instrument=self.inst,
            side="sell",
            qty=1,
            entry_price=100,
            exit_price=100.3,
            pnl_abs=-0.3,
            pnl_pct=-0.003,
            notional_usdt=100,
            margin_used_usdt=20,
            fee_usdt=0,
            leverage=5,
            equity_before=1000,
            equity_after=999.7,
            mode="demo",
            opened_at=now - timedelta(minutes=90),
            closed_at=now - timedelta(minutes=60),
            outcome=OperationReport.Outcome.LOSS,
            reason="exchange_close",
            close_sub_reason="unknown",
            signal_id="2",
            correlation_id="2-ETHUSDT",
        )
        self._seed_post_close_candles(
            start=op_bug.closed_at,
            closes=[100.35, 100.45, 100.55],
            highs=[100.4, 100.5, 100.6],
            lows=[100.2, 100.3, 100.4],
        )

        report = build_recent_loss_audit(days=5, post_minutes=60, symbol="ETHUSDT")

        self.assertEqual(report["total_losses"], 2)
        categories = {row["id"]: row["category"] for row in report["rows"]}
        self.assertEqual(categories[op_timing.id], "timing_candidate")
        self.assertEqual(categories[op_bug.id], "bug_candidate")

        timing_row = next(row for row in report["rows"] if row["id"] == op_timing.id)
        self.assertTrue(timing_row["recovered_to_entry"])
        self.assertGreater(timing_row["close_after_window_pct"], 0)

        by_category = {row["bucket"]: row for row in report["by_category"]}
        self.assertEqual(by_category["timing_candidate"]["count"], 1)
        self.assertEqual(by_category["bug_candidate"]["count"], 1)
