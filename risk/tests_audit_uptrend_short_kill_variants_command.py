from datetime import datetime, timedelta, timezone

from django.test import TestCase, override_settings

from core.models import Instrument
from execution.models import OperationReport
from marketdata.models import Candle
from risk.management.commands.audit_uptrend_short_kill_variants import (
    build_uptrend_short_kill_variant_report,
)


class AuditUptrendShortKillVariantsCommandTest(TestCase):
    def setUp(self):
        self.inst = Instrument.objects.create(
            symbol="SOLUSDT",
            exchange="bingx",
            base="SOL",
            quote="USDT",
            enabled=True,
        )

    def _seed_5m_history(self, *, end_ts: datetime, base_price: float) -> None:
        start = end_ts - timedelta(minutes=5 * 20)
        for idx in range(20):
            px = base_price + (idx * 0.05)
            Candle.objects.create(
                instrument=self.inst,
                timeframe="5m",
                ts=start + timedelta(minutes=5 * idx),
                open=px,
                high=px + 0.25,
                low=px - 0.25,
                close=px + 0.02,
                volume=1000 + idx,
            )

    def _seed_1m_post_close(self, *, start_ts: datetime, rows: list[tuple[float, float, float]]) -> None:
        for idx, (high, low, close) in enumerate(rows, start=1):
            Candle.objects.create(
                instrument=self.inst,
                timeframe="1m",
                ts=start_ts + timedelta(minutes=idx),
                open=close,
                high=high,
                low=low,
                close=close,
                volume=100 + idx,
            )

    @override_settings(
        STOP_LOSS_PCT=0.007,
        MIN_SL_PCT=0.012,
        ATR_MULT_SL=1.5,
    )
    def test_variant_report_counts_both_improvement_and_stop_breach(self):
        now = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)

        op_improve = OperationReport.objects.create(
            instrument=self.inst,
            side="sell",
            qty=1,
            entry_price=100,
            exit_price=100.5,
            pnl_abs=-0.5,
            pnl_pct=-0.005,
            notional_usdt=100,
            margin_used_usdt=20,
            fee_usdt=0,
            leverage=5,
            equity_before=1000,
            equity_after=999.5,
            mode="demo",
            opened_at=now - timedelta(minutes=90),
            closed_at=now - timedelta(minutes=60),
            outcome=OperationReport.Outcome.LOSS,
            reason="uptrend_short_kill",
            close_sub_reason="",
            signal_id="1",
            correlation_id="1-SOLUSDT",
        )
        self._seed_5m_history(end_ts=op_improve.opened_at, base_price=99.0)
        self._seed_1m_post_close(
            start_ts=op_improve.closed_at,
            rows=[
                (100.55, 100.1, 100.2),
                (100.25, 99.9, 99.95),
                (100.0, 99.6, 99.7),
                (99.85, 99.4, 99.5),
            ],
        )

        op_breach = OperationReport.objects.create(
            instrument=self.inst,
            side="sell",
            qty=1,
            entry_price=100,
            exit_price=100.4,
            pnl_abs=-0.4,
            pnl_pct=-0.004,
            notional_usdt=100,
            margin_used_usdt=20,
            fee_usdt=0,
            leverage=5,
            equity_before=1000,
            equity_after=999.6,
            mode="demo",
            opened_at=now + timedelta(minutes=180),
            closed_at=now + timedelta(minutes=205),
            outcome=OperationReport.Outcome.LOSS,
            reason="uptrend_short_kill",
            close_sub_reason="",
            signal_id="2",
            correlation_id="2-SOLUSDT",
        )
        self._seed_5m_history(end_ts=op_breach.opened_at, base_price=99.5)
        self._seed_1m_post_close(
            start_ts=op_breach.closed_at,
            rows=[
                (101.35, 100.2, 101.0),
                (101.1, 100.0, 100.3),
                (100.6, 99.7, 99.8),
                (100.2, 99.5, 99.6),
            ],
        )

        report = build_uptrend_short_kill_variant_report(days=5, grace_windows=[15])

        self.assertEqual(report["trades"], 2)
        summary = report["summary_by_grace"][0]
        self.assertEqual(summary["grace_minutes"], 15)
        self.assertEqual(summary["improved"], 1)
        self.assertEqual(summary["worsened"], 1)
        self.assertEqual(summary["stop_breaches"], 1)

        rows_by_id = {row["id"]: row for row in report["rows"]}
        improve_variant = rows_by_id[op_improve.id]["variants"][0]
        breach_variant = rows_by_id[op_breach.id]["variants"][0]
        self.assertEqual(improve_variant["status"], "window_close")
        self.assertGreater(improve_variant["alt_pnl_abs"], rows_by_id[op_improve.id]["actual_pnl_abs"])
        self.assertEqual(breach_variant["status"], "stop_breach")
        self.assertLess(breach_variant["alt_pnl_abs"], rows_by_id[op_breach.id]["actual_pnl_abs"])
