from datetime import datetime, timedelta, timezone

from django.test import TestCase, override_settings

from core.models import Instrument
from execution.models import OperationReport
from marketdata.models import Candle
from risk.management.commands.audit_session_structure_gate import (
    build_session_structure_audit,
)


class AuditSessionStructureGateCommandTest(TestCase):
    def setUp(self):
        self.inst = Instrument.objects.create(
            symbol="ETHUSDT",
            exchange="bingx",
            base="ETH",
            quote="USDT",
            enabled=True,
        )

    def _seed_5m_history(self, end_ts: datetime) -> None:
        start = end_ts - timedelta(minutes=5 * 20)
        price = 99.0
        for idx in range(20):
            Candle.objects.create(
                instrument=self.inst,
                timeframe="5m",
                ts=start + timedelta(minutes=5 * idx),
                open=price,
                high=price + 0.4,
                low=price - 0.4,
                close=price + 0.1,
                volume=1000 + idx,
            )
            price += 0.05

    def _seed_previous_day(self, day_start: datetime) -> None:
        Candle.objects.create(
            instrument=self.inst,
            timeframe="1d",
            ts=day_start - timedelta(days=1),
            open=96.0,
            high=100.4,
            low=94.0,
            close=99.0,
            volume=10000,
        )

    def _seed_session_1m(self, opened_at: datetime) -> None:
        start = opened_at.replace(hour=13, minute=30, second=0, microsecond=0)
        rows = [
            (99.10, 99.00, 99.05),
            (99.20, 99.05, 99.15),
            (99.35, 99.10, 99.30),
            (99.55, 99.25, 99.50),
            (99.70, 99.45, 99.65),
            (99.82, 99.60, 99.78),
            (99.90, 99.70, 99.88),
            (99.96, 99.80, 99.92),
            (100.00, 99.85, 99.95),
            (100.02, 99.90, 99.98),
            (100.03, 99.92, 99.97),
        ]
        for idx, (high, low, close) in enumerate(rows):
            Candle.objects.create(
                instrument=self.inst,
                timeframe="1m",
                ts=start + timedelta(minutes=idx),
                open=close,
                high=high,
                low=low,
                close=close,
                volume=100 + idx,
            )

    def _op(self, *, opened_at: datetime, pnl_abs: float, entry_price: float) -> None:
        outcome = (
            OperationReport.Outcome.WIN
            if pnl_abs > 0
            else OperationReport.Outcome.LOSS
            if pnl_abs < 0
            else OperationReport.Outcome.BE
        )
        OperationReport.objects.create(
            instrument=self.inst,
            side="buy",
            qty=1,
            entry_price=entry_price,
            exit_price=entry_price + (0.8 if pnl_abs > 0 else -0.8),
            pnl_abs=pnl_abs,
            pnl_pct=pnl_abs / 100.0,
            notional_usdt=100,
            margin_used_usdt=20,
            fee_usdt=0,
            leverage=5,
            equity_before=1000,
            equity_after=1000 + pnl_abs,
            mode="demo",
            opened_at=opened_at,
            closed_at=opened_at + timedelta(minutes=30),
            outcome=outcome,
            reason="tp" if pnl_abs > 0 else "sl",
            close_sub_reason="",
            signal_id=f"sig-{opened_at.isoformat()}",
            correlation_id=f"corr-{opened_at.isoformat()}",
            monthly_regime="bear_confirmed",
            weekly_regime="transition",
            daily_regime="transition",
            btc_lead_state="transition",
            recommended_bias="balanced",
        )

    @override_settings(
        TAKE_PROFIT_PCT=0.008,
        STOP_LOSS_PCT=0.007,
        ATR_MULT_TP=1.6,
        ATR_MULT_SL=1.5,
        MIN_SL_PCT=0.012,
    )
    def test_audit_identifies_chasing_top_variants(self):
        opened_at = datetime(2026, 3, 21, 13, 40, tzinfo=timezone.utc)
        day_start = opened_at.replace(hour=0, minute=0, second=0, microsecond=0)
        self._seed_previous_day(day_start)
        self._seed_5m_history(opened_at)
        self._seed_session_1m(opened_at)

        self._op(opened_at=opened_at, pnl_abs=-1.0, entry_price=99.95)
        self._op(opened_at=opened_at + timedelta(minutes=1), pnl_abs=2.0, entry_price=99.20)

        report = build_session_structure_audit(
            days=5,
            symbol="ETHUSDT",
            sessions={"ny_open"},
            side="buy",
        )

        self.assertEqual(report["baseline"]["trade_count"], 2)
        self.assertEqual(report["baseline"]["total_pnl"], 1.0)

        variants = {row["name"]: row for row in report["variants"]}

        chase = variants["block_chasing_top_needing_extension"]
        self.assertEqual(chase["removed"]["trade_count"], 1)
        self.assertEqual(chase["removed"]["total_pnl"], -1.0)
        self.assertEqual(chase["after_block"]["total_pnl"], 2.0)
        self.assertEqual(chase["delta_total_pnl"], 1.0)

        barrier = variants["block_chasing_top_with_prevday_barrier"]
        self.assertEqual(barrier["removed"]["trade_count"], 1)
        self.assertEqual(barrier["removed"]["total_pnl"], -1.0)
        self.assertEqual(barrier["after_block"]["total_pnl"], 2.0)

        extreme = variants["block_extreme_chase"]
        self.assertEqual(extreme["removed"]["trade_count"], 1)
        row = extreme["affected_rows"][0]
        self.assertTrue(row["prev_day_barrier_between_entry_and_tp"])
        self.assertGreaterEqual(float(row["session_progress"]), 0.85)
