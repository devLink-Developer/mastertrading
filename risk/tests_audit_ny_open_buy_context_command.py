from datetime import datetime, timedelta, timezone

from django.test import TestCase

from core.models import Instrument
from execution.models import OperationReport
from risk.management.commands.audit_ny_open_buy_context import (
    build_ny_open_buy_context_audit,
)


class AuditNyOpenBuyContextCommandTest(TestCase):
    def setUp(self):
        self.inst = Instrument.objects.create(
            symbol="ETHUSDT",
            exchange="bingx",
            base="ETH",
            quote="USDT",
            enabled=True,
        )

    def _op(
        self,
        *,
        opened_at: datetime,
        pnl_abs: float,
        side: str = "buy",
        recommended_bias: str = "balanced",
        btc_lead_state: str = "transition",
        monthly_regime: str = "bear_confirmed",
        weekly_regime: str = "transition",
        daily_regime: str = "transition",
    ) -> OperationReport:
        outcome = (
            OperationReport.Outcome.WIN
            if pnl_abs > 0
            else OperationReport.Outcome.LOSS
            if pnl_abs < 0
            else OperationReport.Outcome.BE
        )
        return OperationReport.objects.create(
            instrument=self.inst,
            side=side,
            qty=1,
            entry_price=100,
            exit_price=101 if pnl_abs >= 0 else 99,
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
            monthly_regime=monthly_regime,
            weekly_regime=weekly_regime,
            daily_regime=daily_regime,
            btc_lead_state=btc_lead_state,
            recommended_bias=recommended_bias,
        )

    def test_audit_reports_expected_variant_deltas(self):
        base_dt = datetime(2026, 3, 21, 13, 40, tzinfo=timezone.utc)  # ny_open

        self._op(opened_at=base_dt, pnl_abs=-2.0, recommended_bias="balanced", btc_lead_state="transition")
        self._op(
            opened_at=base_dt + timedelta(minutes=2),
            pnl_abs=-1.0,
            recommended_bias="balanced",
            btc_lead_state="bull_weak",
        )
        self._op(
            opened_at=base_dt + timedelta(minutes=4),
            pnl_abs=1.5,
            recommended_bias="tactical_long",
            btc_lead_state="transition",
        )
        self._op(
            opened_at=datetime(2026, 3, 21, 14, 10, tzinfo=timezone.utc),  # ny
            pnl_abs=3.0,
            recommended_bias="balanced",
            btc_lead_state="transition",
        )

        report = build_ny_open_buy_context_audit(days=5, symbol="ETHUSDT")

        self.assertEqual(report["baseline"]["trade_count"], 4)
        self.assertEqual(report["baseline"]["total_pnl"], 1.5)

        variants = {row["name"]: row for row in report["variants"]}

        exact = variants["block_ny_open_buy_balanced_transition"]
        self.assertEqual(exact["removed"]["trade_count"], 1)
        self.assertEqual(exact["removed"]["total_pnl"], -2.0)
        self.assertEqual(exact["after_block"]["total_pnl"], 3.5)
        self.assertEqual(exact["delta_total_pnl"], 2.0)

        balanced = variants["block_ny_open_buy_balanced"]
        self.assertEqual(balanced["removed"]["trade_count"], 2)
        self.assertEqual(balanced["removed"]["total_pnl"], -3.0)
        self.assertEqual(balanced["after_block"]["total_pnl"], 4.5)
        self.assertEqual(balanced["delta_total_pnl"], 3.0)

        weak = variants["block_ny_open_buy_weak_long_context"]
        self.assertEqual(weak["removed"]["trade_count"], 2)
        self.assertEqual(weak["removed"]["total_pnl"], -0.5)
        self.assertEqual(weak["after_block"]["total_pnl"], 2.0)
        self.assertEqual(weak["delta_total_pnl"], 0.5)
