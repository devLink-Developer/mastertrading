from datetime import timedelta

from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone as dj_tz

from core.models import Instrument
from execution.management.commands.dedupe_operation_reports import build_operation_report_dedupe_plan
from execution.models import OperationReport


class DedupeOperationReportsCommandTest(TestCase):
    def setUp(self):
        self.inst = Instrument.objects.create(
            symbol="ETHUSDT",
            exchange="bingx",
            base="ETH",
            quote="USDT",
        )

    def _mk_op(self, *, reason: str, close_sub_reason: str = "", minutes: int = 0, correlation_id: str = "1-ETHUSDT"):
        now = dj_tz.now().replace(microsecond=0)
        return OperationReport.objects.create(
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
            mode="live",
            opened_at=now - timedelta(minutes=30),
            closed_at=now + timedelta(minutes=minutes),
            outcome=OperationReport.Outcome.LOSS,
            reason=reason,
            close_sub_reason=close_sub_reason,
            signal_id="1",
            correlation_id=correlation_id,
        )

    def test_build_plan_keeps_higher_priority_row(self):
        weak = self._mk_op(reason="exchange_close", close_sub_reason="unknown", minutes=0)
        strong = self._mk_op(reason="sl", minutes=1)

        plan = build_operation_report_dedupe_plan(days=30, mode="live")

        self.assertEqual(plan["duplicate_group_count"], 1)
        group = plan["groups"][0]
        self.assertEqual(group["canonical_id"], strong.id)
        self.assertEqual(group["duplicate_ids"], [weak.id])

    def test_apply_deletes_only_duplicate_rows(self):
        weak = self._mk_op(reason="exchange_close", close_sub_reason="near_breakeven", minutes=0)
        strong = self._mk_op(reason="tp", minutes=1)
        other = self._mk_op(reason="tp", minutes=2, correlation_id="2-ETHUSDT")

        call_command("dedupe_operation_reports", "--days", "30", "--mode", "live", "--apply")

        remaining_ids = list(OperationReport.objects.order_by("id").values_list("id", flat=True))
        self.assertEqual(remaining_ids, [strong.id, other.id])
        self.assertFalse(OperationReport.objects.filter(id=weak.id).exists())