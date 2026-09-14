"""Durable close-leg accounting tests on the isolated Django database."""

from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import TestCase
from django.utils import timezone

from core.models import Instrument
from execution.close_accounting import capture_close, fee_asset_for_context, get_pending_closes, has_partial_close, refresh_close, summarize_position
from execution.models import OperationReport, Order, Position, TradeFill


class CloseAccountingTests(TestCase):
    def setUp(self):
        self.inst = Instrument.objects.create(symbol="ADAUSDT", base="ADA", quote="USDT", exchange="bingx")
        self.opened = timezone.now() - timedelta(hours=1)
        self.runtime = {"risk_namespace": "bingx:live:live:test", "mode": "live", "primary_asset": "USDT"}
        self.runtime_patch = patch("execution.close_accounting.get_runtime_exchange_context", return_value=self.runtime)
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)
        self.context = dict(inst=self.inst, side="buy", qty="12", entry_price="0.1803", reason="tp",
                            signal_id="12", correlation_id="entry-1", leverage="3", equity_before="10",
                            opened_at=self.opened, contract_size="1")
        self.entry = Order.objects.create(instrument=self.inst, side="buy", type="market", qty="12",
            price="0.1803", status="filled", reduce_only=False, correlation_id="entry-1", opened_at=self.opened)

    def response(self, ident="close-1", qty="12", average="0.1804", fee="0.0010824", **changes):
        order = {"id": ident, "symbol": "ADA/USDT:USDT", "side": "sell", "filled": qty,
                 "average": average, "status": "closed", "remaining": "0", "reduceOnly": True,
                 "lastTradeTimestamp": int(timezone.now().timestamp() * 1000),
                 "fee": {"currency": "USDT", "cost": fee}, "info": {"positionSide": "LONG"}}
        order.update(changes)
        return order

    def capture(self, response=None, adapter=None, **changes):
        context = dict(self.context, **changes)
        return capture_close(adapter, order_response=response, **context)

    def test_duplicate_capture_does_not_duplicate_leg_or_fee(self):
        response = self.response()
        order, first = self.capture(response)
        same, second = self.capture(response)
        self.assertEqual(order.pk, same.pk)
        self.assertEqual(Order.objects.filter(reduce_only=True).count(), 1)
        self.assertEqual(second.status, "confirmed")
        self.assertEqual(second.exit_fee, Decimal("0.0010824"))
        self.assertEqual(second.filled_qty, Decimal("12"))
        self.assertEqual(first.position_key, second.position_key)
        self.assertEqual(len(second.position_key), 64)
        self.assertFalse(TradeFill.objects.exists())

    def test_partial_then_final_aggregate_qty_vwap_and_exit_fees_once(self):
        partial, partial_summary = self.capture(self.response("partial", "4", "0.19", "0.001"), qty="4", is_partial=True)
        self.assertEqual(partial_summary.status, "partial")
        final, summary = self.capture(self.response("final", "8", "0.18", "0.002"), qty="8")
        self.assertEqual(summary.status, "confirmed")
        self.assertEqual(summary.filled_qty, Decimal("12"))
        self.assertEqual(summary.average_price, Decimal("2.20") / Decimal("12"))
        self.assertEqual(summary.exit_fee, Decimal("0.003"))
        self.assertEqual(set(summary.leg_order_ids), {partial.pk, final.pk})

    def test_correlation_unifies_database_and_exchange_opening_timestamps(self):
        partial, first = self.capture(self.response("partial", "4"), qty="4", is_partial=True)
        shifted = self.opened + timedelta(seconds=1)
        self.assertTrue(has_partial_close(self.inst, "buy", "entry-1", shifted))
        final, summary = self.capture(self.response("final", "8"), qty="8", opened_at=shifted)
        self.assertEqual(first.position_key, summary.position_key)
        self.assertEqual(summary.status, "confirmed")
        self.assertEqual(summary.filled_qty, Decimal("12"))
        self.assertEqual(set(summary.leg_order_ids), {partial.pk, final.pk})

    def test_correlation_reused_far_from_original_open_is_not_merged(self):
        original, first = self.capture(self.response("original"))
        conflicted, second = self.capture(self.response("later"), opened_at=self.opened + timedelta(hours=8))
        self.assertEqual(second.status, "pending")
        self.assertEqual(second.reason, "correlation_lifecycle_conflict")
        self.assertNotEqual(first.position_key, second.position_key)
        self.assertEqual(summarize_position(original).status, "confirmed")

    def test_without_correlation_opening_timestamp_remains_identity(self):
        _, first = self.capture(self.response("first-no-corr"), correlation_id="")
        _, second = self.capture(self.response("second-no-corr"), correlation_id="", opened_at=self.opened + timedelta(seconds=1))
        self.assertNotEqual(first.position_key, second.position_key)

    def test_scale_in_entry_quantity_is_included_after_root_opening_window(self):
        self.entry.qty = Decimal("5")
        self.entry.save(update_fields=["qty"])
        Order.objects.create(instrument=self.inst, side="buy", type="market", qty="5", price="0.185",
            status="filled", reduce_only=False, correlation_id="entry-child", parent_correlation_id="entry-1",
            opened_at=self.opened + timedelta(minutes=20))
        _, summary = self.capture(self.response("final", "10"), qty="10")
        self.assertEqual(summary.status, "confirmed")
        self.assertEqual(summary.filled_qty, Decimal("10"))

    def test_partial_then_scale_in_then_final_covers_all_entry_quantity(self):
        self.entry.qty = Decimal("10")
        self.entry.save(update_fields=["qty"])
        self.capture(self.response("partial", "5"), qty="5", is_partial=True)
        Order.objects.create(instrument=self.inst, side="buy", type="market", qty="5", price="0.185",
            status="filled", reduce_only=False, correlation_id="entry-child", parent_correlation_id="entry-1",
            opened_at=self.opened + timedelta(minutes=20))
        _, summary = self.capture(self.response("final", "10"), qty="10")
        self.assertEqual(summary.status, "confirmed")
        self.assertEqual(summary.filled_qty, Decimal("15"))

    def test_missing_unique_root_entry_is_pending(self):
        self.entry.delete()
        _, summary = self.capture(self.response())
        self.assertEqual(summary.status, "pending")
        self.assertEqual(summary.reason, "missing_root_entry_quantity")

    def test_duplicate_exchange_entry_order_cannot_inflate_quantity_coverage(self):
        self.entry.qty = Decimal("5")
        self.entry.exchange_order_id = "entry-root"
        self.entry.save(update_fields=["qty", "exchange_order_id"])
        for suffix in ("a", "b"):
            Order.objects.create(instrument=self.inst, side="buy", type="market", qty="5", price="0.185",
                status="filled", reduce_only=False, exchange_order_id="same-child", correlation_id=f"child-{suffix}",
                parent_correlation_id="entry-1", opened_at=self.opened + timedelta(minutes=20))
        _, summary = self.capture(self.response("final", "15"), qty="15")
        self.assertEqual(summary.status, "pending")
        self.assertEqual(summary.reason, "duplicate_entry_order_identity")

    def test_unfilled_ack_is_durable_pending_not_fabricated_fill(self):
        order, summary = self.capture({"id": "pending"})
        self.assertEqual(summary.status, "pending")
        self.assertIsNone(summary.average_price)
        self.assertIsNone(summary.exit_fee)
        self.assertIsNone(order.price)
        self.assertIsNone(order.raw_response["close_accounting"]["evidence"]["filled_qty"])
        self.assertEqual(get_pending_closes(min_retry_seconds=0), [order])

    def test_pending_exact_order_reconciles_after_position_is_cleared(self):
        order, _ = self.capture({"id": "pending"})
        Position.objects.create(instrument=self.inst, qty=0, avg_price="0.1803", side="long", is_open=False)
        fetch = Mock(return_value=self.response("pending"))
        adapter = SimpleNamespace(client=SimpleNamespace(fetch_order=fetch), _map_symbol=lambda symbol: "ADA/USDT:USDT")
        updated, summary = refresh_close(adapter, order)
        self.assertEqual(updated.pk, order.pk)
        self.assertEqual(summary.status, "confirmed")
        self.assertEqual(summary.context["reason"], "tp")
        self.assertEqual(summary.context["opened_at"], self.opened.isoformat())
        fetch.assert_called_once_with("pending", "ADA/USDT:USDT")

    def test_cumulative_partial_updates_replace_instead_of_append(self):
        first, summary = self.capture(self.response("one", "4", fee="0.001", status="canceled"))
        self.assertEqual(summary.status, "pending")
        second, summary = self.capture(self.response("one", "12", fee="0.003"))
        self.assertEqual(first.pk, second.pk)
        self.assertEqual(summary.filled_qty, Decimal("12"))
        self.assertEqual(summary.exit_fee, Decimal("0.003"))

    def test_missing_partial_commission_keeps_whole_position_pending(self):
        self.capture(self.response("partial", "4", fee=None), qty="4", is_partial=True)
        _, summary = self.capture(self.response("final", "8"), qty="8")
        self.assertEqual(summary.status, "pending")
        self.assertIsNone(summary.exit_fee)

    def test_excess_closed_quantity_is_pending(self):
        self.capture(self.response("partial", "4"), qty="4", is_partial=True)
        _, summary = self.capture(self.response("final", "12"))
        self.assertEqual(summary.status, "pending")
        self.assertIn("entry_quantity", summary.reason)

    def test_unrecorded_partial_cannot_be_silently_omitted_from_whole_position(self):
        _, summary = self.capture(self.response("final", "8"), qty="8")
        self.assertEqual(summary.status, "pending")
        self.assertEqual(summary.reason, "close_legs_do_not_cover_entry_quantity")

    def test_missing_actual_fill_timestamp_keeps_report_pending(self):
        _, summary = self.capture(self.response(lastTradeTimestamp=None))
        self.assertEqual(summary.status, "pending")
        self.assertIsNone(summary.closed_at)

    def test_close_before_lifecycle_entry_cannot_be_confirmed(self):
        stamp = int(self.opened.timestamp() * 1000) - 1000
        _, summary = self.capture(self.response(lastTradeTimestamp=stamp))
        self.assertEqual(summary.status, "pending")
        self.assertEqual(summary.reason, "close_leg_timestamp_outside_lifecycle")

    def test_close_materially_after_original_intent_is_pending(self):
        stamp = int((timezone.now() + timedelta(hours=1)).timestamp() * 1000)
        _, summary = self.capture(self.response(lastTradeTimestamp=stamp))
        self.assertEqual(summary.status, "pending")
        self.assertEqual(summary.reason, "close_leg_timestamp_outside_lifecycle")

    def test_partial_after_final_timestamp_cannot_be_confirmed(self):
        stamp = int((timezone.now() + timedelta(seconds=30)).timestamp() * 1000)
        self.capture(self.response("partial", "4", lastTradeTimestamp=stamp), qty="4", is_partial=True)
        _, summary = self.capture(self.response("final", "8"), qty="8")
        self.assertEqual(summary.status, "pending")
        self.assertEqual(summary.reason, "close_leg_timestamp_after_final")

    def test_exchange_timestamp_rounding_at_entry_is_allowed(self):
        stamp = int(self.opened.replace(microsecond=0).timestamp() * 1000)
        _, summary = self.capture(self.response(lastTradeTimestamp=stamp))
        self.assertEqual(summary.status, "confirmed")

    def test_partial_without_fill_timestamp_remains_retryable(self):
        partial, _ = self.capture(self.response("partial", "4", lastTradeTimestamp=None), qty="4", is_partial=True)
        self.assertEqual(get_pending_closes(min_retry_seconds=0), [partial])

    def test_bingx_demo_fee_asset_mapping_is_explicit_and_narrow(self):
        demo = dict(self.runtime, service="bingx", sandbox=True, mode="demo", primary_asset="VST",
                    risk_namespace="bingx:demo:demo:test")
        self.assertEqual(fee_asset_for_context(demo), "USDT")
        for update in ({"service": "other"}, {"sandbox": False}, {"mode": "live"}):
            self.assertEqual(fee_asset_for_context(dict(demo, **update)), "VST")
        with patch("execution.close_accounting.get_runtime_exchange_context", return_value=demo):
            _, summary = self.capture(self.response())
        self.assertEqual(summary.status, "confirmed")
        self.assertEqual(summary.context["fee_asset_mapping"], "bingx_demo_virtual_settlement")
        self.assertEqual(summary.context["account_asset"], "VST")
        self.assertEqual(summary.context["exchange_fee_asset"], "USDT")

    def test_confirmed_report_removes_close_from_pending_dispatch(self):
        order, summary = self.capture(self.response())
        self.assertEqual(get_pending_closes(min_retry_seconds=0), [order])
        OperationReport.objects.create(instrument=self.inst, side="buy", qty="12", entry_price="0.1803",
            exit_price="0.1804", pnl_abs="-0.0001", pnl_pct="-0.001", outcome="loss", closed_at=timezone.now(),
            accounting_key=summary.position_key, accounting_status="confirmed")
        self.assertEqual(get_pending_closes(min_retry_seconds=0), [])

    def test_confirmed_history_cannot_starve_newer_pending_close(self):
        closed_orders, reports = [], []
        for index in range(55):
            key = f"{index + 100:064x}"
            closed_orders.append(Order(instrument=self.inst, side="sell", type="market", qty="12",
                reduce_only=True, status="filled", exchange_order_id=f"historical-{index}",
                correlation_id=f"historical-{index}", raw_response={"close_accounting": {
                    "version": 1, "namespace": self.runtime["risk_namespace"], "position_key": key, "status": "pending"}}))
            reports.append(OperationReport(instrument=self.inst, side="buy", qty="12", entry_price="0.18",
                exit_price="0.19", pnl_abs="0.1", pnl_pct="0.01", outcome="win", closed_at=timezone.now(),
                accounting_key=key, accounting_status="confirmed"))
        Order.objects.bulk_create(closed_orders)
        OperationReport.objects.bulk_create(reports)
        Order.objects.filter(reduce_only=True).update(updated_at=timezone.now() - timedelta(minutes=5))
        pending, _ = self.capture({"id": "actual-pending"})
        self.assertEqual(get_pending_closes(limit=10, min_retry_seconds=0), [pending])

    def test_pending_partial_retry_exposes_final_reporting_context(self):
        partial, _ = self.capture(self.response("partial", "4", fee=None), qty="4", is_partial=True,
                                  reason="partial_close", signal_id="partial-signal")
        final, _ = self.capture(self.response("final", "8"), qty="8", reason="tp", signal_id="final-signal")
        fetch = Mock(return_value=self.response("partial", "4", fee=None))
        adapter = SimpleNamespace(client=SimpleNamespace(fetch_order=fetch), _map_symbol=lambda symbol: "ADA/USDT:USDT")
        _, summary = refresh_close(adapter, partial)
        self.assertEqual(summary.status, "pending")
        self.assertEqual(summary.final_order_id, final.pk)
        self.assertEqual(summary.context["reason"], "tp")
        self.assertEqual(summary.context["signal_id"], "final-signal")
        self.assertEqual(summary.context["requested_qty"], "8")
        self.assertIsNone(summary.exit_fee)

    def test_partial_only_never_finalizes_and_missing_context_is_pending(self):
        _, summary = self.capture(self.response("partial", "4"), qty="4", is_partial=True)
        self.assertEqual(summary.status, "partial")
        _, summary = self.capture(self.response("other"), opened_at=None, correlation_id="")
        self.assertEqual(summary.status, "pending")

    def test_namespace_separates_same_exchange_id_and_pending_retry(self):
        first, _ = self.capture({"id": "same-id"})
        other = dict(self.runtime, risk_namespace="bingx:demo:demo:test", mode="demo", primary_asset="VST")
        with patch("execution.close_accounting.get_runtime_exchange_context", return_value=other):
            second, _ = self.capture({"id": "same-id"})
            self.assertNotEqual(first.pk, second.pk)
            self.assertEqual(get_pending_closes(min_retry_seconds=0), [second])
            fetch = Mock()
            _, summary = refresh_close(SimpleNamespace(client=SimpleNamespace(fetch_order=fetch)), first)
            self.assertEqual(summary.status, "pending")
            self.assertEqual(summary.reason, "runtime_namespace_mismatch")
            fetch.assert_not_called()

    def test_missing_order_id_uses_stable_pending_position_record(self):
        first, _ = self.capture({})
        second, _ = self.capture({})
        self.assertEqual(first.pk, second.pk)
        self.assertEqual(Order.objects.filter(reduce_only=True).count(), 1)

    def test_missing_lifecycle_identity_cannot_merge_unrelated_closes(self):
        first, first_summary = self.capture(self.response("first"), opened_at=None, correlation_id="")
        second, second_summary = self.capture(self.response("second"), opened_at=None, correlation_id="")
        self.assertNotEqual(first.pk, second.pk)
        self.assertNotEqual(first_summary.position_key, second_summary.position_key)
        unidentified1, _ = self.capture({}, opened_at=None, correlation_id="")
        unidentified2, _ = self.capture({}, opened_at=None, correlation_id="")
        self.assertNotEqual(unidentified1.pk, unidentified2.pk)

    def test_missing_ack_id_can_recover_from_strict_original_history_window(self):
        order, _ = self.capture({})
        fetch = Mock(return_value=[self.response("history-id")])
        adapter = SimpleNamespace(client=SimpleNamespace(fetch_closed_orders=fetch), _map_symbol=lambda symbol: "ADA/USDT:USDT")
        updated, summary = refresh_close(adapter, order)
        self.assertEqual(updated.pk, order.pk)
        self.assertEqual(updated.exchange_order_id, "history-id")
        self.assertEqual(summary.status, "confirmed")
        self.assertEqual(fetch.call_args.kwargs["since"], int(self.opened.timestamp() * 1000))

    def test_pending_partial_is_durable_submission_guard(self):
        self.assertFalse(has_partial_close(self.inst, "buy", "entry-1", self.opened))
        self.capture({"id": "pending-partial"}, qty="4", is_partial=True)
        self.assertTrue(has_partial_close(self.inst, "buy", "entry-1", self.opened))
        self.assertFalse(has_partial_close(self.inst, "sell", "entry-1", self.opened))

    def test_retry_throttle_and_degraded_fetch_fairness(self):
        order, _ = self.capture(self.response(fee=None))
        self.assertEqual(get_pending_closes(), [])
        old = timezone.now() - timedelta(seconds=90)
        Order.objects.filter(pk=order.pk).update(updated_at=old)
        self.assertEqual(get_pending_closes(), [order])
        order.refresh_from_db()
        fetch = Mock(return_value={"id": "close-1"})
        adapter = SimpleNamespace(client=SimpleNamespace(fetch_order=fetch), _map_symbol=lambda symbol: "ADA/USDT:USDT")
        refreshed, _ = refresh_close(adapter, order)
        self.assertGreater(refreshed.updated_at, old)
        self.assertEqual(get_pending_closes(), [])
        self.assertEqual(refreshed.raw_response["close_accounting"]["evidence"]["filled_qty"], "12")

    def test_rebate_preserved_in_sum_and_no_entry_fee_added(self):
        self.entry.fee_usdt = Decimal("0.5")
        self.entry.save(update_fields=["fee_usdt"])
        _, summary = self.capture(self.response(fee="-0.0001"))
        self.assertEqual(summary.exit_fee, Decimal("-0.0001"))

    def test_two_distinct_final_orders_fail_closed(self):
        self.capture(self.response("final1"))
        order, summary = self.capture(self.response("final2"))
        self.assertEqual(summary.status, "pending")
        self.assertEqual(summary.reason, "multiple_final_close_legs")
        self.assertEqual(summarize_position(order).status, "pending")
