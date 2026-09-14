"""Integration contracts for persisted execution evidence, reports and notices."""
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import TestCase, override_settings
from django.utils import timezone

from core.models import Instrument
from execution.close_accounting import capture_close, refresh_close
from execution.models import OperationReport, Order, Position
from execution.tasks import (
    _check_trailing_stop, _log_operation, _manage_open_position, _notify_accounted_close,
    _report_pnl_for_notification, _sync_positions,
)


@override_settings(MODE="live", ML_ENTRY_FILTER_RETRAIN_ON_OPERATION_ENABLED=True)
class ExecutionAccountingIntegrationTests(TestCase):
    def setUp(self):
        self.inst = Instrument.objects.create(symbol="BTCUSDT", base="BTC", quote="USDT", exchange="bingx")
        self.opened = (timezone.now() - timedelta(hours=1)).replace(microsecond=0)
        self.filled = (timezone.now() - timedelta(minutes=1)).replace(microsecond=0)
        self.runtime = {"service": "bingx", "sandbox": False, "mode": "live",
                        "primary_asset": "USDT", "risk_namespace": "bingx:live:live:test"}
        for target, value in (("execution.tasks.get_runtime_exchange_context", self.runtime),
                              ("execution.close_accounting.get_runtime_exchange_context", self.runtime),
                              ("execution.tasks._redis_client", None),
                              ("execution.tasks._operation_regime_snapshot", {})):
            active = patch(target, return_value=value)
            active.start()
            self.addCleanup(active.stop)
        queue_patch = patch("execution.tasks._queue_ml_retrain_after_operation")
        self.queue = queue_patch.start()
        self.addCleanup(queue_patch.stop)
        self.responses = {}
        fetch = Mock(side_effect=lambda ident, _symbol: self.responses.get(str(ident)))
        self.adapter = SimpleNamespace(client=SimpleNamespace(fetch_order=fetch),
                                       _map_symbol=lambda symbol: "BTC/USDT:USDT")
        self.entry_response = self.response("entry-1", side="buy", price="100", fee="1", reduce_only=False,
                                            filled_at=self.opened)
        self.responses["entry-1"] = self.entry_response
        self.entry = Order.objects.create(
            instrument=self.inst, exchange_order_id="entry-1", side="buy", type="market", qty="10",
            price="100", fee_usdt="99", status="filled", reduce_only=False,
            correlation_id="position-1", opened_at=self.opened, closed_at=self.opened,
            raw_response=self.entry_response,
        )

    def response(self, ident="close-1", qty="10", price="101", fee="0.5", side="sell",
                 reduce_only=True, filled_at=None, **updates):
        stamp = filled_at or self.filled
        value = {"id": ident, "symbol": "BTC/USDT:USDT", "side": side,
                 "filled": qty, "average": price, "remaining": "0", "status": "closed",
                 "reduceOnly": reduce_only, "lastTradeTimestamp": int(stamp.timestamp() * 1000),
                 "fee": {"currency": "USDT", "cost": fee}, "info": {"positionSide": "LONG"}}
        value.update(updates)
        return value

    def log(self, response=None, execute_commit_callbacks=True, **updates):
        values = dict(inst=self.inst, side="buy", qty=10, entry_price=100,
                      exit_price=140, reason="tp", signal_id="1", correlation_id="position-1",
                      leverage=5, equity_before=1000, fee_usdt=999,
                      opened_at=self.opened, contract_size=1, adapter=self.adapter,
                      close_response=response)
        values.update(updates)
        with self.captureOnCommitCallbacks(execute=execute_commit_callbacks):
            report = _log_operation(**values)
        report.refresh_from_db()
        return report

    def test_confirmed_fill_replaces_ticker_and_estimated_fees(self):
        report = self.log(self.response(), entry_price=105)
        self.assertEqual(report.accounting_status, OperationReport.AccountingStatus.CONFIRMED)
        self.assertEqual(report.entry_price, Decimal("100"))
        self.assertEqual(report.exit_price, Decimal("101"))
        self.assertEqual(report.qty, Decimal("10"))
        self.assertEqual(report.fee_usdt, Decimal("1.5"))
        self.assertEqual(report.pnl_abs, Decimal("8.5"))
        self.assertEqual(report.pnl_pct, Decimal("0.008500"))
        self.assertEqual(report.outcome, OperationReport.Outcome.WIN)
        self.assertFalse(report.accounting_details["funding_included"])
        self.assertEqual(Order.objects.filter(reduce_only=True).count(), 1)
        self.queue.assert_called_once_with("BTCUSDT", "live", "tp")

    def test_ack_pending_keeps_unknowns_null_and_does_not_train(self):
        report = self.log({"id": "close-1"})
        self.assertEqual(report.accounting_status, OperationReport.AccountingStatus.PENDING)
        self.assertEqual(report.outcome, OperationReport.Outcome.PENDING)
        for name in ("exit_price", "pnl_abs", "pnl_pct", "fee_usdt"):
            self.assertIsNone(getattr(report, name), name)
        self.assertFalse(OperationReport.objects.with_accounted_pnl().exists())
        self.assertEqual(Order.objects.filter(reduce_only=True, exchange_order_id="close-1").count(), 1)
        self.queue.assert_not_called()
        self.assertEqual(_report_pnl_for_notification(report, 99, 999), (None, None))

    def test_missing_entry_fee_cannot_be_replaced_by_stored_estimate(self):
        self.entry_response.pop("fee")
        self.entry.raw_response = self.entry_response
        self.entry.save(update_fields=["raw_response"])
        report = self.log(self.response())
        self.assertEqual(report.accounting_status, OperationReport.AccountingStatus.PENDING)
        self.assertIsNone(report.pnl_abs)
        self.assertIsNone(report.fee_usdt)
        self.queue.assert_not_called()

    def test_stored_entry_fill_cannot_confirm_a_different_current_account(self):
        # The DB row is complete, but the active account cannot retrieve that ID.
        self.responses.pop("entry-1")
        report = self.log(self.response())
        self.assertEqual(report.accounting_status, "pending")
        self.assertIsNone(report.pnl_abs)
        self.assertIsNone(report.fee_usdt)
        self.assertEqual(report.accounting_details["entry_evidence"]["verification"],
                         "exact_order_fetch_current_account")
        self.queue.assert_not_called()

    def test_exact_entry_fill_after_final_exit_cannot_confirm_lifecycle(self):
        self.responses["entry-1"] = self.response(
            "entry-1", side="buy", price="100", fee="1", reduce_only=False,
            filled_at=self.filled + timedelta(seconds=60),
        )
        report = self.log(self.response())
        self.assertEqual(report.accounting_status, "pending")
        self.assertIsNone(report.pnl_abs)
        self.assertIsNone(report.fee_usdt)
        self.assertEqual(report.accounting_details["entry_evidence"]["reason"],
                         "entry_fill_outside_position_lifecycle")
        self.queue.assert_not_called()

    def test_exact_entry_fill_without_timestamp_cannot_confirm_lifecycle(self):
        self.responses["entry-1"] = dict(self.entry_response)
        self.responses["entry-1"].pop("lastTradeTimestamp")
        report = self.log(self.response())
        self.assertEqual(report.accounting_status, "pending")
        self.assertIsNone(report.pnl_abs)
        self.assertIsNone(report.fee_usdt)
        self.assertEqual(report.accounting_details["entry_evidence"]["reason"],
                         "missing_entry_fill_timestamp")
        self.queue.assert_not_called()

    def test_partial_and_final_legs_use_total_quantity_vwap_and_entry_fee_once(self):
        partial, _ = capture_close(
            self.adapter, inst=self.inst, side="buy", qty=5, entry_price=100, reason="partial_close",
            signal_id="1", correlation_id="position-1", leverage=5, equity_before=1000,
            opened_at=self.opened, order_response=self.response("partial", qty="5", price="110"),
            is_partial=True,
        )
        report = self.log(self.response("final", qty="5", price="90"), qty=5, reason="sl")
        self.assertEqual(report.accounting_status, OperationReport.AccountingStatus.CONFIRMED)
        self.assertEqual(report.qty, Decimal("10"))
        self.assertEqual(report.exit_price, Decimal("100"))
        self.assertEqual(report.fee_usdt, Decimal("2"))
        self.assertEqual(report.pnl_abs, Decimal("-2"))
        self.assertEqual(report.outcome, OperationReport.Outcome.LOSS)
        self.assertEqual(len(report.accounting_details["leg_order_ids"]), 2)
        self.assertIn(partial.pk, report.accounting_details["leg_order_ids"])
        self.assertEqual(Order.objects.filter(reduce_only=True).count(), 2)

    def test_repeated_confirmed_capture_has_one_report_order_and_training_event(self):
        response = self.response()
        first = self.log(response)
        second = self.log(response)
        self.assertEqual(first.pk, second.pk)
        self.assertEqual(first.pnl_abs, second.pnl_abs)
        self.assertEqual(first.accounting_key, second.accounting_key)
        self.assertEqual(OperationReport.objects.count(), 1)
        self.assertEqual(Order.objects.filter(reduce_only=True).count(), 1)
        self.queue.assert_called_once()

    def test_pending_retry_confirms_lower_priority_evidence_and_preserves_best_reason(self):
        pending = self.log({"id": "close-1"}, reason="tp")
        close = Order.objects.get(reduce_only=True)
        self.responses["close-1"] = self.response()
        close, summary = refresh_close(self.adapter, close)
        self.assertEqual(summary.status, "confirmed")
        report = self.log(close_record=close, reason="exchange_close", close_sub_reason="unknown")
        repeated = self.log(close_record=close, reason="exchange_close", close_sub_reason="unknown")
        self.assertEqual(pending.pk, report.pk)
        self.assertEqual(report.pk, repeated.pk)
        self.assertEqual(report.accounting_status, OperationReport.AccountingStatus.CONFIRMED)
        self.assertEqual(report.reason, "tp")
        self.assertEqual(report.close_sub_reason, "")
        self.assertEqual(report.pnl_abs, Decimal("8.5"))
        self.queue.assert_called_once()

    @override_settings(MODE="demo")
    def test_bingx_demo_usdt_commission_maps_only_to_virtual_vst_context(self):
        self.runtime.update(sandbox=True, mode="demo", primary_asset="VST",
                            risk_namespace="bingx:demo:demo:test")
        report = self.log(self.response())
        self.assertEqual(report.accounting_status, OperationReport.AccountingStatus.CONFIRMED)
        self.assertEqual(report.mode, "demo")
        self.assertEqual(report.fee_usdt, Decimal("1.5"))
        close = Order.objects.get(reduce_only=True)
        context = close.raw_response["close_accounting"]["context"]
        self.assertEqual(context["fee_asset_mapping"], "bingx_demo_virtual_settlement")
        self.assertEqual(context["account_asset"], "VST")

    @override_settings(MODE="demo")
    def test_different_exchange_cannot_reuse_virtual_fee_mapping(self):
        self.runtime.update(service="other", sandbox=True, mode="demo", primary_asset="VST",
                            risk_namespace="other:demo:demo:test")
        report = self.log(self.response())
        self.assertEqual(report.accounting_status, OperationReport.AccountingStatus.PENDING)
        self.assertIsNone(report.pnl_abs)

    @patch("execution.tasks.notify_trade_closed")
    def test_notification_uses_saved_confirmed_net_not_caller_ticker_or_pnl(self, notify):
        report = self.log(self.response())
        _notify_accounted_close(report, "BTCUSDT", "tp", 9, pnl_abs=999,
                                exit_price=140, entry_price=105, qty=5, side="buy")
        args, kwargs = notify.call_args
        self.assertEqual(args[2], .0085)
        self.assertEqual(kwargs["pnl_abs"], 8.5)
        self.assertEqual(kwargs["exit_price"], 101)
        self.assertEqual(kwargs["qty"], 10)
        self.assertFalse(kwargs["accounting_pending"])

    @patch("execution.tasks.notify_trade_closed")
    def test_pending_notification_suppresses_all_unverified_numeric_fallbacks(self, notify):
        report = self.log({"id": "close-1"})
        _notify_accounted_close(report, "BTCUSDT", "tp", 9, pnl_abs=999,
                                exit_price=140, entry_price=100, qty=10, side="buy")
        args, kwargs = notify.call_args
        self.assertIsNone(args[2])
        self.assertIsNone(kwargs["pnl_abs"])
        self.assertIsNone(kwargs["exit_price"])
        self.assertTrue(kwargs["accounting_pending"])

    @patch("execution.tasks.notify_trade_closed")
    def test_missing_report_cannot_notify_unverified_exit_as_real_profit(self, notify):
        _notify_accounted_close(None, "BTCUSDT", "tp", 9, pnl_abs=999,
                                exit_price=140, entry_price=100, qty=10, side="buy")
        args, kwargs = notify.call_args
        self.assertTrue(kwargs.get("accounting_pending"))
        self.assertIsNone(args[2])
        self.assertIsNone(kwargs.get("pnl_abs"))
        self.assertIsNone(kwargs.get("exit_price"))

    def test_confirmed_report_uses_actual_exchange_fill_timestamp(self):
        report = self.log(self.response())
        self.assertEqual(report.closed_at, self.filled)

    def test_confirmed_training_is_enqueued_only_after_commit(self):
        with self.captureOnCommitCallbacks(execute=False) as callbacks:
            report = self.log(self.response(), execute_commit_callbacks=False)
            self.queue.assert_not_called()
        self.assertEqual(len(callbacks), 1)
        callbacks[0]()
        self.queue.assert_called_once_with("BTCUSDT", "live", "tp")
        self.assertEqual(OperationReport.objects.get(pk=report.pk).accounting_status, "confirmed")

    def test_durable_close_survives_failed_report_creation_and_can_be_retried(self):
        with patch("execution.tasks.OperationReport.objects.create", side_effect=RuntimeError("report write failed")):
            with self.assertRaisesRegex(RuntimeError, "report write failed"):
                self.log(self.response())
        self.assertFalse(OperationReport.objects.exists())
        close = Order.objects.get(reduce_only=True, exchange_order_id="close-1")
        self.assertEqual(close.price, Decimal("101"))
        self.assertEqual(close.qty, Decimal("10"))
        self.assertTrue(close.raw_response["close_accounting"]["evidence"]["fee_known"])
        self.queue.assert_not_called()
        report = self.log(close_record=close)
        self.assertEqual(report.accounting_status, "confirmed")
        self.assertEqual(report.pnl_abs, Decimal("8.5"))
        self.assertEqual(OperationReport.objects.count(), 1)
        self.assertEqual(Order.objects.filter(reduce_only=True).count(), 1)
        self.queue.assert_called_once()

    def test_refreshed_fill_survives_failed_pending_report_update(self):
        pending = self.log({"id": "close-1"})
        close = Order.objects.get(reduce_only=True)
        self.responses["close-1"] = self.response()
        close, summary = refresh_close(self.adapter, close)
        self.assertEqual(summary.status, "confirmed")
        with patch("execution.tasks.OperationReport.save", side_effect=RuntimeError("report update failed")):
            with self.assertRaisesRegex(RuntimeError, "report update failed"):
                self.log(close_record=close)
        pending.refresh_from_db()
        close.refresh_from_db()
        self.assertEqual(pending.accounting_status, "pending")
        self.assertIsNone(pending.pnl_abs)
        self.assertEqual(close.price, Decimal("101"))
        self.assertTrue(close.raw_response["close_accounting"]["evidence"]["fee_known"])
        self.queue.assert_not_called()
        confirmed = self.log(close_record=close)
        self.assertEqual(confirmed.pk, pending.pk)
        self.assertEqual(confirmed.accounting_status, "confirmed")
        self.assertEqual(Order.objects.filter(reduce_only=True).count(), 1)
        self.queue.assert_called_once()

    def configure_sync(self, history):
        """Use exchange-shaped responses while exercising the actual sync/book path."""
        self.history = history
        self.adapter.client.fetch_closed_orders = Mock(side_effect=lambda *a, **kw: self.history)
        self.adapter.client.market = Mock(return_value={"contractSize": 1})
        self.adapter.fetch_closed_orders = Mock(return_value=[])
        self.adapter.fetch_balance = Mock(return_value={"USDT": {"free": 1000, "total": 1000}})
        for target, value in (("execution.tasks._cleanup_orphan_reduce_only_orders", None),
                              ("execution.tasks._atr_pct", None),
                              ("execution.tasks._load_protective_stop_price", None)):
            active = patch(target, return_value=value)
            active.start()
            self.addCleanup(active.stop)

    def open_position(self, opened_at=None):
        return Position.objects.create(
            instrument=self.inst, qty="10", avg_price="100", last_price="140",
            pnl_pct="0.4", unrealized_pnl="400", notional_usdt="1400",
            margin_used_usdt="280", leverage_eff="5", side="long", mode="live",
            is_open=True, opened_at=opened_at or self.opened,
        )

    @patch("execution.tasks.notify_trade_closed")
    def test_sync_vanished_position_uses_execution_fill_not_last_mark(self, notify):
        self.configure_sync([self.response()])
        position = self.open_position(opened_at=self.opened + timedelta(seconds=1))
        with self.captureOnCommitCallbacks(execute=True):
            _sync_positions(self.adapter, positions=[])
        report = OperationReport.objects.get()
        position.refresh_from_db()
        self.assertFalse(position.is_open)
        self.assertEqual(position.qty, Decimal("0"))
        self.assertEqual(report.accounting_status, "confirmed")
        self.assertEqual(report.opened_at, self.opened)
        self.assertEqual(report.closed_at, self.filled)
        self.assertEqual(report.exit_price, Decimal("101"))
        self.assertEqual(report.fee_usdt, Decimal("1.5"))
        self.assertEqual(report.pnl_abs, Decimal("8.5"))
        self.assertEqual(notify.call_args.kwargs["exit_price"], 101)
        self.assertEqual(notify.call_args.kwargs["pnl_abs"], 8.5)
        self.queue.assert_called_once()

    @patch("execution.tasks.notify_trade_closed")
    def test_sync_retries_pending_history_after_position_is_already_closed(self, notify):
        self.configure_sync([])
        position = self.open_position()
        with self.captureOnCommitCallbacks(execute=True):
            _sync_positions(self.adapter, positions=[])
        pending = OperationReport.objects.get()
        position.refresh_from_db()
        self.assertFalse(position.is_open)
        self.assertEqual(pending.accounting_status, "pending")
        self.assertIsNone(pending.exit_price)
        self.assertIsNone(pending.pnl_abs)
        self.assertIsNone(pending.fee_usdt)
        self.assertTrue(notify.call_args.kwargs["accounting_pending"])
        self.assertIsNone(notify.call_args.kwargs["pnl_abs"])
        self.queue.assert_not_called()

        # Advance only the retry eligibility clock; retain the original observed
        # close time so the later exchange fill still has to match that lifecycle.
        close = Order.objects.get(reduce_only=True)
        Order.objects.filter(pk=close.pk).update(updated_at=timezone.now() - timedelta(seconds=61))
        self.history = [self.response()]
        with self.captureOnCommitCallbacks(execute=True):
            _sync_positions(self.adapter, positions=[])
            _sync_positions(self.adapter, positions=[])
        pending.refresh_from_db()
        self.assertEqual(pending.accounting_status, "confirmed")
        self.assertEqual(pending.pnl_abs, Decimal("8.5"))
        self.assertEqual(pending.exit_price, Decimal("101"))
        self.assertEqual(OperationReport.objects.count(), 1)
        self.assertEqual(Order.objects.filter(reduce_only=True).count(), 1)
        self.assertEqual(notify.call_count, 2)  # One pending notice, one reconciliation.
        self.assertFalse(notify.call_args.kwargs["accounting_pending"])
        self.queue.assert_called_once()

    @patch("execution.tasks.notify_trade_closed")
    def test_sync_does_not_deduplicate_a_distinct_position_against_recent_bot_close(self, notify):
        first = self.log(self.response(), reason="tp")
        second_opened = self.filled + timedelta(seconds=10)
        second_filled = self.filled + timedelta(seconds=40)
        entry_response = self.response("entry-2", side="buy", price="100", fee="1",
                                       reduce_only=False, filled_at=second_opened)
        self.responses["entry-2"] = entry_response
        Order.objects.create(
            instrument=self.inst, exchange_order_id="entry-2", side="buy", type="market",
            qty="10", price="100", status="filled", reduce_only=False,
            correlation_id="position-2", opened_at=second_opened, closed_at=second_opened,
            raw_response=entry_response,
        )
        self.configure_sync([self.response("close-2", price="99", filled_at=second_filled)])
        self.open_position(opened_at=second_opened - timedelta(seconds=1))
        with self.captureOnCommitCallbacks(execute=True):
            _sync_positions(self.adapter, positions=[])
        second = OperationReport.objects.exclude(pk=first.pk).get()
        self.assertEqual(OperationReport.objects.count(), 2)
        self.assertNotEqual(first.accounting_key, second.accounting_key)
        self.assertEqual(second.accounting_status, "confirmed")
        self.assertEqual(second.correlation_id, "position-2")
        self.assertEqual(second.opened_at, second_opened)
        self.assertEqual(second.pnl_abs, Decimal("-11.5"))
        self.assertNotEqual(second.close_sub_reason, "bot_close_missed")
        self.assertEqual(self.queue.call_count, 2)
        notify.assert_called_once()

    def test_scale_in_entry_quantity_fees_and_vwap_share_canonical_root_timestamp(self):
        scale_opened = self.opened + timedelta(minutes=10)
        scale_response = self.response("scale-1", qty="5", side="buy", price="110", fee="0.5",
                                       reduce_only=False, filled_at=scale_opened)
        self.responses["scale-1"] = scale_response
        scale = Order.objects.create(
            instrument=self.inst, exchange_order_id="scale-1", side="buy", type="market",
            qty="5", price="110", fee_usdt="99", status="filled", reduce_only=False,
            correlation_id="scale-1", parent_correlation_id="position-1",
            opened_at=scale_opened, closed_at=scale_opened, raw_response=scale_response,
        )
        response = self.response(qty="15", price="105", fee="0.75")
        for delta in (-1, 1):
            with self.subTest(exchange_open_timestamp_delta=delta):
                report = self.log(response, qty=15, entry_price=104,
                                  opened_at=self.opened + timedelta(seconds=delta))
                self.assertEqual(report.accounting_status, "confirmed")
                self.assertEqual(report.opened_at, self.opened)
                self.assertEqual(report.qty, Decimal("15"))
                self.assertAlmostEqual(report.entry_price, Decimal(1550) / Decimal(15), places=9)
                self.assertEqual(report.exit_price, Decimal("105"))
                self.assertEqual(report.fee_usdt, Decimal("2.25"))
                self.assertEqual(report.pnl_abs, Decimal("22.75"))
                self.assertEqual(set(report.accounting_details["entry_evidence"]["entry_order_db_ids"]),
                                 {self.entry.pk, scale.pk})
        self.assertEqual(OperationReport.objects.count(), 1)
        self.assertEqual(Order.objects.filter(reduce_only=True).count(), 1)
        self.queue.assert_called_once()

    def trailing_context(self):
        return {"inst": self.inst, "signal_id": "1", "correlation_id": "position-1",
                "leverage": 5, "equity_before": 1000, "accounting_opened_at": self.opened}

    @override_settings(TRAILING_STOP_ENABLED=True, PARTIAL_CLOSE_AT_R=1,
                       PARTIAL_CLOSE_PCT=.5, BREAKEVEN_STOP_ENABLED=True)
    @patch("execution.tasks._has_sl_stop_order")
    @patch("execution.tasks._normalize_order_qty", side_effect=lambda adapter, symbol, qty: qty)
    def test_trailing_partial_ack_is_durable_and_returns_before_using_old_quantity(self, normalize, stop_lookup):
        self.adapter.create_order = Mock(return_value={"id": "partial-ack"})
        context = self.trailing_context()
        result = _check_trailing_stop(self.adapter, "BTCUSDT", "buy", 10, 100, 102, .01,
                                      opened_at=self.opened, close_context=context)
        self.assertEqual(result, (False, 0))
        self.assertTrue(context["partial_attempted"])
        self.adapter.create_order.assert_called_once_with(
            "BTCUSDT", "sell", "market", 5, params={"reduceOnly": True})
        close = Order.objects.get(reduce_only=True, exchange_order_id="partial-ack")
        self.assertTrue(close.raw_response["close_accounting"]["is_partial"])
        self.assertEqual(close.raw_response["close_accounting"]["status"], "pending")
        self.assertFalse(OperationReport.objects.exists())
        stop_lookup.assert_not_called()  # Even breakeven may not place a stop for stale qty=10.

    @override_settings(TRAILING_STOP_ENABLED=True, PARTIAL_CLOSE_AT_R=1,
                       PARTIAL_CLOSE_PCT=.5, BREAKEVEN_STOP_ENABLED=False,
                       TRAILING_STOP_ACTIVATION_R=99)
    @patch("execution.close_accounting.capture_close", side_effect=RuntimeError("DB write failed"))
    @patch("execution.tasks._normalize_order_qty", side_effect=lambda adapter, symbol, qty: qty)
    def test_partial_ack_cache_marker_survives_failed_durable_capture_and_prevents_repeat(self, normalize, capture):
        state = {}
        cache = SimpleNamespace(get=lambda key: state.get(key),
                                set=lambda key, value, **kwargs: state.__setitem__(key, value))
        self.adapter.create_order = Mock(return_value={"id": "partial-ack"})
        context = self.trailing_context()
        with patch("execution.tasks._redis_client", return_value=cache):
            first = _check_trailing_stop(self.adapter, "BTCUSDT", "buy", 10, 100, 102, .01,
                                        opened_at=self.opened, close_context=context)
            self.assertTrue(context["partial_attempted"])
            self.assertTrue(any(key.startswith("trail:partial_done:") for key in state))
            # A subsequent cycle receives the actual reduced position size.
            second = _check_trailing_stop(self.adapter, "BTCUSDT", "buy", 5, 100, 102, .01,
                                         opened_at=self.opened, close_context=self.trailing_context())
        self.assertEqual(first, (False, 0))
        self.assertEqual(second, (False, 0))
        self.adapter.create_order.assert_called_once()
        capture.assert_called_once()
        self.assertFalse(Order.objects.filter(reduce_only=True).exists())

    def manage_position(self, **updates):
        # Stop maintenance and runtime feature lookup are unrelated to this race;
        # position refresh, exit submission and subsequent sync remain real code.
        with patch("execution.tasks._reconcile_sl"), \
                patch("execution.tasks.get_runtime_bool", return_value=False), \
                patch("execution.tasks._compute_tp_sl_prices", return_value=(101, 99, .01, .01)):
            values = dict(
                adapter=self.adapter, inst=self.inst, sig=None, sig_payload={}, strategy_name="trend",
                symbol="BTCUSDT", ticker_used={"last": 140}, last_price=140,
                current_qty=10, entry_price=100, pos_opened_at=self.opened,
                signal_direction="long", side="buy", direction_allowed=True,
                atr=None, contract_size=1, leverage=5, equity_usdt=1000,
                current_session="london", btc_recommended_bias="", account_ai_enabled=False,
                account_ai_config_id=None, account_owner_id=None, account_alias="test",
                account_service="bingx",
            )
            values.update(updates)
            return _manage_open_position(**values)

    @override_settings(TRAILING_STOP_ENABLED=False, TP_PROGRESS_EARLY_EXIT_ENABLED=False,
                       DOWNTREND_LONG_KILLER_ENABLED=False, STALE_POSITION_CLEANUP_ENABLED=False)
    @patch("execution.tasks.notify_trade_closed")
    def test_signal_flip_ack_returns_manage_only_until_fresh_exchange_snapshot(self, notify):
        self.configure_sync([])
        self.open_position()
        self.adapter.fetch_positions = Mock(return_value=[{
            "symbol": "BTC/USDT:USDT", "contracts": 10, "side": "long", "entryPrice": 100,
            "openingTimestamp": int(self.opened.timestamp() * 1000),
        }])
        self.adapter.create_order = Mock(return_value={"id": "flip-ack"})
        result = self.manage_position(ticker_used={"last": 100}, last_price=100,
                                      signal_direction="short", side="sell")
        self.assertTrue(result[0])  # Caller must skip the opposite-side entry in this cycle.
        self.assertFalse(result[1])
        self.adapter.create_order.assert_called_once_with(
            "BTCUSDT", "sell", "market", 10, params={"reduceOnly": True})
        report = OperationReport.objects.get()
        self.assertEqual(report.reason, "signal_flip")
        self.assertEqual(report.accounting_status, "pending")
        self.assertIsNone(report.exit_price)
        self.assertIsNone(report.pnl_abs)
        self.assertTrue(notify.call_args.kwargs["accounting_pending"])
        self.assertIsNone(notify.call_args.kwargs["pnl_abs"])
        self.queue.assert_not_called()

    @override_settings(TRAILING_STOP_ENABLED=False, TP_PROGRESS_EARLY_EXIT_ENABLED=False,
                       DOWNTREND_LONG_KILLER_ENABLED=False, STALE_POSITION_CLEANUP_ENABLED=False)
    @patch("execution.tasks.notify_trade_closed")
    def test_tp_no_position_error_preserves_context_until_sync_accounts_actual_close(self, notify):
        self.configure_sync([self.response()])
        position = self.open_position()
        self.adapter.fetch_positions = Mock(return_value=[{
            "symbol": "BTC/USDT:USDT", "contracts": 10, "side": "long", "entryPrice": 100,
            "openingTimestamp": int(self.opened.timestamp() * 1000),
        }])
        self.adapter.create_order = Mock(side_effect=RuntimeError("No position to close"))
        result = self.manage_position()
        self.assertTrue(result[0])
        self.adapter.create_order.assert_called_once_with(
            "BTCUSDT", "sell", "market", 10, params={"reduceOnly": True})
        position.refresh_from_db()
        self.assertTrue(position.is_open)
        self.assertEqual(position.qty, Decimal("10"))
        self.assertEqual(position.opened_at, self.opened)
        self.assertFalse(OperationReport.objects.exists())
        notify.assert_not_called()

        with self.captureOnCommitCallbacks(execute=True):
            _sync_positions(self.adapter, positions=[])
        position.refresh_from_db()
        report = OperationReport.objects.get()
        self.assertFalse(position.is_open)
        self.assertEqual(report.accounting_status, "confirmed")
        self.assertEqual(report.qty, Decimal("10"))
        self.assertEqual(report.exit_price, Decimal("101"))
        self.assertEqual(report.fee_usdt, Decimal("1.5"))
        self.assertEqual(report.pnl_abs, Decimal("8.5"))
        self.adapter.create_order.assert_called_once()
        self.queue.assert_called_once()
        notify.assert_called_once()

    @override_settings(TRAILING_STOP_ENABLED=False)
    @patch("execution.tasks.notify_trade_closed")
    def test_manage_refresh_missing_position_preserves_context_for_sync(self, notify):
        self.configure_sync([self.response()])
        position = self.open_position()
        self.adapter.fetch_positions = Mock(return_value=[])
        self.adapter.create_order = Mock()
        self.assertTrue(self.manage_position()[0])
        position.refresh_from_db()
        self.assertTrue(position.is_open)
        self.assertEqual(position.qty, Decimal("10"))
        self.assertEqual(position.opened_at, self.opened)
        self.adapter.create_order.assert_not_called()
        self.assertFalse(OperationReport.objects.exists())
        with self.captureOnCommitCallbacks(execute=True):
            _sync_positions(self.adapter, positions=[])
        self.assertEqual(OperationReport.objects.get().pnl_abs, Decimal("8.5"))
        self.adapter.create_order.assert_not_called()
        notify.assert_called_once()

    @override_settings(TRAILING_STOP_ENABLED=True, PARTIAL_CLOSE_AT_R=99,
                       BREAKEVEN_STOP_ENABLED=False, TRAILING_STOP_ACTIVATION_R=1,
                       TRAILING_STOP_LOCK_IN_PCT=.5, TRAILING_DYNAMIC_LOCK_ENABLED=False)
    @patch("execution.tasks._has_sl_stop_order", return_value=(False, 0, []))
    def test_trailing_no_position_error_sets_context_and_stops_without_another_order(self, stop_lookup):
        cache = SimpleNamespace(get=lambda key: ".10" if key.startswith("trail:max_fav:") else None,
                                set=lambda *args, **kwargs: True)
        self.adapter.create_order = Mock(side_effect=RuntimeError("No position to close"))
        context = self.trailing_context()
        with patch("execution.tasks._redis_client", return_value=cache):
            result = _check_trailing_stop(self.adapter, "BTCUSDT", "buy", 10, 100, 102, .01,
                                         opened_at=self.opened, close_context=context)
        self.assertEqual(result, (False, 0))
        self.assertTrue(context["position_missing"])
        self.adapter.create_order.assert_called_once_with(
            "BTCUSDT", "sell", "market", 10, params={"reduceOnly": True})
        self.assertNotIn("order_response", context)
        self.assertFalse(Order.objects.filter(reduce_only=True).exists())
        self.assertFalse(OperationReport.objects.exists())
