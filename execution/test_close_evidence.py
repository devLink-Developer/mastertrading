"""Portable tests for close-fill evidence; no Django or real exchange calls."""

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from execution.close_evidence import resolve_close_evidence, resolve_exchange_close_evidence


NOW = 1788513877000


def filled_order(**changes):
    order = {"id": "close-1", "symbol": "ADA/USDT:USDT", "side": "sell", "status": "closed",
             "filled": "12", "amount": "12", "remaining": "0", "average": "0.1804",
             "lastTradeTimestamp": NOW, "reduceOnly": True,
             "fee": {"currency": "USDT", "cost": "0.0010824"},
             "info": {"positionSide": "LONG"}}
    order.update(changes)
    return order


def adapter(**methods):
    client = SimpleNamespace(**methods)
    return SimpleNamespace(client=client, _map_symbol=lambda symbol: "ADA/USDT:USDT")


class CloseEvidenceTests(unittest.TestCase):
    def resolve(self, order=None, adapter_obj=None, **kwargs):
        return resolve_close_evidence(adapter_obj, "ADAUSDT", "12", order_response=order,
                                      account_asset="USDT", close_side="sell", **kwargs)

    def test_acknowledgement_is_not_a_fill_and_fetch_uses_exact_id_and_mapped_symbol(self):
        fetch = Mock(return_value={"id": "close-1", "status": "open"})
        result = self.resolve({"id": "close-1"}, adapter(fetch_order=fetch))
        fetch.assert_called_once_with("close-1", "ADA/USDT:USDT")
        self.assertFalse(result.is_full_close)
        self.assertIsNone(result.filled_qty)
        self.assertIsNone(result.average_price)
        self.assertIsNone(result.fee)
        self.assertFalse(result.fee_known)

    def test_exact_fetch_resolves_actual_fill(self):
        result = self.resolve({"id": "close-1"}, adapter(fetch_order=Mock(return_value=filled_order())))
        self.assertTrue(result.is_full_close)
        self.assertEqual(result.source, "fetch_order")
        self.assertEqual(result.average_price, Decimal("0.1804"))
        self.assertEqual(result.filled_qty, Decimal("12"))
        self.assertEqual(result.fee, Decimal("0.0010824"))
        self.assertEqual(result.filled_timestamp_ms, NOW)

    def test_incomplete_fetch_does_not_discard_known_partial_execution(self):
        result = self.resolve(filled_order(status="canceled", filled="4"),
                              adapter(fetch_order=Mock(return_value={"id": "close-1"})))
        self.assertEqual(result.filled_qty, Decimal("4"))
        self.assertEqual(result.status, "partial")

    def test_none_fetch_keeps_known_acknowledgement_id_for_durable_retry(self):
        result = self.resolve({"id": "close-1"}, adapter(fetch_order=Mock(return_value=None)))
        self.assertEqual(result.order_id, "close-1")
        self.assertFalse(result.is_full_close)
        partial = self.resolve(filled_order(status="canceled", filled="4"), adapter(fetch_order=Mock(return_value=None)))
        self.assertEqual(partial.filled_qty, Decimal("4"))
        self.assertEqual(partial.order_id, "close-1")

    def test_fully_evidenced_response_does_not_fetch(self):
        fetch = Mock()
        self.assertTrue(self.resolve(filled_order(), adapter(fetch_order=fetch)).is_full_close)
        fetch.assert_not_called()

    def test_requested_price_and_amount_are_never_execution_evidence(self):
        result = self.resolve({"id": "close-1", "status": "closed", "amount": "12", "price": "0.19"})
        self.assertFalse(result.is_full_close)
        self.assertIsNone(result.filled_qty)
        self.assertIsNone(result.average_price)

    def test_cost_fallback_is_contract_size_aware(self):
        result = self.resolve(filled_order(average=None, cost="21.648", info={}), contract_size="10")
        self.assertEqual(result.average_price, Decimal("0.1804"))
        self.assertTrue(result.is_full_close)

    def test_ccxt_inferred_cost_from_requested_price_is_not_a_fill_price(self):
        result = self.resolve(filled_order(average=None, price="0.19", cost="2.28",
                              info={"executedQty": "12", "avgPrice": "0", "price": "0.19"}))
        self.assertIsNone(result.average_price)
        self.assertFalse(result.is_full_close)

    def test_raw_quote_evidence_allows_cost_fallback(self):
        result = self.resolve(filled_order(average=None, cost="2.1648",
                              info={"executedQty": "12", "avgPrice": "0", "cumQuote": "2.1648"}))
        self.assertEqual(result.average_price, Decimal("0.1804"))

    def test_raw_zero_execution_overrides_ccxt_inferred_filled_amount(self):
        result = self.resolve(filled_order(info={"executedQty": "0", "avgPrice": "0", "origQty": "12"}))
        self.assertFalse(result.is_full_close)
        self.assertEqual(result.filled_qty, Decimal("0"))
        self.assertIsNone(result.average_price)

    def test_raw_order_without_executed_quantity_cannot_use_inferred_filled(self):
        result = self.resolve(filled_order(info={"orderId": "close-1", "origQty": "12",
                                                "status": "FILLED", "avgPrice": "0.1804"}))
        self.assertFalse(result.is_full_close)
        self.assertIsNone(result.filled_qty)

    def test_raw_shorthand_without_executed_quantity_cannot_use_inference(self):
        result = self.resolve(filled_order(info={"i": "close-1", "q": "12", "X": "FILLED", "ap": "0.1804"}))
        self.assertFalse(result.is_full_close)
        self.assertIsNone(result.filled_qty)

    def test_conflicting_raw_and_unified_order_identity_is_rejected(self):
        result = self.resolve(filled_order(info={"orderId": "different", "executedQty": "12", "avgPrice": "0.1804"}))
        self.assertFalse(result.is_full_close)

    def test_conflicting_raw_symbol_or_side_cannot_hide_behind_unified_fields(self):
        for extra in ({"symbol": "BTC-USDT"}, {"side": "BUY"}):
            raw = {"orderId": "close-1", "executedQty": "12", "avgPrice": "0.1804", **extra}
            with self.subTest(extra=extra):
                self.assertFalse(self.resolve(filled_order(info=raw)).is_full_close)

    def test_anonymous_duplicate_fees_are_unknown_not_guessed(self):
        fee = {"currency": "USDT", "cost": "0.001"}
        result = self.resolve(filled_order(fees=[fee, dict(fee)]))
        self.assertFalse(result.fee_known)
        self.assertIsNone(result.fee)

    def test_mixed_fee_currencies_cannot_be_partially_reported_as_complete(self):
        result = self.resolve(filled_order(fees=[{"currency": "USDT", "cost": "0.001"},
                                                {"currency": "BTC", "cost": "0.000001"}]))
        self.assertFalse(result.fee_known)
        self.assertIsNone(result.fee)

    def test_canceled_partial_and_quantity_mismatch_are_not_full_closes(self):
        for status, qty in [("canceled", "4"), ("closed", "4"), ("closed", "13"), ("open", "12")]:
            with self.subTest(status=status, qty=qty):
                result = self.resolve(filled_order(status=status, filled=qty, remaining="8"))
                self.assertFalse(result.is_full_close)
                self.assertEqual(result.filled_qty, Decimal(qty))

    def test_nonfinite_negative_or_zero_average_is_unknown(self):
        for price in ("NaN", "Infinity", "-1", "0"):
            with self.subTest(price=price):
                result = self.resolve(filled_order(average=price))
                self.assertFalse(result.is_full_close)
                self.assertIsNone(result.average_price)

    def test_missing_fee_remains_unknown_without_invalidating_filled_quantity(self):
        result = self.resolve(filled_order(fee=None))
        self.assertTrue(result.is_full_close)
        self.assertFalse(result.fee_known)
        self.assertIsNone(result.fee)

    def test_fee_and_fees_are_alternative_representations_not_additive(self):
        result = self.resolve(filled_order(fees=[{"currency": "USDT", "cost": "0.0010824"}]))
        self.assertEqual(result.fee, Decimal("0.0010824"))

    def test_identified_fee_duplicates_are_not_counted_twice(self):
        fee = {"id": "commission-1", "currency": "USDT", "cost": "0.001"}
        result = self.resolve(filled_order(fees=[fee, dict(fee)]))
        self.assertEqual(result.fee, Decimal("0.001"))

    def test_conflicting_identified_fee_duplicates_are_unknown(self):
        result = self.resolve(filled_order(fees=[
            {"id": "fee1", "currency": "USDT", "cost": "0.001"},
            {"id": "fee1", "currency": "USDT", "cost": "0.002"}]))
        self.assertFalse(result.fee_known)
        self.assertIsNone(result.fee)

    def test_rebate_sign_and_explicit_zero_fee_are_preserved(self):
        for fee in ("-0.0001", "0"):
            with self.subTest(fee=fee):
                result = self.resolve(filled_order(fee={"currency": "USDT", "cost": fee}))
                self.assertTrue(result.fee_known)
                self.assertEqual(result.fee, Decimal(fee))

    def test_currency_context_is_exact_and_mixed_assets_are_unknown(self):
        for currency in ("VST", "USDC", "BTC", None):
            with self.subTest(currency=currency):
                result = self.resolve(filled_order(fee={"currency": currency, "cost": "0.001"}))
                self.assertFalse(result.fee_known)
                self.assertIsNone(result.fee)
        for currency in ("VST", "USDC"):
            result = resolve_close_evidence(None, "ADAUSDT", "12", order_response=filled_order(
                fee={"currency": currency, "cost": "0.001"}), account_asset=currency)
            self.assertEqual(result.fee, Decimal("0.001"))

    def test_wrong_identity_or_side_is_never_accepted(self):
        for changes in ({"symbol": "BTC/USDT:USDT"}, {"side": "buy"}):
            with self.subTest(changes=changes):
                self.assertFalse(self.resolve(filled_order(**changes)).is_full_close)
        fetch = Mock(return_value=filled_order(id="different"))
        self.assertFalse(self.resolve({"id": "close-1"}, adapter(fetch_order=fetch)).is_full_close)

    def test_fetch_failure_is_bounded_unknown_and_does_not_leak_exception_text(self):
        fetch = Mock(side_effect=RuntimeError("secret-value"))
        result = self.resolve({"id": "close-1"}, adapter(fetch_order=fetch))
        fetch.assert_called_once()
        self.assertFalse(result.is_full_close)
        self.assertNotIn("secret-value", result.reason)


class ExchangeCloseEvidenceTests(unittest.TestCase):
    def resolve(self, orders, **kwargs):
        fetch = Mock(return_value=orders)
        result = resolve_exchange_close_evidence(adapter(fetch_closed_orders=fetch), "ADAUSDT", "12",
                    close_side="sell", position_side="long", opened_at=NOW - 3600000,
                    closed_at=NOW, account_asset="USDT", **kwargs)
        return result, fetch

    def test_single_strict_candidate_resolves_with_bounded_request(self):
        result, fetch = self.resolve([filled_order()])
        self.assertTrue(result.is_full_close)
        self.assertEqual(result.source, "fetch_closed_orders")
        fetch.assert_called_once_with("ADA/USDT:USDT", since=NOW - 3600000, limit=50)

    def test_protective_order_created_at_entry_but_filled_now_is_visible(self):
        stop = filled_order(timestamp=NOW - 3600000)
        fetch = Mock(side_effect=lambda symbol, since, limit: [stop] if stop["timestamp"] >= since else [])
        result = resolve_exchange_close_evidence(adapter(fetch_closed_orders=fetch), "ADAUSDT", "12",
            close_side="sell", position_side="long", opened_at=NOW - 3600000,
            closed_at=NOW, account_asset="USDT")
        self.assertTrue(result.is_full_close)

    def test_wrong_symbol_side_old_position_or_missing_identity_are_excluded(self):
        invalid = [filled_order(symbol="BTCUSDT"), filled_order(side="buy"),
                   filled_order(lastTradeTimestamp=NOW - 3600001),
                   filled_order(info={"positionSide": "SHORT"}),
                   filled_order(reduceOnly=False), filled_order(id=None),
                   filled_order(lastTradeTimestamp=None), filled_order(filled="4")]
        result, _ = self.resolve(invalid)
        self.assertFalse(result.is_full_close)
        self.assertEqual(result.status, "unknown")

    def test_two_eligible_orders_are_ambiguous_not_nearest_winner(self):
        result, _ = self.resolve([filled_order(), filled_order(id="other", lastTradeTimestamp=NOW - 1000)])
        self.assertEqual(result.status, "ambiguous")
        self.assertFalse(result.is_full_close)

    def test_duplicate_order_id_does_not_create_fake_ambiguity(self):
        order = filled_order()
        result, _ = self.resolve([order, dict(order)])
        self.assertTrue(result.is_full_close)

    def test_canceled_partial_can_never_be_exchange_full_close(self):
        result, _ = self.resolve([filled_order(status="canceled", filled="4")])
        self.assertFalse(result.is_full_close)

    def test_datetime_bounds_are_supported_and_unknown_entry_is_rejected(self):
        result = resolve_exchange_close_evidence(adapter(fetch_closed_orders=Mock(return_value=[filled_order()])),
            "ADAUSDT", "12", close_side="sell", position_side="long",
            opened_at=datetime.fromtimestamp((NOW - 3600000) / 1000, timezone.utc),
            closed_at=datetime.fromtimestamp(NOW / 1000, timezone.utc), account_asset="USDT")
        self.assertTrue(result.is_full_close)
        result = resolve_exchange_close_evidence(None, "ADAUSDT", "12", close_side="sell",
            position_side="long", opened_at=None, closed_at=NOW, account_asset="USDT")
        self.assertFalse(result.is_full_close)


class RetainedExchangeResponseTests(unittest.TestCase):
    def test_retained_main_demo_fill_uses_real_average_and_exchange_fee_asset(self):
        path = Path(__file__).resolve().parents[1] / "tmp/profit_fix/order_evidence_main.txt"
        if not path.exists():
            self.skipTest("Private read-only exchange probe is not retained here")
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        for retained in data["symbols"]["BTCUSDT"]["orders"]:
            order = dict(retained)
            order["info"] = order.pop("raw")
            result = resolve_close_evidence(None, "BTCUSDT", "0.0835", order_response=order,
                                             account_asset="USDT", close_side=order["side"])
            self.assertTrue(result.is_full_close)
            self.assertTrue(result.fee_known)
            self.assertEqual(result.average_price, Decimal(str(order["average"])))
            self.assertEqual(result.fee, Decimal(str(order["fee"]["cost"])))


if __name__ == "__main__":
    unittest.main()
