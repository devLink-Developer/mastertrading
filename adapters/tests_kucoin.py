from unittest import mock

import ccxt
from django.test import SimpleTestCase

from .kucoin import KucoinFuturesAdapter


class KucoinAdapterCreateOrderTests(SimpleTestCase):
    def _build_adapter(self, client):
        adapter = object.__new__(KucoinFuturesAdapter)
        adapter.client = client
        adapter.sandbox = True
        adapter.margin_mode = "cross"
        adapter.leverage = 3
        adapter._markets_loaded = True
        adapter._leverage_set_symbols = set()
        return adapter

    def test_create_order_retries_transient_errors_with_stable_client_oid(self):
        client = mock.Mock()
        client.create_order.side_effect = [
            ccxt.RequestTimeout("timeout-1"),
            ccxt.RequestTimeout("timeout-2"),
            {"id": "ok-1", "clientOrderId": "oid-ok"},
        ]
        adapter = self._build_adapter(client)

        resp = adapter.create_order(
            symbol="BTCUSDT",
            side="buy",
            type_="market",
            amount=0.2,
            params={},
        )

        self.assertEqual(resp["id"], "ok-1")
        self.assertEqual(client.create_order.call_count, 3)
        first_params = client.create_order.call_args_list[0].args[5]
        second_params = client.create_order.call_args_list[1].args[5]
        third_params = client.create_order.call_args_list[2].args[5]
        self.assertTrue(first_params.get("clientOid"))
        self.assertEqual(first_params.get("clientOid"), second_params.get("clientOid"))
        self.assertEqual(first_params.get("clientOid"), third_params.get("clientOid"))
        self.assertEqual(first_params.get("marginMode"), "cross")
        self.assertEqual(first_params.get("leverage"), 3)

    def test_create_order_recovers_existing_exchange_order_by_client_oid(self):
        client = mock.Mock()
        client.create_order.side_effect = [
            ccxt.RequestTimeout("timeout-1"),
            ccxt.RequestTimeout("timeout-2"),
            ccxt.RequestTimeout("timeout-3"),
        ]
        client.fetch_open_orders.return_value = [
            {"id": "recovered-1", "clientOrderId": "oid-fixed"}
        ]
        client.fetch_closed_orders.return_value = []
        adapter = self._build_adapter(client)

        resp = adapter.create_order(
            symbol="BTCUSDT",
            side="buy",
            type_="market",
            amount=0.1,
            params={"clientOid": "oid-fixed"},
        )

        self.assertEqual(resp["id"], "recovered-1")
        self.assertEqual(client.create_order.call_count, 3)
        client.fetch_open_orders.assert_called()

    def test_create_order_raises_when_retry_and_recovery_fail(self):
        client = mock.Mock()
        client.create_order.side_effect = [
            ccxt.RequestTimeout("timeout-1"),
            ccxt.RequestTimeout("timeout-2"),
            ccxt.RequestTimeout("timeout-3"),
        ]
        client.fetch_open_orders.return_value = []
        client.fetch_closed_orders.return_value = []
        adapter = self._build_adapter(client)

        with self.assertRaises(ccxt.RequestTimeout):
            adapter.create_order(
                symbol="BTCUSDT",
                side="buy",
                type_="market",
                amount=0.1,
                params={"clientOid": "oid-fixed"},
            )
