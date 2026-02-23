from unittest import mock

import ccxt
from django.test import SimpleTestCase

from .bingx import BingXFuturesAdapter


class BingXAdapterCreateOrderTests(SimpleTestCase):
    def _build_adapter(self, client):
        adapter = object.__new__(BingXFuturesAdapter)
        adapter.client = client
        adapter.sandbox = True
        adapter.margin_mode = "cross"
        adapter.leverage = 0
        adapter._markets_loaded = True
        adapter._leverage_set_symbols = set()
        return adapter

    def test_trigger_reduce_only_retries_without_reduce_only(self):
        reduce_only_err = Exception(
            "bingx {'code':109400,'msg':'In the Hedge mode, the \"ReduceOnly\" field can not be filled.'}"
        )
        client = mock.Mock()
        client.create_order.side_effect = [
            reduce_only_err,  # first try
            reduce_only_err,  # retry with positionSide + reduceOnly
            {"id": "ok-1"},  # hedged retry without reduceOnly
        ]
        adapter = self._build_adapter(client)

        resp = adapter.create_order(
            symbol="BTCUSDT",
            side="sell",
            type_="market",
            amount=0.1,
            params={"triggerPrice": 68000.0, "reduceOnly": True},
        )

        self.assertEqual(resp["id"], "ok-1")
        self.assertEqual(client.create_order.call_count, 3)
        second_params = client.create_order.call_args_list[1].args[5]
        third_params = client.create_order.call_args_list[2].args[5]
        self.assertTrue(second_params.get("reduceOnly"))
        self.assertEqual(second_params.get("positionSide"), "LONG")
        self.assertNotIn("reduceOnly", third_params)
        self.assertTrue(third_params.get("hedged"))
        self.assertEqual(third_params.get("positionSide"), "LONG")

    def test_non_trigger_reduce_only_retries_without_reduce_only(self):
        reduce_only_err = Exception(
            "bingx {'code':109400,'msg':'In the Hedge mode, the \"ReduceOnly\" field can not be filled.'}"
        )
        client = mock.Mock()
        client.create_order.side_effect = [
            reduce_only_err,
            reduce_only_err,
            {"id": "ok-2"},
        ]
        adapter = self._build_adapter(client)

        resp = adapter.create_order(
            symbol="BTCUSDT",
            side="sell",
            type_="market",
            amount=0.1,
            params={"reduceOnly": True},
        )

        self.assertEqual(resp["id"], "ok-2")
        self.assertEqual(client.create_order.call_count, 3)
        second_params = client.create_order.call_args_list[1].args[5]
        third_params = client.create_order.call_args_list[2].args[5]
        self.assertTrue(second_params.get("reduceOnly"))
        self.assertEqual(second_params.get("positionSide"), "LONG")
        self.assertNotIn("reduceOnly", third_params)
        self.assertTrue(third_params.get("hedged"))
        self.assertEqual(third_params.get("positionSide"), "LONG")

    def test_non_trigger_reduce_only_raises_no_position_after_hedged_retry(self):
        reduce_only_err = Exception(
            "bingx {'code':109400,'msg':'In the Hedge mode, the \"ReduceOnly\" field can not be filled.'}"
        )
        no_position_err = Exception("Position does not exist")
        client = mock.Mock()
        client.create_order.side_effect = [
            reduce_only_err,
            reduce_only_err,
            no_position_err,
        ]
        adapter = self._build_adapter(client)

        with self.assertRaises(ccxt.InvalidOrder):
            adapter.create_order(
                symbol="BTCUSDT",
                side="sell",
                type_="market",
                amount=0.1,
                params={"reduceOnly": True},
            )

        self.assertEqual(client.create_order.call_count, 3)

    def test_create_order_retries_network_errors(self):
        client = mock.Mock()
        client.create_order.side_effect = [
            ccxt.RequestTimeout("timeout-1"),
            ccxt.RequestTimeout("timeout-2"),
            {"id": "ok-net"},
        ]
        adapter = self._build_adapter(client)

        resp = adapter.create_order(
            symbol="BTCUSDT",
            side="buy",
            type_="market",
            amount=0.1,
            params={},
        )

        self.assertEqual(resp["id"], "ok-net")
        self.assertEqual(client.create_order.call_count, 3)
