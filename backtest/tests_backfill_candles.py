from __future__ import annotations

from datetime import datetime, timezone
from unittest import mock

from django.test import SimpleTestCase

from backtest.management.commands.backfill_candles import (
    _is_rate_limit_error,
    _parse_retry_after_seconds,
    _resolve_ccxt_symbol,
)
from core.models import Instrument


class BackfillCandlesHelpersTests(SimpleTestCase):
    def test_parse_retry_after_seconds_uses_bingx_epoch_ms(self):
        exc = Exception(
            'bingx {"code":109429,"msg":"can retry after time: 1773274715335","data":{}}'
        )
        fake_now = datetime.fromtimestamp(1773274700, tz=timezone.utc)
        with mock.patch(
            "backtest.management.commands.backfill_candles.datetime"
        ) as dt_mock:
            dt_mock.now.return_value = fake_now
            dt_mock.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            wait_s = _parse_retry_after_seconds(exc, default_sleep=1.0)
        self.assertGreaterEqual(wait_s, 16.0)

    def test_parse_retry_after_seconds_falls_back_to_default(self):
        wait_s = _parse_retry_after_seconds(Exception("plain error"), default_sleep=2.5)
        self.assertEqual(wait_s, 2.5)

    def test_is_rate_limit_error(self):
        self.assertTrue(_is_rate_limit_error(Exception("109429 too many requests")))
        self.assertTrue(_is_rate_limit_error(Exception("please retry after time: 123")))
        self.assertFalse(_is_rate_limit_error(Exception("invalid symbol")))

    def test_resolve_ccxt_symbol_prefers_loaded_markets(self):
        inst = Instrument(symbol="BTCUSDT", base="BTC", quote="USDT")

        class DummyAdapter:
            def __init__(self):
                self.client = mock.Mock(markets={"BTC/USDT:USDT": {}})

            def _map_symbol(self, symbol):
                if "/" in symbol:
                    return symbol
                return "BTC/USDT:USDT"

        self.assertEqual(_resolve_ccxt_symbol(DummyAdapter(), inst), "BTC/USDT:USDT")
