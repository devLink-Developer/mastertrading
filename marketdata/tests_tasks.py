from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from marketdata.tasks import (
    _has_active_marketdata_fetches,
    _ohlcv_fetch_params,
    fetch_instrument_data,
    fetch_ohlcv_and_funding,
)


class OhlcvFetchParamsTest(SimpleTestCase):
    def test_uses_incremental_fetch_when_symbol_is_near_real_time(self):
        now_utc = datetime(2026, 3, 10, 13, 30, tzinfo=timezone.utc)
        latest_ts = now_utc - timedelta(minutes=2)

        since_ms, limit, catchup_mode = _ohlcv_fetch_params(
            latest_ts,
            "1m",
            now_utc,
        )

        self.assertFalse(catchup_mode)
        self.assertEqual(limit, 20)
        self.assertIsNotNone(since_ms)

    def test_switches_to_tail_fetch_when_symbol_is_far_behind(self):
        now_utc = datetime(2026, 3, 10, 13, 30, tzinfo=timezone.utc)
        latest_ts = now_utc - timedelta(hours=11)

        since_ms, limit, catchup_mode = _ohlcv_fetch_params(
            latest_ts,
            "1m",
            now_utc,
        )

        self.assertTrue(catchup_mode)
        self.assertIsNone(since_ms)
        self.assertEqual(limit, 200)


class MarketdataLockingTest(SimpleTestCase):
    def test_detects_active_instrument_fetch_lock(self):
        client = MagicMock()
        client.mget.return_value = [None, b"token"]

        active = _has_active_marketdata_fetches(client, [1, 2])

        self.assertTrue(active)

    @patch("marketdata.tasks.fetch_instrument_data.delay")
    @patch("marketdata.tasks.Instrument.objects.filter")
    @patch("marketdata.tasks._redis_client")
    def test_dispatcher_skips_when_any_instrument_fetch_is_active(
        self,
        redis_client_mock,
        filter_mock,
        delay_mock,
    ):
        client = MagicMock()
        client.mget.return_value = [b"busy", None]
        redis_client_mock.return_value = client
        filter_mock.return_value.values_list.return_value = [1, 2]

        fetch_ohlcv_and_funding()

        delay_mock.assert_not_called()
        client.set.assert_not_called()

    @patch("marketdata.tasks.Instrument.objects.get")
    @patch("marketdata.tasks._acquire_marketdata_lock")
    def test_fetch_instrument_data_skips_when_instrument_lock_is_active(
        self,
        acquire_lock_mock,
        get_mock,
    ):
        acquire_lock_mock.return_value = (MagicMock(), "")

        fetch_instrument_data(123)

        get_mock.assert_not_called()
