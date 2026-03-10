from datetime import datetime, timedelta, timezone

from django.test import SimpleTestCase

from marketdata.tasks import _ohlcv_fetch_params


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
