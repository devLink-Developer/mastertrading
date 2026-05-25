from __future__ import annotations

from datetime import timedelta
from io import StringIO
from unittest.mock import patch

import numpy as np
import pandas as pd
from django.core.management import call_command
from django.test import TestCase, override_settings
from django.utils import timezone

from core.models import Instrument
from execution.models import OperationReport
from marketdata.models import Candle
from signals.models import StrategyConfig
from signals.modules import trend as trend_module
from signals.runtime_overrides import RUNTIME_OVERRIDES_VERSION, invalidate_runtime_overrides_cache


def _build_df(values: list[float]) -> pd.DataFrame:
    rows = []
    prev = float(values[0])
    for close in values:
        close_f = float(close)
        open_f = prev
        rows.append(
            {
                "open": open_f,
                "high": max(open_f, close_f) + 0.2,
                "low": min(open_f, close_f) - 0.2,
                "close": close_f,
                "volume": 1.0,
            }
        )
        prev = close_f
    return pd.DataFrame(rows)


class KalmanTrendDetectorTest(TestCase):
    def tearDown(self):
        invalidate_runtime_overrides_cache()
        super().tearDown()

    @override_settings(MODULE_TREND_KALMAN_Q_RATIO=0.01)
    def test_kalman_smooth_is_deterministic(self):
        prices = np.array([100.0 + i * 0.1 + ((i % 3) * 0.02) for i in range(80)])
        first = trend_module._kalman_smooth(prices)
        second = trend_module._kalman_smooth(prices)

        self.assertIsNotNone(first)
        self.assertEqual(len(first), len(prices))
        np.testing.assert_allclose(first, second)

    @override_settings(
        MODULE_ADX_TREND_MIN=5.0,
        MODULE_TREND_HTF_ADX_MIN=0.0,
        MODULE_TREND_KALMAN_ENABLED=True,
        MODULE_TREND_KALMAN_BOOST=0.03,
        MODULE_IMPULSE_FILTER_ENABLED=False,
        MODULE_TREND_VOLUME_CONFIRM_ENABLED=False,
        MODULE_BOUNCE_BLOCK_PCT=999.0,
    )
    def test_kalman_boost_setting_is_used(self):
        df_htf = _build_df([100 + i * 0.25 for i in range(120)])
        df_ltf = _build_df([100 + i * 0.10 for i in range(120)])

        with patch("signals.modules.trend.compute_adx", return_value=30.0):
            out = trend_module.detect(df_ltf, df_htf, [], "london", symbol="BTCUSDT")

        self.assertIsNotNone(out)
        self.assertEqual(out["direction"], "long")
        self.assertAlmostEqual(out["reasons"]["kalman_boost"], 0.03)

    @override_settings(
        MODULE_ADX_TREND_MIN=5.0,
        MODULE_TREND_HTF_ADX_MIN=0.0,
        MODULE_TREND_KALMAN_ENABLED=True,
        MODULE_TREND_KALMAN_BOOST=0.03,
        MODULE_IMPULSE_FILTER_ENABLED=False,
        MODULE_TREND_VOLUME_CONFIRM_ENABLED=False,
        MODULE_BOUNCE_BLOCK_PCT=999.0,
    )
    def test_kalman_boost_db_override_is_used(self):
        StrategyConfig.objects.create(
            name="MODULE_TREND_KALMAN_BOOST",
            version=RUNTIME_OVERRIDES_VERSION,
            enabled=True,
            params_json={"value": 0.07},
        )
        invalidate_runtime_overrides_cache()
        df_htf = _build_df([100 + i * 0.25 for i in range(120)])
        df_ltf = _build_df([100 + i * 0.10 for i in range(120)])

        with patch("signals.modules.trend.compute_adx", return_value=30.0):
            out = trend_module.detect(df_ltf, df_htf, [], "london", symbol="BTCUSDT")

        self.assertIsNotNone(out)
        self.assertEqual(out["direction"], "long")
        self.assertAlmostEqual(out["reasons"]["kalman_boost"], 0.07)

    @override_settings(
        MODULE_ADX_TREND_MIN=5.0,
        MODULE_TREND_HTF_ADX_MIN=0.0,
        MODULE_TREND_EMA20_PULLBACK_TOLERANCE_PCT=0.0,
        MODULE_TREND_KALMAN_ENABLED=True,
        MODULE_TREND_KALMAN_SLOPE_MIN=0.001,
        MODULE_IMPULSE_FILTER_ENABLED=False,
        MODULE_TREND_VOLUME_CONFIRM_ENABLED=False,
        MODULE_BOUNCE_BLOCK_PCT=999.0,
    )
    def test_kalman_fallback_can_select_direction_when_ema_pullback_blocks(self):
        df_htf = _build_df([100.0] * 120)
        df_ltf = _build_df([100 + i * 0.08 for i in range(120)])
        fake_kalman = np.linspace(100.0, 104.0, len(df_htf))

        with (
            patch("signals.modules.trend.compute_adx", return_value=30.0),
            patch("signals.modules.trend._kalman_smooth", return_value=fake_kalman),
        ):
            out = trend_module.detect(df_ltf, df_htf, [], "london", symbol="BTCUSDT")

        self.assertIsNotNone(out)
        self.assertEqual(out["direction"], "long")
        self.assertTrue(out["reasons"]["kalman_fallback"])
        self.assertIn("kalman_slope", out["reasons"])


class AuditKalmanTrendCommandTest(TestCase):
    def setUp(self):
        self.instrument = Instrument.objects.create(
            symbol="ETHUSDT",
            exchange="bingx",
            base="ETH",
            quote="USDT",
            enabled=True,
        )

    def _create_candles(self):
        now = timezone.now().replace(second=0, microsecond=0)
        for idx in range(140):
            ts = now - timedelta(minutes=139 - idx)
            price = 100.0 + idx * 0.03
            Candle.objects.create(
                instrument=self.instrument,
                timeframe="1m",
                ts=ts,
                open=price,
                high=price + 0.2,
                low=price - 0.2,
                close=price + 0.05,
                volume=100,
            )
        for idx in range(120):
            ts = now - timedelta(hours=119 - idx)
            price = 100.0 + idx * 0.10
            Candle.objects.create(
                instrument=self.instrument,
                timeframe="1h",
                ts=ts,
                open=price,
                high=price + 0.3,
                low=price - 0.3,
                close=price + 0.08,
                volume=1000,
            )
        OperationReport.objects.create(
            instrument=self.instrument,
            side="buy",
            qty=1,
            entry_price=100,
            exit_price=101,
            pnl_abs=0.10,
            pnl_pct=0.01,
            notional_usdt=100,
            margin_used_usdt=20,
            fee_usdt=0,
            leverage=5,
            equity_before=100,
            equity_after=100.10,
            mode="demo",
            opened_at=now - timedelta(minutes=20),
            closed_at=now - timedelta(minutes=5),
            outcome=OperationReport.Outcome.WIN,
            reason="tp",
        )

    def test_command_prints_expected_shape(self):
        self._create_candles()
        out = StringIO()
        fixed_result = {
            "direction": "long",
            "raw_score": 0.8,
            "confidence": 0.8,
            "reasons": {
                "kalman_fallback": True,
                "kalman_boost": 0.05,
            },
        }

        with patch(
            "signals.management.commands.audit_kalman_trend.trend_module.detect",
            return_value=fixed_result,
        ):
            call_command(
                "audit_kalman_trend",
                "--days",
                "1",
                "--symbols",
                "ETHUSDT",
                "--q-grid",
                "0.01",
                "--slope-grid",
                "0.003",
                "--boost-grid",
                "0.05",
                "--cadence-minutes",
                "30",
                "--lookback",
                "80",
                stdout=out,
            )

        text = out.getvalue()
        self.assertIn("Kalman trend audit", text)
        self.assertIn("Best combo", text)
        self.assertIn("ETHUSDT", text)
