"""
Tests for signals.garch — GARCH(1,1) volatility forecasting.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from django.test import TestCase, override_settings


def _make_returns(n: int = 300, vol: float = 0.01, seed: int = 42) -> np.ndarray:
    """Generate synthetic log-returns with known volatility."""
    rng = np.random.RandomState(seed)
    return rng.normal(0, vol, n)


def _make_candles(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 1h OHLCV candles."""
    rng = np.random.RandomState(seed)
    log_prices = np.cumsum(rng.normal(0.0005, 0.008, n))
    prices = 50000 * np.exp(log_prices)
    high = prices * (1 + rng.uniform(0.001, 0.008, n))
    low = prices * (1 - rng.uniform(0.001, 0.008, n))
    return pd.DataFrame({
        "open": np.roll(prices, 1),
        "high": high,
        "low": low,
        "close": prices,
        "volume": rng.uniform(100, 1000, n),
    })


class TestFitGarch(TestCase):
    """Tests for _fit_garch()."""

    def test_returns_dict_on_success(self):
        from signals.garch import _fit_garch
        returns = _make_returns(300, vol=0.01)
        result = _fit_garch(returns)
        self.assertIsNotNone(result)
        self.assertIn("cond_vol", result)
        self.assertIn("alpha", result)
        self.assertIn("beta", result)
        self.assertIn("persistence", result)

    def test_cond_vol_positive(self):
        from signals.garch import _fit_garch
        returns = _make_returns(300, vol=0.01)
        result = _fit_garch(returns)
        self.assertGreater(result["cond_vol"], 0)

    def test_persistence_bounded(self):
        from signals.garch import _fit_garch
        returns = _make_returns(300, vol=0.01)
        result = _fit_garch(returns)
        # alpha + beta should be < 1 for stationarity (usually)
        self.assertLess(result["persistence"], 1.05)  # small tolerance
        self.assertGreater(result["persistence"], 0)

    def test_returns_none_for_too_short(self):
        from signals.garch import _fit_garch
        returns = _make_returns(5, vol=0.01)
        result = _fit_garch(returns)
        # With only 5 obs, fit should either fail gracefully or return a result
        # (arch library may still fit, so we just check no crash)
        # This is a "doesn't crash" test
        self.assertTrue(result is None or isinstance(result, dict))

    def test_high_vol_returns_higher_cond_vol(self):
        from signals.garch import _fit_garch
        low_vol = _fit_garch(_make_returns(300, vol=0.005, seed=1))
        high_vol = _fit_garch(_make_returns(300, vol=0.02, seed=1))
        self.assertIsNotNone(low_vol)
        self.assertIsNotNone(high_vol)
        # High vol series should produce higher conditional vol forecast
        self.assertGreater(high_vol["cond_vol"], low_vol["cond_vol"])

    @override_settings(GARCH_MAX_PERSISTENCE=1.0)
    def test_returns_none_when_persistence_above_limit(self):
        from signals.garch import _fit_garch

        mock_result = MagicMock()
        mock_result.forecast.return_value = SimpleNamespace(
            variance=pd.DataFrame([[1.0]]),
        )
        mock_result.params = {"omega": 0.1, "alpha[1]": 0.7, "beta[1]": 0.5}
        mock_result.loglikelihood = -100.0
        mock_result.aic = 250.0
        mock_result.bic = 260.0
        mock_result.nobs = 300

        mock_model = MagicMock()
        mock_model.fit.return_value = mock_result

        with patch("arch.arch_model", return_value=mock_model):
            result = _fit_garch(_make_returns(300, vol=0.01))
        self.assertIsNone(result)


class TestForecastVolFromDf(TestCase):
    """Tests for forecast_vol_from_df() (backtest helper)."""

    def test_returns_float(self):
        from signals.garch import forecast_vol_from_df
        df = _make_candles(300)
        result = forecast_vol_from_df(df)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_returns_none_for_short_data(self):
        from signals.garch import forecast_vol_from_df
        df = _make_candles(30)
        result = forecast_vol_from_df(df)
        self.assertIsNone(result)


class TestBlendedVol(TestCase):
    """Tests for blended_vol()."""

    @override_settings(GARCH_ENABLED=False)
    def test_returns_atr_when_garch_disabled(self):
        from signals.garch import blended_vol
        result = blended_vol("BTCUSDT", 0.01)
        self.assertEqual(result, 0.01)

    @override_settings(GARCH_ENABLED=True, GARCH_BLEND_WEIGHT=0.6)
    @patch("signals.garch.garch_vol_pct")
    def test_blends_correctly(self, mock_garch):
        from signals.garch import blended_vol
        mock_garch.return_value = 0.008  # GARCH forecast
        result = blended_vol("BTCUSDT", 0.01)  # ATR = 0.01
        # blend = 0.6 * 0.008 + 0.4 * 0.01 = 0.0048 + 0.004 = 0.0088
        self.assertAlmostEqual(result, 0.0088, places=5)

    @override_settings(GARCH_ENABLED=True, GARCH_BLEND_WEIGHT=0.6)
    @patch("signals.garch.garch_vol_pct")
    def test_returns_garch_when_no_atr(self, mock_garch):
        from signals.garch import blended_vol
        mock_garch.return_value = 0.008
        result = blended_vol("BTCUSDT", None)
        self.assertEqual(result, 0.008)

    @override_settings(GARCH_ENABLED=True, GARCH_BLEND_WEIGHT=0.6)
    @patch("signals.garch.garch_vol_pct")
    def test_returns_atr_when_no_garch(self, mock_garch):
        from signals.garch import blended_vol
        mock_garch.return_value = None
        result = blended_vol("BTCUSDT", 0.01)
        self.assertEqual(result, 0.01)

    @override_settings(GARCH_ENABLED=True, GARCH_BLEND_WEIGHT=0.6)
    @patch("signals.garch.garch_vol_pct")
    def test_returns_none_when_both_missing(self, mock_garch):
        from signals.garch import blended_vol
        mock_garch.return_value = None
        result = blended_vol("BTCUSDT", None)
        self.assertIsNone(result)

    @override_settings(
        GARCH_ENABLED=True,
        GARCH_BLEND_WEIGHT=0.6,
        GARCH_BLEND_VOL_FLOOR_PCT=0.003,
        GARCH_BLEND_FLOOR_ATR_RATIO=0.5,
    )
    @patch("signals.garch.garch_vol_pct")
    def test_blended_vol_applies_atr_ratio_floor(self, mock_garch):
        from signals.garch import blended_vol

        mock_garch.return_value = 0.0001
        result = blended_vol("BTCUSDT", 0.008)
        # Raw blend ~= 0.00326; floor=max(0.003, 0.008*0.5=0.004) => 0.004
        self.assertAlmostEqual(result, 0.004, places=6)

    @override_settings(
        GARCH_ENABLED=True,
        GARCH_BLEND_WEIGHT=0.6,
        GARCH_BLEND_VOL_FLOOR_PCT=0.003,
    )
    @patch("signals.garch.garch_vol_pct")
    def test_blended_vol_applies_absolute_floor_when_only_garch(self, mock_garch):
        from signals.garch import blended_vol

        mock_garch.return_value = 0.0002
        result = blended_vol("BTCUSDT", None)
        self.assertAlmostEqual(result, 0.003, places=6)


class TestGarchVolPct(TestCase):
    """Tests for garch_vol_pct()."""

    @override_settings(GARCH_ENABLED=False)
    def test_returns_none_when_disabled(self):
        from signals.garch import garch_vol_pct
        result = garch_vol_pct("BTCUSDT")
        self.assertIsNone(result)

    @override_settings(GARCH_ENABLED=True)
    @patch("signals.garch.get_cached_forecast")
    def test_returns_cached_value(self, mock_cache):
        from signals.garch import garch_vol_pct
        mock_cache.return_value = {"cond_vol": 0.0064}
        result = garch_vol_pct("BTCUSDT")
        self.assertEqual(result, 0.0064)

    @override_settings(GARCH_ENABLED=True)
    @patch("signals.garch.get_cached_forecast")
    def test_returns_none_when_no_cache(self, mock_cache):
        from signals.garch import garch_vol_pct
        mock_cache.return_value = None
        result = garch_vol_pct("BTCUSDT")
        self.assertIsNone(result)
