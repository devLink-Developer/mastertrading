"""
Tests for signals.regime — HMM market regime detection.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
from django.test import TestCase, override_settings


def _make_candles(n: int = 200, trend: bool = True, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 1h OHLCV candles for testing.

    If trend=True, generates a trending market (smooth upward drift).
    If trend=False, generates choppy mean-reverting noise.
    """
    rng = np.random.RandomState(seed)

    if trend:
        # Trending: strong drift + low noise
        log_prices = np.cumsum(rng.normal(0.001, 0.005, n))
    else:
        # Choppy: no drift + high noise
        log_prices = np.cumsum(rng.normal(0.0, 0.015, n))

    prices = 50000 * np.exp(log_prices)
    high = prices * (1 + rng.uniform(0.001, 0.008, n))
    low = prices * (1 - rng.uniform(0.001, 0.008, n))
    close = prices
    open_ = np.roll(prices, 1)
    open_[0] = prices[0]
    volume = rng.uniform(100, 1000, n)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df


class TestComputeFeatures(TestCase):
    """Tests for _compute_features()."""

    def test_returns_none_insufficient_data(self):
        from signals.regime import _compute_features
        df = _make_candles(n=30)
        result = _compute_features(df)
        self.assertIsNone(result)

    def test_returns_features_shape(self):
        from signals.regime import _compute_features
        df = _make_candles(n=200)
        features = _compute_features(df)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[1], 3)  # 3 features
        self.assertGreater(features.shape[0], 100)

    def test_no_nans_in_output(self):
        from signals.regime import _compute_features
        df = _make_candles(n=200)
        features = _compute_features(df)
        self.assertFalse(np.any(np.isnan(features)))

    def test_volatility_column_positive(self):
        from signals.regime import _compute_features
        df = _make_candles(n=200)
        features = _compute_features(df)
        # Column 1 is realised vol — should be non-negative
        self.assertTrue(np.all(features[:, 1] >= 0))

    def test_adx_column_bounded(self):
        from signals.regime import _compute_features
        df = _make_candles(n=200)
        features = _compute_features(df)
        # Column 2 is ADX / 100 — should be in [0, 1]
        self.assertTrue(np.all(features[:, 2] >= 0))
        self.assertTrue(np.all(features[:, 2] <= 1))


class TestAdxSeries(TestCase):
    """Tests for _adx_series()."""

    def test_output_length(self):
        from signals.regime import _adx_series
        n = 100
        highs = np.random.uniform(100, 101, n)
        lows = np.random.uniform(99, 100, n)
        closes = np.random.uniform(99.5, 100.5, n)
        adx = _adx_series(highs, lows, closes, period=14)
        self.assertEqual(len(adx), n - 1)

    def test_returns_nan_for_insufficient_data(self):
        from signals.regime import _adx_series
        adx = _adx_series(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.5, 1.5, 2.5]),
            np.array([0.8, 1.8, 2.8]),
            period=14,
        )
        self.assertTrue(np.all(np.isnan(adx)))


class TestFitHMM(TestCase):
    """Tests for _fit_hmm()."""

    def test_fits_two_states(self):
        from signals.regime import _compute_features, _fit_hmm
        df = _make_candles(n=200)
        features = _compute_features(df)
        model, states = _fit_hmm(features, n_states=2)
        self.assertEqual(len(np.unique(states)), 2)
        self.assertEqual(len(states), len(features))

    def test_model_has_means(self):
        from signals.regime import _compute_features, _fit_hmm
        df = _make_candles(n=200)
        features = _compute_features(df)
        model, _ = _fit_hmm(features, n_states=2)
        self.assertEqual(model.means_.shape, (2, 3))


class TestLabelStates(TestCase):
    """Tests for _label_states()."""

    @override_settings(HMM_REGIME_TRENDING_RISK_MULT=1.0, HMM_REGIME_CHOPPY_RISK_MULT=0.7)
    def test_labels_assigned(self):
        from signals.regime import _compute_features, _fit_hmm, _label_states
        df = _make_candles(n=200)
        features = _compute_features(df)
        model, _ = _fit_hmm(features, n_states=2)
        labels = _label_states(model, n_states=2)
        self.assertEqual(len(labels), 2)
        names = {v["name"] for v in labels.values()}
        self.assertEqual(names, {"trending", "choppy"})

    @override_settings(HMM_REGIME_TRENDING_RISK_MULT=1.0, HMM_REGIME_CHOPPY_RISK_MULT=0.7)
    def test_choppy_has_lower_risk_mult(self):
        from signals.regime import _compute_features, _fit_hmm, _label_states
        df = _make_candles(n=200)
        features = _compute_features(df)
        model, _ = _fit_hmm(features, n_states=2)
        labels = _label_states(model, n_states=2)
        choppy = [v for v in labels.values() if v["name"] == "choppy"][0]
        trending = [v for v in labels.values() if v["name"] == "trending"][0]
        self.assertLessEqual(choppy["risk_mult"], trending["risk_mult"])


class TestPredictRegimeFromDf(TestCase):
    """Tests for predict_regime_from_df() (backtest helper)."""

    @override_settings(HMM_REGIME_TRENDING_RISK_MULT=1.0, HMM_REGIME_CHOPPY_RISK_MULT=0.7)
    def test_returns_dict(self):
        from signals.regime import predict_regime_from_df
        df = _make_candles(n=200)
        result = predict_regime_from_df(df)
        self.assertIsNotNone(result)
        self.assertIn("state", result)
        self.assertIn("name", result)
        self.assertIn("risk_mult", result)
        self.assertIn("confidence", result)

    @override_settings(HMM_REGIME_TRENDING_RISK_MULT=1.0, HMM_REGIME_CHOPPY_RISK_MULT=0.7)
    def test_confidence_bounded(self):
        from signals.regime import predict_regime_from_df
        df = _make_candles(n=200)
        result = predict_regime_from_df(df)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_returns_none_for_short_data(self):
        from signals.regime import predict_regime_from_df
        df = _make_candles(n=30)
        result = predict_regime_from_df(df)
        self.assertIsNone(result)


class TestRegimeRiskMult(TestCase):
    """Tests for regime_risk_mult()."""

    @override_settings(HMM_REGIME_ENABLED=False)
    def test_returns_1_when_disabled(self):
        from signals.regime import regime_risk_mult
        result = regime_risk_mult("BTCUSDT")
        self.assertEqual(result, 1.0)

    @override_settings(HMM_REGIME_ENABLED=True)
    @patch("signals.regime.get_cached_regime")
    def test_returns_cached_value(self, mock_cache):
        from signals.regime import regime_risk_mult
        mock_cache.return_value = {"risk_mult": 0.7, "name": "choppy"}
        result = regime_risk_mult("BTCUSDT")
        self.assertEqual(result, 0.7)

    @override_settings(HMM_REGIME_ENABLED=True)
    @patch("signals.regime.get_cached_regime")
    def test_returns_1_when_no_cache(self, mock_cache):
        from signals.regime import regime_risk_mult
        mock_cache.return_value = None
        result = regime_risk_mult("BTCUSDT")
        self.assertEqual(result, 1.0)
