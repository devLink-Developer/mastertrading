from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from signals.regime_mtf import (
    build_symbol_regime_snapshot,
    consolidate_lead_state,
    recommended_bias,
)


def _make_daily_df(days: int = 420, regime: str = "bull", seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=days, freq="D")
    if regime == "bull":
        rets = rng.normal(0.004, 0.01, days)
    elif regime == "bear":
        rets = rng.normal(-0.004, 0.01, days)
    else:
        rets = rng.normal(0.0, 0.005, days)
    prices = 100 * np.exp(np.cumsum(rets))
    high = prices * (1 + rng.uniform(0.001, 0.01, days))
    low = prices * (1 - rng.uniform(0.001, 0.01, days))
    open_ = np.roll(prices, 1)
    open_[0] = prices[0]
    volume = rng.uniform(1000, 3000, days)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": prices,
            "volume": volume,
        },
        index=idx,
    )


class RegimeMtfHelpersTest(SimpleTestCase):
    def test_consolidate_lead_state_prefers_confirmed_bear(self):
        state = consolidate_lead_state("bear_confirmed", "bear_weak", "bull_weak")
        self.assertEqual(state, "bear_confirmed")

    def test_recommended_bias_allows_tactical_long_inside_bear(self):
        bias = recommended_bias(
            "bear_confirmed",
            "bear_confirmed",
            "bull_weak",
            lead_state="bear_confirmed",
        )
        self.assertEqual(bias, "tactical_long")

    def test_recommended_bias_is_balanced_in_range(self):
        bias = recommended_bias("range", "range", "transition", lead_state="range")
        self.assertEqual(bias, "balanced")

    def test_build_symbol_regime_snapshot_detects_bullish_series(self):
        inst = MagicMock()
        inst.symbol = "BTCUSDT"
        df = _make_daily_df(regime="bull")
        with patch("signals.regime_mtf.latest_candles", return_value=df):
            snapshot = build_symbol_regime_snapshot(inst)
        self.assertIn(snapshot["daily_regime"], {"bull_confirmed", "bull_weak"})
        self.assertIn(snapshot["weekly_regime"], {"bull_confirmed", "bull_weak"})
        self.assertIn(snapshot["monthly_regime"], {"bull_confirmed", "bull_weak"})
