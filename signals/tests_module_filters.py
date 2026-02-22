from __future__ import annotations

from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase, override_settings

from signals.modules import carry as carry_module
from signals.modules import meanrev as meanrev_module
from signals.modules import trend as trend_module


def _build_df(values: list[float]) -> pd.DataFrame:
    rows = []
    prev = float(values[0])
    for close in values:
        close_f = float(close)
        open_f = prev
        high_f = max(open_f, close_f) + 0.3
        low_f = min(open_f, close_f) - 0.3
        rows.append(
            {
                "open": open_f,
                "high": high_f,
                "low": low_f,
                "close": close_f,
                "volume": 1.0,
            }
        )
        prev = close_f
    return pd.DataFrame(rows)


class ModuleImpulseFiltersTest(SimpleTestCase):
    @override_settings(
        MODULE_ADX_TREND_MIN=5.0,
        MODULE_IMPULSE_FILTER_ENABLED=True,
        MODULE_IMPULSE_LOOKBACK=20,
        MODULE_IMPULSE_BODY_MULT=1.5,
        MODULE_IMPULSE_MIN_BODY_PCT=0.004,
        MODULE_IMPULSE_MAX_EMA20_DIST_PCT=0.004,
    )
    def test_trend_detector_blocks_impulse_chase(self):
        df_htf = _build_df([100 + i * 0.4 for i in range(120)])
        # LTF with large bullish displacement in the latest bar.
        vals = [100 + i * 0.12 for i in range(110)] + [119.0, 122.0, 126.0, 131.0]
        df_ltf = _build_df(vals)
        out = trend_module.detect(df_ltf, df_htf, [], "london")
        self.assertIsNone(out)

    @override_settings(
        MODULE_ADX_RANGE_MAX=35.0,
        MODULE_MEANREV_Z_ENTRY=0.7,
        MODULE_MEANREV_IMPULSE_BLOCK_ENABLED=True,
        MODULE_IMPULSE_LOOKBACK=20,
        MODULE_IMPULSE_BODY_MULT=1.5,
        MODULE_IMPULSE_MIN_BODY_PCT=0.004,
    )
    def test_meanrev_blocks_knife_catch_after_displacement(self):
        # Flat-ish HTF so meanrev gate can pass.
        df_htf = _build_df([100 + ((i % 4) - 2) * 0.05 for i in range(120)])
        # Last bar is a strong bearish displacement that would trigger long mean-reversion.
        vals = [100 + ((i % 6) - 3) * 0.08 for i in range(130)] + [97.0, 95.0, 92.5]
        df_ltf = _build_df(vals)
        with patch("signals.modules.meanrev.compute_adx", return_value=12.0):
            out = meanrev_module.detect(df_ltf, df_htf, [], "ny")
        self.assertIsNone(out)

    @override_settings(
        FUNDING_EXTREME_PERCENTILE=0.00008,
        MODULE_CARRY_FUNDING_MULT=1.0,
        MODULE_CARRY_MAX_ATR_PCT=0.020,
    )
    def test_carry_blocks_extreme_volatility(self):
        df_ltf = _build_df([100 + ((i % 3) - 1) * 0.2 for i in range(120)])
        funding = [0.00001] * 20 + [0.0002]
        with patch("signals.modules.carry.compute_atr_pct", return_value=0.03):
            out = carry_module.detect(df_ltf, df_ltf, funding, "overlap")
        self.assertIsNone(out)
