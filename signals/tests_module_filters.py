from __future__ import annotations

from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase, override_settings

from signals.modules import carry as carry_module
from signals.modules import grid as grid_module
from signals.modules import meanrev as meanrev_module
from signals.modules import microvol as microvol_module
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
        MODULE_TREND_HTF_ADX_MIN=5.0,
        MODULE_TREND_EMA20_PULLBACK_TOLERANCE_PCT=0.003,
        MODULE_IMPULSE_FILTER_ENABLED=False,
    )
    def test_trend_detector_allows_small_pullback_below_htf_ema20(self):
        df_htf = _build_df([100 + i * 0.25 for i in range(119)] + [128.4])
        df_ltf = _build_df([100 + i * 0.08 for i in range(120)])
        out = trend_module.detect(df_ltf, df_htf, [], "asia")
        self.assertIsNotNone(out)
        self.assertEqual(out["direction"], "long")
        self.assertIn("ema20_pullback_tolerance_pct", out["reasons"])

    @override_settings(
        MODULE_ADX_TREND_MIN=5.0,
        MODULE_TREND_HTF_ADX_MIN=5.0,
        MODULE_TREND_EMA20_PULLBACK_TOLERANCE_PCT=0.001,
        MODULE_IMPULSE_FILTER_ENABLED=False,
    )
    def test_trend_detector_still_blocks_when_pullback_exceeds_tolerance(self):
        df_htf = _build_df([100 + i * 0.25 for i in range(119)] + [126.9])
        df_ltf = _build_df([100 + i * 0.08 for i in range(120)])
        out = trend_module.detect(df_ltf, df_htf, [], "asia")
        self.assertIsNone(out)

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

    @override_settings(
        MODULE_GRID_ADX_MIN=8.0,
        MODULE_GRID_ADX_MAX=22.0,
        MODULE_GRID_ATR_MIN_PCT=0.006,
        MODULE_GRID_ATR_MAX_PCT=0.030,
        MODULE_GRID_Z_ENTRY=1.0,
        MODULE_GRID_RANGE_LOOKBACK=60,
        MODULE_GRID_MIN_RANGE_WIDTH_PCT=0.004,
        MODULE_GRID_EMA_GAP_MAX_PCT=0.05,
        MODULE_GRID_IMPULSE_BLOCK_ENABLED=False,
        MODULE_GRID_MIN_CONFIDENCE=0.20,
        MODULE_GRID_ALLOWED_SESSIONS={"london", "ny", "overlap", "asia"},
        MODULE_GRID_ALLOWED_SYMBOLS={"BTCUSDT", "ETHUSDT"},
        MODULE_GRID_MTF_RANGE_ENABLED=False,
    )
    def test_grid_emits_short_on_range_upper_extreme(self):
        # Sideways/choppy structure with last candles stretched to the upper edge.
        base = [100 + ((i % 12) - 6) * 0.12 for i in range(140)]
        vals = base + [102.0, 102.6, 103.2, 103.5]
        df_ltf = _build_df(vals)
        df_htf = _build_df([100 + ((i % 10) - 5) * 0.10 for i in range(140)])
        with (
            patch("signals.modules.grid.compute_adx", return_value=14.0),
            patch("signals.modules.grid.compute_atr_pct", return_value=0.012),
            patch("signals.modules.grid._regime_gate", return_value=(True, {"status": "ok"})),
        ):
            out = grid_module.detect(df_ltf, df_htf, [], "london", symbol="BTCUSDT")
        self.assertIsNotNone(out)
        self.assertEqual(out["direction"], "short")
        self.assertIn("zscore", out["reasons"])

    @override_settings(
        MODULE_GRID_ADX_MIN=8.0,
        MODULE_GRID_ADX_MAX=22.0,
        MODULE_GRID_IMPULSE_BLOCK_ENABLED=False,
        MODULE_GRID_ALLOWED_SYMBOLS={"BTCUSDT", "ETHUSDT"},
    )
    def test_grid_blocks_when_regime_gate_fails(self):
        vals = [100 + ((i % 8) - 4) * 0.15 for i in range(160)]
        df_ltf = _build_df(vals)
        df_htf = _build_df([100 + ((i % 8) - 4) * 0.12 for i in range(160)])
        with (
            patch("signals.modules.grid.compute_adx", return_value=14.0),
            patch("signals.modules.grid.compute_atr_pct", return_value=0.012),
            patch("signals.modules.grid._regime_gate", return_value=(False, {"status": "blocked"})),
        ):
            out = grid_module.detect(df_ltf, df_htf, [], "london", symbol="BTCUSDT")
        self.assertIsNone(out)

    @override_settings(
        MODULE_GRID_ADX_MIN=8.0,
        MODULE_GRID_ADX_MAX=22.0,
        MODULE_GRID_ATR_MIN_PCT=0.006,
        MODULE_GRID_ATR_MAX_PCT=0.030,
        MODULE_GRID_Z_ENTRY=1.0,
        MODULE_GRID_RANGE_LOOKBACK=60,
        MODULE_GRID_MIN_RANGE_WIDTH_PCT=0.004,
        MODULE_GRID_EMA_GAP_MAX_PCT=0.05,
        MODULE_GRID_IMPULSE_BLOCK_ENABLED=False,
        MODULE_GRID_MIN_CONFIDENCE=0.20,
        MODULE_GRID_ALLOWED_SESSIONS={"london", "ny", "overlap", "asia"},
        MODULE_GRID_ALLOWED_SYMBOLS={"BTCUSDT", "ETHUSDT"},
        MODULE_GRID_MTF_RANGE_ENABLED=False,
    )
    def test_grid_blocks_symbol_outside_allowed_set(self):
        base = [100 + ((i % 12) - 6) * 0.12 for i in range(140)]
        vals = base + [102.0, 102.6, 103.2, 103.5]
        df_ltf = _build_df(vals)
        df_htf = _build_df([100 + ((i % 10) - 5) * 0.10 for i in range(140)])
        with (
            patch("signals.modules.grid.compute_adx", return_value=14.0),
            patch("signals.modules.grid.compute_atr_pct", return_value=0.012),
            patch("signals.modules.grid._regime_gate", return_value=(True, {"status": "ok"})),
        ):
            out = grid_module.detect(df_ltf, df_htf, [], "london", symbol="XRPUSDT")
        self.assertIsNone(out)

    @override_settings(
        MODULE_GRID_ADX_MIN=8.0,
        MODULE_GRID_ADX_MAX=22.0,
        MODULE_GRID_ATR_MIN_PCT=0.006,
        MODULE_GRID_ATR_MAX_PCT=0.030,
        MODULE_GRID_Z_ENTRY=0.3,
        MODULE_GRID_MIN_RANGE_WIDTH_PCT=0.004,
        MODULE_GRID_EMA_GAP_MAX_PCT=0.05,
        MODULE_GRID_IMPULSE_BLOCK_ENABLED=False,
        MODULE_GRID_MIN_CONFIDENCE=0.10,
        MODULE_GRID_ALLOWED_SESSIONS={"london", "ny", "overlap", "asia"},
        MODULE_GRID_ALLOWED_SYMBOLS={"BTCUSDT", "ETHUSDT"},
        MODULE_GRID_MTF_RANGE_ENABLED=True,
        MODULE_GRID_BUY_ZONE_PCT=0.15,
        MODULE_GRID_SELL_ZONE_PCT=0.15,
        MODULE_GRID_SL_BUFFER_PCT=0.003,
        MODULE_GRID_TP_BUFFER_PCT=0.002,
        MODULE_GRID_D1_LOOKBACK=45,
    )
    def test_grid_mtf_emits_long_near_weekly_low_with_sl_tp_hints(self):
        """Grid detects long near weekly low and returns structural SL/TP hints."""
        import numpy as np

        # Price near a weekly low (100.0 floor, high was 110.0)
        base = [100.2 + ((i % 12) - 6) * 0.05 for i in range(140)]
        vals = base + [100.3, 100.1, 100.05, 100.02]
        df_ltf = _build_df(vals)
        df_htf = _build_df([100 + ((i % 10) - 5) * 0.05 for i in range(140)])

        # Build mock daily candles: weekly low~100, high~110
        d1_dates = pd.date_range("2026-01-01", periods=30, freq="D")
        d1_data = {
            "open": np.linspace(104, 105, 30),
            "high": np.array([110.0] * 5 + [107.0] * 25),
            "low": np.array([102.0] * 23 + [100.0] * 7),
            "close": np.linspace(105, 100.02, 30),
            "volume": [1000] * 30,
        }
        df_d1 = pd.DataFrame(d1_data, index=d1_dates)

        with (
            patch("signals.modules.grid.compute_adx", return_value=14.0),
            patch("signals.modules.grid.compute_atr_pct", return_value=0.012),
            patch("signals.modules.grid._regime_gate", return_value=(True, {"status": "ok"})),
            patch("signals.modules.grid._fetch_daily_candles", return_value=df_d1),
        ):
            out = grid_module.detect(df_ltf, df_htf, [], "london", symbol="BTCUSDT")
        self.assertIsNotNone(out, "Grid should emit on MTF low zone")
        self.assertEqual(out["direction"], "long")
        self.assertIn("sl_price_hint", out)
        self.assertIn("tp_price_hint", out)
        # SL should be below the weekly low
        self.assertLess(out["sl_price_hint"], 100.0)
        # TP should be near the weekly high (last 7d high = 107.0)
        self.assertGreater(out["tp_price_hint"], 105.0)
        # Reasons should contain MTF info
        self.assertIn("mtf", out["reasons"])
        self.assertTrue(out["reasons"]["mtf_enabled"])

    @override_settings(
        MODULE_MICROVOL_ALLOWED_SESSIONS={"ny_open", "overlap", "ny"},
        MODULE_MICROVOL_ALLOWED_SYMBOLS={"BTCUSDT", "ETHUSDT"},
        MODULE_MICROVOL_HTF_PULLBACK_TOLERANCE_PCT=0.003,
        MODULE_MICROVOL_HTF_ADX_MIN=18.0,
        MODULE_MICROVOL_HTF_ADX_MAX=55.0,
        MODULE_MICROVOL_ATR_MIN_PCT=0.0025,
        MODULE_MICROVOL_ATR_MAX_PCT=0.025,
        MODULE_MICROVOL_BREAKOUT_LOOKBACK=20,
        MODULE_MICROVOL_BREAKOUT_BUFFER_PCT=0.001,
        MODULE_MICROVOL_VOLUME_LOOKBACK=20,
        MODULE_MICROVOL_MIN_VOLUME_RATIO=1.4,
        MODULE_MICROVOL_IMPULSE_LOOKBACK=20,
        MODULE_MICROVOL_IMPULSE_BODY_MULT=1.4,
        MODULE_MICROVOL_IMPULSE_MIN_BODY_PCT=0.002,
        MODULE_MICROVOL_MAX_EMA20_DIST_PCT=0.03,
        MODULE_MICROVOL_MIN_CONFIDENCE=0.30,
    )
    def test_microvol_emits_long_on_impulse_breakout(self):
        vals = [100 + (i * 0.03) for i in range(120)]
        vals[-3:] = [103.2, 104.0, 105.4]
        df_ltf = _build_df(vals)
        df_ltf.loc[df_ltf.index[:-1], "volume"] = 10.0
        df_ltf.loc[df_ltf.index[-1], "volume"] = 28.0
        df_htf = _build_df([100 + (i * 0.20) for i in range(80)])
        with patch("signals.modules.microvol.compute_adx", return_value=24.0):
            out = microvol_module.detect(df_ltf, df_htf, [], "ny_open", symbol="BTCUSDT")
        self.assertIsNotNone(out)
        self.assertEqual(out["direction"], "long")
        self.assertIn("volume_ratio", out["reasons"])

    @override_settings(
        MODULE_MICROVOL_ALLOWED_SESSIONS={"ny_open", "overlap", "ny"},
        MODULE_MICROVOL_ALLOWED_SYMBOLS={"BTCUSDT", "ETHUSDT"},
        MODULE_MICROVOL_HTF_PULLBACK_TOLERANCE_PCT=0.003,
        MODULE_MICROVOL_HTF_ADX_MIN=18.0,
        MODULE_MICROVOL_HTF_ADX_MAX=55.0,
        MODULE_MICROVOL_ATR_MIN_PCT=0.0025,
        MODULE_MICROVOL_ATR_MAX_PCT=0.025,
        MODULE_MICROVOL_BREAKOUT_LOOKBACK=20,
        MODULE_MICROVOL_BREAKOUT_BUFFER_PCT=0.001,
        MODULE_MICROVOL_VOLUME_LOOKBACK=20,
        MODULE_MICROVOL_MIN_VOLUME_RATIO=1.4,
        MODULE_MICROVOL_IMPULSE_LOOKBACK=20,
        MODULE_MICROVOL_IMPULSE_BODY_MULT=1.4,
        MODULE_MICROVOL_IMPULSE_MIN_BODY_PCT=0.002,
        MODULE_MICROVOL_MAX_EMA20_DIST_PCT=0.03,
        MODULE_MICROVOL_MIN_CONFIDENCE=0.30,
    )
    def test_microvol_allows_small_pullback_below_htf_ema20(self):
        vals = [100 + (i * 0.03) for i in range(120)]
        vals[-3:] = [103.2, 104.0, 105.4]
        df_ltf = _build_df(vals)
        df_ltf.loc[df_ltf.index[:-1], "volume"] = 10.0
        df_ltf.loc[df_ltf.index[-1], "volume"] = 28.0
        df_htf = _build_df([100 + (i * 0.20) for i in range(79)] + [114.9])
        with patch("signals.modules.microvol.compute_adx", return_value=24.0):
            out = microvol_module.detect(df_ltf, df_htf, [], "ny_open", symbol="BTCUSDT")
        self.assertIsNotNone(out)
        self.assertEqual(out["direction"], "long")
        self.assertIn("htf_pullback_tolerance_pct", out["reasons"])

    @override_settings(
        MODULE_MICROVOL_ALLOWED_SESSIONS={"ny_open"},
        MODULE_MICROVOL_ALLOWED_SYMBOLS={"BTCUSDT", "ETHUSDT"},
        MODULE_MICROVOL_HTF_ADX_MIN=18.0,
        MODULE_MICROVOL_HTF_ADX_MAX=55.0,
        MODULE_MICROVOL_ATR_MIN_PCT=0.0025,
        MODULE_MICROVOL_ATR_MAX_PCT=0.025,
        MODULE_MICROVOL_MAX_EMA20_DIST_PCT=0.03,
    )
    def test_microvol_blocks_symbol_outside_allowed_set(self):
        vals = [100 + (i * 0.03) for i in range(120)]
        vals[-3:] = [103.2, 104.0, 105.4]
        df_ltf = _build_df(vals)
        df_ltf.loc[df_ltf.index[:-1], "volume"] = 10.0
        df_ltf.loc[df_ltf.index[-1], "volume"] = 28.0
        df_htf = _build_df([100 + (i * 0.20) for i in range(80)])
        with patch("signals.modules.microvol.compute_adx", return_value=24.0):
            out = microvol_module.detect(df_ltf, df_htf, [], "ny_open", symbol="DOGEUSDT")
        self.assertIsNone(out)

    @override_settings(
        MODULE_MICROVOL_ALLOWED_SESSIONS={"asia", "ny_open"},
        MODULE_MICROVOL_ALLOWED_SYMBOLS={"BTCUSDT", "ETHUSDT"},
        MODULE_MICROVOL_HTF_PULLBACK_TOLERANCE_PCT=0.003,
        MODULE_MICROVOL_HTF_ADX_MIN=18.0,
        MODULE_MICROVOL_HTF_ADX_MAX=55.0,
        MODULE_MICROVOL_ATR_MIN_PCT=0.0025,
        MODULE_MICROVOL_ATR_MAX_PCT=0.025,
        MODULE_MICROVOL_BREAKOUT_LOOKBACK=20,
        MODULE_MICROVOL_BREAKOUT_BUFFER_PCT=0.001,
        MODULE_MICROVOL_VOLUME_LOOKBACK=20,
        MODULE_MICROVOL_MIN_VOLUME_RATIO=1.4,
        MODULE_MICROVOL_IMPULSE_LOOKBACK=20,
        MODULE_MICROVOL_IMPULSE_BODY_MULT=1.4,
        MODULE_MICROVOL_IMPULSE_MIN_BODY_PCT=0.002,
        MODULE_MICROVOL_MAX_EMA20_DIST_PCT=0.03,
        MODULE_MICROVOL_MIN_CONFIDENCE=0.30,
    )
    def test_microvol_explain_reports_reject_stage(self):
        vals = [100 + (i * 0.03) for i in range(120)]
        vals[-3:] = [103.2, 104.0, 105.4]
        df_ltf = _build_df(vals)
        df_ltf.loc[df_ltf.index[:-1], "volume"] = 10.0
        df_ltf.loc[df_ltf.index[-1], "volume"] = 11.0
        df_htf = _build_df([100 + (i * 0.20) for i in range(80)])
        with patch("signals.modules.microvol.compute_adx", return_value=24.0):
            diag = microvol_module.explain(df_ltf, df_htf, [], "asia", symbol="BTCUSDT")
        self.assertFalse(diag["accepted"])
        self.assertEqual(diag["stage"], "volume")
        self.assertIn("volume_ratio", diag)
