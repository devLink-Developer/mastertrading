from django.test import SimpleTestCase, override_settings

from execution.risk_policy import max_daily_trades_for_adx, volatility_adjusted_risk


class RiskPolicyHelpersTest(SimpleTestCase):
    @override_settings(
        MAX_DAILY_TRADES_LOW_ADX=2,
        MAX_DAILY_TRADES=5,
        MAX_DAILY_TRADES_HIGH_ADX=9,
        MAX_DAILY_TRADES_LOW_ADX_THRESHOLD=15.0,
        MAX_DAILY_TRADES_HIGH_ADX_THRESHOLD=30.0,
    )
    def test_max_daily_trades_for_adx_uses_configurable_thresholds(self):
        self.assertEqual(max_daily_trades_for_adx(None), 5)
        self.assertEqual(max_daily_trades_for_adx(12.0), 2)
        self.assertEqual(max_daily_trades_for_adx(20.0), 5)
        self.assertEqual(max_daily_trades_for_adx(32.0), 9)

    @override_settings(PER_INSTRUMENT_RISK={"BTCUSDT": 0.0015})
    def test_volatility_adjusted_risk_caps_symbol_risk_to_base_budget(self):
        # Per-symbol config must not increase allocator/base risk.
        self.assertAlmostEqual(
            volatility_adjusted_risk("BTCUSDT", atr_pct=None, base_risk=0.0005),
            0.0005,
            places=8,
        )

    @override_settings(
        PER_INSTRUMENT_RISK={"BTCUSDT": 0.0015},
        VOL_RISK_LOW_ATR_PCT=0.008,
        VOL_RISK_HIGH_ATR_PCT=0.015,
        VOL_RISK_MIN_SCALE=0.6,
        BTC_VOL_RISK_HARDEN_ENABLED=False,
    )
    def test_volatility_adjusted_risk_applies_atr_scaling_after_symbol_cap(self):
        # High ATR must still scale down capped per-symbol risk (regression for early-return bug).
        self.assertAlmostEqual(
            volatility_adjusted_risk("BTCUSDT", atr_pct=0.02, base_risk=0.003),
            0.0009,
            places=8,
        )

    @override_settings(
        PER_INSTRUMENT_RISK={},
        INSTRUMENT_RISK_TIERS_ENABLED=False,
        VOL_RISK_LOW_ATR_PCT=0.01,
        VOL_RISK_HIGH_ATR_PCT=0.02,
        VOL_RISK_MIN_SCALE=0.5,
    )
    def test_volatility_adjusted_risk_uses_configurable_ramp(self):
        # Midpoint ATR between low/high gives midpoint scale when min_scale=0.5.
        self.assertAlmostEqual(
            volatility_adjusted_risk("ETHUSDT", atr_pct=0.015, base_risk=0.01),
            0.0075,
            places=8,
        )

    @override_settings(
        PER_INSTRUMENT_RISK={},
        INSTRUMENT_RISK_TIERS_ENABLED=True,
        INSTRUMENT_RISK_TIERS={"base": 0.002, "mid": 0.0025, "alt": 0.0015},
        INSTRUMENT_TIER_MAP={"ETHUSDT": "base", "XRPUSDT": "mid", "ENAUSDT": "alt"},
    )
    def test_volatility_adjusted_risk_tiers_do_not_raise_allocator_budget(self):
        self.assertAlmostEqual(
            volatility_adjusted_risk("ETHUSDT", atr_pct=None, base_risk=0.0009),
            0.0009,
            places=8,
        )
        self.assertAlmostEqual(
            volatility_adjusted_risk("XRPUSDT", atr_pct=None, base_risk=0.0009),
            0.0009,
            places=8,
        )
        self.assertAlmostEqual(
            volatility_adjusted_risk("ENAUSDT", atr_pct=None, base_risk=0.0009),
            0.0009,
            places=8,
        )

    @override_settings(
        PER_INSTRUMENT_RISK={},
        INSTRUMENT_RISK_TIERS_ENABLED=False,
        VOL_RISK_LOW_ATR_PCT=0.008,
        VOL_RISK_HIGH_ATR_PCT=0.015,
        VOL_RISK_MIN_SCALE=0.6,
        BTC_VOL_RISK_HARDEN_ENABLED=True,
        BTC_VOL_RISK_ATR_THRESHOLD=0.012,
        BTC_VOL_RISK_MULT=0.75,
    )
    def test_volatility_adjusted_risk_applies_btc_hardening_above_threshold(self):
        # base=0.003, ATR>=high -> *0.6 => 0.0018, then BTC harden *0.75 => 0.00135
        self.assertAlmostEqual(
            volatility_adjusted_risk("BTCUSDT", atr_pct=0.02, base_risk=0.003),
            0.00135,
            places=8,
        )

    @override_settings(
        PER_INSTRUMENT_RISK={},
        INSTRUMENT_RISK_TIERS_ENABLED=False,
        VOL_RISK_LOW_ATR_PCT=0.008,
        VOL_RISK_HIGH_ATR_PCT=0.015,
        VOL_RISK_MIN_SCALE=0.6,
        BTC_VOL_RISK_HARDEN_ENABLED=True,
        BTC_VOL_RISK_ATR_THRESHOLD=0.012,
        BTC_VOL_RISK_MULT=0.75,
    )
    def test_volatility_adjusted_risk_btc_hardening_not_applied_below_threshold(self):
        self.assertAlmostEqual(
            volatility_adjusted_risk("BTCUSDT", atr_pct=0.009, base_risk=0.003),
            0.002828571428571429,
            places=8,
        )
