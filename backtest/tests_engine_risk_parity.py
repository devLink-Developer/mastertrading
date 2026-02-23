from django.test import SimpleTestCase, override_settings

from backtest.engine import _max_daily_trades_for_adx, _volatility_adjusted_risk


class BacktestRiskParityTest(SimpleTestCase):
    @override_settings(
        MAX_DAILY_TRADES_LOW_ADX=1,
        MAX_DAILY_TRADES=4,
        MAX_DAILY_TRADES_HIGH_ADX=8,
        MAX_DAILY_TRADES_LOW_ADX_THRESHOLD=10.0,
        MAX_DAILY_TRADES_HIGH_ADX_THRESHOLD=30.0,
    )
    def test_backtest_uses_configurable_adx_thresholds(self):
        self.assertEqual(_max_daily_trades_for_adx(9.0), 1)
        self.assertEqual(_max_daily_trades_for_adx(20.0), 4)
        self.assertEqual(_max_daily_trades_for_adx(31.0), 8)

    @override_settings(PER_INSTRUMENT_RISK={"BTCUSDT": 0.0015})
    def test_backtest_symbol_risk_is_capped_by_base_budget(self):
        # Backtest should mirror live and never increase risk over base budget.
        self.assertAlmostEqual(
            _volatility_adjusted_risk("BTCUSDT", atr_pct=0.01, base_risk=0.0005),
            0.0005,
            places=8,
        )
