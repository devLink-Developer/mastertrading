from django.test import SimpleTestCase

from execution.management.commands.train_entry_filter_ml import (
    _backtest_direction_and_strategies,
)


class TrainEntryFilterCommandHelpersTest(SimpleTestCase):
    def test_backtest_direction_and_strategies_for_buy(self):
        direction, strategies = _backtest_direction_and_strategies("buy")
        self.assertEqual(direction, "long")
        self.assertEqual(strategies, {"smc_long", "backtest_long"})

    def test_backtest_direction_and_strategies_for_sell(self):
        direction, strategies = _backtest_direction_and_strategies("sell")
        self.assertEqual(direction, "short")
        self.assertEqual(strategies, {"smc_short", "backtest_short"})
