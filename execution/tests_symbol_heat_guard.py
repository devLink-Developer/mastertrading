from decimal import Decimal

from django.test import SimpleTestCase, TestCase, override_settings

from execution.tasks import _symbol_heat_guard


class SymbolHeatGuardUnitTests(SimpleTestCase):
    """Tests for _symbol_heat_guard that don't need DB."""

    @override_settings(SYMBOL_HEAT_GUARD_ENABLED=False)
    def test_disabled_returns_no_reduction(self):
        mult, reason = _symbol_heat_guard("BTCUSDT")
        self.assertEqual(mult, 1.0)
        self.assertEqual(reason, "")

    @override_settings(SYMBOL_HEAT_GUARD_ENABLED=True)
    def test_unknown_symbol_returns_no_reduction(self):
        mult, reason = _symbol_heat_guard("FAKECOINUSDT")
        self.assertEqual(mult, 1.0)


class SymbolHeatGuardDBTests(TestCase):
    """Tests for _symbol_heat_guard that need OperationReport data."""

    @classmethod
    def setUpTestData(cls):
        from core.models import Instrument
        cls.inst = Instrument.objects.create(
            symbol="TESTUSDT",
            base="TEST",
            quote="USDT",
            exchange="bingx",
            enabled=True,
        )

    def _create_reports(self, symbol_inst, pnl_list):
        """Create OperationReport entries with given PnLs (most recent first)."""
        from django.utils import timezone
        from datetime import timedelta
        from execution.models import OperationReport

        now = timezone.now()
        for i, pnl in enumerate(pnl_list):
            OperationReport.objects.create(
                instrument=symbol_inst,
                side="sell",
                qty=1,
                entry_price=100,
                exit_price=100 + float(pnl),
                pnl_abs=Decimal(str(pnl)),
                pnl_pct=Decimal(str(pnl / 100)),
                outcome="win" if pnl > 0 else "loss",
                reason="tp" if pnl > 0 else "sl",
                closed_at=now - timedelta(minutes=i),
            )

    @override_settings(
        SYMBOL_HEAT_GUARD_ENABLED=True,
        SYMBOL_HEAT_GUARD_WINDOW=7,
        SYMBOL_HEAT_GUARD_WR_NEUTRAL=0.50,
        SYMBOL_HEAT_GUARD_WR_FLOOR=0.25,
        SYMBOL_HEAT_GUARD_MIN_RISK_MULT=0.35,
        SYMBOL_HEAT_GUARD_MIN_TRADES=3,
    )
    def test_too_few_trades_returns_no_reduction(self):
        self._create_reports(self.inst, [0.1, -0.1])  # only 2
        mult, reason = _symbol_heat_guard("TESTUSDT")
        self.assertEqual(mult, 1.0)

    @override_settings(
        SYMBOL_HEAT_GUARD_ENABLED=True,
        SYMBOL_HEAT_GUARD_WINDOW=7,
        SYMBOL_HEAT_GUARD_WR_NEUTRAL=0.50,
        SYMBOL_HEAT_GUARD_WR_FLOOR=0.25,
        SYMBOL_HEAT_GUARD_MIN_RISK_MULT=0.35,
        SYMBOL_HEAT_GUARD_MIN_TRADES=3,
    )
    def test_good_wr_no_reduction(self):
        # 5 wins, 2 losses = WR 71%
        self._create_reports(self.inst, [0.1, 0.1, -0.05, 0.1, 0.1, -0.05, 0.1])
        mult, reason = _symbol_heat_guard("TESTUSDT")
        self.assertEqual(mult, 1.0)
        self.assertEqual(reason, "")

    @override_settings(
        SYMBOL_HEAT_GUARD_ENABLED=True,
        SYMBOL_HEAT_GUARD_WINDOW=7,
        SYMBOL_HEAT_GUARD_WR_NEUTRAL=0.50,
        SYMBOL_HEAT_GUARD_WR_FLOOR=0.25,
        SYMBOL_HEAT_GUARD_MIN_RISK_MULT=0.35,
        SYMBOL_HEAT_GUARD_MIN_TRADES=3,
    )
    def test_all_losses_returns_floor(self):
        # 0 wins, 7 losses = WR 0%
        self._create_reports(self.inst, [-0.1] * 7)
        mult, reason = _symbol_heat_guard("TESTUSDT")
        self.assertEqual(mult, 0.35)
        self.assertIn("heat_guard", reason)
        self.assertIn("0%", reason)

    @override_settings(
        SYMBOL_HEAT_GUARD_ENABLED=True,
        SYMBOL_HEAT_GUARD_WINDOW=7,
        SYMBOL_HEAT_GUARD_WR_NEUTRAL=0.50,
        SYMBOL_HEAT_GUARD_WR_FLOOR=0.25,
        SYMBOL_HEAT_GUARD_MIN_RISK_MULT=0.35,
        SYMBOL_HEAT_GUARD_MIN_TRADES=3,
    )
    def test_intermediate_wr_interpolates(self):
        # 2 wins, 5 losses in window of 7 = WR 28.6%
        # WR is between floor (0.25) and neutral (0.50)
        self._create_reports(self.inst, [0.1, -0.1, -0.1, -0.1, 0.1, -0.1, -0.1])
        mult, reason = _symbol_heat_guard("TESTUSDT")
        self.assertGreater(mult, 0.35)
        self.assertLess(mult, 1.0)
        self.assertIn("heat_guard", reason)

    @override_settings(
        SYMBOL_HEAT_GUARD_ENABLED=True,
        SYMBOL_HEAT_GUARD_WINDOW=7,
        SYMBOL_HEAT_GUARD_WR_NEUTRAL=0.50,
        SYMBOL_HEAT_GUARD_WR_FLOOR=0.25,
        SYMBOL_HEAT_GUARD_MIN_RISK_MULT=0.35,
        SYMBOL_HEAT_GUARD_MIN_TRADES=3,
    )
    def test_exactly_neutral_wr_no_reduction(self):
        # 3 wins, 3 losses in window of 6 (< 7, but >= min 3) = WR 50%
        self._create_reports(self.inst, [0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
        mult, reason = _symbol_heat_guard("TESTUSDT")
        self.assertEqual(mult, 1.0)

    @override_settings(
        SYMBOL_HEAT_GUARD_ENABLED=True,
        SYMBOL_HEAT_GUARD_WINDOW=5,
        SYMBOL_HEAT_GUARD_WR_NEUTRAL=0.50,
        SYMBOL_HEAT_GUARD_WR_FLOOR=0.25,
        SYMBOL_HEAT_GUARD_MIN_RISK_MULT=0.35,
        SYMBOL_HEAT_GUARD_MIN_TRADES=3,
    )
    def test_window_limits_lookback(self):
        # 10 reports, window=5 → only uses last 5
        # last 5: 4 losses, 1 win = WR 20% (below floor)
        self._create_reports(
            self.inst,
            [-0.1, -0.1, -0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        )
        mult, reason = _symbol_heat_guard("TESTUSDT")
        self.assertEqual(mult, 0.35)

    @override_settings(
        SYMBOL_HEAT_GUARD_ENABLED=True,
        SYMBOL_HEAT_GUARD_WINDOW=7,
        SYMBOL_HEAT_GUARD_WR_NEUTRAL=0.50,
        SYMBOL_HEAT_GUARD_WR_FLOOR=0.25,
        SYMBOL_HEAT_GUARD_MIN_RISK_MULT=0.35,
        SYMBOL_HEAT_GUARD_MIN_TRADES=3,
    )
    def test_recovery_restores_full_risk(self):
        """After a bad streak, winning trades restore full risk."""
        # first: bad streak
        self._create_reports(self.inst, [-0.1] * 7)
        mult1, _ = _symbol_heat_guard("TESTUSDT")
        self.assertEqual(mult1, 0.35)

        # clear and add recovery
        from execution.models import OperationReport
        OperationReport.objects.filter(instrument=self.inst).delete()
        self._create_reports(self.inst, [0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1])
        mult2, _ = _symbol_heat_guard("TESTUSDT")
        self.assertEqual(mult2, 1.0)
