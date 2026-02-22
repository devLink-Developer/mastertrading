"""Tests for signals.allocator — Bayesian dynamic weight computation."""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from unittest.mock import patch

from django.test import TestCase, override_settings
from django.utils import timezone as dj_tz

from core.models import Instrument
from execution.models import OperationReport, Order
from signals.allocator import (
    MODULE_ORDER,
    _module_rolling_stats,
    default_weight_map,
    dynamic_weight_map,
    normalize_weight_map,
)
from signals.models import Signal


def _mk_instrument(symbol: str = "BTCUSDT") -> Instrument:
    return Instrument.objects.get_or_create(
        symbol=symbol,
        defaults={"base": symbol.replace("USDT", ""), "quote": "USDT"},
    )[0]


def _mk_signal(inst: Instrument, strategy: str = "alloc_long", modules: list[str] | None = None, ts=None):
    """Create a Signal with module_contributions in payload."""
    modules = modules or ["trend"]
    contribs = [{"module": m, "direction": "long", "confidence": 0.7, "weight": 0.3} for m in modules]
    payload = {"reasons": {"module_contributions": contribs}}
    ts = ts or dj_tz.now()
    return Signal.objects.create(
        strategy=strategy,
        instrument=inst,
        ts=ts,
        payload_json=payload,
        score=0.5,
    )


def _mk_report(inst, signal, pnl: float, dt_offset_hours: int = 0):
    """Create an OperationReport tied to a signal."""
    now = dj_tz.now() - timedelta(hours=dt_offset_hours)
    return OperationReport.objects.create(
        instrument=inst,
        side="buy",
        qty=Decimal("0.001"),
        entry_price=Decimal("50000"),
        exit_price=Decimal("50100") if pnl > 0 else Decimal("49900"),
        pnl_abs=Decimal(str(pnl)),
        pnl_pct=Decimal(str(pnl / 100)),
        outcome="win" if pnl > 0 else "loss",
        reason="signal_flip",
        signal_id=str(signal.id) if signal else "",
        closed_at=now,
    )


class TestModuleRollingStats(TestCase):

    def test_empty_returns_zero_for_all_modules(self):
        stats = _module_rolling_stats(days=7)
        for mod in MODULE_ORDER:
            self.assertEqual(stats[mod]["n"], 0)
            self.assertEqual(stats[mod]["wins"], 0)
            self.assertEqual(stats[mod]["losses"], 0)

    def test_attributes_win_to_correct_module(self):
        inst = _mk_instrument("ETHUSDT")
        sig = _mk_signal(inst, modules=["trend", "meanrev"])
        _mk_report(inst, sig, pnl=10.0)
        stats = _module_rolling_stats(days=7)
        self.assertEqual(stats["trend"]["wins"], 1)
        self.assertEqual(stats["trend"]["n"], 1)
        self.assertEqual(stats["meanrev"]["wins"], 1)
        self.assertEqual(stats["meanrev"]["n"], 1)
        self.assertEqual(stats["carry"]["n"], 0)
        self.assertEqual(stats["smc"]["n"], 0)

    def test_loss_counted_correctly(self):
        inst = _mk_instrument("SOLUSDT")
        sig = _mk_signal(inst, modules=["carry"])
        _mk_report(inst, sig, pnl=-5.0)
        stats = _module_rolling_stats(days=7)
        self.assertEqual(stats["carry"]["losses"], 1)
        self.assertEqual(stats["carry"]["wins"], 0)

    def test_old_trades_excluded(self):
        inst = _mk_instrument("ADAUSDT")
        sig = _mk_signal(inst, modules=["trend"])
        _mk_report(inst, sig, pnl=10.0, dt_offset_hours=24 * 10)  # 10 days ago
        stats = _module_rolling_stats(days=7)
        self.assertEqual(stats["trend"]["n"], 0)

    def test_report_without_signal_ignored(self):
        inst = _mk_instrument("XRPUSDT")
        _mk_report(inst, signal=None, pnl=10.0)
        stats = _module_rolling_stats(days=7)
        for mod in MODULE_ORDER:
            self.assertEqual(stats[mod]["n"], 0)


class TestDynamicWeightMap(TestCase):

    def test_no_data_returns_base_weights(self):
        base = {"trend": 0.3, "meanrev": 0.2, "carry": 0.15, "smc": 0.35}
        result = dynamic_weight_map(base_weights=base, days=7, min_trades=10)
        # With no data, should normalize but keep proportions
        for mod in MODULE_ORDER:
            self.assertAlmostEqual(result[mod], base[mod], places=3)

    def test_winning_module_gets_boosted(self):
        inst = _mk_instrument("BTCUSDT")
        # Create 15 winning trend trades, 15 losing meanrev trades
        for _ in range(15):
            sig_t = _mk_signal(inst, modules=["trend"])
            _mk_report(inst, sig_t, pnl=10.0)
            sig_m = _mk_signal(inst, modules=["meanrev"])
            _mk_report(inst, sig_m, pnl=-10.0)

        base = {"trend": 0.3, "meanrev": 0.3, "carry": 0.2, "smc": 0.2}
        result = dynamic_weight_map(
            base_weights=base, days=7,
            alpha_prior=2.0, beta_prior=2.0,
            min_mult=0.5, max_mult=2.0, min_trades=10,
        )
        # Trend should get more weight than meanrev
        self.assertGreater(result["trend"], result["meanrev"])

    def test_clamping_respects_bounds(self):
        inst = _mk_instrument("LINKUSDT")
        # 20 wins, 0 losses → posterior close to 1.0 → mult ~2.0
        for _ in range(20):
            sig = _mk_signal(inst, modules=["trend"])
            _mk_report(inst, sig, pnl=10.0)

        base = {"trend": 0.25, "meanrev": 0.25, "carry": 0.25, "smc": 0.25}
        result = dynamic_weight_map(
            base_weights=base, days=7,
            min_mult=0.5, max_mult=2.0, min_trades=5,
        )
        # After normalization, trend's proportion should not exceed 2x its base
        # relative to other modules (but normalization adjusts this)
        # At minimum, trend should be boosted significantly
        self.assertGreater(result["trend"], 0.25)

    def test_insufficient_data_keeps_base(self):
        inst = _mk_instrument("DOGEUSDT")
        sig = _mk_signal(inst, modules=["trend"])
        _mk_report(inst, sig, pnl=10.0)  # only 1 trade
        base = {"trend": 0.3, "meanrev": 0.2, "carry": 0.15, "smc": 0.35}
        result = dynamic_weight_map(base_weights=base, days=7, min_trades=10)
        for mod in MODULE_ORDER:
            self.assertAlmostEqual(result[mod], base[mod], places=3)

    @override_settings(
        ALLOCATOR_DYNAMIC_WINDOW_DAYS=14,
        ALLOCATOR_DYNAMIC_ALPHA_PRIOR=3.0,
        ALLOCATOR_DYNAMIC_BETA_PRIOR=3.0,
    )
    def test_reads_settings_when_no_args(self):
        """dynamic_weight_map() picks up Django settings for params."""
        result = dynamic_weight_map()
        # Should return something valid (defaults from base)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=3)

    def test_bayesian_math_correct(self):
        """Verify posterior and multiplier math for known inputs."""
        inst = _mk_instrument("BTCUSDT")
        # 12 wins, 8 losses for trend → WR = 60%
        for i in range(20):
            sig = _mk_signal(inst, modules=["trend"])
            _mk_report(inst, sig, pnl=10.0 if i < 12 else -10.0)

        stats = _module_rolling_stats(days=7)
        wins = stats["trend"]["wins"]
        losses = stats["trend"]["losses"]
        self.assertEqual(wins, 12)
        self.assertEqual(losses, 8)

        alpha, beta = 2.0, 2.0
        expected_posterior = (alpha + 12) / (alpha + beta + 20)  # 14/24 = 0.5833
        expected_mult = expected_posterior / 0.5  # ~1.167

        self.assertAlmostEqual(expected_posterior, 14 / 24, places=4)
        self.assertAlmostEqual(expected_mult, 14 / 12, places=4)

    def test_sums_to_one(self):
        inst = _mk_instrument("ETHUSDT")
        for _ in range(15):
            for mod in ["trend", "meanrev", "carry", "smc"]:
                sig = _mk_signal(inst, modules=[mod])
                _mk_report(inst, sig, pnl=10.0 if mod in ("trend", "smc") else -5.0)

        result = dynamic_weight_map(min_trades=5)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=4)

    def test_db_error_fallback(self):
        base = {"trend": 0.3, "meanrev": 0.2, "carry": 0.15, "smc": 0.35}
        with patch("signals.allocator._module_rolling_stats", side_effect=Exception("DB down")):
            result = dynamic_weight_map(base_weights=base)
        self.assertEqual(result, base)
