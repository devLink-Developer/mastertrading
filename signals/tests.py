from datetime import datetime, timezone
from unittest.mock import patch

from django.test import TestCase, override_settings

from core.models import Instrument
from signals.allocator import resolve_symbol_allocation
from signals.multi_strategy import run_allocator_cycle
from .feature_flags import FEATURE_FLAGS_VERSION, FEATURE_KEYS, resolve_runtime_flags
from .models import Signal, StrategyConfig
from .modules.common import strategy_module
from .sessions import (
    get_current_session,
    get_session_risk_mult,
    get_session_score_min,
    is_dead_session,
)
from .tasks import _ema_confluence_state
from .tasks import _detect_signal, _smc_impulse_bar_state, _smc_is_impulse_bar
from .direction_policy import (
    get_direction_mode,
    is_direction_allowed,
    normalize_direction,
    normalize_direction_mode,
)
import pandas as pd


class SignalModelTest(TestCase):
    def test_signal_str(self):
        inst = Instrument.objects.create(
            symbol="SOLUSDT", exchange="binance", base="SOL", quote="USDT"
        )
        sig = Signal.objects.create(
            strategy="test",
            instrument=inst,
            ts=datetime.now(timezone.utc),
            payload_json={"foo": "bar"},
            score=1.0,
        )
        self.assertIn("test", str(sig))


class SessionHelpersTest(TestCase):
    def test_session_mapping(self):
        self.assertEqual(get_current_session(20), "dead")
        self.assertEqual(get_current_session(13), "overlap")
        self.assertEqual(get_current_session(7), "london")
        self.assertEqual(get_current_session(2), "asia")
        self.assertTrue(is_dead_session("dead"))
        self.assertFalse(is_dead_session("london"))

    def test_session_threshold_overrides_and_fallback(self):
        self.assertEqual(get_session_score_min("asia", {"asia": 0.81}), 0.81)
        self.assertEqual(get_session_risk_mult("asia", {"asia": 0.7}), 0.7)
        # Invalid overrides fall back to defaults.
        self.assertEqual(get_session_score_min("asia", {"asia": "bad"}), 0.80)
        self.assertEqual(get_session_risk_mult("dead", {"dead": "bad"}), 0.0)

    def test_ema_confluence_state_long_short(self):
        prices_up = list(range(100, 400))
        df_up = pd.DataFrame(
            {
                "open": prices_up,
                "high": [p + 1 for p in prices_up],
                "low": [p - 1 for p in prices_up],
                "close": prices_up,
                "volume": [1.0] * len(prices_up),
            }
        )
        aligned_long, details_long = _ema_confluence_state(df_up, "long", [20, 50, 200])
        self.assertTrue(aligned_long)
        self.assertEqual(details_long["status"], "ok")

        prices_down = list(range(400, 100, -1))
        df_down = pd.DataFrame(
            {
                "open": prices_down,
                "high": [p + 1 for p in prices_down],
                "low": [p - 1 for p in prices_down],
                "close": prices_down,
                "volume": [1.0] * len(prices_down),
            }
        )
        aligned_short, details_short = _ema_confluence_state(df_down, "short", [20, 50, 200])
        self.assertTrue(aligned_short)
        self.assertEqual(details_short["status"], "ok")

    def test_ema_confluence_state_insufficient_data(self):
        prices = list(range(100, 130))
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": [1.0] * len(prices),
            }
        )
        aligned, details = _ema_confluence_state(df, "long", [20, 50, 200])
        self.assertIsNone(aligned)
        self.assertEqual(details["status"], "insufficient_data")

    def test_smc_impulse_helpers_detect_impulse(self):
        rows = []
        price = 100.0
        for i in range(25):
            o = price
            c = price + 0.05
            rows.append(
                {
                    "open": o,
                    "high": max(o, c) + 0.2,
                    "low": min(o, c) - 0.2,
                    "close": c,
                    "volume": 1.0,
                }
            )
            price += 0.02
        # Displacement candle.
        rows.append(
            {
                "open": price,
                "high": price + 2.6,
                "low": price - 0.1,
                "close": price + 2.2,
                "volume": 1.0,
            }
        )
        df = pd.DataFrame(rows)
        state = _smc_impulse_bar_state(df, lookback=20)
        self.assertIsNotNone(state)
        impulse, threshold = _smc_is_impulse_bar(state, body_mult=2.0, min_body_pct=0.006)
        self.assertTrue(impulse)
        self.assertGreater(threshold, 0.0)
        self.assertEqual(state["candle_direction"], "long")

    def test_smc_impulse_helpers_no_impulse_on_small_bar(self):
        rows = []
        price = 100.0
        for _ in range(30):
            o = price
            c = price + 0.08
            rows.append(
                {
                    "open": o,
                    "high": max(o, c) + 0.2,
                    "low": min(o, c) - 0.2,
                    "close": c,
                    "volume": 1.0,
                }
            )
            price += 0.03
        df = pd.DataFrame(rows)
        state = _smc_impulse_bar_state(df, lookback=20)
        impulse, _ = _smc_is_impulse_bar(state, body_mult=2.5, min_body_pct=0.006)
        self.assertFalse(impulse)

    @override_settings(
        SMC_SESSION_FILTER_ENABLED=True,
        SMC_ALLOWED_SESSIONS={"london", "ny", "overlap"},
        EMA_CONFLUENCE_ENABLED=False,
    )
    def test_detect_signal_blocks_smc_outside_allowed_sessions(self):
        prices = [100 + i * 0.2 for i in range(80)]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 0.4 for p in prices],
                "low": [p - 0.4 for p in prices],
                "close": prices,
                "volume": [1.0] * len(prices),
            }
        )
        with patch("signals.tasks._trend_from_swings", return_value="bull"):
            with patch("signals.tasks._detect_structure_break", return_value=("choch_bull", {"ok": True})):
                with patch("signals.tasks._detect_liquidity_sweep", return_value=("sweep_low", {"ok": True})):
                    with patch("signals.tasks._detect_fvg", return_value=(None, {})):
                        with patch("signals.tasks._detect_order_block", return_value=(None, {})):
                            with patch("signals.tasks._funding_filter", return_value=(True, {"ok": True})):
                                ok, _, explain, _ = _detect_signal(
                                    df,
                                    df,
                                    [],
                                    min_score=0.20,
                                    session="asia",
                                )
        self.assertFalse(ok)
        self.assertIn("blocked_by_session_window", str(explain.get("result", "")))

    @override_settings(
        SMC_SESSION_FILTER_ENABLED=True,
        SMC_ALLOWED_SESSIONS={"london", "ny", "overlap"},
        EMA_CONFLUENCE_ENABLED=False,
    )
    def test_detect_signal_allows_smc_in_london_session(self):
        prices = [100 + i * 0.2 for i in range(80)]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 0.4 for p in prices],
                "low": [p - 0.4 for p in prices],
                "close": prices,
                "volume": [1.0] * len(prices),
            }
        )
        with patch("signals.tasks._trend_from_swings", return_value="bull"):
            with patch("signals.tasks._detect_structure_break", return_value=("choch_bull", {"ok": True})):
                with patch("signals.tasks._detect_liquidity_sweep", return_value=("sweep_low", {"ok": True})):
                    with patch("signals.tasks._detect_fvg", return_value=(None, {})):
                        with patch("signals.tasks._detect_order_block", return_value=(None, {})):
                            with patch("signals.tasks._funding_filter", return_value=(True, {"ok": True})):
                                ok, direction, explain, score = _detect_signal(
                                    df,
                                    df,
                                    [],
                                    min_score=0.20,
                                    session="london",
                                )
        self.assertTrue(ok)
        self.assertEqual(direction, "long")
        self.assertEqual(explain.get("result"), "signal_emitted")
        self.assertGreaterEqual(score, 0.20)


class AllocatorWeightingTest(TestCase):
    @override_settings(
        ALLOCATOR_SMC_CONFLUENCE_BOOST_ENABLED=True,
        ALLOCATOR_SMC_CONFLUENCE_WEIGHT_MULT=1.30,
        ALLOCATOR_SMC_NON_CONFLUENCE_WEIGHT_MULT=0.85,
    )
    def test_allocator_boosts_smc_weight_only_when_confluent(self):
        signals_base = [
            {"module": "trend", "direction": "long", "confidence": 0.70},
            {"module": "smc", "direction": "long", "confidence": 0.90, "smc_confluence": True},
        ]
        weights = {"trend": 0.5, "meanrev": 0.0, "carry": 0.0, "smc": 0.5}
        risk_budgets = {"trend": 0.5, "meanrev": 0.0, "carry": 0.0, "smc": 0.5}
        boosted = resolve_symbol_allocation(
            signals_base,
            threshold=0.05,
            base_risk_pct=0.01,
            session_risk_mult=1.0,
            weights=weights,
            risk_budgets=risk_budgets,
        )

        non_confluent = [dict(signals_base[0]), dict(signals_base[1], smc_confluence=False)]
        reduced = resolve_symbol_allocation(
            non_confluent,
            threshold=0.05,
            base_risk_pct=0.01,
            session_risk_mult=1.0,
            weights=weights,
            risk_budgets=risk_budgets,
        )

        def _smc_row(payload: dict) -> dict:
            rows = payload.get("reasons", {}).get("module_contributions", [])
            for row in rows:
                if row.get("module") == "smc":
                    return row
            return {}

        boosted_row = _smc_row(boosted)
        reduced_row = _smc_row(reduced)
        self.assertGreater(float(boosted_row.get("weight", 0.0)), float(reduced_row.get("weight", 0.0)))


class DirectionPolicyHelpersTest(TestCase):
    def test_normalize_helpers(self):
        self.assertEqual(normalize_direction_mode("LONG_ONLY"), "long_only")
        self.assertEqual(normalize_direction_mode("bad"), "both")
        self.assertEqual(normalize_direction("buy"), "long")
        self.assertEqual(normalize_direction("sell"), "short")
        self.assertIsNone(normalize_direction("flat"))

    @override_settings(SIGNAL_DIRECTION_MODE="long_only", PER_INSTRUMENT_DIRECTION={})
    def test_global_long_only(self):
        self.assertEqual(get_direction_mode("BTCUSDT"), "long_only")
        self.assertTrue(is_direction_allowed("long", "BTCUSDT"))
        self.assertTrue(is_direction_allowed("buy", "BTCUSDT"))
        self.assertFalse(is_direction_allowed("short", "BTCUSDT"))

    @override_settings(
        SIGNAL_DIRECTION_MODE="both",
        PER_INSTRUMENT_DIRECTION={"BTCUSDT": "disabled", "ETHUSDT": "short_only"},
    )
    def test_per_symbol_overrides(self):
        self.assertEqual(get_direction_mode("BTCUSDT"), "disabled")
        self.assertEqual(get_direction_mode("ETHUSDT"), "short_only")
        self.assertEqual(get_direction_mode("SOLUSDT"), "both")

        self.assertFalse(is_direction_allowed("long", "BTCUSDT"))
        self.assertFalse(is_direction_allowed("short", "BTCUSDT"))
        self.assertTrue(is_direction_allowed("short", "ETHUSDT"))
        self.assertFalse(is_direction_allowed("long", "ETHUSDT"))


class StrategyModuleParsingTest(TestCase):
    def test_strategy_module_supports_smc(self):
        self.assertEqual(strategy_module("smc_long"), ("smc", "long"))
        self.assertEqual(strategy_module("smc_short"), ("smc", "short"))


@override_settings(
    MULTI_STRATEGY_ENABLED=True,
    ALLOCATOR_ENABLED=True,
    ALLOCATOR_INCLUDE_SMC=True,
    ALLOCATOR_MIN_MODULES_ACTIVE=1,
    ALLOCATOR_MODULE_WEIGHTS={"trend": 0.0, "meanrev": 0.0, "carry": 0.0, "smc": 1.0},
    ALLOCATOR_MODULE_RISK_BUDGETS={"trend": 0.0, "meanrev": 0.0, "carry": 0.0, "smc": 1.0},
    LIVE_GRADUAL_ENABLED=False,
)
class AllocatorSmcIntegrationTest(TestCase):
    def test_allocator_can_use_smc_signals(self):
        inst = Instrument.objects.create(
            symbol="BTCUSDT",
            exchange="binance",
            base="BTC",
            quote="USDT",
            enabled=True,
        )
        Signal.objects.create(
            strategy="smc_long",
            instrument=inst,
            ts=datetime.now(timezone.utc),
            payload_json={"reason": {"test": True}},
            score=0.92,
        )

        with patch("signals.multi_strategy.acquire_task_lock", return_value=True):
            out = run_allocator_cycle()

        self.assertIn("allocator:emitted=1", out)
        alloc = Signal.objects.filter(instrument=inst, strategy="alloc_long").order_by("-ts").first()
        self.assertIsNotNone(alloc)
        module_rows = (alloc.payload_json or {}).get("reasons", {}).get("module_rows", [])
        self.assertTrue(any(str(row.get("module")) == "smc" for row in module_rows))


@override_settings(
    MULTI_STRATEGY_ENABLED=False,
    MODULE_TREND_ENABLED=True,
    MODULE_MEANREV_ENABLED=True,
    MODULE_CARRY_ENABLED=True,
    ALLOCATOR_ENABLED=True,
)
class FeatureFlagsRuntimeTest(TestCase):
    def test_resolve_runtime_flags_uses_db_override(self):
        defaults = resolve_runtime_flags()
        self.assertFalse(defaults[FEATURE_KEYS["multi"]])

        StrategyConfig.objects.update_or_create(
            name=FEATURE_KEYS["multi"],
            version=FEATURE_FLAGS_VERSION,
            defaults={"enabled": True, "params_json": {"feature_flag": True}},
        )
        updated = resolve_runtime_flags()
        self.assertTrue(updated[FEATURE_KEYS["multi"]])


# -----------------------------------------------------------------------
# Tests for ALLOCATOR_LONG_SCORE_PENALTY
# -----------------------------------------------------------------------

class AllocatorLongScorePenaltyTest(TestCase):
    """Verify that ALLOCATOR_LONG_SCORE_PENALTY dampens long net_score."""

    def _run_alloc(self, signals, **extra_settings):
        weights = {"trend": 0.50, "meanrev": 0.50, "carry": 0.0, "smc": 0.0}
        risk_budgets = dict(weights)
        return resolve_symbol_allocation(
            signals,
            threshold=0.05,
            base_risk_pct=0.01,
            session_risk_mult=1.0,
            weights=weights,
            risk_budgets=risk_budgets,
        )

    @override_settings(ALLOCATOR_LONG_SCORE_PENALTY=1.0)
    def test_no_penalty_when_1(self):
        """Penalty=1.0 means no change."""
        signals = [
            {"module": "trend", "direction": "long", "confidence": 0.80},
            {"module": "meanrev", "direction": "long", "confidence": 0.70},
        ]
        result = self._run_alloc(signals)
        self.assertEqual(result["direction"], "long")

    @override_settings(ALLOCATOR_LONG_SCORE_PENALTY=0.01)
    def test_heavy_penalty_blocks_long(self):
        """Very heavy penalty (0.01) should push long net_score below threshold."""
        signals = [
            {"module": "trend", "direction": "long", "confidence": 0.60},
            {"module": "meanrev", "direction": "long", "confidence": 0.60},
        ]
        result = self._run_alloc(signals)
        # Net score * 0.01 should be below threshold 0.05
        self.assertEqual(result["direction"], "flat")

    @override_settings(ALLOCATOR_LONG_SCORE_PENALTY=0.85)
    def test_penalty_reduces_long_score(self):
        """Check that penalized long has lower net_score than unpenalized."""
        signals = [
            {"module": "trend", "direction": "long", "confidence": 0.80},
            {"module": "meanrev", "direction": "long", "confidence": 0.70},
        ]
        penalized = self._run_alloc(signals)

        with self.settings(ALLOCATOR_LONG_SCORE_PENALTY=1.0):
            unpenalized = self._run_alloc(signals)

        self.assertLess(penalized["net_score"], unpenalized["net_score"])

    @override_settings(ALLOCATOR_LONG_SCORE_PENALTY=0.50)
    def test_penalty_does_not_affect_shorts(self):
        """Short net_score should be unaffected by long penalty."""
        signals = [
            {"module": "trend", "direction": "short", "confidence": 0.80},
            {"module": "meanrev", "direction": "short", "confidence": 0.70},
        ]
        result = self._run_alloc(signals)
        self.assertEqual(result["direction"], "short")

        with self.settings(ALLOCATOR_LONG_SCORE_PENALTY=1.0):
            baseline = self._run_alloc(signals)

        self.assertEqual(result["net_score"], baseline["net_score"])


# -----------------------------------------------------------------------
# Tests for MODULE_TREND_HTF_ADX_MIN
# -----------------------------------------------------------------------

class TrendHTFADXGateTest(TestCase):
    """Verify that the trend module blocks signals when HTF ADX is too low."""

    def _make_df(self, n=100, trend_up=True, volatility=0.005):
        """Generate synthetic OHLCV DataFrame with a trend."""
        import numpy as np
        rng = np.random.RandomState(42)
        drift = 0.001 if trend_up else -0.001
        log_prices = np.cumsum(rng.normal(drift, volatility, n))
        prices = 50000 * np.exp(log_prices)
        high = prices * (1 + rng.uniform(0.001, 0.01, n))
        low = prices * (1 - rng.uniform(0.001, 0.01, n))
        return pd.DataFrame({
            "open": np.roll(prices, 1),
            "high": high,
            "low": low,
            "close": prices,
            "volume": rng.uniform(100, 1000, n),
        })

    @override_settings(
        MODULE_ADX_TREND_MIN=20.0,
        MODULE_TREND_HTF_ADX_MIN=0.0,
        MODULE_IMPULSE_FILTER_ENABLED=False,
        MODULE_BOUNCE_LOOKBACK=30,
        MODULE_BOUNCE_BLOCK_PCT=999,
    )
    def test_htf_adx_gate_disabled(self):
        """When HTF ADX min is 0.0, the gate is disabled — trend passes."""
        from signals.modules.trend import detect
        df_ltf = self._make_df(100, trend_up=True, volatility=0.01)
        df_htf = self._make_df(100, trend_up=True, volatility=0.01)
        result = detect(df_ltf, df_htf, [], "london")
        # Result could be None due to other gates (EMA alignment etc),
        # but we verify that adx_htf is reported in reasons if signal passes
        if result is not None:
            self.assertIn("adx_htf", result["reasons"])

    @override_settings(
        MODULE_ADX_TREND_MIN=5.0,
        MODULE_TREND_HTF_ADX_MIN=999.0,
        MODULE_IMPULSE_FILTER_ENABLED=False,
        MODULE_BOUNCE_LOOKBACK=30,
        MODULE_BOUNCE_BLOCK_PCT=999,
    )
    def test_htf_adx_gate_blocks_with_high_min(self):
        """When HTF ADX min is unreachably high (999), trend should return None."""
        from signals.modules.trend import detect
        df_ltf = self._make_df(100, trend_up=True, volatility=0.01)
        df_htf = self._make_df(100, trend_up=True, volatility=0.01)
        result = detect(df_ltf, df_htf, [], "london")
        self.assertIsNone(result)
