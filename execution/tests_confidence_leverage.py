from django.test import SimpleTestCase, override_settings

from execution.tasks import (
    _confidence_adjusted_entry_leverage,
    _ensure_entry_leverage,
)


class _DummyClient:
    def __init__(self):
        self.calls = []

    def set_leverage(self, leverage, symbol, params=None):
        self.calls.append((int(leverage), str(symbol), dict(params or {})))
        return {"ok": True}


class _DummyBingxAdapter:
    def __init__(self):
        self.client = _DummyClient()
        self.margin_mode = "cross"
        self._leverage_set_symbols = set()
        self._symbol_leverage_cache = {}

    def _map_symbol(self, symbol: str) -> str:
        return f"{symbol}/USDT:USDT" if "/" not in symbol and symbol.endswith("USDT") else symbol


class _DummyKucoinAdapter:
    def __init__(self):
        self.client = _DummyClient()
        self.margin_mode = "cross"
        self._leverage_set_symbols = set()
        self._symbol_leverage_cache = {}

    def _map_symbol(self, symbol: str) -> str:
        return f"{symbol[:-4]}/USDT:USDT" if "/" not in symbol and symbol.endswith("USDT") else symbol


class ConfidenceLeverageBoostTests(SimpleTestCase):
    @override_settings(
        CONFIDENCE_LEVERAGE_BOOST_ENABLED=False,
    )
    def test_disabled_returns_base(self):
        lev, reason = _confidence_adjusted_entry_leverage(
            base_leverage=5.0,
            strategy_name="alloc_long",
            sig_score=0.99,
            ml_prob=0.95,
            ml_enabled=True,
        )
        self.assertEqual(lev, 5.0)
        self.assertEqual(reason, "disabled")

    @override_settings(
        CONFIDENCE_LEVERAGE_BOOST_ENABLED=True,
        CONFIDENCE_LEVERAGE_ONLY_ALLOCATOR=True,
        CONFIDENCE_LEVERAGE_SCORE_THRESHOLD=0.9,
        CONFIDENCE_LEVERAGE_ML_PROB_THRESHOLD=0.7,
        CONFIDENCE_LEVERAGE_REQUIRE_BOTH=False,
        CONFIDENCE_LEVERAGE_MULT=1.3,
        CONFIDENCE_LEVERAGE_MAX=8.0,
    )
    def test_allocator_score_boost_applies_and_caps(self):
        lev, reason = _confidence_adjusted_entry_leverage(
            base_leverage=5.0,
            strategy_name="alloc_short",
            sig_score=0.95,
            ml_prob=None,
            ml_enabled=False,
        )
        self.assertEqual(reason, "score")
        self.assertAlmostEqual(lev, 6.5, places=6)

        lev_cap, _ = _confidence_adjusted_entry_leverage(
            base_leverage=7.0,
            strategy_name="alloc_short",
            sig_score=0.95,
            ml_prob=0.99,
            ml_enabled=True,
        )
        self.assertAlmostEqual(lev_cap, 8.0, places=6)

    @override_settings(
        CONFIDENCE_LEVERAGE_BOOST_ENABLED=True,
        CONFIDENCE_LEVERAGE_ONLY_ALLOCATOR=True,
        CONFIDENCE_LEVERAGE_SCORE_THRESHOLD=0.9,
        CONFIDENCE_LEVERAGE_ML_PROB_THRESHOLD=0.7,
        CONFIDENCE_LEVERAGE_REQUIRE_BOTH=True,
        CONFIDENCE_LEVERAGE_MULT=1.3,
        CONFIDENCE_LEVERAGE_MAX=8.0,
    )
    def test_require_both_needs_ml_and_score(self):
        lev1, reason1 = _confidence_adjusted_entry_leverage(
            base_leverage=5.0,
            strategy_name="alloc_long",
            sig_score=0.95,
            ml_prob=None,
            ml_enabled=False,
        )
        self.assertEqual(lev1, 5.0)
        self.assertEqual(reason1, "below_threshold")

        lev2, reason2 = _confidence_adjusted_entry_leverage(
            base_leverage=5.0,
            strategy_name="alloc_long",
            sig_score=0.95,
            ml_prob=0.8,
            ml_enabled=True,
        )
        self.assertGreater(lev2, 5.0)
        self.assertEqual(reason2, "score+ml")

    @override_settings(
        CONFIDENCE_LEVERAGE_BOOST_ENABLED=True,
        CONFIDENCE_LEVERAGE_ONLY_ALLOCATOR=True,
    )
    def test_non_allocator_is_not_boosted(self):
        lev, reason = _confidence_adjusted_entry_leverage(
            base_leverage=5.0,
            strategy_name="mod_trend_long",
            sig_score=0.99,
            ml_prob=0.99,
            ml_enabled=True,
        )
        self.assertEqual(lev, 5.0)
        self.assertEqual(reason, "non_allocator")


class EnsureEntryLeverageTests(SimpleTestCase):
    def test_sets_bingx_leverage_with_both_side(self):
        adapter = _DummyBingxAdapter()
        ok, reason = _ensure_entry_leverage(adapter, "BTCUSDT", 6.0)
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")
        self.assertEqual(len(adapter.client.calls), 1)
        lev, sym, params = adapter.client.calls[0]
        self.assertEqual(lev, 6)
        self.assertTrue(sym.endswith("/USDT:USDT"))
        self.assertIn("BTC", sym)
        self.assertEqual(params.get("side"), "BOTH")

    def test_cache_avoids_duplicate_set_calls(self):
        adapter = _DummyKucoinAdapter()
        ok1, _ = _ensure_entry_leverage(adapter, "ETHUSDT", 5.0)
        ok2, reason2 = _ensure_entry_leverage(adapter, "ETHUSDT", 5.0)
        self.assertTrue(ok1)
        self.assertTrue(ok2)
        self.assertEqual(reason2, "cached")
        self.assertEqual(len(adapter.client.calls), 1)
