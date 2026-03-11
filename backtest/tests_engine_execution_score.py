from django.test import SimpleTestCase, override_settings

from backtest.engine import (
    _execution_min_signal_score,
    _passes_execution_score_gate,
    _resolve_backtest_symbol_allocation,
)


class BacktestExecutionScoreGateTest(SimpleTestCase):
    @override_settings(EXECUTION_MIN_SIGNAL_SCORE=0.5, MIN_SIGNAL_SCORE=0.45)
    def test_uses_global_execution_score_when_session_policy_disabled(self):
        self.assertEqual(
            _execution_min_signal_score(
                "ny",
                session_policy_enabled=False,
                session_score_overrides=None,
            ),
            0.5,
        )
        self.assertEqual(
            _passes_execution_score_gate(
                0.49,
                "ny",
                session_policy_enabled=False,
                session_score_overrides=None,
            ),
            (False, 0.5),
        )
        self.assertEqual(
            _passes_execution_score_gate(
                0.5,
                "ny",
                session_policy_enabled=False,
                session_score_overrides=None,
            ),
            (True, 0.5),
        )

    def test_uses_session_score_when_policy_enabled(self):
        overrides = {"ny": 0.58}
        self.assertEqual(
            _execution_min_signal_score(
                "ny",
                session_policy_enabled=True,
                session_score_overrides=overrides,
            ),
            0.58,
        )
        self.assertEqual(
            _passes_execution_score_gate(
                0.57,
                "ny",
                session_policy_enabled=True,
                session_score_overrides=overrides,
            ),
            (False, 0.58),
        )
        self.assertEqual(
            _passes_execution_score_gate(
                0.58,
                "ny",
                session_policy_enabled=True,
                session_score_overrides=overrides,
            ),
            (True, 0.58),
        )

    def test_session_override_wins_when_policy_enabled(self):
        overrides = {"ny": 0.55}
        self.assertEqual(
            _execution_min_signal_score(
                "ny",
                session_policy_enabled=True,
                session_score_overrides=overrides,
            ),
            0.55,
        )
        self.assertEqual(
            _passes_execution_score_gate(
                0.56,
                "ny",
                session_policy_enabled=True,
                session_score_overrides=overrides,
            ),
            (True, 0.55),
        )


class BacktestAllocatorBridgeTest(SimpleTestCase):
    @override_settings(
        ALLOCATOR_MIN_MODULES_ACTIVE=2,
        ALLOCATOR_STRONG_TREND_SOLO_ENABLED=True,
        ALLOCATOR_STRONG_TREND_ADX_MIN=25.0,
        ALLOCATOR_STRONG_TREND_CONFIDENCE_MIN=0.80,
        ALLOCATOR_STRONG_TREND_ADX_MIN_BY_CONTEXT={"BTCUSDT:london": 21.0},
        ALLOCATOR_STRONG_TREND_SOLO_DISABLED_SESSIONS={"ny_open"},
    )
    def test_backtest_allocator_passes_symbol_and_session_context(self):
        alloc = _resolve_backtest_symbol_allocation(
            [
                {
                    "module": "trend",
                    "direction": "long",
                    "confidence": 0.85,
                    "reasons": {"adx_htf": 22.0},
                }
            ],
            threshold=0.05,
            base_risk_pct=0.01,
            session_risk_mult=1.0,
            weights={"trend": 1.0, "meanrev": 0.0, "carry": 0.0, "grid": 0.0, "smc": 0.0},
            risk_budgets={"trend": 1.0, "meanrev": 0.0, "carry": 0.0, "grid": 0.0, "smc": 0.0},
            symbol="BTCUSDT",
            session_name="london",
        )
        self.assertEqual(alloc["direction"], "long")
        self.assertGreater(alloc["confidence"], 0.0)

    @override_settings(
        ALLOCATOR_MIN_MODULES_ACTIVE=2,
        ALLOCATOR_STRONG_TREND_SOLO_ENABLED=True,
        ALLOCATOR_STRONG_TREND_ADX_MIN=25.0,
        ALLOCATOR_STRONG_TREND_CONFIDENCE_MIN=0.80,
        ALLOCATOR_STRONG_TREND_ADX_MIN_BY_CONTEXT={"BTCUSDT:london": 21.0},
        ALLOCATOR_STRONG_TREND_SOLO_DISABLED_SESSIONS={"ny_open"},
    )
    def test_backtest_allocator_respects_disabled_session_for_trend_solo(self):
        alloc = _resolve_backtest_symbol_allocation(
            [
                {
                    "module": "trend",
                    "direction": "long",
                    "confidence": 0.85,
                    "reasons": {"adx_htf": 22.0},
                }
            ],
            threshold=0.05,
            base_risk_pct=0.01,
            session_risk_mult=1.0,
            weights={"trend": 1.0, "meanrev": 0.0, "carry": 0.0, "grid": 0.0, "smc": 0.0},
            risk_budgets={"trend": 1.0, "meanrev": 0.0, "carry": 0.0, "grid": 0.0, "smc": 0.0},
            symbol="BTCUSDT",
            session_name="ny_open",
        )
        self.assertEqual(alloc["direction"], "flat")
        self.assertEqual(alloc["reasons"]["required_modules"], 2)
