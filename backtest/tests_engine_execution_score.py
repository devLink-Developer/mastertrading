from django.test import SimpleTestCase, override_settings

from backtest.engine import (
    _execution_min_signal_score,
    _passes_execution_score_gate,
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
