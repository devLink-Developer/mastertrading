from django.test import TestCase, override_settings

from execution.ai_entry_gate import evaluate_ai_entry_gate
from execution.ai_exit_gate import evaluate_ai_exit_gate
from execution.tasks import _bull_short_retrace_precheck
from signals.models import StrategyConfig
from signals.runtime_overrides import (
    RUNTIME_OVERRIDES_VERSION,
    invalidate_runtime_overrides_cache,
)


class RuntimeOverridesIntegrationTest(TestCase):
    def tearDown(self):
        invalidate_runtime_overrides_cache()
        super().tearDown()

    @override_settings(
        REGIME_BULL_SHORT_RETRACE_STRICT_ENABLED=True,
        REGIME_BULL_SHORT_RETRACE_MIN_SCORE=0.88,
        REGIME_BULL_SHORT_RETRACE_MIN_ALLOWED_MODULES=1,
        REGIME_BULL_SHORT_RETRACE_ALLOWED_MODULES={"meanrev", "smc", "carry"},
    )
    def test_db_override_can_harden_bull_short_retrace_rules(self):
        StrategyConfig.objects.create(
            name="REGIME_BULL_SHORT_RETRACE_MIN_ALLOWED_MODULES",
            version=RUNTIME_OVERRIDES_VERSION,
            enabled=True,
            params_json={"value": 2},
        )
        StrategyConfig.objects.create(
            name="REGIME_BULL_SHORT_RETRACE_ALLOWED_MODULES",
            version=RUNTIME_OVERRIDES_VERSION,
            enabled=True,
            params_json={"value": ["meanrev", "smc"]},
        )
        invalidate_runtime_overrides_cache()

        ok, reason = _bull_short_retrace_precheck(
            symbol="BTCUSDT",
            strategy_name="alloc_short",
            signal_direction="short",
            regime_bias="bull",
            sig_score=0.93,
            sig_payload={"reasons": {"module_rows": [{"module": "carry"}, {"module": "trend"}]}},
        )
        self.assertFalse(ok)
        self.assertIn("bull_short_low_retrace_modules", reason)

    @override_settings(AI_ENTRY_GATE_ENABLED=True)
    def test_db_override_can_disable_ai_entry_gate_globally(self):
        StrategyConfig.objects.create(
            name="AI_ENTRY_GATE_ENABLED",
            version=RUNTIME_OVERRIDES_VERSION,
            enabled=True,
            params_json={"value": False},
        )
        invalidate_runtime_overrides_cache()

        allow, risk_mult, reason, meta = evaluate_ai_entry_gate(
            account_ai_enabled=True,
            account_ai_config_id=None,
            account_owner_id=None,
            account_alias="rortigoza",
            account_service="bingx",
            symbol="BTCUSDT",
            strategy_name="alloc_long",
            signal_direction="long",
            sig_score=0.91,
            atr=0.01,
            spread_bps=5.0,
            sl_pct=0.01,
            session_name="london",
            sig_payload={},
        )
        self.assertTrue(allow)
        self.assertEqual(risk_mult, 1.0)
        self.assertEqual(reason, "ai_gate_disabled_global")
        self.assertEqual(meta.get("account_alias"), "rortigoza")

    @override_settings(AI_EXIT_GATE_ENABLED=True)
    def test_db_override_can_disable_ai_exit_gate_globally(self):
        StrategyConfig.objects.create(
            name="AI_EXIT_GATE_ENABLED",
            version=RUNTIME_OVERRIDES_VERSION,
            enabled=True,
            params_json={"value": False},
        )
        invalidate_runtime_overrides_cache()

        should_close, reason, meta = evaluate_ai_exit_gate(
            account_ai_enabled=True,
            account_ai_config_id=None,
            account_owner_id=None,
            account_alias="eudy",
            account_service="bingx",
            symbol="ETHUSDT",
            strategy_name="alloc_long",
            position_direction="long",
            sig_score=0.91,
            atr=0.01,
            spread_bps=4.0,
            tp_pct=0.02,
            sl_pct=0.01,
            pnl_pct_gross=0.015,
            pnl_pct_gate=0.015,
            r_multiple=1.5,
            remaining_tp_pct=0.003,
            position_age_min=35.0,
            session_name="london",
            sig_payload={},
        )
        self.assertFalse(should_close)
        self.assertEqual(reason, "ai_exit_disabled_global")
        self.assertEqual(meta.get("account_alias"), "eudy")
