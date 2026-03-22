from django.core.management import call_command
from django.test import TestCase, override_settings

from core.models import Instrument
from signals.allocator import resolve_symbol_allocation
from signals.models import StrategyConfig
from signals.runtime_overrides import RUNTIME_OVERRIDES_VERSION, invalidate_runtime_overrides_cache


class RuntimeOverrideCommandTest(TestCase):
    def tearDown(self):
        invalidate_runtime_overrides_cache()
        super().tearDown()

    def test_set_runtime_override_persists_json_value(self):
        call_command(
            "set_runtime_override",
            "ALLOCATOR_DIRECTION_SCORE_MULT_BY_CONTEXT",
            "--value",
            '{"ADAUSDT:ny:short":0.0}',
        )
        row = StrategyConfig.objects.get(
            version=RUNTIME_OVERRIDES_VERSION,
            name="ALLOCATOR_DIRECTION_SCORE_MULT_BY_CONTEXT",
        )
        self.assertEqual((row.params_json or {}).get("value"), {"ADAUSDT:ny:short": 0.0})


class AllocatorRuntimeOverridePenaltyTest(TestCase):
    def tearDown(self):
        StrategyConfig.objects.filter(version=RUNTIME_OVERRIDES_VERSION).delete()
        invalidate_runtime_overrides_cache()
        super().tearDown()

    @override_settings(ALLOCATOR_DIRECTION_SCORE_MULT_BY_CONTEXT={})
    def test_runtime_override_dict_penalizes_context_without_env_setting(self):
        StrategyConfig.objects.create(
            name="ALLOCATOR_DIRECTION_SCORE_MULT_BY_CONTEXT",
            version=RUNTIME_OVERRIDES_VERSION,
            enabled=True,
            params_json={"value": {"ADAUSDT:ny:short": 0.5}},
        )
        invalidate_runtime_overrides_cache()

        signals = [
            {"module": "trend", "direction": "short", "confidence": 0.06},
            {"module": "meanrev", "direction": "short", "confidence": 0.06},
        ]
        weights = {"trend": 0.50, "meanrev": 0.50, "carry": 0.0, "grid": 0.0, "smc": 0.0}
        result = resolve_symbol_allocation(
            signals,
            threshold=0.05,
            base_risk_pct=0.01,
            session_risk_mult=1.0,
            weights=weights,
            risk_budgets=dict(weights),
            symbol="ADAUSDT",
            session_name="ny",
        )

        self.assertEqual(result["direction"], "flat")
        self.assertEqual(result["reasons"].get("direction_score_context_key"), "ADAUSDT:ny:short")