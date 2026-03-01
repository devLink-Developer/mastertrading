from unittest.mock import patch

from django.test import TestCase

from risk.drawdown_state import (
    compute_drawdown,
    mark_drawdown_event_emitted,
    should_emit_drawdown_event,
)
from risk.models import DrawdownBaseline


class DrawdownStateTest(TestCase):
    def test_compute_drawdown_creates_and_updates_baseline(self):
        baseline0, dd0 = compute_drawdown(
            risk_ns="stack_a",
            period_type="daily",
            period_key="2026-03-01",
            equity=1000.0,
        )
        self.assertIsNotNone(baseline0)
        self.assertAlmostEqual(dd0, 0.0, places=8)

        baseline1, dd1 = compute_drawdown(
            risk_ns="stack_a",
            period_type="daily",
            period_key="2026-03-01",
            equity=900.0,
        )
        self.assertIsNotNone(baseline1)
        assert baseline0 is not None
        assert baseline1 is not None
        self.assertEqual(baseline0.id, baseline1.id)
        self.assertAlmostEqual(dd1, -0.1, places=8)

    def test_emit_marking_and_delta_gate(self):
        baseline, dd = compute_drawdown(
            risk_ns="stack_b",
            period_type="weekly",
            period_key="2026-W09",
            equity=1000.0,
        )
        assert baseline is not None
        self.assertTrue(should_emit_drawdown_event(baseline, -0.05, min_delta=0.01))
        mark_drawdown_event_emitted(baseline, -0.05)
        baseline.refresh_from_db()
        self.assertFalse(should_emit_drawdown_event(baseline, -0.055, min_delta=0.01))
        self.assertTrue(should_emit_drawdown_event(baseline, -0.07, min_delta=0.01))

    def test_unique_namespace_period(self):
        DrawdownBaseline.objects.create(
            risk_namespace="stack_c",
            period_type=DrawdownBaseline.PeriodType.DAILY,
            period_key="2026-03-01",
            start_equity=1000,
            last_equity=1000,
            last_dd=0,
        )
        self.assertEqual(
            DrawdownBaseline.objects.filter(
                risk_namespace="stack_c",
                period_type=DrawdownBaseline.PeriodType.DAILY,
                period_key="2026-03-01",
            ).count(),
            1,
        )

    def test_baseline_continues_across_cache_miss_simulated_restart(self):
        with patch("risk.drawdown_state._redis_client", return_value=None):
            baseline_a, dd_a = compute_drawdown(
                risk_ns="stack_restart",
                period_type="daily",
                period_key="2026-03-02",
                equity=1000.0,
            )
        self.assertIsNotNone(baseline_a)
        self.assertAlmostEqual(dd_a, 0.0, places=8)

        # Simulate a second cycle after restart/cache miss: baseline must come from DB.
        with patch("risk.drawdown_state._redis_client", return_value=None):
            baseline_b, dd_b = compute_drawdown(
                risk_ns="stack_restart",
                period_type="daily",
                period_key="2026-03-02",
                equity=930.0,
            )
        assert baseline_a is not None
        assert baseline_b is not None
        self.assertEqual(baseline_a.id, baseline_b.id)
        self.assertAlmostEqual(dd_b, -0.07, places=8)
