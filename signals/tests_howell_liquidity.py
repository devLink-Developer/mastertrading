from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings
from django.utils import timezone

from core.models import Instrument
from signals.models import MacroLiquiditySnapshot, Signal
from signals.multi_strategy import run_allocator_cycle


def _component(
    key: str,
    *,
    score: float,
    momentum: float,
    weight: float,
    asof: str = "2026-04-15",
) -> dict:
    return {
        "key": key,
        "weight": weight,
        "score": score,
        "momentum": momentum,
        "level_z": score,
        "change_z": momentum,
        "asof": asof,
        "latest_value": 1.0,
        "source_ids": [key.upper()],
    }


class HowellLiquidityComposeTest(TestCase):
    @override_settings(
        HOWELL_LIQUIDITY_EXPANDING_SCORE_MIN=0.35,
        HOWELL_LIQUIDITY_EXPANDING_MOMENTUM_MIN=0.10,
        HOWELL_LIQUIDITY_LATE_EXPANDING_SCORE_MIN=0.05,
        HOWELL_LIQUIDITY_ROLLOVER_MOMENTUM_MAX=-0.20,
        HOWELL_LIQUIDITY_STRESS_SCORE_MAX=-0.75,
        HOWELL_LIQUIDITY_STRESS_FCI_Z_MAX=-0.90,
    )
    def test_compose_snapshot_marks_expanding(self):
        from signals.howell_liquidity import _compose_snapshot

        payload = _compose_snapshot(
            {
                "fed_net_liquidity_z": _component(
                    "fed_net_liquidity_z",
                    score=1.10,
                    momentum=0.85,
                    weight=0.55,
                ),
                "financial_conditions_z": _component(
                    "financial_conditions_z",
                    score=0.75,
                    momentum=0.30,
                    weight=0.25,
                ),
                "dollar_z": _component(
                    "dollar_z",
                    score=0.25,
                    momentum=0.18,
                    weight=0.20,
                ),
            }
        )

        self.assertEqual(payload["regime"], "expanding")
        self.assertGreater(payload["composite_score"], 0.35)
        self.assertGreater(payload["composite_momentum"], 0.10)
        self.assertGreater(payload["confidence"], 0.0)

    @override_settings(
        HOWELL_LIQUIDITY_STRESS_SCORE_MAX=-0.75,
        HOWELL_LIQUIDITY_STRESS_FCI_Z_MAX=-0.90,
    )
    def test_compose_snapshot_marks_stress_when_fci_breaks(self):
        from signals.howell_liquidity import _compose_snapshot

        payload = _compose_snapshot(
            {
                "fed_net_liquidity_z": _component(
                    "fed_net_liquidity_z",
                    score=-0.40,
                    momentum=-0.30,
                    weight=0.55,
                ),
                "financial_conditions_z": _component(
                    "financial_conditions_z",
                    score=-1.20,
                    momentum=-0.60,
                    weight=0.25,
                ),
                "dollar_z": _component(
                    "dollar_z",
                    score=-0.25,
                    momentum=-0.10,
                    weight=0.20,
                ),
            }
        )

        self.assertEqual(payload["regime"], "stress")
        self.assertLess(payload["financial_conditions_z"], -0.90)


class HowellLiquidityPersistenceTest(TestCase):
    @override_settings(HOWELL_LIQUIDITY_ENABLED=True)
    @patch("redis.from_url")
    @patch("signals.howell_liquidity._build_core_components")
    def test_refresh_persists_snapshot_and_caches(self, mock_build_components, mock_from_url):
        from signals.howell_liquidity import refresh_liquidity_snapshot

        mock_build_components.return_value = {
            "fed_net_liquidity_z": _component(
                "fed_net_liquidity_z",
                score=0.90,
                momentum=0.55,
                weight=0.55,
            ),
            "financial_conditions_z": _component(
                "financial_conditions_z",
                score=0.35,
                momentum=0.10,
                weight=0.25,
            ),
            "dollar_z": _component(
                "dollar_z",
                score=0.20,
                momentum=0.05,
                weight=0.20,
            ),
        }
        redis_client = MagicMock()
        mock_from_url.return_value = redis_client

        result = refresh_liquidity_snapshot()

        self.assertIsNotNone(result)
        self.assertEqual(MacroLiquiditySnapshot.objects.count(), 1)
        row = MacroLiquiditySnapshot.objects.get()
        self.assertEqual(row.regime, result["regime"])
        redis_client.setex.assert_called_once()


class HowellLiquidityDiagnosticTest(TestCase):
    @override_settings(
        HOWELL_LIQUIDITY_ENABLED=True,
        HOWELL_LIQUIDITY_PREVIEW_BASE_MULT_LATE_EXPANDING=0.91,
        INSTRUMENT_TIER_MAP={"BTCUSDT": "base", "DOGEUSDT": "alt"},
    )
    def test_shadow_diagnostic_uses_bucket_preview_multiplier(self):
        from signals.howell_liquidity import howell_shadow_diagnostic

        diag = howell_shadow_diagnostic(
            "BTCUSDT",
            snapshot={
                "asof": timezone.now().isoformat(),
                "regime": "late_expanding",
                "confidence": 0.82,
                "composite_score": 0.48,
                "composite_momentum": 0.06,
                "details_json": {"component_count": 3, "active_components": ["fed_net_liquidity_z"]},
            },
        )

        self.assertIsNotNone(diag)
        self.assertEqual(diag["bucket"], "base")
        self.assertEqual(diag["regime"], "late_expanding")
        self.assertEqual(diag["preview_risk_mult"], 0.91)


class HowellAllocatorIntegrationTest(TestCase):
    @override_settings(
        MULTI_STRATEGY_ENABLED=True,
        MODULE_TREND_ENABLED=True,
        MODULE_MEANREV_ENABLED=True,
        MODULE_CARRY_ENABLED=False,
        MODULE_GRID_ENABLED=False,
        MODULE_MICROVOL_ENABLED=False,
        ALLOCATOR_ENABLED=True,
        FEATURE_FLAGS_SOURCE="env",
        HOWELL_LIQUIDITY_ENABLED=True,
        ALLOCATOR_MIN_MODULES_ACTIVE=2,
        ALLOCATOR_NET_THRESHOLD=0.05,
        RISK_PER_TRADE_PCT=0.01,
        HMM_REGIME_ENABLED=False,
    )
    def test_allocator_attaches_howell_shadow_diag(self):
        inst = Instrument.objects.create(
            symbol="BTCUSDT",
            exchange="binance",
            base="BTC",
            quote="USDT",
        )
        now = timezone.now() - timedelta(seconds=5)
        Signal.objects.create(
            strategy="mod_trend_long",
            instrument=inst,
            ts=now,
            payload_json={
                "module": "trend",
                "direction": "long",
                "raw_score": 0.80,
                "confidence": 0.76,
                "reasons": {"test": True},
            },
            score=0.76,
        )
        Signal.objects.create(
            strategy="mod_meanrev_long",
            instrument=inst,
            ts=now,
            payload_json={
                "module": "meanrev",
                "direction": "long",
                "raw_score": 0.62,
                "confidence": 0.64,
                "reasons": {"test": True},
            },
            score=0.64,
        )

        snapshot = {
            "asof": timezone.now().isoformat(),
            "regime": "late_expanding",
            "confidence": 0.81,
            "composite_score": 0.42,
            "composite_momentum": 0.04,
            "details_json": {
                "component_count": 3,
                "active_components": [
                    "fed_net_liquidity_z",
                    "financial_conditions_z",
                    "dollar_z",
                ],
                "source_max_age_days": 2,
            },
        }

        with patch("signals.multi_strategy.acquire_task_lock", return_value=True):
            with patch("signals.multi_strategy._btc_allocator_context", return_value=("transition", "balanced")):
                with patch("signals.multi_strategy.get_cached_liquidity_snapshot", return_value=snapshot):
                    result = run_allocator_cycle()

        self.assertIn("allocator:emitted=", result)
        alloc = Signal.objects.filter(strategy="alloc_long", instrument=inst).latest("ts")
        diag = alloc.payload_json.get("reasons", {}).get("howell_liquidity")
        self.assertIsNotNone(diag)
        self.assertEqual(diag["regime"], "late_expanding")
        self.assertTrue(diag["shadow_only"])
