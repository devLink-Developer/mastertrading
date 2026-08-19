from decimal import Decimal
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone as dj_tz

from core.models import Instrument
from execution.eudy_recovery import (
    classify_eudy_edge_sample,
    eudy_exploration_risk_integrity_allows,
    evaluate_eudy_edge_guard,
)
from execution.models import OperationReport
from execution.tasks import _entry_quality_profile_precheck, _flat_signal_policy
from signals.models import Signal
from signals.multi_strategy import run_allocator_cycle


class EudyRecoveryClassifierTests(SimpleTestCase):
    def test_insufficient_sample_explores_at_reduced_risk(self):
        decision = classify_eudy_edge_sample(
            [0.002, -0.001],
            min_trades=3,
            min_profit_factor=1.15,
            min_expectancy_pct=0.0,
            exploration_risk_mult=0.5,
            context="transition_sell",
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.status, "explore")
        self.assertEqual(decision.risk_mult, 0.5)
        self.assertGreater(decision.profit_factor, 1.0)

    def test_negative_early_sample_brakes_before_full_validation_window(self):
        decision = classify_eudy_edge_sample(
            [0.008957, -0.006797, 0.002905, -0.011923],
            min_trades=12,
            min_profit_factor=1.15,
            min_expectancy_pct=0.0,
            exploration_risk_mult=0.5,
            context="transition_bear_short",
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.status, "exploration_brake")
        self.assertEqual(decision.risk_mult, 0.0)
        self.assertLess(decision.profit_factor, 1.0)
        self.assertLess(decision.expectancy_pct, 0.0)

    def test_positive_early_sample_keeps_reduced_risk(self):
        decision = classify_eudy_edge_sample(
            [0.004, -0.002, 0.003, -0.001],
            min_trades=12,
            min_profit_factor=1.15,
            min_expectancy_pct=0.0,
            exploration_risk_mult=0.5,
            context="transition_bear_short",
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.status, "explore")
        self.assertEqual(decision.risk_mult, 0.5)

    @override_settings(
        EUDY_RECOVERY_ENABLED=True,
        EUDY_RECOVERY_ACCOUNT_ALIASES={"eudy"},
        MIN_QTY_DYNAMIC_ALLOWLIST_WATCH_MULTIPLIER=2.0,
    )
    def test_min_qty_cannot_erase_eudy_exploration_risk_reduction(self):
        allowed, reason = eudy_exploration_risk_integrity_allows(
            account_alias="eudy",
            edge_status="explore",
            actual_risk_mult=10.25,
        )

        self.assertFalse(allowed)
        self.assertIn("min_qty_block", reason)

    @override_settings(
        EUDY_RECOVERY_ENABLED=True,
        EUDY_RECOVERY_ACCOUNT_ALIASES={"eudy"},
        MIN_QTY_DYNAMIC_ALLOWLIST_WATCH_MULTIPLIER=2.0,
    )
    def test_absolute_cap_preserves_bounded_tiny_account_exploration(self):
        allowed, reason = eudy_exploration_risk_integrity_allows(
            account_alias="eudy",
            edge_status="explore",
            actual_risk_mult=10.25,
            absolute_cap_allows=True,
        )

        self.assertTrue(allowed)
        self.assertIn("absolute_cap", reason)

    @override_settings(
        EUDY_RECOVERY_ENABLED=True,
        EUDY_RECOVERY_ACCOUNT_ALIASES={"eudy"},
        MIN_QTY_DYNAMIC_ALLOWLIST_WATCH_MULTIPLIER=2.0,
    )
    def test_risk_integrity_does_not_change_ricardo_or_validated_eudy(self):
        ricardo_allowed, _ = eudy_exploration_risk_integrity_allows(
            account_alias="rortigoza",
            edge_status="explore",
            actual_risk_mult=10.25,
        )
        validated_allowed, _ = eudy_exploration_risk_integrity_allows(
            account_alias="eudy",
            edge_status="validated",
            actual_risk_mult=10.25,
        )

        self.assertTrue(ricardo_allowed)
        self.assertTrue(validated_allowed)

    def test_positive_net_sample_restores_normal_risk(self):
        decision = classify_eudy_edge_sample(
            [0.006, 0.004, -0.002, 0.003],
            min_trades=4,
            min_profit_factor=1.15,
            min_expectancy_pct=0.0,
            exploration_risk_mult=0.5,
            context="transition_sell",
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.status, "validated")
        self.assertEqual(decision.risk_mult, 1.0)
        self.assertGreater(decision.profit_factor, 1.15)

    def test_negative_net_sample_is_blocked(self):
        decision = classify_eudy_edge_sample(
            [0.002, -0.004, -0.003, 0.001],
            min_trades=4,
            min_profit_factor=1.15,
            min_expectancy_pct=0.0,
            exploration_risk_mult=0.5,
            context="bear_buy",
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.status, "blocked")
        self.assertEqual(decision.risk_mult, 0.0)


@override_settings(
    MULTI_STRATEGY_ENABLED=True,
    MODULE_TREND_ENABLED=True,
    MODULE_MEANREV_ENABLED=False,
    MODULE_CARRY_ENABLED=True,
    MODULE_GRID_ENABLED=False,
    ALLOCATOR_ENABLED=True,
    ALLOCATOR_INCLUDE_SMC=False,
    ALLOCATOR_MIN_MODULES_ACTIVE=2,
    ALLOCATOR_STRONG_TREND_SOLO_ENABLED=False,
    ALLOCATOR_NET_THRESHOLD=0.20,
    ALLOCATOR_DYNAMIC_WEIGHTS_ENABLED=False,
    META_ALLOCATOR_ENABLED=True,
    HMM_REGIME_ENABLED=False,
    LIVE_GRADUAL_ENABLED=False,
    FEATURE_FLAGS_SOURCE="env",
    EUDY_RECOVERY_ENABLED=True,
    ALLOCATOR_LONG_SCORE_PENALTY=1.0,
    ALLOCATOR_TREND_BALANCED_TRANSITION_DAMPEN_ENABLED=True,
    ALLOCATOR_TREND_BALANCED_TRANSITION_DAMPEN_LEAD_STATES={"transition"},
    ALLOCATOR_TREND_BALANCED_TRANSITION_DAMPEN_RECOMMENDED_BIASES={"balanced"},
    ALLOCATOR_TREND_BALANCED_TRANSITION_DAMPEN_MULT=0.65,
    ALLOCATOR_MODULE_WEIGHTS={
        "trend": 0.25,
        "meanrev": 0.20,
        "carry": 0.15,
        "grid": 0.15,
        "smc": 0.25,
    },
    ALLOCATOR_MODULE_RISK_BUDGETS={
        "trend": 0.25,
        "meanrev": 0.20,
        "carry": 0.15,
        "grid": 0.15,
        "smc": 0.25,
    },
)
class EudyAllocatorRecoveryTests(TestCase):
    _META_OVERLAY = {
        "weights": {
            "trend": 0.202422,
            "meanrev": 0.240217,
            "carry": 0.076928,
            "grid": 0.180163,
            "smc": 0.300271,
        },
        "risk_budgets": {
            "trend": 0.10,
            "meanrev": 0.20,
            "carry": 0.05,
            "grid": 0.15,
            "smc": 0.25,
        },
        "diag": {"enabled": True, "summary": {}},
    }

    def _seed_aligned_signals(self) -> Instrument:
        inst = Instrument.objects.create(
            symbol="ADAUSDT",
            exchange="bingx",
            base="ADA",
            quote="USDT",
            enabled=True,
        )
        now = dj_tz.now()
        Signal.objects.create(
            strategy="mod_trend_short",
            instrument=inst,
            ts=now,
            payload_json={
                "module": "trend",
                "direction": "short",
                "confidence": 0.577,
                "raw_score": 0.577,
                "reasons": {"adx_htf": 24.0, "volume_ratio": 1.0},
            },
            score=0.577,
        )
        Signal.objects.create(
            strategy="mod_carry_short",
            instrument=inst,
            ts=now,
            payload_json={
                "module": "carry",
                "direction": "short",
                "confidence": 0.9795,
                "raw_score": 0.9795,
            },
            score=0.9795,
        )
        return inst

    def _run_allocator(self) -> str:
        with (
            patch("signals.multi_strategy.acquire_task_lock", return_value=True),
            patch(
                "signals.allocator.get_runtime_bool",
                side_effect=lambda _key, fallback: fallback,
            ),
            patch(
                "signals.allocator.get_runtime_float",
                side_effect=lambda _key, fallback, **_kwargs: fallback,
            ),
            patch(
                "signals.allocator.get_runtime_str_list",
                side_effect=lambda _key, fallback: set(fallback or set()),
            ),
            patch(
                "signals.multi_strategy.get_runtime_float",
                side_effect=lambda _key, fallback, **_kwargs: fallback,
            ),
            patch(
                "signals.multi_strategy.compute_meta_allocator_overlay",
                return_value=self._META_OVERLAY,
            ),
            patch(
                "signals.multi_strategy._btc_allocator_context",
                return_value=("transition", "balanced"),
            ),
        ):
            return run_allocator_cycle()

    def test_meta_overlay_limits_risk_without_suppressing_actionable_signal(self):
        inst = self._seed_aligned_signals()
        out = self._run_allocator()

        self.assertIn("allocator:emitted=1", out)
        alloc = Signal.objects.filter(
            instrument=inst,
            strategy="alloc_short",
        ).first()
        self.assertIsNotNone(alloc)
        self.assertGreater(abs(float(alloc.payload_json["net_score"])), 0.20)
        self.assertAlmostEqual(
            float(alloc.payload_json["reasons"]["budget_mix"]),
            0.106675,
            places=5,
        )
        meta = alloc.payload_json["reasons"]["meta_allocator"]
        self.assertEqual(meta["score_weight_source"], "pre_meta_eudy_recovery")

    @override_settings(EUDY_RECOVERY_ENABLED=False)
    def test_non_eudy_stack_keeps_meta_weights_for_signal_scoring(self):
        inst = self._seed_aligned_signals()
        out = self._run_allocator()

        self.assertIn("allocator:emitted=1", out)
        alloc = Signal.objects.filter(
            instrument=inst,
            strategy="alloc_flat",
        ).first()
        self.assertIsNotNone(alloc)
        self.assertLess(abs(float(alloc.payload_json["net_score"])), 0.20)
        meta = alloc.payload_json["reasons"]["meta_allocator"]
        self.assertEqual(meta["score_weight_source"], "meta_overlay")


@override_settings(
    MODE="live",
    EUDY_RECOVERY_ENABLED=True,
    EUDY_RECOVERY_ACCOUNT_ALIASES={"eudy"},
    EUDY_RECOVERY_BYPASS_STATIC_PROFILE=True,
    EUDY_EDGE_GUARD_LOOKBACK_DAYS=120,
    EUDY_EDGE_GUARD_MAX_TRADES=60,
    EUDY_EDGE_GUARD_MIN_TRADES=3,
    EUDY_EDGE_GUARD_MIN_PROFIT_FACTOR=1.15,
    EUDY_EDGE_GUARD_MIN_EXPECTANCY_PCT=0.0,
    EUDY_EDGE_GUARD_EXPLORATION_RISK_MULT=0.5,
    EUDY_EDGE_GUARD_RESET_AT="",
)
class EudyRecoveryIntegrationTests(TestCase):
    def setUp(self):
        self.inst = Instrument.objects.create(
            symbol="LINKUSDT",
            exchange="bingx",
            base="LINK",
            quote="USDT",
        )

    def _report(
        self,
        pnl_pct: float,
        *,
        daily_regime: str = "transition",
        lead: str = "transition",
        bias: str = "balanced",
        side: str = "sell",
    ) -> None:
        pnl_abs = Decimal(str(pnl_pct * 100.0))
        OperationReport.objects.create(
            instrument=self.inst,
            side=side,
            qty=Decimal("1"),
            entry_price=Decimal("100"),
            exit_price=Decimal("100"),
            pnl_abs=pnl_abs,
            pnl_pct=Decimal(str(pnl_pct)),
            outcome=(
                OperationReport.Outcome.WIN
                if pnl_pct > 0
                else OperationReport.Outcome.LOSS
            ),
            reason="tp" if pnl_pct > 0 else "sl",
            mode="live",
            daily_regime=daily_regime,
            btc_lead_state=lead,
            recommended_bias=bias,
            closed_at=dj_tz.now(),
        )

    def test_guard_uses_exact_regime_bias_and_side_cohort(self):
        self._report(0.006)
        self._report(0.004)
        self._report(-0.002)
        # This large loss belongs to another cohort and must not contaminate it.
        self._report(-0.50, daily_regime="bear_confirmed")

        decision = evaluate_eudy_edge_guard(
            account_alias="eudy",
            side="sell",
            daily_regime="transition",
            btc_lead_state="transition",
            recommended_bias="balanced",
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.status, "validated")
        self.assertEqual(decision.sample_size, 3)

    def test_non_eudy_account_is_always_bypassed(self):
        decision = evaluate_eudy_edge_guard(
            account_alias="rortigoza",
            side="buy",
            daily_regime="bear_confirmed",
            btc_lead_state="bear_confirmed",
            recommended_bias="short_bias",
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.status, "bypass")
        self.assertEqual(decision.risk_mult, 1.0)

    @override_settings(
        FLAT_SIGNAL_TIMEOUT_ENABLED=True,
        FLAT_SIGNAL_TIMEOUT_MINUTES=10,
        FLAT_SIGNAL_EARLY_EXIT_ENABLED=True,
        EUDY_RECOVERY_FLAT_TIMEOUT_ENABLED=True,
        EUDY_RECOVERY_FLAT_TIMEOUT_MINUTES=120,
        EUDY_RECOVERY_FLAT_EARLY_EXIT_ENABLED=False,
    )
    def test_flat_policy_is_extended_only_for_eudy(self):
        with (
            patch(
                "execution.tasks.get_runtime_bool",
                side_effect=lambda _key, fallback: fallback,
            ),
            patch(
                "execution.tasks.get_runtime_float",
                side_effect=lambda _key, fallback, **_kwargs: fallback,
            ),
        ):
            self.assertEqual(_flat_signal_policy("eudy"), (True, 120.0, False))
            self.assertEqual(_flat_signal_policy("rortigoza"), (True, 10.0, True))

    @override_settings(
        ENTRY_QUALITY_PROFILE_ENABLED=True,
        ENTRY_QUALITY_PROFILE_ALLOWED_SYMBOLS=set(),
        ENTRY_QUALITY_PROFILE_ALLOWED_SESSIONS=set(),
        ENTRY_QUALITY_PROFILE_ALLOWED_MODULE_SETS=set(),
    )
    def test_eudy_bypasses_stale_static_entry_profile_only(self):
        with patch(
            "execution.tasks.get_runtime_bool",
            side_effect=lambda _key, fallback: fallback,
        ):
            eudy_ok, eudy_reason = _entry_quality_profile_precheck(
                inst=self.inst,
                strategy_name="alloc_short",
                current_session="london",
                sig_payload={},
                account_alias="eudy",
            )
            ricardo_ok, ricardo_reason = _entry_quality_profile_precheck(
                inst=self.inst,
                strategy_name="alloc_short",
                current_session="london",
                sig_payload={},
                account_alias="rortigoza",
            )

        self.assertTrue(eudy_ok)
        self.assertEqual(eudy_reason, "eudy_recovery_bypass_static_profile")
        self.assertFalse(ricardo_ok)
        self.assertIn("config_missing:allowed_symbols", ricardo_reason)
