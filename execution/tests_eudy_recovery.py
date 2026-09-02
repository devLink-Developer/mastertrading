from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone as dj_tz

from core.models import Instrument
from execution.eudy_recovery import (
    classify_eudy_edge_sample,
    eudy_exploration_risk_integrity_allows,
    evaluate_eudy_edge_guard,
)
from execution.models import OperationReport, Order
from execution.tasks import (
    _attempt_entry_open,
    _entry_quality_profile_precheck,
    _flat_signal_policy,
    _record_eudy_entry_block,
)
from risk.models import RiskEvent
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
    def test_absolute_cap_does_not_override_relative_exploration_limit(self):
        allowed, reason = eudy_exploration_risk_integrity_allows(
            account_alias="eudy",
            edge_status="explore",
            actual_risk_mult=10.25,
            absolute_cap_allows=True,
        )

        self.assertFalse(allowed)
        self.assertIn("min_qty_block", reason)

    @override_settings(
        EUDY_RECOVERY_ENABLED=True,
        EUDY_RECOVERY_ACCOUNT_ALIASES={"eudy"},
        MIN_QTY_DYNAMIC_ALLOWLIST_WATCH_MULTIPLIER=2.0,
        EUDY_EXPLORATION_MIN_QTY_ABSOLUTE_CAP_OVERRIDE_ENABLED=True,
    )
    def test_enabled_absolute_cap_override_allows_safe_eudy_exploration(self):
        allowed, reason = eudy_exploration_risk_integrity_allows(
            account_alias="eudy",
            edge_status="explore",
            actual_risk_mult=10.25,
            absolute_cap_allows=True,
        )

        self.assertTrue(allowed)
        self.assertIn("absolute_cap_override", reason)

    @override_settings(
        EUDY_RECOVERY_ENABLED=True,
        EUDY_RECOVERY_ACCOUNT_ALIASES={"eudy"},
        MIN_QTY_DYNAMIC_ALLOWLIST_WATCH_MULTIPLIER=2.0,
        EUDY_EXPLORATION_MIN_QTY_ABSOLUTE_CAP_OVERRIDE_ENABLED=True,
    )
    def test_enabled_absolute_cap_override_still_blocks_unsafe_eudy_exploration(self):
        allowed, reason = eudy_exploration_risk_integrity_allows(
            account_alias="eudy",
            edge_status="explore",
            actual_risk_mult=10.25,
            absolute_cap_allows=False,
        )

        self.assertFalse(allowed)
        self.assertIn("min_qty_block", reason)

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

    @override_settings(
        EUDY_RECOVERY_ENABLED=True,
        EUDY_RECOVERY_ACCOUNT_ALIASES={"eudy"},
        EUDY_RECOVERY_BYPASS_STATIC_PROFILE=True,
        EUDY_CARRY_TREND_STRONG_REQUIRED=True,
    )
    def test_eudy_static_profile_bypass_rejects_weak_carry_trend_pair(self):
        payload = {
            "reasons": {
                "module_rows": [
                    {"module": "carry", "direction": "long"},
                    {"module": "trend", "direction": "long"},
                ],
                "trend_context": {
                    "direction": "long",
                    "is_strong": False,
                },
            }
        }

        with patch(
            "execution.tasks.get_runtime_bool",
            side_effect=lambda _key, fallback: fallback,
        ):
            allowed, reason = _entry_quality_profile_precheck(
                inst=SimpleNamespace(symbol="LINKUSDT"),
                strategy_name="alloc_long",
                current_session="london",
                sig_payload=payload,
                account_alias="eudy",
            )

        self.assertFalse(allowed)
        self.assertIn("carry_trend_not_strong", reason)

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
        closed_at=None,
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
            closed_at=closed_at or dj_tz.now(),
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

    @override_settings(
        EUDY_EDGE_GUARD_RESET_AT="2026-08-25T13:50:00Z",
        EUDY_EDGE_GUARD_MIN_TRADES=1,
    )
    def test_guard_reset_excludes_pre_rollout_outcomes(self):
        self._report(
            -0.50,
            closed_at=datetime(2026, 8, 25, 13, 49, tzinfo=timezone.utc),
        )
        self._report(
            0.004,
            closed_at=datetime(2026, 8, 25, 13, 51, tzinfo=timezone.utc),
        )

        decision = evaluate_eudy_edge_guard(
            account_alias="eudy",
            side="sell",
            daily_regime="transition",
            btc_lead_state="transition",
            recommended_bias="balanced",
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.status, "validated")
        self.assertEqual(decision.sample_size, 1)
        self.assertAlmostEqual(decision.expectancy_pct, 0.004)

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

    @override_settings(EUDY_ENTRY_BLOCK_AUDIT_ENABLED=True)
    def test_eudy_entry_block_audit_is_persistent_deduplicated_and_scoped(self):
        sig = Signal.objects.create(
            instrument=self.inst,
            strategy="alloc_long",
            score=0.31,
            ts=dj_tz.now(),
            payload_json={"direction": "long"},
        )

        _record_eudy_entry_block(
            inst=self.inst,
            sig=sig,
            strategy_name="alloc_long",
            signal_direction="long",
            reason="min_qty_exploration_risk",
            account_alias="eudy",
        )
        _record_eudy_entry_block(
            inst=self.inst,
            sig=sig,
            strategy_name="alloc_long",
            signal_direction="long",
            reason="min_qty_exploration_risk",
            account_alias="eudy",
        )
        _record_eudy_entry_block(
            inst=self.inst,
            sig=sig,
            strategy_name="alloc_long",
            signal_direction="long",
            reason="direction_policy",
            account_alias="rortigoza",
        )

        events = RiskEvent.objects.filter(kind="entry_blocked", instrument=self.inst)
        self.assertEqual(events.count(), 1)
        event = events.get()
        self.assertEqual(event.severity, RiskEvent.Severity.INFO)
        self.assertEqual(event.details_json["signal_id"], sig.id)
        self.assertEqual(event.details_json["strategy"], "alloc_long")
        self.assertEqual(event.details_json["direction"], "long")
        self.assertEqual(event.details_json["reason"], "min_qty_exploration_risk")

    def test_entry_attempt_exposes_terminal_block_reason_to_audit_caller(self):
        sig = Signal.objects.create(
            instrument=self.inst,
            strategy="alloc_long",
            score=0.31,
            ts=dj_tz.now(),
            payload_json={"direction": "long"},
        )
        trace: dict[str, str] = {}

        result = _attempt_entry_open(
            adapter=SimpleNamespace(),
            inst=self.inst,
            sig=sig,
            sig_payload=sig.payload_json,
            strategy_name=sig.strategy,
            side="buy",
            signal_direction="long",
            direction_allowed=True,
            signal_expired=False,
            can_open=False,
            macro_active=False,
            macro_context={},
            macro_block_entries=False,
            macro_risk_mult=1.0,
            regime_blocked_symbols=set(),
            regime_adx_by_symbol={},
            regime_adx_min_by_symbol={},
            regime_bias_by_symbol={},
            regime_adx_min=17.0,
            market_regime_adx=None,
            mtf_symbol_snapshot={},
            btc_lead_state="neutral",
            btc_recommended_bias="balanced",
            allow_scale_entry=False,
            scale_parent_correlation="",
            scale_add_index=0,
            session_policy_enabled=True,
            session_dead_zone_block=True,
            current_session="london",
            session_min_score=0.20,
            session_risk_mult=1.0,
            ml_entry_filter_enabled=False,
            ml_entry_filter_default_min_prob=0.50,
            ml_entry_filter_fail_open=True,
            use_allocator_signals=True,
            symbol=self.inst.symbol,
            last_price=10.0,
            contract_size=1.0,
            market_info={},
            atr=0.01,
            sl_pct=0.012,
            spread_bps_selected=1.0,
            free_usdt=10.0,
            equity_usdt=10.0,
            leverage=5.0,
            total_notional=0.0,
            cycle_notional_added=0.0,
            account_ai_enabled=False,
            account_ai_config_id=None,
            account_owner_id=None,
            account_alias="eudy",
            account_service="trading",
            positions_snapshot=[],
            decision_trace=trace,
        )

        self.assertEqual(result, (0, 0.0))
        self.assertEqual(trace["block_reason"], "account_or_risk_gate")

    def _attempt_min_qty_entry(self, *, account_alias: str, min_qty: float):
        sig = Signal.objects.create(
            instrument=self.inst,
            strategy="alloc_long",
            score=0.95,
            ts=dj_tz.now(),
            payload_json={
                "direction": "long",
                "risk_budget_pct": 0.0003,
            },
        )
        sent_orders: list[tuple] = []

        def _create_order(*args, **kwargs):
            sent_orders.append((args, kwargs))
            return {
                "id": f"test-{sig.id}",
                "average": 10.0,
                "fee": {"cost": 0.0},
            }

        adapter = SimpleNamespace(
            client=SimpleNamespace(
                precisionMode=4,
                amount_to_precision=lambda _symbol, amount: str(amount),
            ),
            _map_symbol=lambda symbol: symbol,
            create_order=_create_order,
            margin_mode="cross",
        )
        trace: dict[str, str] = {}
        market_info = {
            "limits": {"amount": {"min": min_qty}},
            "precision": {"amount": 0.1},
        }

        task_patches = {
            "get_runtime_bool": Mock(side_effect=lambda _key, fallback: fallback),
            "get_runtime_float": Mock(
                side_effect=lambda _key, fallback, **_kwargs: fallback
            ),
            "_get_daily_trade_count": Mock(return_value=0),
            "_bull_short_retrace_precheck": Mock(return_value=(True, "ok")),
            "_ny_open_weak_long_precheck": Mock(return_value=(True, "ok")),
            "_weak_long_bear_weak_precheck": Mock(return_value=(True, "ok")),
            "_asia_weak_short_precheck": Mock(return_value=(True, "ok")),
            "_weak_short_transition_precheck": Mock(return_value=(True, "ok")),
            "_long_bias_short_precheck": Mock(return_value=(True, "ok")),
            "_entry_quality_profile_precheck": Mock(return_value=(True, "ok")),
            "_symbol_health_precheck": Mock(return_value=(True, "ok")),
            "_symbol_side_health_precheck": Mock(return_value=(True, "ok")),
            "_volume_delta_check": Mock(return_value=(True, 0.0, "ok")),
            "_post_tp_alt_reentry_quality_precheck": Mock(return_value=(True, "ok")),
            "_volume_gate_allowed": Mock(return_value=(True, 1.0)),
            "_ai_entry_should_suppress_retry": Mock(return_value=(False, "")),
            "_ai_entry_should_suppress_retry_from_feedback": Mock(return_value=(False, "")),
            "evaluate_ai_entry_gate": Mock(return_value=(True, 1.0, "ok", {})),
            "_ai_entry_clear_reject_cache": Mock(),
            "_regime_directional_risk_mult": Mock(return_value=(1.0, False, "ok")),
            "_symbol_heat_guard": Mock(return_value=(1.0, "ok")),
            "_ensure_entry_leverage": Mock(return_value=(True, "cached")),
            "_compute_tp_sl_prices": Mock(return_value=(10.08, 9.88, 0.008, 0.012)),
            "_resolve_regime_label": Mock(return_value="transition"),
            "_strong_trend_safety_execution_allowed": Mock(
                return_value=(True, "not_applicable")
            ),
            "get_current_session": Mock(return_value="london"),
            "_reserve_daily_trade_slot": Mock(return_value=(True, None, "ok")),
            "_place_sl_order": Mock(return_value=None),
            "_increment_daily_trade_count": Mock(),
            "_record_min_qty_risk_guard_event": Mock(),
            "notify_trade_opened": Mock(),
            "_track_consecutive_errors": Mock(),
        }
        with patch.multiple("execution.tasks", **task_patches):
            result = _attempt_entry_open(
                adapter=adapter,
                inst=self.inst,
                sig=sig,
                sig_payload=sig.payload_json,
                strategy_name=sig.strategy,
                side="buy",
                signal_direction="long",
                direction_allowed=True,
                signal_expired=False,
                can_open=True,
                macro_active=False,
                macro_context={},
                macro_block_entries=False,
                macro_risk_mult=1.0,
                regime_blocked_symbols=set(),
                regime_adx_by_symbol={},
                regime_adx_min_by_symbol={},
                regime_bias_by_symbol={},
                regime_adx_min=17.0,
                market_regime_adx=None,
                mtf_symbol_snapshot={
                    "daily_regime": "transition",
                    "monthly_regime": "transition",
                },
                btc_lead_state="transition",
                btc_recommended_bias="balanced",
                allow_scale_entry=False,
                scale_parent_correlation="",
                scale_add_index=0,
                session_policy_enabled=True,
                session_dead_zone_block=True,
                current_session="london",
                session_min_score=0.20,
                session_risk_mult=1.0,
                ml_entry_filter_enabled=False,
                ml_entry_filter_default_min_prob=0.50,
                ml_entry_filter_fail_open=True,
                use_allocator_signals=True,
                symbol=self.inst.symbol,
                last_price=10.0,
                contract_size=1.0,
                market_info=market_info,
                atr=0.01,
                sl_pct=0.012,
                spread_bps_selected=1.0,
                free_usdt=10.0,
                equity_usdt=10.0,
                leverage=5.0,
                total_notional=0.0,
                cycle_notional_added=0.0,
                account_ai_enabled=False,
                account_ai_config_id=None,
                account_owner_id=None,
                account_alias=account_alias,
                account_service="trading",
                positions_snapshot=[],
                decision_trace=trace,
            )

        return result, trace, sent_orders

    @override_settings(
        EUDY_EXPLORATION_MIN_QTY_ABSOLUTE_CAP_OVERRIDE_ENABLED=True,
        MIN_QTY_RISK_ABSOLUTE_CAP_ENABLED=True,
        MIN_QTY_RISK_ABSOLUTE_CAP_PCT=0.003,
        MIN_QTY_DYNAMIC_ALLOWLIST_ENABLED=True,
        MIN_QTY_DYNAMIC_ALLOWLIST_WATCH_MULTIPLIER=2.0,
        MIN_QTY_DYNAMIC_ALLOWLIST_BLOCK_MULTIPLIER=3.0,
        SIGNAL_COOLDOWN_MINUTES=0,
        PER_INSTRUMENT_COOLDOWN={},
        SHADOW_TRADING_ENABLED=False,
    )
    def test_entry_attempt_opens_eudy_when_min_qty_stop_risk_is_within_absolute_cap(self):
        result, trace, sent_orders = self._attempt_min_qty_entry(
            account_alias="eudy",
            min_qty=0.2,
        )

        self.assertEqual(result, (1, 2.0))
        self.assertEqual(len(sent_orders), 1)
        self.assertNotIn("block_reason", trace)
        self.assertTrue(
            Order.objects.filter(
                instrument=self.inst,
                status=Order.OrderStatus.FILLED,
            ).exists()
        )

    @override_settings(
        EUDY_EXPLORATION_MIN_QTY_ABSOLUTE_CAP_OVERRIDE_ENABLED=True,
        MIN_QTY_RISK_ABSOLUTE_CAP_ENABLED=True,
        MIN_QTY_RISK_ABSOLUTE_CAP_PCT=0.003,
        MIN_QTY_DYNAMIC_ALLOWLIST_ENABLED=True,
        MIN_QTY_DYNAMIC_ALLOWLIST_WATCH_MULTIPLIER=2.0,
        MIN_QTY_DYNAMIC_ALLOWLIST_BLOCK_MULTIPLIER=3.0,
        SIGNAL_COOLDOWN_MINUTES=0,
        PER_INSTRUMENT_COOLDOWN={},
        SHADOW_TRADING_ENABLED=False,
    )
    def test_entry_attempt_blocks_eudy_when_min_qty_stop_risk_exceeds_absolute_cap(self):
        result, trace, sent_orders = self._attempt_min_qty_entry(
            account_alias="eudy",
            min_qty=0.3,
        )

        self.assertEqual(result, (0, 0.0))
        self.assertEqual(sent_orders, [])
        self.assertTrue(trace["block_reason"].startswith("min_qty_exploration_risk:"))

    @override_settings(
        EUDY_EXPLORATION_MIN_QTY_ABSOLUTE_CAP_OVERRIDE_ENABLED=True,
        MIN_QTY_RISK_ABSOLUTE_CAP_ENABLED=True,
        MIN_QTY_RISK_ABSOLUTE_CAP_PCT=0.003,
        MIN_QTY_DYNAMIC_ALLOWLIST_ENABLED=True,
        MIN_QTY_DYNAMIC_ALLOWLIST_WATCH_MULTIPLIER=2.0,
        MIN_QTY_DYNAMIC_ALLOWLIST_BLOCK_MULTIPLIER=3.0,
        SIGNAL_COOLDOWN_MINUTES=0,
        PER_INSTRUMENT_COOLDOWN={},
        SHADOW_TRADING_ENABLED=False,
    )
    def test_entry_attempt_does_not_apply_eudy_override_to_ricardo(self):
        result, trace, sent_orders = self._attempt_min_qty_entry(
            account_alias="rortigoza",
            min_qty=0.3,
        )

        self.assertEqual(result, (0, 0.0))
        self.assertEqual(sent_orders, [])
        self.assertEqual(trace["block_reason"], "dynamic_min_qty_allowlist")

    @override_settings(
        ALLOCATOR_STRONG_TREND_SOLO_SAFETY_ENVELOPE_ENABLED=True,
        ALLOCATOR_STRONG_TREND_SOLO_SAFETY_ALLOWED_SYMBOLS={"LINKUSDT"},
        ALLOCATOR_STRONG_TREND_SOLO_SAFETY_ALLOWED_SESSIONS={"london", "overlap"},
    )
    def test_entry_attempt_revalidates_strong_trend_session_at_execution_time(self):
        sig = Signal.objects.create(
            instrument=self.inst,
            strategy="alloc_long",
            score=0.95,
            ts=dj_tz.now(),
            payload_json={
                "direction": "long",
                "risk_budget_pct": 0.001,
                "reasons": {"strong_trend_solo_applied": True},
            },
        )
        trace: dict[str, str] = {}
        adapter = SimpleNamespace(
            client=SimpleNamespace(
                precisionMode=4,
                amount_to_precision=lambda _symbol, amount: str(amount),
            ),
            _map_symbol=lambda symbol: symbol,
        )

        with patch("execution.tasks.get_current_session", return_value="ny_open"):
            result = _attempt_entry_open(
                adapter=adapter,
                inst=self.inst,
                sig=sig,
                sig_payload=sig.payload_json,
                strategy_name=sig.strategy,
                side="buy",
                signal_direction="long",
                direction_allowed=True,
                signal_expired=False,
                can_open=True,
                macro_active=False,
                macro_context={},
                macro_block_entries=False,
                macro_risk_mult=1.0,
                regime_blocked_symbols=set(),
                regime_adx_by_symbol={},
                regime_adx_min_by_symbol={},
                regime_bias_by_symbol={},
                regime_adx_min=17.0,
                market_regime_adx=None,
                mtf_symbol_snapshot={},
                btc_lead_state="neutral",
                btc_recommended_bias="balanced",
                allow_scale_entry=False,
                scale_parent_correlation="",
                scale_add_index=0,
                session_policy_enabled=True,
                session_dead_zone_block=True,
                current_session="overlap",
                session_min_score=0.20,
                session_risk_mult=1.0,
                ml_entry_filter_enabled=False,
                ml_entry_filter_default_min_prob=0.50,
                ml_entry_filter_fail_open=True,
                use_allocator_signals=True,
                symbol=self.inst.symbol,
                last_price=10.0,
                contract_size=1.0,
                market_info={},
                atr=0.01,
                sl_pct=0.012,
                spread_bps_selected=1.0,
                free_usdt=10.0,
                equity_usdt=10.0,
                leverage=5.0,
                total_notional=0.0,
                cycle_notional_added=0.0,
                account_ai_enabled=False,
                account_ai_config_id=None,
                account_owner_id=None,
                account_alias="eudy",
                account_service="trading",
                positions_snapshot=[],
                decision_trace=trace,
            )

        self.assertEqual(result, (0, 0.0))
        self.assertEqual(trace["block_reason"], "strong_trend_safety_session")

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
