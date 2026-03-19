import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone as dj_tz

from core.models import Instrument, AiFeedbackEvent
from execution.models import OperationReport, Order
from marketdata.models import Candle
from risk.models import RiskEvent
from signals.models import Signal
from execution.tasks import (
    _ai_entry_market_fingerprint,
    _ai_entry_mark_rejected,
    _ai_entry_reject_cache_key,
    _ai_entry_should_suppress_retry,
    _ai_entry_should_suppress_retry_from_feedback,
    _acquire_task_lock,
    _check_trailing_stop,
    _classify_exchange_close,
    _count_pyramid_adds,
    _create_risk_event,
    _check_drawdown,
    _compute_tp_sl_prices,
    _extract_trigger_price,
    _extract_fee_usdt,
    _is_insufficient_margin_error,
    _ml_entry_filter_model_path,
    _ml_entry_filter_min_prob,
    _align_min_order_qty,
    _minimum_order_amount_from_error,
    _market_min_qty,
    _macro_high_impact_allows_entry,
    _min_qty_risk_guard,
    _is_no_position_error,
    _load_enabled_instruments_and_latest_signals,
    _log_operation,
    _normalize_order_qty,
    _actual_stop_risk_amount,
    _position_root_correlation,
    _position_origin_refs,
    _release_task_lock,
    _reconcile_sl,
    _resolve_signal_direction,
    _safe_correlation_id,
    _signal_active_modules,
    _signal_entry_reason,
    _manage_open_position,
    _is_macro_high_impact_window,
    _regime_adx_min_for_symbol_session,
    _bull_short_retrace_precheck,
    _btc_lead_directional_risk_mult,
    _regime_directional_risk_mult,
    _volume_activity_ratio,
    _volume_gate_allowed,
    _volume_gate_min_ratio,
    _tp_sl_gate_pnl_pct,
    _track_data_staleness_transition,
    _evaluate_tp_progress_exit,
    _volatility_adjusted_risk,
)


class _DummyRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.expiry: dict[str, int] = {}

    def ping(self):
        return True

    def get(self, key: str):
        return self.store.get(key)

    def set(self, key: str, value, nx: bool = False, ex: int | None = None):
        if nx and key in self.store:
            return False
        self.store[key] = str(value)
        if ex is not None:
            self.expiry[key] = int(ex)
        return True

    def delete(self, key: str):
        self.store.pop(key, None)
        return 1

    def expire(self, key: str, seconds: int):
        if key not in self.store:
            return False
        self.expiry[key] = int(seconds)
        return True


class _DummyAdapter:
    class _Client:
        @staticmethod
        def amount_to_precision(_symbol: str, amount: float) -> str:
            # Truncate to 1 decimal (floor-like) for deterministic tests.
            return f"{int(amount * 10) / 10:.1f}"

    client = _Client()

    @staticmethod
    def _map_symbol(symbol: str) -> str:
        return symbol

    def create_order(self, symbol: str, side: str, type_: str, amount: float, price=None, params=None):
        return {
            "id": "dummy-order",
            "symbol": symbol,
            "side": side,
            "type": type_,
            "amount": amount,
            "price": price,
            "params": params or {},
        }

    @staticmethod
    def fetch_closed_orders(_symbol: str, since: int | None = None, limit: int = 10):
        return []

    @staticmethod
    def fetch_ticker(_symbol: str):
        return {"last": 100.0}


class _DummyAdapterPrecisionError:
    class _Client:
        @staticmethod
        def amount_to_precision(_symbol: str, _amount: float) -> str:
            raise Exception("amount of SOL/USDT:USDT must be greater than minimum amount precision of 1")

        @staticmethod
        def market(_symbol: str) -> dict:
            return {"precision": {"amount": 0}}

    client = _Client()

    @staticmethod
    def _map_symbol(symbol: str) -> str:
        return symbol


class _DummyAdapterPrecisionTickSize:
    class _Client:
        precisionMode = 4

        @staticmethod
        def amount_to_precision(_symbol: str, _amount: float) -> str:
            raise Exception("amount of SOL/USDT:USDT must be greater than minimum amount precision of 1")

        @staticmethod
        def market(_symbol: str) -> dict:
            return {
                "limits": {"amount": {"min": 0.02}},
                "precision": {"amount": 1.0},
            }

    client = _Client()

    @staticmethod
    def _map_symbol(symbol: str) -> str:
        return symbol


class _DummyAdapterAlignUp:
    class _Client:
        precisionMode = 4

        @staticmethod
        def amount_to_precision(_symbol: str, amount: float) -> str:
            return f"{int(amount):d}"

    client = _Client()

    @staticmethod
    def _map_symbol(symbol: str) -> str:
        return symbol


class TaskHelpersTest(SimpleTestCase):
    def test_extract_fee_from_fees_list(self):
        fee = _extract_fee_usdt({"fees": [{"cost": 0.11}, {"cost": "0.04"}]})
        self.assertAlmostEqual(fee, 0.15, places=8)

    def test_extract_fee_fallback_info(self):
        fee = _extract_fee_usdt({"info": {"commission": "0.23"}})
        self.assertAlmostEqual(fee, 0.23, places=8)

    def test_error_classifiers(self):
        self.assertTrue(_is_no_position_error(Exception("No position to close")))
        self.assertTrue(_is_insufficient_margin_error(Exception("Insufficient margin")))
        self.assertFalse(_is_no_position_error(Exception("network timeout")))

    def test_drawdown_skips_invalid_equity(self):
        with patch("execution.tasks._compute_drawdown_state", return_value=(None, 0.0)) as dd_mock:
            allowed, dd, meta = _check_drawdown(0.0, risk_ns="t")
            self.assertTrue(allowed)
            self.assertEqual(dd, 0.0)
            self.assertFalse(meta.get("emit_event", True))
            dd_mock.assert_not_called()

        fake_baseline = object()
        with patch("execution.tasks._compute_drawdown_state", return_value=(fake_baseline, -0.01)) as dd_mock:
            allowed2, dd2, meta2 = _check_drawdown(100.0, risk_ns="t")
            self.assertTrue(allowed2)
            self.assertEqual(dd2, -0.01)
            self.assertFalse(meta2.get("emit_event", True))
            dd_mock.assert_called_once()

    def test_market_min_qty_prefers_market_limits(self):
        market = {"limits": {"amount": {"min": 0.2}}}
        self.assertEqual(_market_min_qty(market, fallback=1.0), 0.2)
        self.assertEqual(_market_min_qty({}, fallback=1.0), 1.0)

    def test_market_min_qty_uses_precision_when_limits_missing(self):
        market = {"precision": {"amount": 0}}
        self.assertEqual(_market_min_qty(market, fallback=0.0), 1.0)

    def test_market_min_qty_uses_tick_size_precision_mode(self):
        market = {
            "limits": {"amount": {"min": 0.02}},
            "precision": {"amount": 1.0},
        }
        self.assertEqual(_market_min_qty(market, fallback=0.0, precision_mode=4), 1.0)

    @override_settings(
        TP_PROGRESS_EARLY_EXIT_ENABLED=True,
        TP_PROGRESS_EARLY_EXIT_MIN_PROGRESS=0.70,
        TP_PROGRESS_EARLY_EXIT_MIN_R=0.8,
        TP_PROGRESS_EARLY_EXIT_MAX_GIVEBACK_RATIO=0.25,
        TP_PROGRESS_EARLY_EXIT_FORCE_PROGRESS=0.90,
        TP_PROGRESS_EARLY_EXIT_FORCE_GIVEBACK_RATIO=0.18,
        TP_PROGRESS_EARLY_EXIT_CLOSE_SCORE=2,
    )
    def test_tp_progress_exit_closes_on_signal_mismatch(self):
        opened_at = dj_tz.now().replace(microsecond=0)
        state_key = f"BTCUSDT:{int(opened_at.timestamp())}"
        redis_stub = _DummyRedis()
        redis_stub.set(f"trail:max_fav:{state_key}", "0.016")
        with patch("execution.tasks._redis_client", return_value=redis_stub):
            should_close, reason, meta = _evaluate_tp_progress_exit(
                symbol="BTCUSDT",
                side="buy",
                entry_price=100.0,
                last_price=101.60,   # 1.60% gross => 75% net of a 2% TP after fee adjustment
                tp_pct=0.02,
                sl_pct=0.01,
                opened_at=opened_at,
                signal_direction="short",
                recommended_bias="balanced",
                strategy_name="alloc_long",
            )
        self.assertTrue(should_close)
        self.assertIn("signal_mismatch", reason)
        self.assertGreaterEqual(float(meta["progress"] or 0.0), 0.70)

    @override_settings(
        TP_PROGRESS_EARLY_EXIT_ENABLED=True,
        TP_PROGRESS_EARLY_EXIT_MIN_PROGRESS=0.70,
        TP_PROGRESS_EARLY_EXIT_MIN_R=0.8,
        TP_PROGRESS_EARLY_EXIT_MAX_GIVEBACK_RATIO=0.25,
        TP_PROGRESS_EARLY_EXIT_FORCE_PROGRESS=0.90,
        TP_PROGRESS_EARLY_EXIT_FORCE_GIVEBACK_RATIO=0.18,
        TP_PROGRESS_EARLY_EXIT_CLOSE_SCORE=2,
    )
    def test_tp_progress_exit_holds_when_trade_is_still_clean(self):
        opened_at = dj_tz.now().replace(microsecond=0)
        state_key = f"BTCUSDT:{int(opened_at.timestamp())}"
        redis_stub = _DummyRedis()
        redis_stub.set(f"trail:max_fav:{state_key}", "0.0148")
        with patch("execution.tasks._redis_client", return_value=redis_stub):
            should_close, reason, meta = _evaluate_tp_progress_exit(
                symbol="BTCUSDT",
                side="buy",
                entry_price=100.0,
                last_price=101.60,
                tp_pct=0.02,
                sl_pct=0.01,
                opened_at=opened_at,
                signal_direction="long",
                recommended_bias="long_bias",
                strategy_name="alloc_long",
            )
        self.assertFalse(should_close)
        self.assertEqual(reason, "tp_progress_exit_hold")
        self.assertLess(float(meta["giveback_ratio"] or 0.0), 0.25)

    @override_settings(
        TP_PROGRESS_EARLY_EXIT_ENABLED=True,
        TP_PROGRESS_EARLY_EXIT_MIN_PROGRESS=0.70,
        TP_PROGRESS_EARLY_EXIT_MIN_R=0.8,
        TP_PROGRESS_EARLY_EXIT_MAX_GIVEBACK_RATIO=0.25,
        TP_PROGRESS_EARLY_EXIT_FORCE_PROGRESS=0.90,
        TP_PROGRESS_EARLY_EXIT_FORCE_GIVEBACK_RATIO=0.18,
        TP_PROGRESS_EARLY_EXIT_CLOSE_SCORE=2,
        TP_PROGRESS_EARLY_EXIT_MICROVOL_AGE_RATIO=0.50,
        MODULE_MICROVOL_MAX_HOLD_MINUTES=18,
    )
    def test_tp_progress_exit_closes_microvol_on_age_plus_giveback(self):
        opened_at = (dj_tz.now() - timedelta(minutes=10)).replace(microsecond=0)
        state_key = f"ETHUSDT:{int(opened_at.timestamp())}"
        redis_stub = _DummyRedis()
        redis_stub.set(f"trail:max_fav:{state_key}", "0.020")
        with patch("execution.tasks._redis_client", return_value=redis_stub):
            should_close, reason, meta = _evaluate_tp_progress_exit(
                symbol="ETHUSDT",
                side="buy",
                entry_price=100.0,
                last_price=101.50,   # 1.5% => 75% of a 2% TP
                tp_pct=0.02,
                sl_pct=0.01,
                opened_at=opened_at,
                signal_direction="long",
                recommended_bias="balanced",
                strategy_name="mod_microvol_long",
            )
        self.assertTrue(should_close)
        self.assertIn("microvol_age", reason)
        self.assertGreaterEqual(int(meta["score"] or 0), 2)

    def test_market_min_qty_uses_cost_floor_when_higher(self):
        market = {
            "limits": {"amount": {"min": 6.0}, "cost": {"min": 2.0}},
            "precision": {"amount": 1.0},
        }
        self.assertEqual(
            _market_min_qty(
                market,
                fallback=0.0,
                precision_mode=4,
                last_price=0.25,
                contract_size=1.0,
            ),
            8.0,
        )

    def test_align_min_order_qty_rounds_up_to_step(self):
        adapter = _DummyAdapterAlignUp()
        market = {
            "limits": {"amount": {"min": 19.0}, "cost": {"min": 2.0}},
            "precision": {"amount": 1.0},
        }
        aligned = _align_min_order_qty(
            adapter,
            "DOGE/USDT:USDT",
            21.98,
            market=market,
            precision_mode=4,
        )
        self.assertEqual(aligned, 22.0)

    def test_minimum_order_amount_from_error_parses_exchange_message(self):
        qty = _minimum_order_amount_from_error(
            Exception('bingx {"code":101400,"msg":"The minimum order amount is 7 ADA.","data":{}}')
        )
        self.assertEqual(qty, 7.0)

    def test_actual_stop_risk_amount_matches_expected_math(self):
        risk = _actual_stop_risk_amount(
            qty=1.0,
            entry_price=90.24,
            stop_distance_pct=0.012,
            contract_size=1.0,
        )
        self.assertAlmostEqual(risk, 1.08288, places=8)

    @override_settings(
        MIN_QTY_RISK_GUARD_ENABLED=True,
        MIN_QTY_RISK_MULTIPLIER_MAX=3.0,
    )
    def test_min_qty_risk_guard_blocks_when_lot_min_explodes_risk(self):
        blocked, actual_risk, risk_mult = _min_qty_risk_guard(
            qty=1.0,
            risk_qty=0.12,
            min_qty=1.0,
            entry_price=90.24,
            stop_distance_pct=0.0075,
            contract_size=1.0,
            target_risk_amount=0.08,
        )
        self.assertTrue(blocked)
        self.assertGreater(actual_risk, 0.67)
        self.assertGreater(risk_mult, 8.0)

    @override_settings(
        MIN_QTY_RISK_GUARD_ENABLED=True,
        MIN_QTY_RISK_MULTIPLIER_MAX=3.0,
    )
    def test_min_qty_risk_guard_allows_normal_sizing(self):
        blocked, actual_risk, risk_mult = _min_qty_risk_guard(
            qty=0.4,
            risk_qty=0.4,
            min_qty=0.1,
            entry_price=90.24,
            stop_distance_pct=0.0075,
            contract_size=1.0,
            target_risk_amount=0.30,
        )
        self.assertFalse(blocked)
        self.assertEqual(actual_risk, 0.0)
        self.assertEqual(risk_mult, 0.0)

    @override_settings(
        TP_SL_FEE_ADJUST_ENABLED=True,
        TP_SL_ESTIMATED_ROUNDTRIP_FEE_PCT=0.0010,
    )
    def test_tp_sl_gate_pnl_pct_applies_fee_adjustment(self):
        pnl_gate, fee_est = _tp_sl_gate_pnl_pct(0.0100)
        self.assertAlmostEqual(fee_est, 0.0010, places=8)
        self.assertAlmostEqual(pnl_gate, 0.0090, places=8)

    @override_settings(TP_SL_FEE_ADJUST_ENABLED=False)
    def test_tp_sl_gate_pnl_pct_can_be_disabled(self):
        pnl_gate, fee_est = _tp_sl_gate_pnl_pct(0.0100)
        self.assertAlmostEqual(fee_est, 0.0, places=8)
        self.assertAlmostEqual(pnl_gate, 0.0100, places=8)

    @override_settings(MODULE_MICROVOL_TP_MULT=0.50)
    def test_microvol_tp_profile_is_tighter(self):
        generic_tp, _, generic_tp_pct, _ = _compute_tp_sl_prices("buy", 100.0, 0.01)
        micro_tp, _, micro_tp_pct, _ = _compute_tp_sl_prices(
            "buy",
            100.0,
            0.01,
            strategy_name="mod_microvol_long",
        )
        self.assertLess(micro_tp, generic_tp)
        self.assertLess(micro_tp_pct, generic_tp_pct)

    @override_settings(PER_INSTRUMENT_RISK={"BTCUSDT": 0.0015})
    def test_volatility_adjusted_risk_caps_per_symbol_to_base_allocator_risk(self):
        # Per-instrument config should never increase a low-confidence allocator budget.
        risk = _volatility_adjusted_risk("BTCUSDT", atr_pct=None, base_risk=0.0005)
        self.assertAlmostEqual(risk, 0.0005, places=8)

    @override_settings(PER_INSTRUMENT_RISK={"BTCUSDT": 0.0015})
    def test_volatility_adjusted_risk_still_honors_lower_per_symbol_cap(self):
        # If allocator budget is higher than per-symbol cap, the lower cap wins.
        risk = _volatility_adjusted_risk("BTCUSDT", atr_pct=None, base_risk=0.0025)
        self.assertAlmostEqual(risk, 0.0015, places=8)

    @override_settings(
        PER_INSTRUMENT_RISK={},
        INSTRUMENT_RISK_TIERS_ENABLED=False,
        VOL_RISK_LOW_ATR_PCT=0.01,
        VOL_RISK_HIGH_ATR_PCT=0.02,
        VOL_RISK_MIN_SCALE=0.5,
    )
    def test_volatility_adjusted_risk_uses_configurable_atr_ramp(self):
        # atr=1.5% is midpoint between 1.0% and 2.0% -> scale 0.75 when min_scale=0.5
        risk = _volatility_adjusted_risk("ETHUSDT", atr_pct=0.015, base_risk=0.01)
        self.assertAlmostEqual(risk, 0.0075, places=8)

    def test_normalize_order_qty_uses_exchange_precision(self):
        adapter = _DummyAdapter()
        self.assertAlmostEqual(_normalize_order_qty(adapter, "BTC/USDT:USDT", 1.29), 1.2, places=8)
        self.assertEqual(_normalize_order_qty(adapter, "BTC/USDT:USDT", -1), 0.0)

    def test_normalize_order_qty_does_not_fallback_to_raw_when_precision_fails(self):
        adapter = _DummyAdapterPrecisionError()
        self.assertEqual(_normalize_order_qty(adapter, "SOL/USDT:USDT", 0.1024), 0.0)

    def test_normalize_order_qty_respects_tick_size_precision_mode(self):
        adapter = _DummyAdapterPrecisionTickSize()
        self.assertEqual(_normalize_order_qty(adapter, "SOL/USDT:USDT", 0.3), 0.0)

    def test_extract_trigger_price_prefers_unified_and_info(self):
        self.assertEqual(_extract_trigger_price({"triggerPrice": "123.4"}), 123.4)
        self.assertEqual(_extract_trigger_price({"info": {"stopPrice": "55.1"}}), 55.1)
        self.assertEqual(_extract_trigger_price({}), 0.0)

    def test_ml_entry_filter_min_prob_uses_redis_override(self):
        dummy = _DummyRedis()
        dummy.set("ml:entry_filter:min_prob", "0.73")
        with patch("execution.tasks._redis_client", return_value=dummy):
            self.assertAlmostEqual(_ml_entry_filter_min_prob(0.52), 0.73, places=8)

    def test_ml_entry_filter_min_prob_falls_back_on_invalid(self):
        dummy = _DummyRedis()
        dummy.set("ml:entry_filter:min_prob", "bad")
        with patch("execution.tasks._redis_client", return_value=dummy):
            self.assertAlmostEqual(_ml_entry_filter_min_prob(0.52), 0.52, places=8)

    def test_ml_entry_filter_min_prob_prefers_symbol_override(self):
        dummy = _DummyRedis()
        dummy.set("ml:entry_filter:min_prob", "0.61")
        dummy.set("ml:entry_filter:min_prob:BTCUSDT", "0.72")
        with patch("execution.tasks._redis_client", return_value=dummy):
            self.assertAlmostEqual(_ml_entry_filter_min_prob(0.52, symbol="BTCUSDT"), 0.72, places=8)
            self.assertAlmostEqual(_ml_entry_filter_min_prob(0.52, symbol="ETHUSDT"), 0.61, places=8)

    def test_ml_entry_filter_min_prob_prefers_symbol_strategy_override(self):
        dummy = _DummyRedis()
        dummy.set("ml:entry_filter:min_prob", "0.61")
        dummy.set("ml:entry_filter:min_prob:BTCUSDT", "0.72")
        dummy.set("ml:entry_filter:min_prob:strategy:ALLOC_LONG", "0.66")
        dummy.set("ml:entry_filter:min_prob:BTCUSDT:ALLOC_LONG", "0.81")
        with patch("execution.tasks._redis_client", return_value=dummy):
            self.assertAlmostEqual(
                _ml_entry_filter_min_prob(0.52, symbol="BTCUSDT", strategy_name="alloc_long"),
                0.81,
                places=8,
            )
            self.assertAlmostEqual(
                _ml_entry_filter_min_prob(0.52, symbol="BTCUSDT", strategy_name="alloc_short"),
                0.72,
                places=8,
            )
            self.assertAlmostEqual(
                _ml_entry_filter_min_prob(0.52, symbol="ETHUSDT", strategy_name="alloc_long"),
                0.66,
                places=8,
            )

    @override_settings(AI_ENTRY_GATE_REJECT_COOLDOWN_ENABLED=True)
    def test_ai_entry_reject_suppresses_retry_when_fingerprint_is_unchanged(self):
        dummy = _DummyRedis()
        fp = _ai_entry_market_fingerprint(
            symbol="LINKUSDT",
            strategy_name="alloc_short",
            signal_direction="short",
            session_name="london",
            sig_score=0.73456,
            atr=0.01022,
            spread_bps=4.38,
            sl_pct=0.012,
            sig_payload={
                "reasons": {
                    "net_score": -0.3345,
                    "module_rows": [
                        {
                            "module": "trend",
                            "direction": "short",
                            "confidence": 0.81,
                            "raw_score": 0.42,
                        }
                    ],
                }
            },
        )
        with patch("execution.tasks._redis_client", return_value=dummy):
            _ai_entry_mark_rejected(
                account_alias="rortigoza",
                account_service="bingx",
                symbol="LINKUSDT",
                strategy_name="alloc_short",
                signal_direction="short",
                market_fingerprint=fp,
                reason="short",
            )
            suppressed, reason = _ai_entry_should_suppress_retry(
                account_alias="rortigoza",
                account_service="bingx",
                symbol="LINKUSDT",
                strategy_name="alloc_short",
                signal_direction="short",
                market_fingerprint=fp,
            )
        self.assertTrue(suppressed)
        self.assertEqual(reason, "short")

    @override_settings(AI_ENTRY_GATE_REJECT_COOLDOWN_ENABLED=True)
    def test_ai_entry_reject_does_not_suppress_when_market_fingerprint_changes(self):
        dummy = _DummyRedis()
        fp_old = _ai_entry_market_fingerprint(
            symbol="LINKUSDT",
            strategy_name="alloc_short",
            signal_direction="short",
            session_name="london",
            sig_score=0.70,
            atr=0.0100,
            spread_bps=4.0,
            sl_pct=0.012,
            sig_payload={},
        )
        fp_new = _ai_entry_market_fingerprint(
            symbol="LINKUSDT",
            strategy_name="alloc_short",
            signal_direction="short",
            session_name="london",
            sig_score=0.81,
            atr=0.0112,
            spread_bps=5.8,
            sl_pct=0.012,
            sig_payload={},
        )
        with patch("execution.tasks._redis_client", return_value=dummy):
            _ai_entry_mark_rejected(
                account_alias="rortigoza",
                account_service="bingx",
                symbol="LINKUSDT",
                strategy_name="alloc_short",
                signal_direction="short",
                market_fingerprint=fp_old,
                reason="short",
            )
            suppressed, _ = _ai_entry_should_suppress_retry(
                account_alias="rortigoza",
                account_service="bingx",
                symbol="LINKUSDT",
                strategy_name="alloc_short",
                signal_direction="short",
                market_fingerprint=fp_new,
            )
        self.assertFalse(suppressed)

    @override_settings(AI_ENTRY_GATE_REJECT_COOLDOWN_ENABLED=True)
    def test_ai_entry_reject_suppresses_retry_when_coarse_fingerprint_matches(self):
        dummy = _DummyRedis()
        fp_old = _ai_entry_market_fingerprint(
            symbol="SOLUSDT",
            strategy_name="alloc_long",
            signal_direction="long",
            session_name="asia",
            sig_score=0.7121,
            atr=0.01211,
            spread_bps=7.21,
            sl_pct=0.0122,
            sig_payload={
                "reasons": {
                    "net_score": 0.214,
                    "module_rows": [
                        {
                            "module": "carry",
                            "direction": "long",
                            "confidence": 0.61,
                            "raw_score": 0.35,
                        }
                    ],
                }
            },
        )
        fp_old_coarse = _ai_entry_market_fingerprint(
            symbol="SOLUSDT",
            strategy_name="alloc_long",
            signal_direction="long",
            session_name="asia",
            sig_score=0.7121,
            atr=0.01211,
            spread_bps=7.21,
            sl_pct=0.0122,
            sig_payload={
                "reasons": {
                    "net_score": 0.214,
                    "module_rows": [
                        {
                            "module": "carry",
                            "direction": "long",
                            "confidence": 0.61,
                            "raw_score": 0.35,
                        }
                    ],
                }
            },
            coarse=True,
        )
        fp_new = _ai_entry_market_fingerprint(
            symbol="SOLUSDT",
            strategy_name="alloc_long",
            signal_direction="long",
            session_name="asia",
            sig_score=0.7199,  # Drift that may change exact fp.
            atr=0.01224,
            spread_bps=7.24,
            sl_pct=0.0121,
            sig_payload={
                "reasons": {
                    "net_score": 0.221,
                    "module_rows": [
                        {
                            "module": "carry",
                            "direction": "long",
                            "confidence": 0.62,
                            "raw_score": 0.33,
                        }
                    ],
                }
            },
        )
        fp_new_coarse = _ai_entry_market_fingerprint(
            symbol="SOLUSDT",
            strategy_name="alloc_long",
            signal_direction="long",
            session_name="asia",
            sig_score=0.7199,
            atr=0.01224,
            spread_bps=7.24,
            sl_pct=0.0121,
            sig_payload={
                "reasons": {
                    "net_score": 0.221,
                    "module_rows": [
                        {
                            "module": "carry",
                            "direction": "long",
                            "confidence": 0.62,
                            "raw_score": 0.33,
                        }
                    ],
                }
            },
            coarse=True,
        )
        with patch("execution.tasks._redis_client", return_value=dummy):
            _ai_entry_mark_rejected(
                account_alias="rortigoza",
                account_service="bingx",
                symbol="SOLUSDT",
                strategy_name="alloc_long",
                signal_direction="long",
                market_fingerprint=fp_old,
                market_fingerprint_coarse=fp_old_coarse,
                reason="spr_hi,sl_tight,rb_low",
            )
            suppressed, reason = _ai_entry_should_suppress_retry(
                account_alias="rortigoza",
                account_service="bingx",
                symbol="SOLUSDT",
                strategy_name="alloc_long",
                signal_direction="long",
                market_fingerprint=fp_new,
                market_fingerprint_coarse=fp_new_coarse,
            )
        self.assertNotEqual(fp_old, fp_new)
        self.assertEqual(fp_old_coarse, fp_new_coarse)
        self.assertTrue(suppressed)
        self.assertEqual(reason, "spr_hi,sl_tight,rb_low")

    def test_ai_entry_reject_cache_key_is_stable(self):
        key = _ai_entry_reject_cache_key(
            account_alias="Rortigoza",
            account_service="BingX",
            symbol="LINKUSDT",
            strategy_name="alloc_short",
            signal_direction="short",
        )
        self.assertEqual(
            key,
            "ai:entry:reject:rortigoza:bingx:linkusdt:alloc_short:short",
        )

    def test_signal_active_modules_from_module_rows(self):
        payload = {
            "reasons": {
                "module_rows": [
                    {"module": "trend"},
                    {"module": "meanrev"},
                    {"module": "carry"},
                    {"module": "smc"},
                    {"module": "trend"},
                ]
            }
        }
        self.assertEqual(
            _signal_active_modules(payload, "alloc_long"),
            ["trend", "meanrev", "carry", "smc"],
        )

    def test_signal_active_modules_falls_back_to_strategy_module(self):
        self.assertEqual(_signal_active_modules({}, "mod_meanrev_short"), ["meanrev"])
        self.assertEqual(_signal_active_modules({}, "smc_long"), ["smc"])

    def test_signal_entry_reason_from_module_rows(self):
        payload = {
            "risk_budget_pct": 0.0007,
            "reasons": {
                "net_score": 1.234,
                "module_rows": [
                    {"module": "carry", "direction": "short", "confidence": 0.9886},
                    {"module": "trend", "direction": "short", "confidence": 0.6238},
                ],
            },
        }
        reason = _signal_entry_reason(payload, "alloc_short")
        self.assertIn("signal=alloc_short", reason)
        self.assertIn("confluencia: carry short (0.99), trend short (0.62)", reason)
        self.assertIn("risk_budget=0.070%", reason)
        self.assertIn("net_score=1.234", reason)

    def test_signal_entry_reason_falls_back_to_strategy_module(self):
        reason = _signal_entry_reason({}, "mod_meanrev_long")
        self.assertIn("signal=mod_meanrev_long", reason)
        self.assertIn("confluencia: meanrev", reason)

    def test_resolve_signal_direction_for_allocator(self):
        self.assertEqual(_resolve_signal_direction("alloc_long"), ("long", "buy"))
        self.assertEqual(_resolve_signal_direction("alloc_short"), ("short", "sell"))

    def test_resolve_signal_direction_for_legacy_and_flat(self):
        self.assertEqual(_resolve_signal_direction("mod_trend_long"), ("long", "buy"))
        self.assertEqual(_resolve_signal_direction("mod_meanrev_short"), ("short", "sell"))
        self.assertEqual(_resolve_signal_direction("mod_carry"), ("flat", ""))

    def test_task_lock_acquire_and_release(self):
        dummy = _DummyRedis()
        with patch("execution.tasks._redis_client", return_value=dummy):
            client, token = _acquire_task_lock("lock:test", 30)
            self.assertIsNotNone(client)
            self.assertTrue(token)
            self.assertEqual(dummy.get("lock:test"), token)

            client2, token2 = _acquire_task_lock("lock:test", 30)
            self.assertIsNotNone(client2)
            self.assertEqual(token2, "")

            _release_task_lock(client, "lock:test", token)
            self.assertIsNone(dummy.get("lock:test"))

    def test_task_lock_release_ignores_mismatched_token(self):
        dummy = _DummyRedis()
        with patch("execution.tasks._redis_client", return_value=dummy):
            client, token = _acquire_task_lock("lock:test", 30)
            self.assertTrue(token)
            _release_task_lock(client, "lock:test", "wrong-token")
            self.assertEqual(dummy.get("lock:test"), token)

    @override_settings(
        ML_ENTRY_FILTER_MODEL_PATH="/tmp/global_model.json",
        ML_ENTRY_FILTER_MODEL_DIR="/tmp/models",
        ML_ENTRY_FILTER_PER_SYMBOL_ENABLED=True,
        ML_ENTRY_FILTER_PER_SYMBOL_FALLBACK_GLOBAL=True,
    )
    def test_ml_entry_filter_model_path_prefers_per_symbol_when_exists(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            per_symbol = model_dir / "entry_filter_model_BTCUSDT.json"
            per_symbol.write_text("{}", encoding="utf-8")
            with patch("execution.tasks.settings.ML_ENTRY_FILTER_MODEL_DIR", str(model_dir)):
                resolved = _ml_entry_filter_model_path("BTCUSDT")
                self.assertEqual(resolved, str(per_symbol))

    @override_settings(
        ML_ENTRY_FILTER_MODEL_PATH="/tmp/global_model.json",
        ML_ENTRY_FILTER_MODEL_DIR="/tmp/models",
        ML_ENTRY_FILTER_PER_SYMBOL_ENABLED=True,
        ML_ENTRY_FILTER_PER_SYMBOL_FALLBACK_GLOBAL=True,
    )
    def test_ml_entry_filter_model_path_falls_back_global(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("execution.tasks.settings.ML_ENTRY_FILTER_MODEL_DIR", tmp_dir):
                resolved = _ml_entry_filter_model_path("BTCUSDT")
                self.assertEqual(resolved, "/tmp/global_model.json")

    @override_settings(
        ML_ENTRY_FILTER_MODEL_PATH="/tmp/global_model.json",
        ML_ENTRY_FILTER_MODEL_DIR="/tmp/models",
        ML_ENTRY_FILTER_PER_SYMBOL_ENABLED=True,
        ML_ENTRY_FILTER_PER_SYMBOL_FALLBACK_GLOBAL=True,
        ML_ENTRY_FILTER_PER_STRATEGY_ENABLED=True,
        ML_ENTRY_FILTER_PER_STRATEGY_FALLBACK_GLOBAL=True,
    )
    def test_ml_entry_filter_model_path_prefers_symbol_strategy_when_exists(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            per_symbol = model_dir / "entry_filter_model_BTCUSDT.json"
            per_symbol_strategy = model_dir / "entry_filter_model_BTCUSDT_ALLOC_LONG.json"
            per_symbol.write_text("{}", encoding="utf-8")
            per_symbol_strategy.write_text("{}", encoding="utf-8")
            with patch("execution.tasks.settings.ML_ENTRY_FILTER_MODEL_DIR", str(model_dir)):
                resolved = _ml_entry_filter_model_path("BTCUSDT", strategy_name="alloc_long")
                self.assertEqual(resolved, str(per_symbol_strategy))

    @override_settings(
        ML_ENTRY_FILTER_MODEL_PATH="/tmp/global_model.json",
        ML_ENTRY_FILTER_MODEL_DIR="/tmp/models",
        ML_ENTRY_FILTER_PER_SYMBOL_ENABLED=True,
        ML_ENTRY_FILTER_PER_SYMBOL_FALLBACK_GLOBAL=True,
        ML_ENTRY_FILTER_PER_STRATEGY_ENABLED=True,
        ML_ENTRY_FILTER_PER_STRATEGY_FALLBACK_GLOBAL=True,
    )
    def test_ml_entry_filter_model_path_uses_strategy_when_no_symbol_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            per_strategy = model_dir / "entry_filter_model_strategy_ALLOC_LONG.json"
            per_strategy.write_text("{}", encoding="utf-8")
            with patch("execution.tasks.settings.ML_ENTRY_FILTER_MODEL_DIR", str(model_dir)):
                resolved = _ml_entry_filter_model_path("BTCUSDT", strategy_name="alloc_long")
                self.assertEqual(resolved, str(per_strategy))

    def test_reconcile_sl_replaces_wide_or_duplicate_stops(self):
        stop_orders = [
            {"stopPrice": 90.0},
            {"stopPrice": 80.0},
        ]
        with patch("execution.tasks._has_sl_stop_order", return_value=(True, 90.0, stop_orders)):
            with patch("execution.tasks._cancel_stop_orders") as cancel_mock:
                with patch("execution.tasks._place_sl_order", return_value="sl-1") as place_mock:
                    _reconcile_sl(object(), "BTCUSDT", "buy", 1.0, 100.0, atr_pct=None)
                    cancel_mock.assert_called_once()
                    place_mock.assert_called_once()

    @override_settings(SL_RECONCILE_TOO_TIGHT_MULT=0.20, SL_RECONCILE_TOO_WIDE_MULT=2.0)
    def test_reconcile_sl_tight_multiplier_can_avoid_replacement(self):
        # current SL distance=0.3%; expected=1.0% -> with tight_mult=0.2, threshold=0.2% (not too tight)
        with patch("execution.tasks._compute_tp_sl_prices", return_value=(101.0, 99.0, 0.01, 0.01)):
            with patch("execution.tasks._has_sl_stop_order", return_value=(True, 99.7, [{"stopPrice": 99.7}])):
                with patch("execution.tasks._cancel_stop_orders") as cancel_mock:
                    with patch("execution.tasks._place_sl_order") as place_mock:
                        _reconcile_sl(object(), "BTCUSDT", "buy", 1.0, 100.0, atr_pct=None)
                        cancel_mock.assert_not_called()
                        place_mock.assert_not_called()

    @override_settings(SL_RECONCILE_TOO_TIGHT_MULT=0.50, SL_RECONCILE_TOO_WIDE_MULT=2.0)
    def test_reconcile_sl_tight_multiplier_can_force_replacement(self):
        # current SL distance=0.3%; expected=1.0% -> with tight_mult=0.5, threshold=0.5% (too tight)
        with patch("execution.tasks._compute_tp_sl_prices", return_value=(101.0, 99.0, 0.01, 0.01)):
            with patch("execution.tasks._has_sl_stop_order", return_value=(True, 99.7, [{"stopPrice": 99.7}])):
                with patch("execution.tasks._cancel_stop_orders") as cancel_mock:
                    with patch("execution.tasks._place_sl_order") as place_mock:
                        _reconcile_sl(object(), "BTCUSDT", "buy", 1.0, 100.0, atr_pct=None)
                        cancel_mock.assert_called_once()
                        place_mock.assert_called_once()

    @override_settings(
        TAKE_PROFIT_PCT=0.01,
        STOP_LOSS_PCT=0.007,
        TAKE_PROFIT_DYNAMIC_MULT=0.9,
        TAKE_PROFIT_MIN_PCT=0.006,
        ATR_MULT_TP=2.2,
        ATR_MULT_TP_LONG=2.2,
        ATR_MULT_TP_SHORT=2.2,
        ATR_MULT_SL=1.5,
        VOL_FAST_EXIT_ENABLED=True,
        VOL_FAST_EXIT_ATR_PCT=0.012,
        VOL_FAST_EXIT_TP_MULT=0.75,
        VOL_FAST_EXIT_MIN_TP_PCT=0.006,
    )
    def test_compute_tp_sl_prices_fast_exit_compresses_tp(self):
        _, _, tp_pct, _ = _compute_tp_sl_prices("buy", 100.0, atr_pct=0.02)
        # Base ATR TP=4.4%; dynamic mult (0.9) then fast-exit (0.75) -> 2.97%
        self.assertAlmostEqual(tp_pct, 0.0297, places=6)

    @override_settings(
        TAKE_PROFIT_PCT=0.01,
        STOP_LOSS_PCT=0.007,
        ATR_MULT_TP=2.0,
        ATR_MULT_TP_LONG=1.6,
        ATR_MULT_TP_SHORT=2.2,
        ATR_MULT_SL=1.5,
        TAKE_PROFIT_DYNAMIC_MULT=1.0,
        TAKE_PROFIT_MIN_PCT=0.0,
        VOL_FAST_EXIT_ENABLED=False,
    )
    def test_compute_tp_sl_prices_supports_directional_asymmetry(self):
        _, _, tp_long, sl_long = _compute_tp_sl_prices("buy", 100.0, atr_pct=0.02)
        _, _, tp_short, sl_short = _compute_tp_sl_prices("sell", 100.0, atr_pct=0.02)
        self.assertAlmostEqual(tp_long, 0.032, places=6)   # 2% * 1.6
        self.assertAlmostEqual(tp_short, 0.044, places=6)  # 2% * 2.2
        self.assertAlmostEqual(sl_long, sl_short, places=6)

    @override_settings(
        TAKE_PROFIT_PCT=0.01,
        STOP_LOSS_PCT=0.007,
        ATR_MULT_TP=2.0,
        ATR_MULT_TP_LONG=1.6,
        ATR_MULT_TP_SHORT=2.2,
        ATR_MULT_SL=1.5,
        TAKE_PROFIT_DYNAMIC_MULT=1.0,
        TAKE_PROFIT_MIN_PCT=0.0,
        VOL_FAST_EXIT_ENABLED=False,
        TACTICAL_EXIT_PROFILE_ENABLED=True,
        TACTICAL_EXIT_TP_MULT=0.75,
    )
    def test_compute_tp_sl_prices_compresses_tp_for_tactical_countertrend(self):
        _, _, tp_long, _ = _compute_tp_sl_prices("buy", 100.0, atr_pct=0.02, recommended_bias="tactical_long")
        _, _, tp_short, _ = _compute_tp_sl_prices("sell", 100.0, atr_pct=0.02, recommended_bias="tactical_short")
        self.assertAlmostEqual(tp_long, 0.024, places=6)   # 0.032 * 0.75
        self.assertAlmostEqual(tp_short, 0.033, places=6)  # 0.044 * 0.75

    @override_settings(
        REGIME_DIRECTIONAL_PENALTY_ENABLED=True,
        REGIME_BEAR_LONG_PENALTY=0.15,
        REGIME_BULL_SHORT_PENALTY=0.10,
        BTC_BEAR_LONG_BLOCK_ENABLED=True,
        REGIME_BULL_SHORT_BLOCK_ENABLED=False,
    )
    def test_regime_directional_penalty_and_btc_block(self):
        mult1, blocked1, reason1 = _regime_directional_risk_mult("BTCUSDT", "long", "bear")
        self.assertTrue(blocked1)
        self.assertEqual(mult1, 0.0)
        self.assertEqual(reason1, "btc_bear_long_block")

        mult2, blocked2, reason2 = _regime_directional_risk_mult("ETHUSDT", "long", "bear")
        self.assertFalse(blocked2)
        self.assertAlmostEqual(mult2, 0.85, places=6)
        self.assertEqual(reason2, "bear_long_penalty")

        mult3, blocked3, reason3 = _regime_directional_risk_mult("ETHUSDT", "short", "bull")
        self.assertFalse(blocked3)
        self.assertAlmostEqual(mult3, 0.90, places=6)
        self.assertEqual(reason3, "bull_short_penalty")

    @override_settings(
        REGIME_DIRECTIONAL_PENALTY_ENABLED=True,
        REGIME_BULL_SHORT_BLOCK_ENABLED=True,
    )
    def test_regime_bull_short_can_be_hard_blocked(self):
        mult, blocked, reason = _regime_directional_risk_mult("ETHUSDT", "short", "bull")
        self.assertTrue(blocked)
        self.assertEqual(mult, 0.0)
        self.assertEqual(reason, "bull_short_block")

    @override_settings(
        BTC_LEAD_FILTER_ENABLED=True,
        BTC_LEAD_HARD_BLOCK_ENABLED=True,
        BTC_LEAD_ALT_RISK_PENALTY=0.20,
    )
    def test_btc_lead_filter_blocks_alt_long_in_confirmed_bear_without_tactical_override(self):
        mult, blocked, reason = _btc_lead_directional_risk_mult(
            "ETHUSDT",
            "long",
            "bear_confirmed",
            "short_bias",
        )
        self.assertTrue(blocked)
        self.assertEqual(mult, 0.0)
        self.assertEqual(reason, "btc_lead_bear_alt_long_block")

    @override_settings(
        BTC_LEAD_FILTER_ENABLED=True,
        BTC_LEAD_HARD_BLOCK_ENABLED=True,
        BTC_LEAD_ALT_RISK_PENALTY=0.20,
    )
    def test_btc_lead_filter_reduces_penalty_for_tactical_countertrend(self):
        mult, blocked, reason = _btc_lead_directional_risk_mult(
            "ETHUSDT",
            "long",
            "bear_confirmed",
            "tactical_long",
        )
        self.assertFalse(blocked)
        self.assertAlmostEqual(mult, 0.90, places=6)
        self.assertEqual(reason, "btc_lead_bear_alt_long_penalty")

    @override_settings(
        REGIME_BULL_SHORT_RETRACE_STRICT_ENABLED=True,
        REGIME_BULL_SHORT_RETRACE_MIN_SCORE=0.88,
        REGIME_BULL_SHORT_RETRACE_MIN_ALLOWED_MODULES=1,
        REGIME_BULL_SHORT_RETRACE_ALLOWED_MODULES={"meanrev", "smc", "carry"},
    )
    def test_bull_short_retrace_precheck_blocks_low_score(self):
        ok, reason = _bull_short_retrace_precheck(
            symbol="BTCUSDT",
            strategy_name="alloc_short",
            signal_direction="short",
            regime_bias="bull",
            sig_score=0.72,
            sig_payload={"reasons": {"module_rows": [{"module": "carry"}]}},
        )
        self.assertFalse(ok)
        self.assertIn("bull_short_low_retrace_score", reason)

    @override_settings(
        REGIME_BULL_SHORT_RETRACE_STRICT_ENABLED=True,
        REGIME_BULL_SHORT_RETRACE_MIN_SCORE=0.88,
        REGIME_BULL_SHORT_RETRACE_MIN_ALLOWED_MODULES=1,
        REGIME_BULL_SHORT_RETRACE_ALLOWED_MODULES={"meanrev", "smc", "carry"},
    )
    def test_bull_short_retrace_precheck_blocks_without_retrace_modules(self):
        ok, reason = _bull_short_retrace_precheck(
            symbol="BTCUSDT",
            strategy_name="alloc_short",
            signal_direction="short",
            regime_bias="bull",
            sig_score=0.93,
            sig_payload={"reasons": {"module_rows": [{"module": "trend"}]}},
        )
        self.assertFalse(ok)
        self.assertIn("bull_short_low_retrace_modules", reason)

    @override_settings(
        REGIME_BULL_SHORT_RETRACE_STRICT_ENABLED=True,
        REGIME_BULL_SHORT_RETRACE_MIN_SCORE=0.88,
        REGIME_BULL_SHORT_RETRACE_MIN_ALLOWED_MODULES=1,
        REGIME_BULL_SHORT_RETRACE_ALLOWED_MODULES={"meanrev", "smc", "carry"},
    )
    def test_bull_short_retrace_precheck_allows_high_score_with_retrace_module(self):
        ok, reason = _bull_short_retrace_precheck(
            symbol="BTCUSDT",
            strategy_name="alloc_short",
            signal_direction="short",
            regime_bias="bull",
            sig_score=0.92,
            sig_payload={"reasons": {"module_rows": [{"module": "carry"}, {"module": "trend"}]}},
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    @override_settings(
        REGIME_DIRECTIONAL_PENALTY_ENABLED=False,
        BTC_BEAR_LONG_BLOCK_ENABLED=True,
    )
    def test_regime_directional_penalty_can_be_disabled(self):
        mult, blocked, reason = _regime_directional_risk_mult("BTCUSDT", "long", "bear")
        self.assertFalse(blocked)
        self.assertAlmostEqual(mult, 1.0, places=6)
        self.assertEqual(reason, "disabled")

    @override_settings(
        MARKET_REGIME_ADX_MIN=17.0,
        MARKET_REGIME_ADX_MIN_BY_CONTEXT={},
    )
    def test_regime_adx_min_resolver_falls_back_to_global(self):
        out = _regime_adx_min_for_symbol_session("BTCUSDT", "asia", 17.0)
        self.assertAlmostEqual(out, 17.0, places=6)

    @override_settings(
        MARKET_REGIME_ADX_MIN=17.0,
        MARKET_REGIME_ADX_MIN_BY_CONTEXT={
            "BTCUSDT:asia": 13.0,
            "BTCUSDT:*": 14.0,
            "*:asia": 15.0,
            "*:*": 16.0,
        },
    )
    def test_regime_adx_min_resolver_precedence(self):
        btc_asia = _regime_adx_min_for_symbol_session("BTCUSDT", "asia", 17.0)
        btc_london = _regime_adx_min_for_symbol_session("BTCUSDT", "london", 17.0)
        eth_asia = _regime_adx_min_for_symbol_session("ETHUSDT", "asia", 17.0)
        sol_ny = _regime_adx_min_for_symbol_session("SOLUSDT", "ny", 17.0)
        self.assertAlmostEqual(btc_asia, 13.0, places=6)   # symbol+session
        self.assertAlmostEqual(btc_london, 14.0, places=6)  # symbol wildcard session
        self.assertAlmostEqual(eth_asia, 15.0, places=6)    # session wildcard symbol
        self.assertAlmostEqual(sol_ny, 16.0, places=6)      # global wildcard

    @override_settings(
        TRAILING_STOP_ENABLED=True,
        PARTIAL_CLOSE_AT_R=0.8,
        PARTIAL_CLOSE_PCT=0.5,
        PARTIAL_CLOSE_MIN_REMAINING_QTY=0.0,
    )
    def test_partial_close_supports_fractional_position_size(self):
        adapter = _DummyAdapter()
        calls: list[dict] = []

        def _capture_order(symbol, side, type_, amount, price=None, params=None):
            calls.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "type": type_,
                    "amount": amount,
                    "price": price,
                    "params": params or {},
                }
            )
            return {"id": "pc-1"}

        adapter.create_order = _capture_order

        with patch("execution.tasks._redis_client", return_value=_DummyRedis()):
            closed, fee = _check_trailing_stop(
                adapter=adapter,
                symbol="BTCUSDT",
                side="buy",
                current_qty=1.5,
                entry_price=100.0,
                last_price=102.0,  # pnl=2%, with sl=2% => 1R
                sl_pct=0.02,
            )
        self.assertFalse(closed)
        self.assertEqual(fee, 0.0)
        self.assertEqual(len(calls), 1)
        self.assertGreater(calls[0]["amount"], 0.0)
        self.assertLess(calls[0]["amount"], 1.5)
        self.assertEqual(calls[0]["params"].get("reduceOnly"), True)

    @override_settings(
        TRAILING_STOP_ENABLED=True,
        PARTIAL_CLOSE_AT_R=0.8,
        PARTIAL_CLOSE_PCT=0.5,
        TRAILING_STATE_TTL_SECONDS=111,
        TRAILING_ADAPTIVE_ENABLED=True,
        TRAILING_ACTIVATION_R_LOWVOL=2.5,
        TRAILING_ACTIVATION_R_HIGHVOL=1.5,
        TRAILING_ACTIVATION_ATR_THRESHOLD=0.015,
        TRAILING_LOCKIN_MIN=0.4,
        TRAILING_LOCKIN_MAX=0.7,
        TRAILING_LOCKIN_SLOPE=15.0,
    )
    def test_trailing_state_uses_configured_ttl_seconds(self):
        adapter = _DummyAdapter()
        client = _DummyRedis()
        with patch("execution.tasks._redis_client", return_value=client):
            _check_trailing_stop(
                adapter=adapter,
                symbol="BTCUSDT",
                side="buy",
                current_qty=2.0,
                entry_price=100.0,
                last_price=102.0,
                sl_pct=0.02,
                atr_pct=0.02,
            )
        self.assertTrue(any(k.startswith("trail:partial_done:") for k in client.expiry))
        self.assertTrue(any(k.startswith("trail:max_fav:") for k in client.expiry))
        self.assertTrue(any(k.startswith("trail:max_adv:") for k in client.expiry))
        self.assertTrue(any(k.startswith("trail:sl_pct:") for k in client.expiry))
        self.assertTrue(all(v == 111 for v in client.expiry.values()))

    @override_settings(
        TRAILING_STOP_ENABLED=True,
        BREAKEVEN_STOP_ENABLED=False,
        PARTIAL_CLOSE_AT_R=5.0,
        TRAILING_ADAPTIVE_ENABLED=True,
        TRAILING_ACTIVATION_R_LOWVOL=2.5,
        TRAILING_ACTIVATION_R_HIGHVOL=1.5,
        TRAILING_ACTIVATION_ATR_THRESHOLD=0.015,
        TACTICAL_EXIT_PROFILE_ENABLED=True,
        TACTICAL_EXIT_TRAIL_R_MULT=0.6,
        TACTICAL_EXIT_PARTIAL_R_MULT=1.0,
    )
    def test_tactical_exit_profile_activates_trailing_earlier(self):
        adapter = _DummyAdapter()
        client = _DummyRedis()
        with patch("execution.tasks._redis_client", return_value=client):
            with patch("execution.tasks._has_sl_stop_order", return_value=(False, 0.0, [])):
                with patch("execution.tasks._place_sl_order") as place_mock:
                    _check_trailing_stop(
                        adapter=adapter,
                        symbol="BTCUSDT",
                        side="buy",
                        current_qty=1.0,
                        entry_price=100.0,
                        last_price=103.0,  # 3% pnl, with 2% sl = 1.5R
                        sl_pct=0.02,
                        atr_pct=0.01,  # low-vol => normal activation 2.5R
                        recommended_bias="tactical_long",
                    )
                    place_mock.assert_called_once()

    @override_settings(
        EXCHANGE_CLOSE_CLASSIFY_STOP_SCALE=0.35,
        EXCHANGE_CLOSE_CLASSIFY_TP_SCALE=0.35,
        EXCHANGE_CLOSE_CLASSIFY_MIN_BAND_PCT=0.0015,
        EXCHANGE_CLOSE_CLASSIFY_BREAKEVEN_SCALE=0.20,
        STOP_LOSS_PCT=0.015,
        TAKE_PROFIT_PCT=0.02,
        MIN_SL_PCT=0.012,
    )
    def test_classify_exchange_close_uses_scaled_sl_tp_hints(self):
        adapter = _DummyAdapter()

        stop_reason = _classify_exchange_close(
            adapter=adapter,
            symbol="BTCUSDT",
            pos_side="buy",
            entry_price=100.0,
            liq_price_est=0.0,
            exit_price=99.3,  # -0.70%
            sl_pct_hint=0.012,
            tp_pct_hint=0.02,
        )
        self.assertEqual(stop_reason, "exchange_stop")

        tp_reason = _classify_exchange_close(
            adapter=adapter,
            symbol="BTCUSDT",
            pos_side="buy",
            entry_price=100.0,
            liq_price_est=0.0,
            exit_price=100.8,  # +0.80%
            sl_pct_hint=0.012,
            tp_pct_hint=0.02,
        )
        self.assertEqual(tp_reason, "exchange_tp_limit")

    @override_settings(
        EXCHANGE_CLOSE_CLASSIFY_STOP_SCALE=0.35,
        EXCHANGE_CLOSE_CLASSIFY_TP_SCALE=0.35,
        EXCHANGE_CLOSE_CLASSIFY_MIN_BAND_PCT=0.0015,
        EXCHANGE_CLOSE_CLASSIFY_BREAKEVEN_SCALE=0.20,
        STOP_LOSS_PCT=0.015,
        TAKE_PROFIT_PCT=0.02,
        MIN_SL_PCT=0.012,
    )
    def test_classify_exchange_close_near_breakeven_when_move_small_vs_sl(self):
        adapter = _DummyAdapter()
        reason = _classify_exchange_close(
            adapter=adapter,
            symbol="BTCUSDT",
            pos_side="buy",
            entry_price=100.0,
            liq_price_est=0.0,
            exit_price=99.61,  # -0.39%
            sl_pct_hint=0.02,  # SL is 2%, so -0.4% is still near-breakeven band
            tp_pct_hint=0.03,
        )
        self.assertEqual(reason, "near_breakeven")

    @override_settings(
        MACRO_HIGH_IMPACT_FILTER_ENABLED=True,
        MACRO_HIGH_IMPACT_UTC_HOURS={13, 14},
        MACRO_HIGH_IMPACT_WEEKDAYS={0, 1, 2, 3, 4},
        MACRO_HIGH_IMPACT_SESSIONS={"ny", "overlap"},
    )
    def test_macro_high_impact_window_active_only_when_all_filters_match(self):
        ts_active = datetime(2026, 2, 9, 13, 5, tzinfo=timezone.utc)  # Monday
        active, details = _is_macro_high_impact_window(ts_active, session_name="ny")
        self.assertTrue(active)
        self.assertTrue(details["hour_ok"])
        self.assertTrue(details["weekday_ok"])
        self.assertTrue(details["session_ok"])

        ts_inactive_hour = datetime(2026, 2, 9, 10, 0, tzinfo=timezone.utc)
        inactive_hour, _ = _is_macro_high_impact_window(ts_inactive_hour, session_name="ny")
        self.assertFalse(inactive_hour)

        ts_inactive_session = datetime(2026, 2, 9, 13, 5, tzinfo=timezone.utc)
        inactive_session, _ = _is_macro_high_impact_window(ts_inactive_session, session_name="asia")
        self.assertFalse(inactive_session)

    @override_settings(
        MACRO_HIGH_IMPACT_ALLOW_MICROVOL=True,
        MACRO_HIGH_IMPACT_ALLOW_MICROVOL_SYMBOLS={"BTCUSDT", "ETHUSDT"},
    )
    def test_macro_high_impact_allows_microvol_only_for_allowed_symbols(self):
        self.assertTrue(
            _macro_high_impact_allows_entry(
                strategy_name="mod_microvol_long",
                symbol="BTCUSDT",
            )
        )
        self.assertFalse(
            _macro_high_impact_allows_entry(
                strategy_name="mod_microvol_long",
                symbol="DOGEUSDT",
            )
        )
        self.assertFalse(
            _macro_high_impact_allows_entry(
                strategy_name="alloc_long",
                symbol="BTCUSDT",
            )
        )


class OperationReportFeeNetTest(TestCase):
    @override_settings(ML_ENTRY_FILTER_RETRAIN_ON_OPERATION_ENABLED=False)
    def test_log_operation_applies_fee_to_net_pnl(self):
        inst = Instrument.objects.create(
            symbol="BTCUSDT",
            exchange="bingx",
            base="BTC",
            quote="USDT",
        )

        _log_operation(
            inst=inst,
            side="buy",
            qty=1.0,
            entry_price=100.0,
            exit_price=102.0,
            reason="tp",
            signal_id="1",
            correlation_id="1-BTCUSDT",
            leverage=5.0,
            fee_usdt=0.5,
            opened_at=None,
            contract_size=1.0,
        )

        op = OperationReport.objects.get(instrument=inst)
        self.assertAlmostEqual(float(op.pnl_abs), 1.5, places=8)
        self.assertAlmostEqual(float(op.fee_usdt), 0.5, places=8)
        self.assertEqual(op.outcome, OperationReport.Outcome.WIN)

    @override_settings(
        ML_ENTRY_FILTER_RETRAIN_ON_OPERATION_ENABLED=False,
        NEAR_BREAKEVEN_LOSS_TO_BE_PCT=0.0015,
    )
    def test_log_operation_marks_small_near_breakeven_loss_as_be(self):
        inst = Instrument.objects.create(
            symbol="ETHUSDT",
            exchange="bingx",
            base="ETH",
            quote="USDT",
        )

        _log_operation(
            inst=inst,
            side="buy",
            qty=1.0,
            entry_price=100.0,
            exit_price=99.95,  # -0.05%
            reason="exchange_close",
            signal_id="2",
            correlation_id="2-ETHUSDT",
            leverage=5.0,
            fee_usdt=0.0,
            opened_at=None,
            contract_size=1.0,
            close_sub_reason="near_breakeven",
        )

        op = OperationReport.objects.get(instrument=inst)
        self.assertEqual(op.outcome, OperationReport.Outcome.BE)

    @override_settings(
        ML_ENTRY_FILTER_RETRAIN_ON_OPERATION_ENABLED=False,
        NEAR_BREAKEVEN_LOSS_TO_BE_PCT=0.0015,
    )
    def test_log_operation_keeps_larger_near_breakeven_loss_as_loss(self):
        inst = Instrument.objects.create(
            symbol="XRPUSDT",
            exchange="bingx",
            base="XRP",
            quote="USDT",
        )

        _log_operation(
            inst=inst,
            side="buy",
            qty=1.0,
            entry_price=100.0,
            exit_price=99.6,  # -0.40%
            reason="exchange_close",
            signal_id="3",
            correlation_id="3-XRPUSDT",
            leverage=5.0,
            fee_usdt=0.0,
            opened_at=None,
            contract_size=1.0,
            close_sub_reason="near_breakeven",
        )

        op = OperationReport.objects.get(instrument=inst)
        self.assertEqual(op.outcome, OperationReport.Outcome.LOSS)


class LatestSignalSelectionTest(TestCase):
    def test_allocator_mode_includes_microvol_direct_signal(self):
        inst = Instrument.objects.create(
            symbol="BTCUSDT",
            exchange="bingx",
            base="BTC",
            quote="USDT",
            enabled=True,
        )
        Candle.objects.create(
            instrument=inst,
            timeframe="1m",
            ts=dj_tz.now(),
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1,
        )
        Signal.objects.create(
            instrument=inst,
            strategy="alloc_long",
            ts=dj_tz.now() - timedelta(minutes=2),
            payload_json={"direction": "long"},
            score=0.8,
        )
        micro = Signal.objects.create(
            instrument=inst,
            strategy="mod_microvol_long",
            ts=dj_tz.now() - timedelta(seconds=30),
            payload_json={"direction": "long"},
            score=0.9,
        )

        instruments, latest_signals, _ = _load_enabled_instruments_and_latest_signals(True)
        latest_ids = {inst_obj.latest_signal_id for inst_obj in instruments if inst_obj.symbol == "BTCUSDT"}
        self.assertEqual(latest_ids, {micro.id})
        self.assertIn(micro.id, latest_signals)

    @override_settings(ML_ENTRY_FILTER_RETRAIN_ON_OPERATION_ENABLED=False)
    def test_log_operation_persists_mfe_mae_capture_metrics(self):
        inst = Instrument.objects.create(
            symbol="BTCUSDT",
            exchange="bingx",
            base="BTC",
            quote="USDT",
        )
        opened_at = dj_tz.now().replace(microsecond=0)
        state_key = f"{inst.symbol}:{int(opened_at.timestamp())}"
        redis_stub = _DummyRedis()
        redis_stub.set(f"trail:max_fav:{state_key}", "0.04")
        redis_stub.set(f"trail:max_adv:{state_key}", "0.01")
        redis_stub.set(f"trail:sl_pct:{state_key}", "0.02")
        redis_stub.set(f"trail:partial_done:{state_key}", "1")

        with patch("execution.tasks._redis_client", return_value=redis_stub):
            _log_operation(
                inst=inst,
                side="buy",
                qty=1.0,
                entry_price=100.0,
                exit_price=101.0,
                reason="tp",
                signal_id="4",
                correlation_id="4-BTCUSDT",
                leverage=5.0,
                fee_usdt=0.0,
                opened_at=opened_at,
                contract_size=1.0,
            )
        op = OperationReport.objects.get(instrument=inst)
        self.assertAlmostEqual(float(op.mfe_r or 0.0), 2.0, places=6)
        self.assertAlmostEqual(float(op.mae_r or 0.0), 0.5, places=6)
        self.assertAlmostEqual(float(op.mfe_capture_ratio or 0.0), 0.25, places=6)
        self.assertIsNone(redis_stub.get(f"trail:max_fav:{state_key}"))
        self.assertIsNone(redis_stub.get(f"trail:max_adv:{state_key}"))
        self.assertIsNone(redis_stub.get(f"trail:sl_pct:{state_key}"))
        self.assertIsNone(redis_stub.get(f"trail:partial_done:{state_key}"))

    @override_settings(ML_ENTRY_FILTER_RETRAIN_ON_OPERATION_ENABLED=False)
    def test_log_operation_persists_regime_snapshot_fields(self):
        inst = Instrument.objects.create(
            symbol="ETHUSDT",
            exchange="bingx",
            base="ETH",
            quote="USDT",
        )
        with patch(
            "execution.tasks._operation_regime_snapshot",
            return_value={
                "monthly_regime": "bear_confirmed",
                "weekly_regime": "bear_weak",
                "daily_regime": "bull_weak",
                "btc_lead_state": "bear_confirmed",
                "recommended_bias": "tactical_long",
            },
        ):
            _log_operation(
                inst=inst,
                side="buy",
                qty=1.0,
                entry_price=100.0,
                exit_price=101.0,
                reason="tp",
                signal_id="5",
                correlation_id="5-ETHUSDT",
                leverage=5.0,
                fee_usdt=0.0,
                opened_at=None,
                contract_size=1.0,
            )

        op = OperationReport.objects.get(instrument=inst)
        self.assertEqual(op.monthly_regime, "bear_confirmed")
        self.assertEqual(op.weekly_regime, "bear_weak")
        self.assertEqual(op.daily_regime, "bull_weak")
        self.assertEqual(op.btc_lead_state, "bear_confirmed")
        self.assertEqual(op.recommended_bias, "tactical_long")


class OperationReportMlRetrainTriggerTest(TestCase):
    def _create_inst(self, symbol: str = "SOLUSDT") -> Instrument:
        return Instrument.objects.create(
            symbol=symbol,
            exchange="bingx",
            base=symbol.replace("USDT", ""),
            quote="USDT",
        )

    @override_settings(
        ML_ENTRY_FILTER_ENABLED=True,
        ML_ENTRY_FILTER_AUTO_TRAIN_ENABLED=True,
        ML_ENTRY_FILTER_RETRAIN_ON_OPERATION_ENABLED=True,
        ML_ENTRY_FILTER_RETRAIN_ON_OPERATION_MIN_INTERVAL_SECONDS=0,
    )
    def test_log_operation_queues_retrain_on_close(self):
        inst = self._create_inst("SOLUSDT")
        with patch("execution.tasks._redis_client", return_value=_DummyRedis()):
            with patch("execution.tasks.retrain_entry_filter_model.delay") as delay_mock:
                _log_operation(
                    inst=inst,
                    side="buy",
                    qty=1.0,
                    entry_price=100.0,
                    exit_price=101.0,
                    reason="tp",
                    signal_id="1",
                    correlation_id="1-SOLUSDT",
                    leverage=3.0,
                    fee_usdt=0.0,
                    opened_at=None,
                    contract_size=1.0,
                )
        delay_mock.assert_called_once_with()

    @override_settings(
        ML_ENTRY_FILTER_ENABLED=True,
        ML_ENTRY_FILTER_AUTO_TRAIN_ENABLED=True,
        ML_ENTRY_FILTER_RETRAIN_ON_OPERATION_ENABLED=True,
        ML_ENTRY_FILTER_RETRAIN_ON_OPERATION_MIN_INTERVAL_SECONDS=300,
    )
    def test_log_operation_retrain_throttled_by_redis_interval(self):
        inst = self._create_inst("ADAUSDT")
        dummy = _DummyRedis()
        dummy.set("ml:entry_filter:retrain:last_trigger_at", int(dj_tz.now().timestamp()))
        with patch("execution.tasks._redis_client", return_value=dummy):
            with patch("execution.tasks.retrain_entry_filter_model.delay") as delay_mock:
                _log_operation(
                    inst=inst,
                    side="sell",
                    qty=1.0,
                    entry_price=100.0,
                    exit_price=99.0,
                    reason="sl",
                    signal_id="2",
                    correlation_id="2-ADAUSDT",
                    leverage=3.0,
                    fee_usdt=0.0,
                    opened_at=None,
                    contract_size=1.0,
                )
        delay_mock.assert_not_called()


class ExecutionVolumeGateTest(TestCase):
    def setUp(self):
        self.inst = Instrument.objects.create(
            symbol="SOLUSDT",
            exchange="bingx",
            base="SOL",
            quote="USDT",
        )

    def _seed_5m_volumes(self, values: list[float]) -> None:
        start = dj_tz.now() - timedelta(minutes=5 * len(values))
        rows = []
        for idx, vol in enumerate(values):
            ts = start + timedelta(minutes=5 * idx)
            price = 100 + idx
            rows.append(
                Candle(
                    instrument=self.inst,
                    timeframe="5m",
                    ts=ts,
                    open=price,
                    high=price + 0.5,
                    low=price - 0.5,
                    close=price + 0.1,
                    volume=vol,
                )
            )
        Candle.objects.bulk_create(rows)

    def test_volume_activity_ratio_uses_recent_median(self):
        self._seed_5m_volumes([100.0] * 48 + [50.0])
        ratio = _volume_activity_ratio(self.inst, tf="5m", lookback=48)
        self.assertIsNotNone(ratio)
        self.assertAlmostEqual(float(ratio), 0.5, places=6)

    @override_settings(
        ENTRY_VOLUME_FILTER_ENABLED=True,
        ENTRY_VOLUME_FILTER_TIMEFRAME="5m",
        ENTRY_VOLUME_FILTER_LOOKBACK=48,
        ENTRY_VOLUME_FILTER_MIN_RATIO=0.75,
        ENTRY_VOLUME_FILTER_FAIL_OPEN=False,
    )
    def test_volume_gate_blocks_low_ratio_when_fail_closed(self):
        self._seed_5m_volumes([100.0] * 48 + [50.0])
        allowed, ratio = _volume_gate_allowed(self.inst)
        self.assertFalse(allowed)
        self.assertIsNotNone(ratio)
        self.assertAlmostEqual(float(ratio), 0.5, places=6)

    @override_settings(
        ENTRY_VOLUME_FILTER_ENABLED=True,
        ENTRY_VOLUME_FILTER_TIMEFRAME="5m",
        ENTRY_VOLUME_FILTER_LOOKBACK=48,
        ENTRY_VOLUME_FILTER_MIN_RATIO=0.75,
        ENTRY_VOLUME_FILTER_FAIL_OPEN=True,
    )
    def test_volume_gate_allows_on_missing_data_when_fail_open(self):
        allowed, ratio = _volume_gate_allowed(self.inst)
        self.assertTrue(allowed)
        self.assertIsNone(ratio)

    @override_settings(
        ENTRY_VOLUME_FILTER_ENABLED=True,
        ENTRY_VOLUME_FILTER_TIMEFRAME="5m",
        ENTRY_VOLUME_FILTER_LOOKBACK=48,
        ENTRY_VOLUME_FILTER_MIN_RATIO=0.75,
        ENTRY_VOLUME_FILTER_MIN_RATIO_BY_SESSION={"asia": 0.60, "ny": 0.95},
        ENTRY_VOLUME_FILTER_FAIL_OPEN=False,
    )
    def test_volume_gate_uses_session_specific_threshold(self):
        self._seed_5m_volumes([100.0] * 48 + [70.0])  # ratio=0.70
        allowed_asia, ratio_asia = _volume_gate_allowed(self.inst, session_name="asia")
        allowed_ny, ratio_ny = _volume_gate_allowed(self.inst, session_name="ny")
        self.assertTrue(allowed_asia)
        self.assertFalse(allowed_ny)
        self.assertAlmostEqual(float(ratio_asia or 0.0), 0.7, places=6)
        self.assertAlmostEqual(float(ratio_ny or 0.0), 0.7, places=6)

    @override_settings(
        ENTRY_VOLUME_FILTER_MIN_RATIO=0.75,
        ENTRY_VOLUME_FILTER_MIN_RATIO_BY_SESSION={"asia": 0.60},
    )
    def test_volume_gate_min_ratio_falls_back_to_global(self):
        self.assertAlmostEqual(_volume_gate_min_ratio("asia"), 0.60, places=6)
        self.assertAlmostEqual(_volume_gate_min_ratio("london"), 0.75, places=6)


class PyramidingHelpersTest(TestCase):
    def test_safe_correlation_id_truncates(self):
        raw = "x" * 80
        self.assertEqual(len(_safe_correlation_id(raw)), 64)
        self.assertEqual(_safe_correlation_id(""), "")

    def test_position_root_and_add_count(self):
        inst = Instrument.objects.create(
            symbol="ETHUSDT",
            exchange="bingx",
            base="ETH",
            quote="USDT",
        )
        root = "123-ETHUSDT"
        Order.objects.create(
            instrument=inst,
            side=Order.OrderSide.BUY,
            type=Order.OrderType.MARKET,
            qty=1.0,
            price=2000,
            status=Order.OrderStatus.FILLED,
            correlation_id=root,
            parent_correlation_id=root,
            opened_at=dj_tz.now(),
        )
        Order.objects.create(
            instrument=inst,
            side=Order.OrderSide.BUY,
            type=Order.OrderType.MARKET,
            qty=0.5,
            price=2010,
            status=Order.OrderStatus.FILLED,
            correlation_id=f"{root}:add1",
            parent_correlation_id=root,
            opened_at=dj_tz.now(),
        )
        self.assertEqual(_position_root_correlation(inst, "buy"), root)
        self.assertEqual(_count_pyramid_adds(inst, "buy", root), 1)


class PositionOriginAttributionTest(TestCase):
    def setUp(self):
        self.inst = Instrument.objects.create(
            symbol="ETHUSDT",
            exchange="bingx",
            base="ETH",
            quote="USDT",
        )

    def test_position_origin_refs_prefer_entry_signal_and_root_correlation(self):
        origin_signal = Signal.objects.create(
            strategy="mod_microvol_long",
            instrument=self.inst,
            ts=dj_tz.now() - timedelta(minutes=5),
            payload_json={"origin": "entry"},
            score=0.91,
        )
        current_sig = Signal.objects.create(
            strategy="alloc_flat",
            instrument=self.inst,
            ts=dj_tz.now(),
            payload_json={},
            score=0.0,
        )
        root = f"{origin_signal.id}-{self.inst.symbol}"
        Order.objects.create(
            instrument=self.inst,
            side=Order.OrderSide.BUY,
            type=Order.OrderType.MARKET,
            qty=1.0,
            price=2000,
            status=Order.OrderStatus.FILLED,
            reduce_only=False,
            correlation_id=root,
            parent_correlation_id=root,
            opened_at=dj_tz.now() - timedelta(minutes=5),
        )

        signal_id, correlation_id = _position_origin_refs(self.inst, "buy", origin_signal, current_sig)
        self.assertEqual(signal_id, str(origin_signal.id))
        self.assertEqual(correlation_id, root)

    @override_settings(
        AI_EXIT_GATE_ENABLED=False,
        TRAILING_STOP_ENABLED=True,
        TP_PROGRESS_EARLY_EXIT_ENABLED=False,
    )
    def test_manage_open_position_logs_origin_signal_on_tp_close(self):
        origin_signal = Signal.objects.create(
            strategy="mod_microvol_long",
            instrument=self.inst,
            ts=dj_tz.now() - timedelta(minutes=6),
            payload_json={"origin": "entry"},
            score=0.88,
        )
        current_sig = Signal.objects.create(
            strategy="alloc_flat",
            instrument=self.inst,
            ts=dj_tz.now(),
            payload_json={},
            score=0.0,
        )
        root = f"{origin_signal.id}-{self.inst.symbol}"
        opened_at = dj_tz.now() - timedelta(minutes=5)
        Order.objects.create(
            instrument=self.inst,
            side=Order.OrderSide.BUY,
            type=Order.OrderType.MARKET,
            qty=1.0,
            price=2000,
            status=Order.OrderStatus.FILLED,
            reduce_only=False,
            correlation_id=root,
            parent_correlation_id=root,
            opened_at=opened_at,
        )

        with patch("execution.tasks._redis_client", return_value=None):
            with patch("execution.tasks._reconcile_sl"):
                with patch("execution.tasks._check_trailing_stop", return_value=(False, 0.0)):
                    with patch("execution.tasks._tp_sl_gate_pnl_pct", return_value=(0.02, 0.0)):
                        with patch("execution.tasks._compute_tp_sl_prices", return_value=(102.0, 99.0, 0.01, 0.01)):
                            with patch("execution.tasks.notify_trade_closed"):
                                with patch("execution.tasks._mark_position_closed"):
                                    skip_symbol, _, _, _ = _manage_open_position(
                                        adapter=_DummyAdapter(),
                                        inst=self.inst,
                                        sig=current_sig,
                                        sig_payload={},
                                        strategy_name=current_sig.strategy,
                                        symbol=self.inst.symbol,
                                        ticker_used={"last": 2020.0},
                                        last_price=2020.0,
                                        current_qty=1.0,
                                        entry_price=2000.0,
                                        pos_opened_at=opened_at,
                                        signal_direction="flat",
                                        side="",
                                        direction_allowed=True,
                                        atr=None,
                                        contract_size=1.0,
                                        leverage=5.0,
                                        equity_usdt=1000.0,
                                        current_session="ny",
                                        btc_recommended_bias="",
                                        account_ai_enabled=False,
                                        account_ai_config_id=None,
                                        account_owner_id=None,
                                        account_alias="rortigoza",
                                        account_service="bingx",
                                    )

        self.assertTrue(skip_symbol)
        op = OperationReport.objects.get(instrument=self.inst)
        self.assertEqual(op.reason, "tp")
        self.assertEqual(op.signal_id, str(origin_signal.id))
        self.assertEqual(op.correlation_id, root)


class AiFeedbackRetrySuppressTest(TestCase):
    def test_feedback_dedup_suppresses_when_coarse_fingerprint_matches(self):
        fp = _ai_entry_market_fingerprint(
            symbol="DOGEUSDT",
            strategy_name="alloc_short",
            signal_direction="short",
            session_name="london",
            sig_score=0.9577,
            atr=0.0102,
            spread_bps=2.05,
            sl_pct=0.0120,
            sig_payload={},
        )
        fp_coarse = _ai_entry_market_fingerprint(
            symbol="DOGEUSDT",
            strategy_name="alloc_short",
            signal_direction="short",
            session_name="london",
            sig_score=0.9577,
            atr=0.0102,
            spread_bps=2.05,
            sl_pct=0.0120,
            sig_payload={},
            coarse=True,
        )
        AiFeedbackEvent.objects.create(
            event_type="ai_gate_decision",
            level=AiFeedbackEvent.Level.WARNING,
            account_alias="rortigoza",
            account_service="bingx",
            symbol="DOGEUSDT",
            strategy="alloc_short",
            allow=False,
            risk_mult=0.35,
            reason="spr_hi,sl_tight,rb_low,contra_sig",
            payload_json={
                "direction": "short",
                "session": "london",
                "sig_score": 0.9577,
                "spread_bps": 2.05,
                "market_fp": fp,
                "market_fp_coarse": fp_coarse,
            },
        )

        suppressed, reason = _ai_entry_should_suppress_retry_from_feedback(
            account_alias="rortigoza",
            account_service="bingx",
            symbol="DOGEUSDT",
            strategy_name="alloc_short",
            signal_direction="short",
            session_name="london",
            sig_score=0.9580,
            spread_bps=2.10,
            market_fingerprint=fp,
            market_fingerprint_coarse=fp_coarse,
        )
        self.assertTrue(suppressed)
        self.assertEqual(reason, "spr_hi,sl_tight,rb_low,contra_sig")

    def test_feedback_dedup_does_not_suppress_for_other_direction(self):
        AiFeedbackEvent.objects.create(
            event_type="ai_gate_decision",
            level=AiFeedbackEvent.Level.WARNING,
            account_alias="rortigoza",
            account_service="bingx",
            symbol="DOGEUSDT",
            strategy="alloc_short",
            allow=False,
            reason="contra_sig",
            payload_json={
                "direction": "short",
                "session": "london",
                "sig_score": 0.95,
                "spread_bps": 2.0,
            },
        )
        suppressed, _ = _ai_entry_should_suppress_retry_from_feedback(
            account_alias="rortigoza",
            account_service="bingx",
            symbol="DOGEUSDT",
            strategy_name="alloc_short",
            signal_direction="long",
            session_name="london",
            sig_score=0.95,
            spread_bps=2.0,
            market_fingerprint="",
            market_fingerprint_coarse="",
        )
        self.assertFalse(suppressed)


class RiskEventDedupTest(TestCase):
    @override_settings(RISK_EVENT_DEDUP_SECONDS=120)
    def test_create_risk_event_dedups_by_kind_symbol_namespace_window(self):
        inst = Instrument.objects.create(
            symbol="SOLUSDT",
            exchange="bingx",
            base="SOL",
            quote="USDT",
        )
        redis_stub = _DummyRedis()
        fixed_now = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)

        with patch("execution.tasks._redis_client", return_value=redis_stub):
            with patch("execution.tasks.notify_risk_event"):
                with patch("execution.tasks.dj_tz.now", return_value=fixed_now):
                    _create_risk_event(
                        "daily_dd_limit",
                        "high",
                        instrument=inst,
                        details={"dd": -0.05},
                        risk_ns="stack_rortigoza",
                    )
                    _create_risk_event(
                        "daily_dd_limit",
                        "critical",  # different severity should still dedup in same window
                        instrument=inst,
                        details={"dd": -0.09},
                        risk_ns="stack_rortigoza",
                    )
        self.assertEqual(
            OperationReport.objects.filter(instrument=inst).count(),
            0,
        )
        self.assertEqual(
            RiskEvent.objects.filter(
                instrument=inst,
                kind="daily_dd_limit",
            ).count(),
            1,
        )

    def test_create_risk_event_formats_stale_details_for_notification(self):
        inst = Instrument.objects.create(
            symbol="ETHUSDT",
            exchange="bingx",
            base="ETH",
            quote="USDT",
        )
        redis_stub = _DummyRedis()

        with patch("execution.tasks._redis_client", return_value=redis_stub):
            with patch("execution.tasks.notify_risk_event") as notify_mock:
                _create_risk_event(
                    "data_stale",
                    "medium",
                    instrument=inst,
                    details={
                        "symbol": "ETHUSDT",
                        "latest_ts": "2026-03-10T02:06:00+00:00",
                        "age_seconds": 40556,
                    },
                    risk_ns="stack_rortigoza",
                )

        self.assertEqual(notify_mock.call_count, 1)
        args = notify_mock.call_args[0]
        self.assertEqual(args[0], "data_stale")
        self.assertEqual(args[1], "medium")
        self.assertIn("Symbol: ETHUSDT", args[2])
        self.assertIn("Latest 1m: 2026-03-10T02:06:00+00:00", args[2])
        self.assertIn("Age: 40556s", args[2])


class DataStalenessTransitionTest(TestCase):
    @override_settings(DATA_STALE_SECONDS=300)
    def test_track_data_staleness_emits_only_on_state_transition(self):
        inst = Instrument.objects.create(
            symbol="ETHUSDT",
            exchange="bingx",
            base="ETH",
            quote="USDT",
        )
        redis_stub = _DummyRedis()
        initial_now = datetime(2026, 3, 10, 13, 0, tzinfo=timezone.utc)
        stale_latest = initial_now - timedelta(minutes=10)

        with patch("execution.tasks._redis_client", return_value=redis_stub):
            is_stale, should_emit, details = _track_data_staleness_transition(
                inst,
                stale_latest,
                now_ts=initial_now,
                risk_ns="stack_rortigoza",
            )
            self.assertTrue(is_stale)
            self.assertTrue(should_emit)
            self.assertEqual(details["symbol"], "ETHUSDT")

            is_stale, should_emit, _ = _track_data_staleness_transition(
                inst,
                stale_latest,
                now_ts=initial_now + timedelta(minutes=1),
                risk_ns="stack_rortigoza",
            )
            self.assertTrue(is_stale)
            self.assertFalse(should_emit)

            recovery_now = initial_now + timedelta(minutes=2)
            fresh_latest = recovery_now - timedelta(seconds=60)
            is_stale, should_emit, details = _track_data_staleness_transition(
                inst,
                fresh_latest,
                now_ts=recovery_now,
                risk_ns="stack_rortigoza",
            )
            self.assertFalse(is_stale)
            self.assertFalse(should_emit)
            self.assertEqual(details, {})

            is_stale, should_emit, _ = _track_data_staleness_transition(
                inst,
                recovery_now - timedelta(minutes=9),
                now_ts=recovery_now + timedelta(minutes=1),
                risk_ns="stack_rortigoza",
            )
            self.assertTrue(is_stale)
            self.assertTrue(should_emit)
