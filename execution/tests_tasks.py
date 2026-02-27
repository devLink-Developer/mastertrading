import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone as dj_tz

from core.models import Instrument
from execution.models import OperationReport, Order
from marketdata.models import Candle
from execution.tasks import (
    _acquire_task_lock,
    _check_trailing_stop,
    _classify_exchange_close,
    _count_pyramid_adds,
    _check_drawdown,
    _compute_tp_sl_prices,
    _extract_trigger_price,
    _extract_fee_usdt,
    _is_insufficient_margin_error,
    _ml_entry_filter_model_path,
    _ml_entry_filter_min_prob,
    _market_min_qty,
    _is_no_position_error,
    _log_operation,
    _normalize_order_qty,
    _position_root_correlation,
    _release_task_lock,
    _reconcile_sl,
    _resolve_signal_direction,
    _safe_correlation_id,
    _signal_active_modules,
    _signal_entry_reason,
    _is_macro_high_impact_window,
    _volume_activity_ratio,
    _volume_gate_allowed,
    _volume_gate_min_ratio,
    _tp_sl_gate_pnl_pct,
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
        dummy = _DummyRedis()
        with patch("execution.tasks._redis_client", return_value=dummy):
            allowed, dd = _check_drawdown(0.0, risk_ns="t")
            self.assertTrue(allowed)
            self.assertEqual(dd, 0.0)
            self.assertEqual(dummy.store, {})

            allowed2, dd2 = _check_drawdown(100.0, risk_ns="t")
            self.assertTrue(allowed2)
            self.assertEqual(dd2, 0.0)
            self.assertTrue(any(k.startswith("risk:equity_start:t:") for k in dummy.store.keys()))

    def test_market_min_qty_prefers_market_limits(self):
        market = {"limits": {"amount": {"min": 0.2}}}
        self.assertEqual(_market_min_qty(market, fallback=1.0), 0.2)
        self.assertEqual(_market_min_qty({}, fallback=1.0), 1.0)

    def test_market_min_qty_uses_precision_when_limits_missing(self):
        market = {"precision": {"amount": 0}}
        self.assertEqual(_market_min_qty(market, fallback=0.0), 1.0)

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

    @override_settings(PER_INSTRUMENT_RISK={"BTCUSDT": 0.0015})
    def test_volatility_adjusted_risk_caps_per_symbol_to_base_allocator_risk(self):
        # Per-instrument config should never increase a low-confidence allocator budget.
        risk = _volatility_adjusted_risk("BTCUSDT", atr_pct=0.01, base_risk=0.0005)
        self.assertAlmostEqual(risk, 0.0005, places=8)

    @override_settings(PER_INSTRUMENT_RISK={"BTCUSDT": 0.0015})
    def test_volatility_adjusted_risk_still_honors_lower_per_symbol_cap(self):
        # If allocator budget is higher than per-symbol cap, the lower cap wins.
        risk = _volatility_adjusted_risk("BTCUSDT", atr_pct=0.01, base_risk=0.0025)
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
            )
        self.assertTrue(any(k.startswith("trail:partial_done:") for k in client.expiry))
        self.assertTrue(any(k.startswith("trail:max_fav:") for k in client.expiry))
        self.assertTrue(all(v == 111 for v in client.expiry.values()))

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
