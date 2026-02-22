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
    _safe_correlation_id,
    _signal_active_modules,
    _is_macro_high_impact_window,
    _volume_activity_ratio,
    _volume_gate_allowed,
    _volume_gate_min_ratio,
)


class _DummyRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    def ping(self):
        return True

    def get(self, key: str):
        return self.store.get(key)

    def set(self, key: str, value, nx: bool = False, ex: int | None = None):
        if nx and key in self.store:
            return False
        self.store[key] = str(value)
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

    def test_normalize_order_qty_uses_exchange_precision(self):
        adapter = _DummyAdapter()
        self.assertAlmostEqual(_normalize_order_qty(adapter, "BTC/USDT:USDT", 1.29), 1.2, places=8)
        self.assertEqual(_normalize_order_qty(adapter, "BTC/USDT:USDT", -1), 0.0)

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
