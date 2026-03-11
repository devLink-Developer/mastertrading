from datetime import datetime, timedelta, timezone

from django.test import TestCase

from core.models import Instrument
from execution.models import OperationReport
from risk.management.commands.perf_dashboard import Command
from signals.models import Signal


class PerfDashboardCommandTest(TestCase):
    def setUp(self):
        self.command = Command()
        self.instrument = Instrument.objects.create(
            symbol="ETHUSDT",
            exchange="bingx",
            base="ETH",
            quote="USDT",
            enabled=True,
        )

    def test_load_and_enrich_trade_uses_signal_and_stored_regime_snapshot(self):
        signal = Signal.objects.create(
            strategy="alloc_short",
            instrument=self.instrument,
            ts=datetime(2026, 3, 10, 10, 0, tzinfo=timezone.utc),
            score=0.92,
            payload_json={
                "reasons": {
                    "module_contributions": [
                        {"module": "carry", "contribution": -0.40},
                        {"module": "trend", "contribution": -0.25},
                    ]
                }
            },
        )
        OperationReport.objects.create(
            instrument=self.instrument,
            side="sell",
            qty=1,
            entry_price=2000,
            exit_price=1980,
            pnl_abs=10,
            pnl_pct=0.005,
            notional_usdt=2000,
            margin_used_usdt=400,
            fee_usdt=1,
            leverage=5,
            equity_before=1000,
            equity_after=1010,
            mode="demo",
            opened_at=datetime(2026, 3, 10, 10, 0, tzinfo=timezone.utc),
            closed_at=datetime(2026, 3, 10, 11, 0, tzinfo=timezone.utc),
            outcome=OperationReport.Outcome.WIN,
            reason="exchange_close",
            close_sub_reason="exchange_stop",
            signal_id=str(signal.id),
            correlation_id="abc",
            mfe_r=2.0,
            mae_r=0.5,
            mfe_capture_ratio=0.6,
            monthly_regime="bear_confirmed",
            weekly_regime="transition",
            daily_regime="bear_weak",
            btc_lead_state="bear_confirmed",
            recommended_bias="short_bias",
        )

        trades = self.command._load_trades(days=30, symbol="ETHUSDT")
        self.assertEqual(len(trades), 1)

        self.command._enrich_with_signal(trades)
        self.command._enrich_with_regime(trades)
        trade = trades[0]

        self.assertEqual(trade["session"], "london")
        self.assertEqual(trade["weekday"], "tuesday")
        self.assertEqual(trade["reason_detail"], "exchange_close:exchange_stop")
        self.assertEqual(trade["strategy"], "alloc_short")
        self.assertEqual(trade["dominant_module"], "carry")
        self.assertEqual(trade["regime"], "bear_weak")
        self.assertEqual(
            trade["mtf_snapshot"],
            "bear_confirmed|transition|bear_weak",
        )
        self.assertEqual(trade["btc_lead_state"], "bear_confirmed")
        self.assertEqual(trade["recommended_bias"], "short_bias")

    def test_capture_hotspots_and_stop_clusters(self):
        trades = [
            {
                "symbol": "ETHUSDT",
                "session": "london",
                "side": "sell",
                "reason": "sl",
                "reason_detail": "sl",
                "pnl_pct": -0.01,
                "duration_min": 30,
                "mfe_capture_ratio": 0.20,
                "mfe_r": 1.6,
                "mae_r": 0.8,
            },
            {
                "symbol": "ETHUSDT",
                "session": "london",
                "side": "sell",
                "reason": "sl",
                "reason_detail": "exchange_close:exchange_stop",
                "pnl_pct": -0.008,
                "duration_min": 25,
                "mfe_capture_ratio": 0.25,
                "mfe_r": 1.3,
                "mae_r": 0.9,
            },
            {
                "symbol": "ETHUSDT",
                "session": "london",
                "side": "sell",
                "reason": "tp_progress_exit",
                "reason_detail": "tp_progress_exit:giveback",
                "pnl_pct": 0.004,
                "duration_min": 20,
                "mfe_capture_ratio": 0.30,
                "mfe_r": 1.1,
                "mae_r": 0.2,
            },
            {
                "symbol": "BTCUSDT",
                "session": "ny",
                "side": "buy",
                "reason": "tp",
                "reason_detail": "tp",
                "pnl_pct": 0.012,
                "duration_min": 45,
                "mfe_capture_ratio": 0.80,
                "mfe_r": 2.5,
                "mae_r": 0.4,
            },
            {
                "symbol": "BTCUSDT",
                "session": "ny",
                "side": "buy",
                "reason": "tp",
                "reason_detail": "tp",
                "pnl_pct": 0.009,
                "duration_min": 40,
                "mfe_capture_ratio": 0.78,
                "mfe_r": 2.2,
                "mae_r": 0.5,
            },
            {
                "symbol": "BTCUSDT",
                "session": "ny",
                "side": "buy",
                "reason": "trailing_stop",
                "reason_detail": "trailing_stop",
                "pnl_pct": 0.006,
                "duration_min": 35,
                "mfe_capture_ratio": 0.60,
                "mfe_r": 1.8,
                "mae_r": 0.6,
            },
        ]

        hotspots = self.command._capture_hotspots(trades, min_trades=3)
        self.assertGreaterEqual(len(hotspots), 2)
        self.assertEqual(hotspots[0]["bucket"], "ETHUSDT|london")
        self.assertAlmostEqual(hotspots[0]["mfe_capture_avg"], 0.25, places=4)

        stop_clusters = self.command._stop_clusters(trades, min_trades=2)
        self.assertEqual(len(stop_clusters), 1)
        self.assertEqual(stop_clusters[0]["bucket"], "ETHUSDT|sell|london")
        self.assertEqual(stop_clusters[0]["n"], 2)

    def test_enrich_with_regime_falls_back_to_candle_inference(self):
        base_ts = datetime(2026, 3, 10, 0, 0, tzinfo=timezone.utc)
        from marketdata.models import Candle

        price = 100.0
        for idx in range(60):
            Candle.objects.create(
                instrument=self.instrument,
                timeframe="1h",
                ts=base_ts + timedelta(hours=idx),
                open=price + idx,
                high=price + idx + 1,
                low=price + idx - 1,
                close=price + idx + 0.5,
                volume=1000 + idx,
            )

        trades = [{
            "symbol": "ETHUSDT",
            "closed_at": base_ts + timedelta(hours=59),
            "monthly_regime": "unknown",
            "weekly_regime": "unknown",
            "daily_regime": "unknown",
            "mtf_snapshot": "unknown",
            "regime": "unknown",
        }]

        self.command._enrich_with_regime(trades)
        self.assertEqual(trades[0]["regime"], "bull")
        self.assertEqual(trades[0]["mtf_snapshot"], "unknown|unknown|bull")
