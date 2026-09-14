from io import StringIO
from unittest.mock import patch

from django.db import IntegrityError, transaction
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase, override_settings
from django.utils import timezone

from core.models import Instrument
from execution.models import OperationReport
from risk.management.commands.perf_dashboard import Command as PerfCommand
from risk.management.commands.monte_carlo import Command as MonteCarloCommand
from execution.management.commands.dedupe_operation_reports import build_operation_report_dedupe_plan
from risk.tasks import _build_performance_report
from signals.allocator import _module_rolling_stats
from signals.meta_allocator import _collect_module_metrics
from signals import meta_allocator
from signals.models import Signal


@override_settings(MODE="live")
class OperationAccountingStatusTests(TestCase):
    def setUp(self):
        self.instrument = Instrument.objects.create(symbol="BTCUSDT", base="BTC", quote="USDT")
        self.signal = Signal.objects.create(
            instrument=self.instrument, strategy="alloc_long", ts=timezone.now(), score=1,
            payload_json={"reasons": {"module_contributions": [
                {"module": "trend", "direction": "long", "confidence": 1, "contribution": 1},
            ]}},
        )

    def report(self, **changes):
        values = dict(instrument=self.instrument, side="buy", qty=1, entry_price=100,
                      exit_price=101, pnl_abs=1, pnl_pct=.01, mode="live",
                      outcome=OperationReport.Outcome.WIN, closed_at=timezone.now(),
                      signal_id=str(self.signal.pk), fee_usdt=0)
        values.update(changes)
        return OperationReport.objects.create(**values)

    def test_unknown_values_are_null_and_activity_remains_visible(self):
        report = self.report(accounting_status=OperationReport.AccountingStatus.PENDING,
                             outcome=OperationReport.Outcome.PENDING,
                             exit_price=None, pnl_abs=None, pnl_pct=None, fee_usdt=None)
        report.refresh_from_db()
        self.assertIsNone(report.exit_price)
        self.assertIsNone(report.pnl_abs)
        self.assertIsNone(report.pnl_pct)
        self.assertIsNone(report.fee_usdt)
        self.assertEqual(OperationReport.objects.count(), 1)
        self.assertEqual(OperationReport.objects.with_accounted_pnl().count(), 0)

    def test_legacy_and_confirmed_values_remain_in_performance(self):
        legacy = self.report()
        confirmed = self.report(accounting_status=OperationReport.AccountingStatus.CONFIRMED,
                                pnl_abs=0, pnl_pct=0, outcome=OperationReport.Outcome.BE)
        self.assertEqual(legacy.accounting_status, OperationReport.AccountingStatus.LEGACY)
        self.assertEqual(set(OperationReport.objects.with_accounted_pnl().values_list("pk", flat=True)),
                         {legacy.pk, confirmed.pk})

    def test_pending_numeric_placeholders_and_incomplete_legacy_are_excluded(self):
        self.report(accounting_status=OperationReport.AccountingStatus.PENDING)
        self.report(outcome=OperationReport.Outcome.PENDING)
        self.report(pnl_abs=None)
        self.report(pnl_pct=None)
        self.assertFalse(OperationReport.objects.with_accounted_pnl().exists())

    def test_accounting_key_unique_nonblank_and_legacy_blanks_compatible(self):
        self.report(accounting_key="position-key")
        with self.assertRaises(IntegrityError), transaction.atomic():
            self.report(accounting_key="position-key")
        self.report()
        self.report()
        self.assertEqual(OperationReport.objects.count(), 3)

    def test_dynamic_and_meta_learning_ignore_pending_and_other_mode(self):
        self.report(pnl_abs=1, pnl_pct=.01)
        self.report(accounting_status=OperationReport.AccountingStatus.CONFIRMED, pnl_abs=2, pnl_pct=.02)
        self.report(accounting_status=OperationReport.AccountingStatus.PENDING, pnl_abs=999, pnl_pct=9.99)
        self.report(mode="demo", pnl_abs=100, pnl_pct=1)
        stats = _module_rolling_stats(days=7)
        self.assertEqual(stats["trend"]["n"], 2)
        self.assertEqual(stats["trend"]["wins"], 2)
        self.assertEqual(stats["trend"]["pnl"], 3)
        metrics, diagnostic = _collect_module_metrics(lookback_days=7, min_trades=1)
        self.assertEqual(diagnostic["trade_count"], 2)
        self.assertEqual(metrics["trend"].n, 2)
        with override_settings(MODE="demo"):
            self.assertEqual(_module_rolling_stats(days=7)["trend"]["n"], 1)
            self.assertEqual(_module_rolling_stats(days=7)["trend"]["pnl"], 100)

    @override_settings(META_ALLOCATOR_ENABLED=True)
    def test_meta_cache_is_not_reused_between_demo_and_live(self):
        with patch.dict(meta_allocator._OVERLAY_CACHE, {}, clear=True), patch.object(
            meta_allocator, "_collect_module_metrics", return_value=({}, {"trade_count": 0})
        ) as collect:
            kwargs = {"base_weights": {"trend": 1}, "base_risk_budgets": {"trend": 1}}
            meta_allocator.compute_meta_allocator_overlay(**kwargs)
            meta_allocator.compute_meta_allocator_overlay(**kwargs)
            self.assertEqual(collect.call_count, 1)
            with override_settings(MODE="demo"):
                meta_allocator.compute_meta_allocator_overlay(**kwargs)
            self.assertEqual(collect.call_count, 2)

    def test_performance_dashboard_excludes_pending(self):
        known = self.report()
        self.report(accounting_status=OperationReport.AccountingStatus.PENDING,
                    outcome=OperationReport.Outcome.PENDING, pnl_abs=None, pnl_pct=None,
                    exit_price=None, fee_usdt=None)
        trades = PerfCommand()._load_trades(days=7, symbol="BTCUSDT")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["pnl_abs"], float(known.pnl_abs))
        self.assertEqual(len(MonteCarloCommand()._load_trade_rows(days=7, symbol="BTCUSDT")), 1)

    def test_legacy_dedupe_cannot_delete_confirmed_accounting_evidence(self):
        self.report(correlation_id="same-lifecycle", reason="tp")
        confirmed = self.report(correlation_id="same-lifecycle", reason="exchange_close",
                                accounting_status=OperationReport.AccountingStatus.CONFIRMED,
                                accounting_key="confirmed-position")
        pending = self.report(correlation_id="same-lifecycle", reason="sl",
                              accounting_status=OperationReport.AccountingStatus.PENDING,
                              accounting_key="pending-position")
        plan = build_operation_report_dedupe_plan(days=7)
        self.assertEqual(plan["delete_ids"], [])
        self.assertTrue(OperationReport.objects.filter(pk=confirmed.pk).exists())
        self.assertTrue(OperationReport.objects.filter(pk=pending.pk).exists())

    def test_ml_training_does_not_label_pending_as_losses_or_mix_demo(self):
        self.report(correlation_id="known")
        self.report(correlation_id="demo", mode="demo")
        self.report(correlation_id="pending", accounting_status=OperationReport.AccountingStatus.PENDING,
                    outcome=OperationReport.Outcome.PENDING, pnl_abs=None, pnl_pct=None,
                    exit_price=None, fee_usdt=None)
        with self.assertRaises(CommandError) as error:
            call_command("train_entry_filter_ml", source="live", days=7, min_samples=6,
                         output="tmp/profit_fix_env/unused-test-model.json", stdout=StringIO())
        self.assertIn("Not enough samples (1)", str(error.exception))
        self.assertIn("live=1, backtest=0", str(error.exception))

    @patch("risk.tasks.get_runtime_exchange_context", return_value={"service": "bingx", "sandbox": False})
    def test_periodic_report_counts_pending_activity_without_inventing_pnl(self, _context):
        self.report(accounting_status=OperationReport.AccountingStatus.PENDING,
                    outcome=OperationReport.Outcome.PENDING, pnl_abs=None, pnl_pct=None,
                    exit_price=None, fee_usdt=None)
        report = _build_performance_report(window_minutes=60)
        self.assertIn("ops cerradas=1", report)
        self.assertIn("pendientes=1", report)
        self.assertIn("WR=pendiente", report)
        self.assertIn("pnl cerrada=pendiente de conciliacion", report)
        self.assertNotIn("pnl cerrada=+0.0000", report)
