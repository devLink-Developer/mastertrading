from django.test import TestCase, override_settings

from risk.report_controls import (
    REPORT_CONTROL_NAME,
    REPORT_CONTROL_VERSION,
    resolve_report_config,
    update_report_config,
)
from signals.models import StrategyConfig


@override_settings(
    PERFORMANCE_REPORT_ENABLED=True,
    PERFORMANCE_REPORT_BEAT_ENABLED=True,
    PERFORMANCE_REPORT_BEAT_MODE="interval",
    PERFORMANCE_REPORT_BEAT_MINUTES=15,
    PERFORMANCE_REPORT_BEAT_HOUR=0,
    PERFORMANCE_REPORT_BEAT_MINUTE=0,
    PERFORMANCE_REPORT_WINDOW_MINUTES=15,
)
class ReportControlsRuntimeTest(TestCase):
    def test_resolve_creates_default_row(self):
        cfg = resolve_report_config()
        self.assertTrue(cfg["enabled"])
        self.assertTrue(cfg["beat_enabled"])
        self.assertEqual(cfg["mode"], "interval")
        self.assertEqual(cfg["beat_minutes"], 15)
        self.assertEqual(cfg["window_minutes"], 15)

        row = StrategyConfig.objects.get(
            name=REPORT_CONTROL_NAME,
            version=REPORT_CONTROL_VERSION,
        )
        self.assertTrue(row.enabled)
        self.assertEqual(row.params_json["beat_minutes"], 15)

    def test_update_normalizes_and_persists(self):
        cfg = update_report_config(
            enabled=False,
            beat_enabled=False,
            mode="daily",
            beat_minutes=0,
            beat_hour=27,
            beat_minute=-3,
            window_minutes=5000,
        )

        self.assertFalse(cfg["enabled"])
        self.assertFalse(cfg["beat_enabled"])
        self.assertEqual(cfg["mode"], "daily")
        self.assertEqual(cfg["beat_minutes"], 1)
        self.assertEqual(cfg["beat_hour"], 23)
        self.assertEqual(cfg["beat_minute"], 0)
        self.assertEqual(cfg["window_minutes"], 1440)

