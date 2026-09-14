from unittest.mock import patch

from django.contrib.admin.sites import AdminSite
from django.contrib.admin.utils import display_for_field
from django.test import SimpleTestCase

from execution.admin import OperationReportAdmin
from execution.models import OperationReport
from risk.notifications import notify_trade_closed


class PendingCloseNotificationTest(SimpleTestCase):
    @patch("risk.notifications._runtime_context", return_value=("Main DEMO", "VST"))
    @patch("risk.notifications.send_telegram")
    def test_pending_close_has_context_without_invented_outcome(self, send, _context):
        notify_trade_closed(
            symbol="ADAUSDT", reason="exchange_close", pnl_pct=None, pnl_abs=None,
            entry_price=0.18, exit_price=None, qty=12, side="buy", duration_min=90,
            leverage=3, equity_before=100, strategy_name="mod_trend_long",
            accounting_pending=True,
        )
        send.assert_called_once()
        message = send.call_args.args[0]
        self.assertIn("Cierre pendiente de conciliación", message)
        for expected in ("ADAUSDT", "LONG", "0.1800", "Qty:</b> 12", "1h 30m", "Main DEMO", "mod_trend_long"):
            self.assertIn(expected, message)
        for forbidden in ("WIN", "LOSS", "PnL", "Exit:", "Equity:", "+0.00", "None"):
            self.assertNotIn(forbidden, message)

    @patch("risk.notifications._runtime_context", return_value=("Eudy LIVE", "USDT"))
    @patch("risk.notifications.send_telegram")
    def test_pending_flag_ignores_stale_numeric_fallbacks(self, send, _context):
        notify_trade_closed(
            "BTCUSDT", "tp", 0.03, pnl_abs=42, entry_price=100, exit_price=999,
            equity_before=1000, side="sell", accounting_pending=True,
        )
        message = send.call_args.args[0]
        self.assertIn("SHORT", message)
        self.assertNotIn("999", message)
        self.assertNotIn("42", message)
        self.assertNotIn("WIN", message)
        self.assertNotIn("PnL", message)

    @patch("risk.notifications._runtime_context", return_value=("Main DEMO", "VST"))
    @patch("risk.notifications.send_telegram")
    def test_known_close_keeps_existing_message(self, send, _context):
        notify_trade_closed(
            "ADAUSDT", "sl", -0.01, pnl_abs=-1, entry_price=100, exit_price=99,
            equity_before=100, side="buy",
        )
        message = send.call_args.args[0]
        self.assertIn("[LOSS]", message)
        self.assertIn("<b>PnL:</b> -1.00%", message)
        self.assertIn("<b>-1.0000 VST</b>", message)
        self.assertIn("<b>Exit:</b> 99.0000", message)

    @patch("risk.notifications._runtime_context", return_value=("Main DEMO", "VST"))
    @patch("risk.notifications.send_telegram")
    def test_verified_execution_net_does_not_invent_account_equity(self, send, _context):
        notify_trade_closed(
            "ADAUSDT", "tp", 0.02, pnl_abs=2, entry_price=100, exit_price=102,
            equity_before=999, side="buy", accounting_execution_only=True,
        )
        message = send.call_args.args[0]
        self.assertIn("[WIN]", message)
        self.assertIn("<b>PnL ejecución:</b> +2.00%", message)
        self.assertIn("<b>+2.0000 VST</b>", message)
        self.assertIn("comisiones incluidas; funding excluido", message)
        self.assertNotIn("Equity:", message)
        self.assertNotIn("equity", message)
        self.assertNotIn("999", message)
        self.assertNotIn("1001", message)

    def test_admin_exposes_accounting_state_and_handles_unknown_values(self):
        model_admin = OperationReportAdmin(OperationReport, AdminSite())
        self.assertIn("accounting_status", model_admin.list_display)
        self.assertIn("accounting_status", model_admin.list_filter)
        self.assertIn("accounting_details", model_admin.get_readonly_fields(None))
        for name in ("exit_price", "pnl_abs", "pnl_pct", "fee_usdt"):
            self.assertEqual(display_for_field(None, OperationReport._meta.get_field(name), "-"), "-")
