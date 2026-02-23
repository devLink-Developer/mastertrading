from unittest.mock import patch

from django.test import SimpleTestCase

from risk.notifications import notify_trade_closed, notify_trade_opened


class NotificationStrategyMetadataTest(SimpleTestCase):
    @patch("risk.notifications.send_telegram")
    def test_opened_message_includes_strategy_type_and_commercial_name(self, send_mock):
        notify_trade_opened(
            symbol="BTCUSDT",
            side="buy",
            qty=0.1,
            price=50000.0,
            strategy_name="alloc_long",
            active_modules=["trend", "meanrev", "carry", "smc"],
        )

        self.assertTrue(send_mock.called)
        msg = send_mock.call_args.args[0]
        self.assertIn("Estrategia (tecnica):</b> alloc_long", msg)
        self.assertIn("Tipo estrategia:</b> Allocator", msg)
        self.assertIn("Nombre comercial:</b> Gestor Dinamico de Estrategias", msg)
        self.assertIn("Trend (Seguidor de Tendencia)", msg)
        self.assertIn("Mean Reversion (Reversion a la Media)", msg)
        self.assertIn("Carry (Captura de Funding)", msg)
        self.assertIn("SMC (Smart Money Concepts)", msg)

    @patch("risk.notifications.send_telegram")
    def test_opened_message_includes_entry_reason(self, send_mock):
        notify_trade_opened(
            symbol="BTCUSDT",
            side="sell",
            qty=0.0787,
            price=67631.2,
            strategy_name="alloc_short",
            active_modules=["carry", "trend"],
            entry_reason="signal=alloc_short | confluencia: carry short (0.99), trend short (0.62)",
        )

        self.assertTrue(send_mock.called)
        msg = send_mock.call_args.args[0]
        self.assertIn("Razon de posicion:</b>", msg)
        self.assertIn("signal=alloc_short", msg)
        self.assertIn("confluencia: carry short (0.99), trend short (0.62)", msg)

    @patch("risk.notifications.send_telegram")
    def test_closed_message_includes_strategy_metadata_without_modules(self, send_mock):
        notify_trade_closed(
            symbol="ETHUSDT",
            reason="tp",
            pnl_pct=0.02,
            pnl_abs=20.0,
            entry_price=2000.0,
            exit_price=2040.0,
            qty=1.0,
            side="buy",
            strategy_name="mod_meanrev_long",
        )

        self.assertTrue(send_mock.called)
        msg = send_mock.call_args.args[0]
        self.assertIn("Estrategia (tecnica):</b> mod_meanrev_long", msg)
        self.assertIn("Tipo estrategia:</b> Mean Reversion", msg)
        self.assertIn("Nombre comercial:</b> Reversion a la Media", msg)
