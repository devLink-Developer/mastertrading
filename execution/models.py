from django.db import models

from core.models import Instrument, TimeStampedModel


class Order(TimeStampedModel):
    class OrderSide(models.TextChoices):
        BUY = "buy", "Buy"
        SELL = "sell", "Sell"

    class OrderType(models.TextChoices):
        MARKET = "market", "Market"
        LIMIT = "limit", "Limit"
        STOP_MARKET = "stop_market", "Stop Market"
        STOP_LIMIT = "stop_limit", "Stop Limit"

    class OrderStatus(models.TextChoices):
        NEW = "new", "New"
        PARTIALLY_FILLED = "partially_filled", "Partially Filled"
        FILLED = "filled", "Filled"
        CANCELED = "canceled", "Canceled"
        REJECTED = "rejected", "Rejected"

    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE)
    exchange_order_id = models.CharField(max_length=128, blank=True, default="")
    side = models.CharField(max_length=4, choices=OrderSide.choices)
    type = models.CharField(max_length=12, choices=OrderType.choices)
    qty = models.DecimalField(max_digits=28, decimal_places=10)
    price = models.DecimalField(max_digits=28, decimal_places=10, null=True, blank=True)
    status = models.CharField(
        max_length=20, choices=OrderStatus.choices, default=OrderStatus.NEW
    )
    reduce_only = models.BooleanField(default=False)
    post_only = models.BooleanField(default=False)
    correlation_id = models.CharField(max_length=64, db_index=True)
    leverage = models.DecimalField(max_digits=10, decimal_places=4, default=0)
    margin_mode = models.CharField(max_length=12, default="", blank=True)
    notional_usdt = models.DecimalField(max_digits=28, decimal_places=10, default=0)
    fee_usdt = models.DecimalField(max_digits=18, decimal_places=10, default=0)
    status_reason = models.CharField(max_length=255, default="", blank=True)
    opened_at = models.DateTimeField(null=True, blank=True)
    closed_at = models.DateTimeField(null=True, blank=True)
    raw_response = models.JSONField(null=True, blank=True)
    parent_correlation_id = models.CharField(max_length=64, default="", blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["instrument", "status", "opened_at"]),
        ]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.instrument.symbol} {self.side} {self.qty}"


class TradeFill(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name="fills")
    ts = models.DateTimeField()
    price = models.DecimalField(max_digits=28, decimal_places=10)
    qty = models.DecimalField(max_digits=28, decimal_places=10)
    fee = models.DecimalField(max_digits=18, decimal_places=10, default=0)

    class Meta:
        ordering = ["-ts"]


class Position(TimeStampedModel):
    class Side(models.TextChoices):
        LONG = "long", "Long"
        SHORT = "short", "Short"

    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE)
    qty = models.DecimalField(max_digits=28, decimal_places=10)
    avg_price = models.DecimalField(max_digits=28, decimal_places=10)
    unrealized_pnl = models.DecimalField(max_digits=28, decimal_places=10, default=0)
    realized_pnl = models.DecimalField(max_digits=28, decimal_places=10, default=0)
    leverage_eff = models.DecimalField(max_digits=10, decimal_places=4, default=0)
    side = models.CharField(max_length=5, choices=Side.choices, default=Side.LONG)
    notional_usdt = models.DecimalField(max_digits=28, decimal_places=10, default=0)
    margin_used_usdt = models.DecimalField(max_digits=28, decimal_places=10, default=0)
    pnl_pct = models.DecimalField(max_digits=10, decimal_places=6, default=0)
    last_price = models.DecimalField(max_digits=28, decimal_places=10, default=0)
    liq_price_est = models.DecimalField(max_digits=28, decimal_places=10, default=0)
    is_open = models.BooleanField(default=True)
    mode = models.CharField(max_length=12, default="", blank=True)
    opened_at = models.DateTimeField(null=True, blank=True)
    last_sync = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["instrument"]
        indexes = [
            models.Index(fields=["instrument", "is_open"]),
        ]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.instrument.symbol} pos {self.qty}"


class OperationReport(TimeStampedModel):
    class Outcome(models.TextChoices):
        WIN = "win", "Win"
        LOSS = "loss", "Loss"
        BE = "breakeven", "Breakeven"

    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE)
    side = models.CharField(max_length=4, choices=Order.OrderSide.choices)
    qty = models.DecimalField(max_digits=28, decimal_places=10)
    entry_price = models.DecimalField(max_digits=28, decimal_places=10)
    exit_price = models.DecimalField(max_digits=28, decimal_places=10)
    pnl_abs = models.DecimalField(max_digits=28, decimal_places=10)
    pnl_pct = models.DecimalField(max_digits=10, decimal_places=6)
    notional_usdt = models.DecimalField(max_digits=28, decimal_places=10, default=0)
    margin_used_usdt = models.DecimalField(max_digits=28, decimal_places=10, default=0)
    fee_usdt = models.DecimalField(max_digits=18, decimal_places=10, default=0)
    leverage = models.DecimalField(max_digits=10, decimal_places=4, default=0)
    equity_before = models.DecimalField(max_digits=28, decimal_places=10, null=True, blank=True)
    equity_after = models.DecimalField(max_digits=28, decimal_places=10, null=True, blank=True)
    mode = models.CharField(max_length=12, default="", blank=True)
    opened_at = models.DateTimeField(null=True, blank=True)
    outcome = models.CharField(max_length=10, choices=Outcome.choices)
    reason = models.CharField(max_length=32, default="")
    close_sub_reason = models.CharField(
        max_length=32, default="", blank=True,
        help_text="Sub-classification for exchange_close: exchange_stop, likely_liquidation, bot_close_missed, unknown",
    )
    signal_id = models.CharField(max_length=64, blank=True, default="")
    correlation_id = models.CharField(max_length=64, blank=True, default="")
    closed_at = models.DateTimeField()

    class Meta:
        ordering = ["-closed_at"]
        indexes = [
            models.Index(fields=["instrument", "closed_at"]),
        ]

    def __str__(self):  # pragma: no cover - trivial
        return f"{self.instrument.symbol} {self.side} {self.outcome} {self.pnl_abs}"


class BalanceSnapshot(TimeStampedModel):
    equity_usdt = models.DecimalField(max_digits=28, decimal_places=10)
    free_usdt = models.DecimalField(max_digits=28, decimal_places=10)
    notional_usdt = models.DecimalField(max_digits=28, decimal_places=10, default=0)
    eff_leverage = models.DecimalField(max_digits=18, decimal_places=6, default=0)
    note = models.CharField(max_length=128, default="", blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):  # pragma: no cover - trivial
        return f"eq={self.equity_usdt} free={self.free_usdt} lev={self.eff_leverage}"
