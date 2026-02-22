from django.db import models

from core.models import Instrument


class Candle(models.Model):
    class Timeframe(models.TextChoices):
        M1 = "1m", "1m"
        M5 = "5m", "5m"
        M15 = "15m", "15m"
        H1 = "1h", "1h"
        H4 = "4h", "4h"
        D1 = "1d", "1d"

    # Composite unique index (instrument, timeframe, ts) already covers instrument lookups.
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE, db_index=False)
    timeframe = models.CharField(max_length=3, choices=Timeframe.choices)
    ts = models.DateTimeField()
    open = models.DecimalField(max_digits=20, decimal_places=8)
    high = models.DecimalField(max_digits=20, decimal_places=8)
    low = models.DecimalField(max_digits=20, decimal_places=8)
    close = models.DecimalField(max_digits=20, decimal_places=8)
    volume = models.DecimalField(max_digits=28, decimal_places=10)
    mark_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)

    class Meta:
        unique_together = ("instrument", "timeframe", "ts")
        ordering = ["-ts"]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.instrument.symbol} {self.timeframe} {self.ts}"


class FundingRate(models.Model):
    # Composite unique index (instrument, ts) already covers instrument lookups.
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE, db_index=False)
    ts = models.DateTimeField()
    rate = models.DecimalField(max_digits=10, decimal_places=8)

    class Meta:
        unique_together = ("instrument", "ts")
        ordering = ["-ts"]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.instrument.symbol} {self.rate}"


class OrderBookSnapshot(models.Model):
    # Explicit (instrument, ts) index already exists.
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE, db_index=False)
    ts = models.DateTimeField()
    bid_px = models.DecimalField(max_digits=20, decimal_places=8)
    bid_sz = models.DecimalField(max_digits=20, decimal_places=8)
    ask_px = models.DecimalField(max_digits=20, decimal_places=8)
    ask_sz = models.DecimalField(max_digits=20, decimal_places=8)
    depth_json = models.JSONField(default=dict)

    class Meta:
        ordering = ["-ts"]
        indexes = [models.Index(fields=["instrument", "ts"])]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.instrument.symbol} book {self.ts}"
