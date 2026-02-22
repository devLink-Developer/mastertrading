from django.db import models

from core.models import Instrument


class StrategyConfig(models.Model):
    name = models.CharField(max_length=64)
    version = models.CharField(max_length=16, default="v1")
    params_json = models.JSONField(default=dict)
    enabled = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("name", "version")
        ordering = ["name", "-created_at"]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.name} {self.version}"


class Signal(models.Model):
    strategy = models.CharField(max_length=64)
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE)
    ts = models.DateTimeField()
    payload_json = models.JSONField(default=dict)
    score = models.FloatField(default=0.0)

    class Meta:
        ordering = ["-ts"]
        indexes = [models.Index(fields=["instrument", "ts"])]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.strategy} {self.instrument.symbol} {self.ts}"
