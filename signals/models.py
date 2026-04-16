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


class MacroLiquiditySnapshot(models.Model):
    asof = models.DateTimeField(db_index=True)
    regime = models.CharField(max_length=24, default="unavailable")
    confidence = models.FloatField(default=0.0)
    composite_score = models.FloatField(default=0.0)
    composite_momentum = models.FloatField(default=0.0)
    fed_net_liquidity_z = models.FloatField(default=0.0)
    financial_conditions_z = models.FloatField(default=0.0)
    dollar_z = models.FloatField(default=0.0)
    stablecoin_growth_z = models.FloatField(default=0.0)
    btc_etf_flow_z = models.FloatField(default=0.0)
    details_json = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-asof", "-id"]
        indexes = [
            models.Index(fields=["regime", "asof"]),
        ]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.regime} {self.asof}"
