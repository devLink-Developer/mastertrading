from django.conf import settings
from django.db import models

from core.fields import EncryptedCredentialField


class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Instrument(TimeStampedModel):
    class InstrumentKind(models.TextChoices):
        SPOT = "spot", "Spot"
        PERP = "perp", "Perpetual"

    symbol = models.CharField(max_length=32, unique=True)
    exchange = models.CharField(max_length=32, default="binance")
    base = models.CharField(max_length=16)
    quote = models.CharField(max_length=16)
    enabled = models.BooleanField(default=True)
    kind = models.CharField(
        max_length=8, choices=InstrumentKind.choices, default=InstrumentKind.PERP
    )
    tick_size = models.DecimalField(max_digits=18, decimal_places=8, default=0.0)
    lot_size = models.DecimalField(max_digits=18, decimal_places=8, default=0.0)

    class Meta:
        indexes = [
            models.Index(fields=["enabled", "symbol"], name="core_instr_enabled_symbol_idx"),
        ]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.symbol} ({self.exchange})"


class ExchangeCredential(TimeStampedModel):
    class Service(models.TextChoices):
        KUCOIN = "kucoin", "KuCoin Futures"
        BINANCE = "binance", "Binance Futures"
        BINGX = "bingx", "BingX Perpetual"

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="exchange_credentials",
    )
    name_alias = models.CharField(max_length=64, unique=True)
    service = models.CharField(max_length=16, choices=Service.choices)
    api_key = EncryptedCredentialField(blank=True, default="")
    api_secret = EncryptedCredentialField(blank=True, default="")
    api_passphrase = EncryptedCredentialField(blank=True, default="")
    sandbox = models.BooleanField(default=False)
    margin_mode = models.CharField(max_length=16, default="cross")
    leverage = models.PositiveIntegerField(default=3)
    active = models.BooleanField(default=True)
    label = models.CharField(max_length=128, blank=True, default="")

    class Meta:
        indexes = [
            models.Index(fields=["service", "active"], name="core_excred_service_active_idx"),
            models.Index(fields=["owner", "active"], name="core_excred_owner_active_idx"),
        ]

    def __str__(self) -> str:  # pragma: no cover - trivial
        owner = getattr(self.owner, "username", "unowned")
        return f"{self.name_alias} [{owner}] {self.service} ({'active' if self.active else 'inactive'})"


class AuditLog(models.Model):
    actor = models.CharField(max_length=64)
    action = models.CharField(max_length=128)
    before_json = models.JSONField(null=True, blank=True)
    after_json = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.created_at} - {self.actor} - {self.action}"
