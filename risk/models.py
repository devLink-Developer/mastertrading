from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

from core.models import Instrument


class RiskEvent(models.Model):
    class Severity(models.TextChoices):
        INFO = "info", "Info"
        WARN = "warn", "Warn"
        CRITICAL = "critical", "Critical"

    instrument = models.ForeignKey(
        Instrument, on_delete=models.SET_NULL, null=True, blank=True
    )
    kind = models.CharField(max_length=64)
    severity = models.CharField(
        max_length=10, choices=Severity.choices, default=Severity.INFO
    )
    ts = models.DateTimeField(auto_now_add=True)
    details_json = models.JSONField(default=dict)

    class Meta:
        ordering = ["-ts"]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.kind} {self.severity}"


# ---------------------------------------------------------------------------
# Circuit Breaker — singleton config editable via admin
# ---------------------------------------------------------------------------

class CircuitBreakerConfig(models.Model):
    """
    Admin-editable circuit breaker.  Only ONE row should exist (singleton).
    When triggered, the bot pauses opening new positions.
    """
    enabled = models.BooleanField(
        default=True,
        help_text="Enable / disable the circuit breaker globally.",
    )
    max_daily_dd_pct = models.FloatField(
        default=10.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(100)],
        help_text="Max intra-day drawdown (%) before pausing. E.g. 10 = -10%.",
    )
    max_total_dd_pct = models.FloatField(
        default=15.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(100)],
        help_text="Max total drawdown from equity peak (%) before pausing. E.g. 15 = -15%.",
    )
    max_consecutive_losses = models.PositiveIntegerField(
        default=6,
        help_text="Pause after N consecutive losing trades (0 = disabled).",
    )
    cooldown_minutes_after_trigger = models.PositiveIntegerField(
        default=1440,
        help_text="Minutes to stay paused after circuit breaker fires (default 24h).",
    )
    # Runtime state (updated by the engine, not by the user)
    is_tripped = models.BooleanField(default=False, editable=False)
    tripped_at = models.DateTimeField(null=True, blank=True, editable=False)
    trip_reason = models.CharField(max_length=255, blank=True, default="", editable=False)
    peak_equity = models.FloatField(default=0.0, editable=False)

    class Meta:
        verbose_name = "Circuit Breaker Config"
        verbose_name_plural = "Circuit Breaker Config"

    def __str__(self):
        state = "TRIPPED" if self.is_tripped else "OK"
        return f"CircuitBreaker [{state}] daily={self.max_daily_dd_pct}% total={self.max_total_dd_pct}%"

    @classmethod
    def get(cls) -> "CircuitBreakerConfig":
        """Return the singleton (auto-created with defaults)."""
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj

    def reset(self):
        """Admin action: manually reset the circuit breaker."""
        self.is_tripped = False
        self.tripped_at = None
        self.trip_reason = ""
        self.save(update_fields=["is_tripped", "tripped_at", "trip_reason"])


# ---------------------------------------------------------------------------
# Regime Filter — admin-editable volatility / trend filter
# ---------------------------------------------------------------------------

class RegimeFilterConfig(models.Model):
    """
    Pause signals when market is sideways / low-volatility.
    Admin can tweak thresholds without code changes.
    """
    class FilterType(models.TextChoices):
        ATR_PCT = "atr_pct", "ATR % of price"
        ADX = "adx", "ADX (Average Directional Index)"

    enabled = models.BooleanField(
        default=True,
        help_text="Enable / disable the regime filter.",
    )
    filter_type = models.CharField(
        max_length=10,
        choices=FilterType.choices,
        default=FilterType.ATR_PCT,
        help_text="Which volatility metric to use.",
    )
    # ATR % thresholds
    atr_period = models.PositiveIntegerField(
        default=14,
        help_text="Lookback period for ATR calculation (candles).",
    )
    atr_min_pct = models.FloatField(
        default=0.4,
        validators=[MinValueValidator(0.01), MaxValueValidator(20)],
        help_text="Minimum ATR% to allow signals. Below this = sideways → skip. "
                  "E.g. 0.4 means ATR must be ≥0.4% of price.",
    )
    atr_timeframe = models.CharField(
        max_length=4,
        default="4h",
        help_text="Timeframe for ATR calc (e.g. 5m, 1h, 4h).",
    )
    # ADX thresholds
    adx_period = models.PositiveIntegerField(
        default=14,
        help_text="Lookback period for ADX calculation.",
    )
    adx_min = models.FloatField(
        default=20.0,
        validators=[MinValueValidator(1), MaxValueValidator(100)],
        help_text="Minimum ADX value to allow signals. Below this = no trend → skip.",
    )

    class Meta:
        verbose_name = "Regime Filter Config"
        verbose_name_plural = "Regime Filter Config"

    def __str__(self):
        if self.filter_type == self.FilterType.ATR_PCT:
            return f"RegimeFilter [{'ON' if self.enabled else 'OFF'}] ATR%≥{self.atr_min_pct} ({self.atr_timeframe})"
        return f"RegimeFilter [{'ON' if self.enabled else 'OFF'}] ADX≥{self.adx_min}"

    @classmethod
    def get(cls) -> "RegimeFilterConfig":
        """Return the singleton (auto-created with defaults)."""
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj
