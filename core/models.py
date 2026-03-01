from django.conf import settings
from django.db import models
from django.db.models import Q

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
    ai_enabled = models.BooleanField(default=False)
    ai_provider_config = models.ForeignKey(
        "core.ApiProviderConfig",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="exchange_credentials",
    )
    label = models.CharField(max_length=128, blank=True, default="")

    class Meta:
        indexes = [
            models.Index(fields=["service", "active"], name="core_excred_service_active_idx"),
            models.Index(fields=["owner", "active"], name="core_excred_owner_active_idx"),
            models.Index(fields=["active", "ai_enabled"], name="core_excred_ai_enabled_idx"),
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


class ApiProviderConfig(TimeStampedModel):
    class Provider(models.TextChoices):
        OPENAI = "openai", "OpenAI"
        OPENROUTER = "openrouter", "OpenRouter"
        ANTHROPIC = "anthropic", "Anthropic"
        GEMINI = "gemini", "Google Gemini"
        CUSTOM = "custom", "Custom"

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="api_provider_configs",
    )
    name_alias = models.CharField(max_length=64, unique=True)
    provider = models.CharField(max_length=16, choices=Provider.choices, default=Provider.OPENAI)
    api_key = EncryptedCredentialField(blank=True, default="")
    base_url = models.CharField(max_length=255, blank=True, default="")
    model_name = models.CharField(max_length=128, default="gpt-5")
    organization_id = models.CharField(max_length=128, blank=True, default="")
    project_id = models.CharField(max_length=128, blank=True, default="")
    timeout_seconds = models.PositiveIntegerField(default=30)
    max_input_tokens = models.PositiveIntegerField(default=12000)
    max_output_tokens = models.PositiveIntegerField(default=1200)
    temperature = models.DecimalField(max_digits=4, decimal_places=2, default=0.30)
    top_p = models.DecimalField(max_digits=4, decimal_places=2, default=1.00)
    active = models.BooleanField(default=True)
    is_default = models.BooleanField(default=False)
    label = models.CharField(max_length=128, blank=True, default="")
    extra_headers_json = models.JSONField(default=dict, blank=True)
    extra_params_json = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["provider", "active"], name="core_apicfg_provider_active_idx"),
            models.Index(fields=["owner", "active"], name="core_apicfg_owner_active_idx"),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["provider"],
                condition=Q(is_default=True, owner__isnull=True),
                name="core_apicfg_default_provider_global_uniq",
            ),
        ]

    def __str__(self) -> str:  # pragma: no cover - trivial
        owner = getattr(self.owner, "username", "global")
        return f"{self.name_alias} [{owner}] {self.provider}:{self.model_name}"


class ApiContextFile(TimeStampedModel):
    class TrimMode(models.TextChoices):
        HEAD = "head", "Head (start)"
        TAIL = "tail", "Tail (end)"

    config = models.ForeignKey(
        ApiProviderConfig,
        on_delete=models.CASCADE,
        related_name="context_files",
    )
    name = models.CharField(max_length=96, blank=True, default="")
    file_path = models.CharField(max_length=255)
    enabled = models.BooleanField(default=True)
    required = models.BooleanField(default=False)
    priority = models.IntegerField(default=100)
    max_tokens = models.PositiveIntegerField(default=1200)
    max_chars = models.PositiveIntegerField(default=0)
    trim_mode = models.CharField(max_length=8, choices=TrimMode.choices, default=TrimMode.TAIL)
    include_header = models.BooleanField(default=True)
    notes = models.CharField(max_length=255, blank=True, default="")

    class Meta:
        ordering = ["priority", "id"]
        unique_together = ("config", "file_path")
        indexes = [
            models.Index(fields=["config", "enabled", "priority"], name="core_apictx_cfg_enabled_prio_idx"),
        ]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.config.name_alias}:{self.file_path}"


class ApiTokenUsageLog(TimeStampedModel):
    config = models.ForeignKey(
        ApiProviderConfig,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="token_logs",
    )
    provider = models.CharField(max_length=16, blank=True, default="")
    model_name = models.CharField(max_length=128, blank=True, default="")
    operation = models.CharField(max_length=64, blank=True, default="")
    prompt_tokens = models.PositiveIntegerField(default=0)
    completion_tokens = models.PositiveIntegerField(default=0)
    total_tokens = models.PositiveIntegerField(default=0)
    context_tokens = models.PositiveIntegerField(default=0)
    estimated = models.BooleanField(default=False)
    metadata_json = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["created_at"], name="core_apitok_created_idx"),
            models.Index(fields=["provider", "model_name"], name="core_apitok_provider_model_idx"),
            models.Index(fields=["operation", "created_at"], name="core_apitok_op_created_idx"),
        ]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.provider}:{self.model_name} {self.total_tokens}t ({self.operation})"
