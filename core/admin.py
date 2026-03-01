from django.contrib import admin
from django import forms

from .models import (
    ApiContextFile,
    ApiProviderConfig,
    ApiTokenUsageLog,
    AuditLog,
    ExchangeCredential,
    Instrument,
)


@admin.register(Instrument)
class InstrumentAdmin(admin.ModelAdmin):
    list_display = ("symbol", "exchange", "kind", "enabled")
    list_filter = ("exchange", "kind", "enabled")
    search_fields = ("symbol", "exchange")


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ("actor", "action", "created_at")
    search_fields = ("actor", "action")


class ExchangeCredentialAdminForm(forms.ModelForm):
    class Meta:
        model = ExchangeCredential
        fields = "__all__"
        widgets = {
            "api_secret": forms.PasswordInput(render_value=True),
            "api_passphrase": forms.PasswordInput(render_value=True),
        }


@admin.register(ExchangeCredential)
class ExchangeCredentialAdmin(admin.ModelAdmin):
    form = ExchangeCredentialAdminForm
    list_display = (
        "name_alias",
        "owner",
        "service",
        "active",
        "ai_enabled",
        "ai_provider_config",
        "sandbox",
        "margin_mode",
        "leverage",
        "label",
        "updated_at",
    )
    list_editable = ("active", "ai_enabled", "sandbox")
    list_filter = ("service", "owner", "active", "ai_enabled", "sandbox", "margin_mode")
    search_fields = ("name_alias", "service", "label", "owner__username")

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        if obj.active:
            siblings = ExchangeCredential.objects.exclude(pk=obj.pk).filter(
                service=obj.service,
                sandbox=obj.sandbox,
                active=True,
            )
            if obj.owner_id:
                siblings = siblings.filter(owner_id=obj.owner_id)
            else:
                siblings = siblings.filter(owner__isnull=True)
            siblings.update(active=False)


class ApiProviderConfigAdminForm(forms.ModelForm):
    class Meta:
        model = ApiProviderConfig
        fields = "__all__"
        widgets = {
            "api_key": forms.PasswordInput(render_value=True),
        }


@admin.register(ApiProviderConfig)
class ApiProviderConfigAdmin(admin.ModelAdmin):
    form = ApiProviderConfigAdminForm
    list_display = (
        "name_alias",
        "owner",
        "provider",
        "model_name",
        "active",
        "is_default",
        "max_input_tokens",
        "max_output_tokens",
        "timeout_seconds",
        "updated_at",
    )
    list_editable = ("active", "is_default")
    list_filter = ("provider", "owner", "active", "is_default")
    search_fields = ("name_alias", "model_name", "label", "owner__username")

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        if obj.is_default:
            siblings = ApiProviderConfig.objects.exclude(pk=obj.pk).filter(
                provider=obj.provider,
                is_default=True,
            )
            if obj.owner_id:
                siblings = siblings.filter(owner_id=obj.owner_id)
            else:
                siblings = siblings.filter(owner__isnull=True)
            siblings.update(is_default=False)


@admin.register(ApiContextFile)
class ApiContextFileAdmin(admin.ModelAdmin):
    list_display = (
        "config",
        "name",
        "file_path",
        "enabled",
        "required",
        "priority",
        "max_tokens",
        "trim_mode",
        "updated_at",
    )
    list_editable = ("enabled", "required", "priority", "max_tokens")
    list_filter = ("config__provider", "config", "enabled", "required", "trim_mode")
    search_fields = ("name", "file_path", "notes", "config__name_alias")


@admin.register(ApiTokenUsageLog)
class ApiTokenUsageLogAdmin(admin.ModelAdmin):
    list_display = (
        "created_at",
        "config",
        "provider",
        "model_name",
        "operation",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "context_tokens",
        "estimated",
    )
    list_filter = ("provider", "model_name", "operation", "estimated")
    search_fields = ("operation", "model_name", "provider")
    readonly_fields = (
        "created_at",
        "updated_at",
        "config",
        "provider",
        "model_name",
        "operation",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "context_tokens",
        "estimated",
        "metadata_json",
    )
