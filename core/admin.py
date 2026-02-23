from django.contrib import admin
from django import forms

from .models import AuditLog, ExchangeCredential, Instrument


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
        "sandbox",
        "margin_mode",
        "leverage",
        "label",
        "updated_at",
    )
    list_editable = ("active", "sandbox")
    list_filter = ("service", "owner", "active", "sandbox", "margin_mode")
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
