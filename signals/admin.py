from django.contrib import admin

from .models import Signal, StrategyConfig
from .runtime_overrides import invalidate_runtime_overrides_cache


@admin.register(StrategyConfig)
class StrategyConfigAdmin(admin.ModelAdmin):
    list_display = ("name", "version", "enabled", "runtime_value", "created_at")
    list_filter = ("enabled", "version")
    search_fields = ("name", "version")

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        invalidate_runtime_overrides_cache()

    def delete_model(self, request, obj):
        super().delete_model(request, obj)
        invalidate_runtime_overrides_cache()

    @admin.display(description="Runtime Value")
    def runtime_value(self, obj):
        payload = obj.params_json if isinstance(obj.params_json, dict) else {}
        if "value" not in payload:
            return "-"
        return payload.get("value")


@admin.register(Signal)
class SignalAdmin(admin.ModelAdmin):
    list_display = ("strategy", "instrument", "ts", "score")
    list_filter = ("strategy", "instrument")
    ordering = ("-ts",)
