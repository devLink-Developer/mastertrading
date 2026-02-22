from django.contrib import admin

from .models import RiskEvent, CircuitBreakerConfig, RegimeFilterConfig


@admin.register(RiskEvent)
class RiskEventAdmin(admin.ModelAdmin):
    list_display = ("kind", "severity", "instrument", "ts")
    list_filter = ("severity",)


@admin.register(CircuitBreakerConfig)
class CircuitBreakerConfigAdmin(admin.ModelAdmin):
    list_display = (
        "enabled", "max_daily_dd_pct", "max_total_dd_pct",
        "max_consecutive_losses", "is_tripped", "trip_reason",
    )
    fieldsets = (
        ("Settings", {
            "fields": (
                "enabled",
                "max_daily_dd_pct",
                "max_total_dd_pct",
                "max_consecutive_losses",
                "cooldown_minutes_after_trigger",
            ),
        }),
        ("Runtime State (read-only)", {
            "fields": ("is_tripped", "tripped_at", "trip_reason", "peak_equity"),
            "classes": ("collapse",),
        }),
    )
    readonly_fields = ("is_tripped", "tripped_at", "trip_reason", "peak_equity")
    actions = ["reset_circuit_breaker"]

    def has_add_permission(self, request):
        # Singleton: allow add only if none exists
        return not CircuitBreakerConfig.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False

    @admin.action(description="Reset circuit breaker (un-trip)")
    def reset_circuit_breaker(self, request, queryset):
        for obj in queryset:
            obj.reset()
        self.message_user(request, "Circuit breaker reset.")


@admin.register(RegimeFilterConfig)
class RegimeFilterConfigAdmin(admin.ModelAdmin):
    list_display = ("enabled", "filter_type", "atr_min_pct", "atr_timeframe", "adx_min")
    fieldsets = (
        ("General", {
            "fields": ("enabled", "filter_type"),
        }),
        ("ATR % Filter", {
            "fields": ("atr_period", "atr_min_pct", "atr_timeframe"),
            "description": "Used when filter_type = ATR %",
        }),
        ("ADX Filter", {
            "fields": ("adx_period", "adx_min"),
            "description": "Used when filter_type = ADX",
        }),
    )

    def has_add_permission(self, request):
        return not RegimeFilterConfig.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False
