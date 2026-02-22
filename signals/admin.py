from django.contrib import admin

from .models import Signal, StrategyConfig


@admin.register(StrategyConfig)
class StrategyConfigAdmin(admin.ModelAdmin):
    list_display = ("name", "version", "enabled", "created_at")
    list_filter = ("enabled",)
    search_fields = ("name", "version")


@admin.register(Signal)
class SignalAdmin(admin.ModelAdmin):
    list_display = ("strategy", "instrument", "ts", "score")
    list_filter = ("strategy", "instrument")
    ordering = ("-ts",)
