from django.contrib import admin
from .models import BacktestRun, BacktestTrade


class BacktestTradeInline(admin.TabularInline):
    model = BacktestTrade
    extra = 0
    readonly_fields = [
        "instrument", "side", "qty", "entry_price", "exit_price",
        "entry_ts", "exit_ts", "pnl_abs", "pnl_pct", "fee_paid",
        "reason", "score", "outcome",
    ]
    can_delete = False

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(BacktestRun)
class BacktestRunAdmin(admin.ModelAdmin):
    list_display = ["id", "name", "status", "start_date", "end_date", "duration_seconds", "created_at"]
    list_filter = ["status"]
    readonly_fields = ["metrics_json", "settings_json", "instruments_json", "duration_seconds"]
    inlines = [BacktestTradeInline]


@admin.register(BacktestTrade)
class BacktestTradeAdmin(admin.ModelAdmin):
    list_display = ["id", "run", "instrument", "side", "outcome", "pnl_abs", "pnl_pct", "reason", "entry_ts", "exit_ts"]
    list_filter = ["outcome", "side", "reason"]
    readonly_fields = [
        "run", "instrument", "side", "qty", "entry_price", "exit_price",
        "entry_ts", "exit_ts", "pnl_abs", "pnl_pct", "fee_paid",
        "reason", "score", "outcome",
    ]
