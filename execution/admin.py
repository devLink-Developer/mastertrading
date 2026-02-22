from django.contrib import admin
from django.urls import path
from django.http import HttpResponseRedirect
from django.utils.html import format_html

from adapters import get_default_adapter
from .models import Order, Position, TradeFill, OperationReport, BalanceSnapshot


class TradeFillInline(admin.TabularInline):
    model = TradeFill
    extra = 0


@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ("instrument", "side", "type", "qty", "price", "status", "leverage", "margin_mode", "notional_usdt", "fee_usdt", "created_at", "closed_at", "correlation_id")
    list_filter = ("side", "type", "status", "margin_mode")
    search_fields = ("instrument__symbol", "exchange_order_id", "correlation_id", "status_reason")
    inlines = [TradeFillInline]


@admin.register(Position)
class PositionAdmin(admin.ModelAdmin):
    list_display = (
        "instrument",
        "side",
        "qty",
        "avg_price",
        "last_price",
        "liq_price_badge",
        "pnl_abs_badge",
        "pnl_pct_badge",
        "is_open",
        "leverage_eff",
        "sync_link",
        "close_link",
        "updated_at",
    )
    search_fields = ("instrument__symbol",)
    actions = ["close_positions", "sync_positions"]

    def get_urls(self):
        urls = super().get_urls()
        custom = [
            path("<int:pk>/sync/", self.admin_site.admin_view(self.sync_single), name="execution_position_sync"),
            path("<int:pk>/close/", self.admin_site.admin_view(self.close_single), name="execution_position_close"),
        ]
        return custom + urls

    def close_positions(self, request, queryset):
        adapter = get_default_adapter()
        closed = 0
        for pos in queryset:
            if not pos.is_open or float(pos.qty) == 0:
                continue
            symbol = pos.instrument.symbol
            alt_symbol = f"{pos.instrument.base}/USDT:USDT" if symbol.endswith("USDT") else symbol
            close_side = "sell" if float(pos.qty) > 0 else "buy"
            qty = abs(float(pos.qty))
            for sym in (symbol, alt_symbol):
                try:
                    adapter.create_order(sym, close_side, "market", qty, params={"reduceOnly": True})
                    closed += 1
                    break
                except Exception:
                    continue
        self.message_user(request, f"Cerradas {closed} posiciones (reduceOnly).")

    close_positions.short_description = "Cerrar posiciones seleccionadas (reduce-only)"

    def sync_positions(self, request, queryset):
        from execution.tasks import _sync_positions
        _sync_positions(get_default_adapter())
        self.message_user(request, "Posiciones sincronizadas desde el exchange.")

    sync_positions.short_description = "Actualizar números desde el exchange"

    def pnl_pct_badge(self, obj):
        try:
            val = float(obj.pnl_pct) * 100
        except Exception:
            return "-"
        color = "#1e7e34" if val > 0 else "#c82333" if val < 0 else "#0062cc"
        formatted = f"{val:+.2f}%"
        return format_html('<span style="padding:2px 6px;border-radius:10px;background:{};color:white;font-weight:bold;">{}</span>', color, formatted)

    pnl_pct_badge.short_description = "PnL %"

    def pnl_abs_badge(self, obj):
        try:
            val = float(obj.unrealized_pnl)
        except Exception:
            return "-"
        color = "#1e7e34" if val > 0 else "#c82333" if val < 0 else "#0062cc"
        formatted = f"{val:+.4f}"
        return format_html('<span style="padding:2px 6px;border-radius:10px;background:{};color:white;font-weight:bold;">{}</span>', color, formatted)

    pnl_abs_badge.short_description = "PnL"

    def liq_price_badge(self, obj):
        liq = obj.liq_price_est
        if liq is None:
            return "-"
        color = "#6f42c1"  # purple badge
        try:
            formatted = f"{float(liq):.2f}"
        except Exception:
            return "-"
        return format_html('<span style="padding:2px 6px;border-radius:10px;background:{};color:white;font-weight:bold;">{}</span>', color, formatted)

    liq_price_badge.short_description = "Liq est."

    def sync_link(self, obj):
        return format_html(
            '<a class="button" style="background:#17a2b8;color:white;padding:4px 8px;border-radius:6px;text-decoration:none;" href="{}">Sync</a>',
            f"{obj.id}/sync/",
        )

    sync_link.short_description = "Actualizar"

    def close_link(self, obj):
        if not obj.is_open or float(obj.qty) == 0:
            return "-"
        return format_html(
            '<a class="button" style="background:#d9534f;color:white;padding:4px 8px;border-radius:6px;text-decoration:none;" href="{}">Cerrar</a>',
            f"{obj.id}/close/",
        )

    close_link.short_description = "Cerrar"

    def sync_single(self, request, pk):
        from execution.tasks import _sync_positions
        _sync_positions(get_default_adapter())
        self.message_user(request, "Posición actualizada.")
        return HttpResponseRedirect(request.META.get("HTTP_REFERER", "../"))

    def close_single(self, request, pk):
        adapter = get_default_adapter()
        from execution.models import Position
        try:
            pos = Position.objects.get(pk=pk)
            if float(pos.qty) != 0:
                symbol = pos.instrument.symbol
                alt_symbol = f"{pos.instrument.base}/USDT:USDT" if symbol.endswith("USDT") else symbol
                close_side = "sell" if float(pos.qty) > 0 else "buy"
                qty = abs(float(pos.qty))
                for sym in (symbol, alt_symbol):
                    try:
                        adapter.create_order(sym, close_side, "market", qty, params={"reduceOnly": True})
                        self.message_user(request, f"Posición cerrada en {sym}.")
                        break
                    except Exception:
                        continue
        except Position.DoesNotExist:
            self.message_user(request, "Posición no encontrada.", level=30)
        return HttpResponseRedirect(request.META.get("HTTP_REFERER", "../"))


@admin.register(TradeFill)
class TradeFillAdmin(admin.ModelAdmin):
    list_display = ("order", "ts", "price", "qty", "fee")


@admin.register(OperationReport)
class OperationReportAdmin(admin.ModelAdmin):
    list_display = ("instrument", "side", "pnl_abs", "pnl_pct", "outcome", "opened_at", "closed_at", "reason", "signal_id")
    list_filter = ("outcome", "reason", "side")
    search_fields = ("instrument__symbol", "signal_id", "correlation_id")


@admin.register(BalanceSnapshot)
class BalanceSnapshotAdmin(admin.ModelAdmin):
    list_display = ("created_at", "equity_usdt", "free_usdt", "notional_usdt", "eff_leverage", "note")
    list_filter = ("note",)
