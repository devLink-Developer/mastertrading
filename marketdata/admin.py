from django.contrib import admin

from .models import Candle, FundingRate, OrderBookSnapshot


@admin.register(Candle)
class CandleAdmin(admin.ModelAdmin):
    list_display = ("instrument", "timeframe", "ts", "close", "volume")
    list_filter = ("timeframe", "instrument")
    ordering = ("-ts",)


@admin.register(FundingRate)
class FundingRateAdmin(admin.ModelAdmin):
    list_display = ("instrument", "ts", "rate")
    ordering = ("-ts",)


@admin.register(OrderBookSnapshot)
class OrderBookSnapshotAdmin(admin.ModelAdmin):
    list_display = ("instrument", "ts", "bid_px", "ask_px")
    ordering = ("-ts",)
