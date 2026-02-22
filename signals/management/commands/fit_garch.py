"""
Management command: fit GARCH(1,1) volatility model and display forecasts.

Usage:
    python manage.py fit_garch                         # all instruments
    python manage.py fit_garch --symbol BTCUSDT        # specific symbol
    python manage.py fit_garch --lookback 720          # custom lookback
"""
from __future__ import annotations

from django.core.management.base import BaseCommand

from signals.garch import fit_and_forecast, fit_and_forecast_all


class Command(BaseCommand):
    help = "Fit GARCH(1,1) on 1h candles and display volatility forecasts."

    def add_arguments(self, parser):
        parser.add_argument("--symbol", type=str, default=None,
                            help="Symbol to fit (default: all active)")
        parser.add_argument("--lookback", type=int, default=None,
                            help="Override lookback bars (default: from settings)")

    def handle(self, *args, **options):
        from django.conf import settings as s

        if options["lookback"]:
            s.GARCH_LOOKBACK_BARS = options["lookback"]

        self.stdout.write("=" * 70)
        self.stdout.write("  GARCH(1,1) VOLATILITY FORECAST")
        self.stdout.write("=" * 70)
        self.stdout.write(f"  Timeframe    : {s.GARCH_TIMEFRAME}")
        self.stdout.write(f"  Lookback     : {s.GARCH_LOOKBACK_BARS} bars")
        self.stdout.write(f"  Blend weight : {s.GARCH_BLEND_WEIGHT} (GARCH vs ATR)")
        self.stdout.write("-" * 70)

        if options["symbol"]:
            r = fit_and_forecast(options["symbol"])
            results = {options["symbol"]: r} if r else {}
        else:
            results = fit_and_forecast_all()

        if not results:
            self.stdout.write(self.style.ERROR("  No results — check candle data."))
            return

        self.stdout.write("")
        self.stdout.write(
            f"  {'Symbol':<12} {'CondVol':>10} {'AnnVol':>10} "
            f"{'Alpha':>8} {'Beta':>8} {'Persist':>8} {'AIC':>10} {'Bars':>6}"
        )
        self.stdout.write("  " + "-" * 72)

        for sym, r in sorted(results.items()):
            self.stdout.write(
                f"  {sym:<12} {r['cond_vol']:>10.6f} "
                f"{r['cond_vol_annualized']*100:>9.2f}% "
                f"{r['alpha']:>8.4f} {r['beta']:>8.4f} "
                f"{r['persistence']:>8.4f} {r['aic']:>10.1f} {r['n_obs']:>6}"
            )

        self.stdout.write("")
        self.stdout.write("=" * 70)
