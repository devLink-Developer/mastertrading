"""
Management command: fit HMM regime model and display results.

Usage:
    python manage.py fit_regime                          # BTC only (default)
    python manage.py fit_regime --all                    # all active instruments
    python manage.py fit_regime --symbol ETHUSDT         # specific symbol
    python manage.py fit_regime --lookback 720           # custom lookback (bars)
"""
from __future__ import annotations

from django.core.management.base import BaseCommand

from signals.regime import fit_and_predict, fit_and_predict_all


class Command(BaseCommand):
    help = "Fit HMM regime model on 1h candles and display the predicted regime."

    def add_arguments(self, parser):
        parser.add_argument("--symbol", type=str, default="BTCUSDT",
                            help="Symbol to fit (default: BTCUSDT)")
        parser.add_argument("--all", action="store_true",
                            help="Fit all active instruments")
        parser.add_argument("--lookback", type=int, default=None,
                            help="Override lookback bars (default: from settings)")

    def handle(self, *args, **options):
        from django.conf import settings as s

        if options["lookback"]:
            s.HMM_REGIME_LOOKBACK_BARS = options["lookback"]

        self.stdout.write("=" * 60)
        self.stdout.write("  HMM REGIME MODEL")
        self.stdout.write("=" * 60)
        self.stdout.write(f"  States       : {s.HMM_REGIME_N_STATES}")
        self.stdout.write(f"  Timeframe    : {s.HMM_REGIME_TIMEFRAME}")
        self.stdout.write(f"  Lookback     : {s.HMM_REGIME_LOOKBACK_BARS} bars")
        self.stdout.write(f"  Trending mult: {s.HMM_REGIME_TRENDING_RISK_MULT}")
        self.stdout.write(f"  Choppy mult  : {s.HMM_REGIME_CHOPPY_RISK_MULT}")
        self.stdout.write("-" * 60)

        if options["all"]:
            # Override to fit per-symbol
            old_val = getattr(s, "HMM_REGIME_PER_SYMBOL", False)
            s.HMM_REGIME_PER_SYMBOL = True
            results = fit_and_predict_all()
            s.HMM_REGIME_PER_SYMBOL = old_val
        else:
            r = fit_and_predict(options["symbol"])
            results = {options["symbol"]: r} if r else {}

        if not results:
            self.stdout.write(self.style.ERROR("  No results — check candle data."))
            return

        self.stdout.write("")
        self.stdout.write(f"  {'Symbol':<12} {'State':<10} {'Name':<12} "
                          f"{'RiskMult':>9} {'Conf':>7} {'MeanVol':>10} {'MeanADX':>8} {'Bars':>6}")
        self.stdout.write("  " + "-" * 76)

        for sym, r in sorted(results.items()):
            self.stdout.write(
                f"  {sym:<12} {r['state']:<10} {r['name']:<12} "
                f"{r['risk_mult']:>9.2f} {r['confidence']:>7.4f} "
                f"{r['mean_vol']:>10.6f} {r['mean_adx']:>8.1f} {r['n_bars']:>6}"
            )

        self.stdout.write("")
        self.stdout.write("=" * 60)
