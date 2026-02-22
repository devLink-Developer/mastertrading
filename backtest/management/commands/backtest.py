"""
Management command: python manage.py backtest

Runs a walk-forward backtest over historical candle data using the
production signal engine and simulated execution with risk management.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings as django_settings
from django.utils import timezone as dj_tz

from core.models import Instrument
from backtest.engine import run_backtest
from backtest.models import BacktestRun, BacktestTrade

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Run a walk-forward backtest on historical candle data."

    def add_arguments(self, parser):
        parser.add_argument(
            "--start",
            type=str,
            required=True,
            help="Start date YYYY-MM-DD",
        )
        parser.add_argument(
            "--end",
            type=str,
            default=None,
            help="End date YYYY-MM-DD (default: today)",
        )
        parser.add_argument(
            "--symbols",
            type=str,
            default="",
            help="Comma-separated instrument symbols (default: all enabled)",
        )
        parser.add_argument(
            "--equity",
            type=float,
            default=1000.0,
            help="Initial equity USDT (default: 1000)",
        )
        parser.add_argument(
            "--ltf",
            type=str,
            default="5m",
            help="Low timeframe (default: 5m)",
        )
        parser.add_argument(
            "--htf",
            type=str,
            default="4h",
            help="High timeframe for trend (default: 4h)",
        )
        parser.add_argument(
            "--no-trailing",
            action="store_true",
            help="Disable trailing stop",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            dest="verbose_trades",
            help="Print each trade as it happens",
        )
        parser.add_argument(
            "--name",
            type=str,
            default="",
            help="Optional name for this backtest run",
        )
        parser.add_argument(
            "--no-save",
            action="store_true",
            help="Don't save results to database",
        )

    def handle(self, *args, **options):
        # Parse dates
        try:
            start = datetime.strptime(options["start"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            raise CommandError(f"Invalid start date: {options['start']}. Use YYYY-MM-DD format.")

        if options["end"]:
            try:
                end = datetime.strptime(options["end"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                raise CommandError(f"Invalid end date: {options['end']}. Use YYYY-MM-DD format.")
        else:
            end = dj_tz.now()

        if start >= end:
            raise CommandError("Start date must be before end date.")

        # Resolve instruments
        if options["symbols"]:
            symbols = [s.strip().upper() for s in options["symbols"].split(",")]
            instruments = list(Instrument.objects.filter(symbol__in=symbols))
            missing = set(symbols) - {i.symbol for i in instruments}
            if missing:
                raise CommandError(f"Instruments not found: {missing}")
        else:
            instruments = list(Instrument.objects.filter(enabled=True))

        if not instruments:
            raise CommandError("No instruments found. Seed instruments first with `python manage.py seed_instruments`.")

        initial_equity = options["equity"]
        trailing = not options["no_trailing"]
        verbose = options["verbose_trades"]

        # Header
        self.stdout.write(self.style.HTTP_INFO("=" * 70))
        self.stdout.write(self.style.HTTP_INFO("  MASTERTRADING BACKTEST ENGINE"))
        self.stdout.write(self.style.HTTP_INFO("=" * 70))
        self.stdout.write(f"  Period     : {start:%Y-%m-%d} → {end:%Y-%m-%d}")
        self.stdout.write(f"  Instruments: {', '.join(i.symbol for i in instruments)}")
        self.stdout.write(f"  Equity     : ${initial_equity:,.2f} USDT")
        self.stdout.write(f"  LTF / HTF  : {options['ltf']} / {options['htf']}")
        self.stdout.write(f"  Trailing   : {'ON' if trailing else 'OFF'}")
        self.stdout.write(f"  Risk/trade : {django_settings.RISK_PER_TRADE_PCT * 100:.1f}%")
        self.stdout.write(f"  TP default : {django_settings.TAKE_PROFIT_PCT * 100:.1f}%  |  SL default: {django_settings.STOP_LOSS_PCT * 100:.1f}%")
        self.stdout.write(f"  ATR mult   : TP {django_settings.ATR_MULT_TP}x  |  SL {django_settings.ATR_MULT_SL}x")
        self.stdout.write(f"  Max leverage: {django_settings.MAX_EFF_LEVERAGE}x")
        self.stdout.write(
            f"  Session policy: {'ON' if getattr(django_settings, 'SESSION_POLICY_ENABLED', False) else 'OFF'}"
        )
        self.stdout.write(
            f"  Direction mode: {getattr(django_settings, 'SIGNAL_DIRECTION_MODE', 'both')}"
        )
        self.stdout.write(self.style.HTTP_INFO("-" * 70))
        self.stdout.write("  Running backtest...\n")

        # Run — auto-scale lookback for finer timeframes (1m needs more bars for same window)
        ltf = options["ltf"]
        _lookback_map = {"1m": 1200, "3m": 400, "5m": 240, "15m": 100, "1h": 50}
        lookback = _lookback_map.get(ltf, 240)

        trades, metrics = run_backtest(
            instruments=instruments,
            start=start,
            end=end,
            initial_equity=initial_equity,
            ltf=ltf,
            htf=options["htf"],
            lookback_bars=lookback,
            trailing_stop=trailing,
            verbose=verbose,
        )

        # Results
        self.stdout.write("")
        self.stdout.write(self.style.HTTP_INFO("=" * 70))
        self.stdout.write(self.style.HTTP_INFO("  RESULTS"))
        self.stdout.write(self.style.HTTP_INFO("=" * 70))

        m = metrics
        self.stdout.write(f"  Total trades     : {m['total_trades']}")
        self.stdout.write(f"  Wins / Losses    : {m['wins']} / {m['losses']} (WR: {m['win_rate']}%)")

        # Color the PnL
        pnl_style = self.style.SUCCESS if m["total_pnl"] >= 0 else self.style.ERROR
        pnl_text = f"${m['total_pnl']:+,.4f} ({m['total_pnl_pct']:+.2f}%)"
        self.stdout.write(f"  Total PnL        : {pnl_style(pnl_text)}")
        self.stdout.write(f"  Final equity     : ${m['final_equity']:,.2f}")
        self.stdout.write(f"  Avg win          : ${m['avg_win']:+,.4f}")
        self.stdout.write(f"  Avg loss         : ${m['avg_loss']:+,.4f}")
        self.stdout.write(f"  Best trade       : ${m['best_trade']:+,.4f}")
        self.stdout.write(f"  Worst trade      : ${m['worst_trade']:+,.4f}")
        self.stdout.write(f"  Profit factor    : {m['profit_factor']}")
        self.stdout.write(f"  Sharpe ratio     : {m['sharpe_ratio']}")
        self.stdout.write(f"  Max drawdown     : {m['max_drawdown_pct']:.2f}% (${m['max_drawdown_abs']:,.4f})")
        self.stdout.write(f"  Avg trade duration: {m['avg_trade_duration_min']:.1f} min")
        self.stdout.write(f"  Total fees       : ${m['total_fees']:,.4f}")
        self.stdout.write(f"  Expectancy       : ${m['expectancy']:+,.4f} per trade")
        self.stdout.write(f"  Bars processed   : {m.get('bars_processed', 0):,}")
        self.stdout.write(f"  Elapsed          : {m.get('elapsed_seconds', 0):.1f}s")

        # Close reason breakdown
        reason_counts = m.get("close_reason_counts", {})
        if reason_counts:
            self.stdout.write(f"  Close reasons    : {reason_counts}")
        sf_count = m.get("signal_flip_count", 0)
        sf_pnl = m.get("signal_flip_pnl", 0)
        self.stdout.write(f"  Signal flips     : {sf_count} (PnL: ${sf_pnl:+,.4f})")

        self.stdout.write(self.style.HTTP_INFO("-" * 70))

        # Trade log summary
        if trades:
            self.stdout.write("")
            self.stdout.write(self.style.HTTP_INFO("  TRADE LOG (last 20)"))
            self.stdout.write(self.style.HTTP_INFO("-" * 70))
            self.stdout.write(
                f"  {'#':>3}  {'Symbol':<14} {'Side':<5} {'Entry':>10} {'Exit':>10} "
                f"{'PnL':>10} {'PnL%':>8} {'Reason':<12} {'Score':>5}"
            )
            inst_map = {i.id: i.symbol for i in instruments}
            for i, t in enumerate(trades[-20:], start=max(1, len(trades) - 19)):
                symbol = inst_map.get(t.instrument_id, "?")
                pnl_str = f"${t.pnl_abs:+,.2f}"
                pnl_pct_str = f"{t.pnl_pct * 100:+.2f}%"
                row = f"  {i:>3}  {symbol:<14} {t.side:<5} {t.entry_price:>10.2f} {t.exit_price:>10.2f} {pnl_str:>10} {pnl_pct_str:>8} {t.reason:<12} {t.score:>5.2f}"
                if t.pnl_abs >= 0:
                    self.stdout.write(self.style.SUCCESS(row))
                else:
                    self.stdout.write(self.style.ERROR(row))

        # Per-instrument breakdown
        if trades and len(instruments) > 1:
            self.stdout.write("")
            self.stdout.write(self.style.HTTP_INFO("  PER-INSTRUMENT BREAKDOWN"))
            self.stdout.write(self.style.HTTP_INFO("-" * 70))
            inst_map = {i.id: i.symbol for i in instruments}
            by_inst: dict[int, list] = {}
            for t in trades:
                by_inst.setdefault(t.instrument_id, []).append(t)

            self.stdout.write(
                f"  {'Symbol':<14} {'Trades':>6} {'Wins':>5} {'WR%':>6} {'PnL':>12} {'PF':>6}"
            )
            for inst_id, inst_trades in sorted(by_inst.items(), key=lambda x: sum(t.pnl_abs for t in x[1]), reverse=True):
                sym = inst_map.get(inst_id, "?")
                w = len([t for t in inst_trades if t.pnl_abs > 0])
                wr = w / len(inst_trades) * 100 if inst_trades else 0
                pnl = sum(t.pnl_abs for t in inst_trades)
                gp = sum(t.pnl_abs for t in inst_trades if t.pnl_abs > 0)
                gl = abs(sum(t.pnl_abs for t in inst_trades if t.pnl_abs < 0))
                pf = gp / gl if gl > 0 else float("inf") if gp > 0 else 0
                pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
                row = f"  {sym:<14} {len(inst_trades):>6} {w:>5} {wr:>5.1f}% ${pnl:>+10.2f} {pf_str:>6}"
                self.stdout.write(row)

        self.stdout.write(self.style.HTTP_INFO("=" * 70))

        # Save to DB
        if not options["no_save"]:
            settings_snapshot = {
                "RISK_PER_TRADE_PCT": django_settings.RISK_PER_TRADE_PCT,
                "TAKE_PROFIT_PCT": django_settings.TAKE_PROFIT_PCT,
                "STOP_LOSS_PCT": django_settings.STOP_LOSS_PCT,
                "ATR_MULT_TP": django_settings.ATR_MULT_TP,
                "ATR_MULT_SL": django_settings.ATR_MULT_SL,
                "MAX_EFF_LEVERAGE": django_settings.MAX_EFF_LEVERAGE,
                "TRAILING_STOP_ENABLED": trailing,
                "TRAILING_STOP_ACTIVATION_R": django_settings.TRAILING_STOP_ACTIVATION_R,
                "PARTIAL_CLOSE_AT_R": django_settings.PARTIAL_CLOSE_AT_R,
                "PARTIAL_CLOSE_PCT": django_settings.PARTIAL_CLOSE_PCT,
                "DAILY_DD_LIMIT": django_settings.DAILY_DD_LIMIT,
                "WEEKLY_DD_LIMIT": django_settings.WEEKLY_DD_LIMIT,
                "FUNDING_EXTREME_PERCENTILE": django_settings.FUNDING_EXTREME_PERCENTILE,
                "SESSION_POLICY_ENABLED": getattr(django_settings, "SESSION_POLICY_ENABLED", False),
                "SESSION_DEAD_ZONE_BLOCK": getattr(django_settings, "SESSION_DEAD_ZONE_BLOCK", True),
                "SESSION_SCORE_MIN": getattr(django_settings, "SESSION_SCORE_MIN", {}),
                "SESSION_RISK_MULTIPLIER": getattr(django_settings, "SESSION_RISK_MULTIPLIER", {}),
                "SIGNAL_DIRECTION_MODE": getattr(django_settings, "SIGNAL_DIRECTION_MODE", "both"),
                "PER_INSTRUMENT_DIRECTION": getattr(django_settings, "PER_INSTRUMENT_DIRECTION", {}),
                "initial_equity": initial_equity,
            }
            inst_map = {i.id: i.symbol for i in instruments}

            run = BacktestRun.objects.create(
                name=options["name"] or f"backtest_{start:%Y%m%d}_{end:%Y%m%d}",
                status=BacktestRun.Status.DONE,
                start_date=start,
                end_date=end,
                instruments_json=[i.symbol for i in instruments],
                settings_json=settings_snapshot,
                metrics_json=metrics,
                duration_seconds=metrics.get("elapsed_seconds", 0),
            )

            # Bulk create trades
            inst_obj_map = {i.id: i for i in instruments}
            trade_objs = []
            for t in trades:
                inst_obj = inst_obj_map.get(t.instrument_id)
                if not inst_obj:
                    continue
                outcome = BacktestTrade.Outcome.BE
                if t.pnl_abs > 0:
                    outcome = BacktestTrade.Outcome.WIN
                elif t.pnl_abs < 0:
                    outcome = BacktestTrade.Outcome.LOSS

                trade_objs.append(BacktestTrade(
                    run=run,
                    instrument=inst_obj,
                    side=t.side,
                    qty=t.qty,
                    entry_price=t.entry_price,
                    exit_price=t.exit_price,
                    entry_ts=t.entry_ts,
                    exit_ts=t.exit_ts,
                    pnl_abs=t.pnl_abs,
                    pnl_pct=t.pnl_pct,
                    fee_paid=t.fee_paid,
                    reason=t.reason,
                    score=t.score,
                    outcome=outcome,
                ))

            if trade_objs:
                BacktestTrade.objects.bulk_create(trade_objs, batch_size=500)

            self.stdout.write(self.style.SUCCESS(
                f"\n  ✓ Results saved to DB: BacktestRun #{run.id} with {len(trade_objs)} trades"
            ))

        self.stdout.write("")
