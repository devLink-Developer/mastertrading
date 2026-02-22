from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path

from django.conf import settings as django_settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone as dj_tz

from backtest.engine import run_backtest
from core.models import Instrument


def _parse_grid(raw: str) -> list[float]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise CommandError("Grid cannot be empty.")
    try:
        return [float(v) for v in values]
    except ValueError as exc:
        raise CommandError(f"Invalid grid value: {exc}") from exc


@contextmanager
def _temporary_settings(**overrides):
    previous = {}
    for key, value in overrides.items():
        previous[key] = getattr(django_settings, key)
        setattr(django_settings, key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(django_settings, key, value)


class Command(BaseCommand):
    help = "Walk-forward optimizer for ATR TP/SL multipliers and score threshold."

    def add_arguments(self, parser):
        parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
        parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (default: now)")
        parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols")
        parser.add_argument("--equity", type=float, default=1000.0, help="Initial equity")
        parser.add_argument("--ltf", type=str, default="5m")
        parser.add_argument("--htf", type=str, default="4h")
        parser.add_argument("--train-days", type=int, default=21)
        parser.add_argument("--test-days", type=int, default=7)
        parser.add_argument("--step-days", type=int, default=7)
        parser.add_argument("--tp-grid", type=str, default="3.0,3.5,4.0,4.5")
        parser.add_argument("--sl-grid", type=str, default="0.8,1.0,1.2,1.5")
        parser.add_argument("--score-grid", type=str, default="0.78,0.80,0.82,0.85")
        parser.add_argument("--dd-penalty", type=float, default=0.8)
        parser.add_argument("--max-train-dd", type=float, default=25.0)
        parser.add_argument("--min-trades-train", type=int, default=20)
        parser.add_argument("--no-trailing", action="store_true")
        parser.add_argument("--out", type=str, default="", help="Path to write JSON report")

    def _parse_date(self, raw: str) -> datetime:
        try:
            return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise CommandError(f"Invalid date '{raw}'. Use YYYY-MM-DD.") from exc

    def _resolve_instruments(self, symbols_raw: str) -> list[Instrument]:
        if symbols_raw:
            symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
            instruments = list(Instrument.objects.filter(symbol__in=symbols))
            missing = sorted(set(symbols) - {i.symbol for i in instruments})
            if missing:
                raise CommandError(f"Instruments not found: {missing}")
            return instruments
        instruments = list(Instrument.objects.filter(enabled=True))
        if not instruments:
            raise CommandError("No enabled instruments found.")
        return instruments

    def handle(self, *args, **options):
        start = self._parse_date(options["start"])
        end = self._parse_date(options["end"]) if options["end"] else dj_tz.now()
        if start >= end:
            raise CommandError("Start must be before end.")

        instruments = self._resolve_instruments(options["symbols"])
        tp_grid = _parse_grid(options["tp_grid"])
        sl_grid = _parse_grid(options["sl_grid"])
        score_grid = _parse_grid(options["score_grid"])

        train_days = options["train_days"]
        test_days = options["test_days"]
        step_days = options["step_days"]
        if train_days <= 0 or test_days <= 0 or step_days <= 0:
            raise CommandError("train-days, test-days and step-days must be positive.")

        self.stdout.write(self.style.HTTP_INFO("=" * 78))
        self.stdout.write(self.style.HTTP_INFO("  WALK-FORWARD OPTIMIZER"))
        self.stdout.write(self.style.HTTP_INFO("=" * 78))
        self.stdout.write(f"  Range      : {start:%Y-%m-%d} -> {end:%Y-%m-%d}")
        self.stdout.write(f"  Symbols    : {', '.join(i.symbol for i in instruments)}")
        self.stdout.write(f"  Train/Test : {train_days}d / {test_days}d  (step {step_days}d)")
        self.stdout.write(
            f"  Grid sizes : TP {len(tp_grid)} x SL {len(sl_grid)} x Score {len(score_grid)} "
            f"= {len(tp_grid) * len(sl_grid) * len(score_grid)} combos/window"
        )
        self.stdout.write(self.style.HTTP_INFO("-" * 78))

        windows = []
        cursor = start
        while cursor + timedelta(days=train_days + test_days) <= end:
            train_start = cursor
            train_end = cursor + timedelta(days=train_days)
            test_end = train_end + timedelta(days=test_days)
            windows.append((train_start, train_end, test_end))
            cursor += timedelta(days=step_days)

        if not windows:
            raise CommandError("No complete walk-forward windows fit in the selected range.")

        trailing_stop = not options["no_trailing"]
        equity = float(options["equity"])
        dd_penalty = float(options["dd_penalty"])
        max_train_dd = float(options["max_train_dd"])
        min_trades_train = int(options["min_trades_train"])

        window_results = []
        for idx, (train_start, train_end, test_end) in enumerate(windows, start=1):
            best = None
            best_obj = float("-inf")

            for tp_mult, sl_mult, min_score in product(tp_grid, sl_grid, score_grid):
                overrides = {
                    "ATR_MULT_TP": tp_mult,
                    "ATR_MULT_SL": sl_mult,
                    "MIN_SIGNAL_SCORE": min_score,
                    "EXECUTION_MIN_SIGNAL_SCORE": min_score,
                }
                with _temporary_settings(**overrides):
                    _, train_metrics = run_backtest(
                        instruments=instruments,
                        start=train_start,
                        end=train_end,
                        initial_equity=equity,
                        ltf=options["ltf"],
                        htf=options["htf"],
                        trailing_stop=trailing_stop,
                        verbose=False,
                    )

                if train_metrics["total_trades"] < min_trades_train:
                    continue
                if train_metrics["max_drawdown_pct"] > max_train_dd:
                    continue

                objective = train_metrics["total_pnl"] - (dd_penalty * train_metrics["max_drawdown_abs"])
                if objective > best_obj:
                    best_obj = objective
                    best = {
                        "params": {
                            "ATR_MULT_TP": tp_mult,
                            "ATR_MULT_SL": sl_mult,
                            "MIN_SIGNAL_SCORE": min_score,
                        },
                        "objective": round(objective, 4),
                        "train_metrics": train_metrics,
                    }

            if best is None:
                # Fallback to current live-ish defaults if no combo passed constraints.
                best = {
                    "params": {
                        "ATR_MULT_TP": float(django_settings.ATR_MULT_TP),
                        "ATR_MULT_SL": float(django_settings.ATR_MULT_SL),
                        "MIN_SIGNAL_SCORE": float(django_settings.MIN_SIGNAL_SCORE),
                    },
                    "objective": None,
                    "train_metrics": {
                        "total_trades": 0,
                        "total_pnl": 0.0,
                        "max_drawdown_pct": 0.0,
                    },
                }

            eval_overrides = {
                "ATR_MULT_TP": best["params"]["ATR_MULT_TP"],
                "ATR_MULT_SL": best["params"]["ATR_MULT_SL"],
                "MIN_SIGNAL_SCORE": best["params"]["MIN_SIGNAL_SCORE"],
                "EXECUTION_MIN_SIGNAL_SCORE": best["params"]["MIN_SIGNAL_SCORE"],
            }
            with _temporary_settings(**eval_overrides):
                _, test_metrics = run_backtest(
                    instruments=instruments,
                    start=train_end,
                    end=test_end,
                    initial_equity=equity,
                    ltf=options["ltf"],
                    htf=options["htf"],
                    trailing_stop=trailing_stop,
                    verbose=False,
                )

            equity += float(test_metrics["total_pnl"])
            window_result = {
                "window": idx,
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "test_end": test_end.isoformat(),
                "selected_params": best["params"],
                "train": best["train_metrics"],
                "test": test_metrics,
                "equity_after_test": round(equity, 4),
            }
            window_results.append(window_result)

            self.stdout.write(
                f"  W{idx:02d} | {train_start:%Y-%m-%d}->{train_end:%Y-%m-%d} "
                f"| TPx{best['params']['ATR_MULT_TP']:.2f} SLx{best['params']['ATR_MULT_SL']:.2f} "
                f"Score>={best['params']['MIN_SIGNAL_SCORE']:.2f} "
                f"| test pnl {test_metrics['total_pnl']:+.2f} "
                f"| dd {test_metrics['max_drawdown_pct']:.2f}% "
                f"| trades {test_metrics['total_trades']}"
            )

        total_test_pnl = sum(float(w["test"]["total_pnl"]) for w in window_results)
        total_test_trades = sum(int(w["test"]["total_trades"]) for w in window_results)
        total_wins = sum(int(w["test"]["wins"]) for w in window_results)
        total_losses = sum(int(w["test"]["losses"]) for w in window_results)
        max_test_dd = max(float(w["test"]["max_drawdown_pct"]) for w in window_results)
        weighted_expectancy = (
            total_test_pnl / total_test_trades if total_test_trades else 0.0
        )

        summary = {
            "windows": len(window_results),
            "initial_equity": float(options["equity"]),
            "final_equity": round(equity, 4),
            "total_test_pnl": round(total_test_pnl, 4),
            "total_test_trades": total_test_trades,
            "wins": total_wins,
            "losses": total_losses,
            "win_rate_pct": round((total_wins / total_test_trades * 100), 2) if total_test_trades else 0.0,
            "max_test_drawdown_pct": round(max_test_dd, 2),
            "expectancy_per_trade": round(weighted_expectancy, 4),
        }

        self.stdout.write(self.style.HTTP_INFO("-" * 78))
        self.stdout.write(f"  Out-of-sample PnL : {summary['total_test_pnl']:+.4f}")
        self.stdout.write(f"  Final equity      : {summary['final_equity']:.4f}")
        self.stdout.write(
            f"  Trades/WL/WR      : {summary['total_test_trades']} / "
            f"{summary['wins']}-{summary['losses']} / {summary['win_rate_pct']:.2f}%"
        )
        self.stdout.write(f"  Max test DD       : {summary['max_test_drawdown_pct']:.2f}%")
        self.stdout.write(f"  Expectancy/trade  : {summary['expectancy_per_trade']:+.4f}")
        self.stdout.write(self.style.HTTP_INFO("=" * 78))

        report = {
            "config": {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "symbols": [i.symbol for i in instruments],
                "train_days": train_days,
                "test_days": test_days,
                "step_days": step_days,
                "tp_grid": tp_grid,
                "sl_grid": sl_grid,
                "score_grid": score_grid,
                "dd_penalty": dd_penalty,
                "max_train_dd": max_train_dd,
                "min_trades_train": min_trades_train,
                "trailing_stop": trailing_stop,
            },
            "summary": summary,
            "windows": window_results,
        }

        if options["out"]:
            out_path = Path(options["out"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(f"Report written to {out_path}"))
