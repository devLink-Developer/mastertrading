"""
Statistical validation: Purged K-Fold CV + White's Reality Check (SPA test).

Usage (live trades):
    python manage.py validate_strategy --days 30
    python manage.py validate_strategy --days 60 --folds 5 --embargo-days 2
    python manage.py validate_strategy --days 30 --spa --json reports/validation.json

Usage (backtest trades):
    python manage.py validate_strategy --source backtest --days 60

Combines:
  1. Purged K-Fold CV to measure OOS stability and detect overfitting
  2. White's Reality Check / Hansen's SPA to test statistical significance
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz

from execution.models import OperationReport


class Command(BaseCommand):
    help = "Purged K-Fold CV + White's Reality Check on live or backtest trades"

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30, help="Lookback days")
        parser.add_argument("--symbol", type=str, default="", help="Filter by symbol")
        parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
        parser.add_argument("--embargo-days", type=int, default=1, help="Embargo window in days")
        parser.add_argument("--embargo-pct", type=float, default=0.05, help="Embargo as %% of fold")
        parser.add_argument(
            "--spa", action="store_true",
            help="Run White's Reality Check (SPA) grouping by strategy",
        )
        parser.add_argument("--spa-bootstrap", type=int, default=5000, help="SPA bootstrap samples")
        parser.add_argument("--spa-block-size", type=int, default=10, help="SPA block size")
        parser.add_argument("--seed", type=int, default=42, help="RNG seed")
        parser.add_argument(
            "--source", choices=["live", "backtest"], default="live",
            help="Trade source (live OperationReports or BacktestTrades)",
        )
        parser.add_argument("--json", type=str, default="", help="Write JSON output to file")

    def handle(self, **opts):
        from backtest.statistical_validation import purged_kfold_cv, whites_reality_check

        trades = self._load_trades(opts)
        if not trades:
            self.stderr.write("No trades found for the given filters.\n")
            return

        self.stdout.write(f"\nLoaded {len(trades)} trades\n")

        # ── Purged K-Fold CV ──
        cv_result = purged_kfold_cv(
            trades,
            n_folds=opts["folds"],
            embargo_pct=opts["embargo_pct"],
            embargo_days=opts["embargo_days"],
        )

        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("  PURGED K-FOLD CROSS-VALIDATION")
        self.stdout.write("=" * 60)
        self.stdout.write(f"Folds: {cv_result.n_folds}  |  Embargo: {cv_result.embargo_days}d")
        self.stdout.write(f"Mean test PnL:        {cv_result.mean_test_pnl:+.4f}%")
        self.stdout.write(f"Std test PnL:          {cv_result.std_test_pnl:.4f}%")
        self.stdout.write(f"Mean test Win Rate:    {cv_result.mean_test_win_rate:.2%}")
        self.stdout.write(f"Mean test Sharpe:      {cv_result.mean_test_sharpe:.4f}")
        self.stdout.write(f"Mean test Expectancy:  {cv_result.mean_test_expectancy:.6f}")
        self.stdout.write(f"Train-Test gap (PnL):  {cv_result.mean_train_test_gap:+.4f}%")
        self.stdout.write(f"Deflated Sharpe p-val: {cv_result.deflated_sharpe_pvalue:.4f}")

        if cv_result.mean_train_test_gap > 5.0:
            self.stdout.write("  ⚠  Large train-test gap suggests overfitting")
        if cv_result.deflated_sharpe_pvalue > 0.05:
            self.stdout.write("  ⚠  Deflated Sharpe not significant (p > 0.05)")

        for f in cv_result.folds:
            self.stdout.write(
                f"\n  Fold {f.fold}: "
                f"train={f.n_train_trades}t PnL={f.train_pnl_pct:+.2f}%  |  "
                f"test={f.n_test_trades}t PnL={f.test_pnl_pct:+.2f}% "
                f"WR={f.test_win_rate:.0%} Sharpe={f.test_sharpe:.2f} "
                f"DD={f.test_max_dd:.2f}%"
            )

        # ── White's Reality Check / SPA ──
        spa_result_dict = None
        if opts["spa"]:
            strategy_returns = self._group_by_strategy(trades)
            if len(strategy_returns) < 2:
                self.stdout.write("\nSPA test needs >= 2 strategies, found "
                                  f"{len(strategy_returns)}. Skipping.\n")
            else:
                spa_result = whites_reality_check(
                    strategy_returns,
                    n_bootstrap=opts["spa_bootstrap"],
                    block_size=opts["spa_block_size"],
                    seed=opts["seed"],
                )
                spa_result_dict = spa_result.to_dict()

                self.stdout.write("\n" + "=" * 60)
                self.stdout.write("  WHITE'S REALITY CHECK / HANSEN'S SPA")
                self.stdout.write("=" * 60)
                self.stdout.write(f"Strategies: {spa_result.n_strategies}")
                self.stdout.write(f"Best: {spa_result.best_strategy} "
                                  f"(mean={spa_result.best_mean_return:+.6f})")
                self.stdout.write(f"SPA p-value:        {spa_result.spa_pvalue:.4f}")
                self.stdout.write(f"Consistent p-value: {spa_result.consistent_pvalue:.4f}")

                if spa_result.significant:
                    self.stdout.write("  ✓ Significant at 5%: best strategy beats benchmark")
                else:
                    self.stdout.write("  ✗ NOT significant: edge may be data-snooping artifact")

                for s in spa_result.strategy_stats:
                    self.stdout.write(
                        f"  {s['name']:25s} mean={s['mean_return']:+.6f} "
                        f"std={s['std_return']:.6f} t={s['t_stat']:+.4f}"
                    )

        # ── JSON output ──
        if opts["json"]:
            out = {
                "source": opts["source"],
                "days": opts["days"],
                "symbol": opts["symbol"] or "all",
                "n_trades": len(trades),
                "purged_cv": cv_result.to_dict(),
            }
            if spa_result_dict:
                out["spa_test"] = spa_result_dict
            path = Path(opts["json"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, indent=2, default=str))
            self.stdout.write(f"\nJSON written to {path}\n")

    def _load_trades(self, opts) -> list[dict]:
        """Load trades from live OperationReports or BacktestTrades."""
        since = dj_tz.now() - timedelta(days=opts["days"])

        if opts["source"] == "live":
            qs = OperationReport.objects.filter(closed_at__gte=since)
            if opts["symbol"]:
                qs = qs.filter(symbol__icontains=opts["symbol"])
            qs = qs.order_by("opened_at")
            trades = []
            for r in qs:
                try:
                    pnl = float(r.pnl_pct or 0)
                    trades.append({
                        "entry_ts": r.opened_at,
                        "exit_ts": r.closed_at,
                        "pnl_pct": pnl,
                        "strategy": str(r.strategy or "unknown"),
                        "symbol": str(r.symbol or ""),
                        "side": str(r.side or ""),
                    })
                except Exception:
                    continue
            return trades

        # Backtest source
        try:
            from backtest.models import BacktestTrade
            qs = BacktestTrade.objects.filter(entry_ts__gte=since)
            if opts["symbol"]:
                qs = qs.filter(symbol__icontains=opts["symbol"])
            qs = qs.order_by("entry_ts")
            return [
                {
                    "entry_ts": t.entry_ts,
                    "exit_ts": t.exit_ts,
                    "pnl_pct": float(t.pnl_pct or 0),
                    "strategy": str(t.strategy or "backtest"),
                    "symbol": str(t.symbol or ""),
                    "side": str(t.side or ""),
                }
                for t in qs
            ]
        except Exception:
            return []

    @staticmethod
    def _group_by_strategy(trades: list[dict]) -> dict[str, list[float]]:
        """Group trade PnL by strategy for SPA test."""
        groups: dict[str, list[float]] = {}
        for t in trades:
            key = t.get("strategy", "unknown")
            groups.setdefault(key, []).append(t["pnl_pct"])
        return groups
