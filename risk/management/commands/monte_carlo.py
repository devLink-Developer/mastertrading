"""
Monte Carlo simulation for drawdown distribution and risk-of-ruin estimation.

Usage:
    python manage.py monte_carlo                  # all trades, last 30 days
    python manage.py monte_carlo --days 60
    python manage.py monte_carlo --symbol BTCUSDT
    python manage.py monte_carlo --sims 50000
    python manage.py monte_carlo --ruin-threshold 25   # ruin = -25% equity
    python manage.py monte_carlo --json reports/mc.json
"""
from __future__ import annotations

import json
import math
import random
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz

from execution.models import OperationReport


class Command(BaseCommand):
    help = "Monte Carlo equity-curve simulation & risk-of-ruin from live trades"

    # ------------------------------------------------------------------
    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30, help="Lookback window (days)")
        parser.add_argument("--symbol", type=str, default="", help="Filter by instrument (e.g. BTCUSDT)")
        parser.add_argument("--sims", type=int, default=10_000, help="Number of MC paths")
        parser.add_argument("--ruin-threshold", type=float, default=20.0,
                            help="Ruin defined as equity drop ≥ X%% from starting capital")
        parser.add_argument("--horizon", type=int, default=0,
                            help="Trades per simulated path (0 = same as observed N)")
        parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
        parser.add_argument("--json", type=str, default="", help="Write results to JSON file")

    # ------------------------------------------------------------------
    def handle(self, **opts):
        pnl_pcts = self._load_pnl(opts["days"], opts["symbol"])
        n_trades = len(pnl_pcts)
        if n_trades < 10:
            self.stderr.write(self.style.ERROR(f"Only {n_trades} trades found — need ≥10 for MC."))
            return

        horizon = opts["horizon"] or n_trades
        n_sims = opts["sims"]
        ruin_thr = opts["ruin_threshold"] / 100.0  # fraction
        rng = np.random.default_rng(opts["seed"])

        sym_label = f" ({opts['symbol']})" if opts['symbol'] else ""
        sep = "=" * 60
        self.stdout.write(self.style.NOTICE(
            f"\n{sep}\n"
            f"  Monte Carlo — {n_sims:,} paths × {horizon} trades\n"
            f"  Source: {n_trades} live trades{sym_label}"
            f", last {opts['days']}d\n"
            f"  Ruin threshold: {opts['ruin_threshold']:.1f}%\n"
            f"{sep}"
        ))

        # --- simulation ---
        returns = np.array(pnl_pcts)
        max_dds, final_pnls, ruin_flags = self._simulate(
            returns, n_sims, horizon, ruin_thr, rng
        )

        # --- descriptive stats from observed trades ---
        obs = self._observed_stats(pnl_pcts)

        # --- MC stats ---
        mc = self._mc_stats(max_dds, final_pnls, ruin_flags, ruin_thr)

        self._print_observed(obs)
        self._print_mc(mc, opts["ruin_threshold"])

        if opts["json"]:
            self._write_json(opts["json"], obs, mc, opts)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_pnl(self, days: int, symbol: str) -> list[float]:
        cutoff = dj_tz.now() - timedelta(days=days)
        qs = OperationReport.objects.filter(closed_at__gte=cutoff).order_by("closed_at")
        if symbol:
            qs = qs.filter(instrument__symbol__iexact=symbol)
        return [float(r.pnl_pct) for r in qs]

    # ------------------------------------------------------------------
    # Core simulation (vectorised)
    # ------------------------------------------------------------------
    @staticmethod
    def _simulate(
        returns: np.ndarray,
        n_sims: int,
        horizon: int,
        ruin_thr: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resample *with replacement* from observed trade returns.
        Returns (max_drawdowns, final_pnls, ruin_flags) each of shape (n_sims,).
        """
        # idx matrix: (n_sims, horizon)
        idx = rng.integers(0, len(returns), size=(n_sims, horizon))
        sampled = returns[idx]  # (n_sims, horizon) pnl_pct per trade

        # cumulative equity curve (1.0 = starting equity)
        equity = np.cumprod(1.0 + sampled, axis=1)

        # running peak
        peak = np.maximum.accumulate(equity, axis=1)

        # drawdown at each step
        dd = (equity - peak) / peak  # negative or zero

        max_dds = dd.min(axis=1)  # worst DD per path (negative)
        final_pnls = equity[:, -1] - 1.0  # total return
        ruin_flags = (max_dds <= -ruin_thr).astype(int)

        return max_dds, final_pnls, ruin_flags

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    @staticmethod
    def _observed_stats(pnl_pcts: list[float]) -> dict[str, Any]:
        arr = np.array(pnl_pcts)
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        wr = len(wins) / len(arr) if len(arr) else 0
        avg_w = float(wins.mean()) if len(wins) else 0
        avg_l = float(losses.mean()) if len(losses) else 0
        payoff = abs(avg_w / avg_l) if avg_l != 0 else 0
        expectancy = wr * avg_w + (1 - wr) * avg_l

        # Kelly fraction
        if payoff > 0:
            kelly = wr - (1 - wr) / payoff
        else:
            kelly = 0.0

        # Observed max drawdown (sequential equity curve)
        eq = np.cumprod(1.0 + arr)
        peak = np.maximum.accumulate(eq)
        obs_max_dd = float(((eq - peak) / peak).min())

        return {
            "n_trades": len(arr),
            "win_rate": round(wr, 4),
            "avg_win_pct": round(avg_w * 100, 4),
            "avg_loss_pct": round(avg_l * 100, 4),
            "payoff_ratio": round(payoff, 4),
            "expectancy_pct": round(expectancy * 100, 4),
            "kelly_fraction": round(kelly, 4),
            "total_return_pct": round(float((eq[-1] - 1) * 100), 4) if len(eq) else 0,
            "observed_max_dd_pct": round(obs_max_dd * 100, 2),
        }

    @staticmethod
    def _mc_stats(
        max_dds: np.ndarray,
        final_pnls: np.ndarray,
        ruin_flags: np.ndarray,
        ruin_thr: float,
    ) -> dict[str, Any]:
        dd_pcts = max_dds * 100  # convert to %
        ret_pcts = final_pnls * 100

        pctiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

        return {
            "risk_of_ruin_pct": round(float(ruin_flags.mean()) * 100, 2),
            "ruin_threshold_pct": round(ruin_thr * 100, 1),
            "max_dd_percentiles": {
                f"p{p}": round(float(np.percentile(dd_pcts, p)), 2) for p in pctiles
            },
            "final_return_percentiles": {
                f"p{p}": round(float(np.percentile(ret_pcts, p)), 2) for p in pctiles
            },
            "mean_max_dd_pct": round(float(dd_pcts.mean()), 2),
            "std_max_dd_pct": round(float(dd_pcts.std()), 2),
            "mean_final_return_pct": round(float(ret_pcts.mean()), 2),
            "std_final_return_pct": round(float(ret_pcts.std()), 2),
        }

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------
    def _print_observed(self, obs: dict):
        self.stdout.write(self.style.SUCCESS("\n--- Observed Trade Stats ---"))
        self.stdout.write(f"  Trades:          {obs['n_trades']}")
        self.stdout.write(f"  Win rate:        {obs['win_rate']:.1%}")
        self.stdout.write(f"  Avg win:         {obs['avg_win_pct']:+.3f}%")
        self.stdout.write(f"  Avg loss:        {obs['avg_loss_pct']:+.3f}%")
        self.stdout.write(f"  Payoff ratio:    {obs['payoff_ratio']:.3f}")
        self.stdout.write(f"  Expectancy:      {obs['expectancy_pct']:+.4f}% per trade")
        self.stdout.write(f"  Kelly fraction:  {obs['kelly_fraction']:+.4f}")
        self.stdout.write(f"  Total return:    {obs['total_return_pct']:+.2f}%")
        self.stdout.write(f"  Max drawdown:    {obs['observed_max_dd_pct']:.2f}%")

    def _print_mc(self, mc: dict, ruin_thr: float):
        self.stdout.write(self.style.SUCCESS("\n--- Monte Carlo Results ---"))
        self.stdout.write(f"  Risk of ruin ({ruin_thr:.0f}% DD): {mc['risk_of_ruin_pct']:.2f}%")

        self.stdout.write("\n  Max Drawdown distribution:")
        for k, v in mc["max_dd_percentiles"].items():
            self.stdout.write(f"    {k:>4}: {v:+.2f}%")

        self.stdout.write("\n  Final Return distribution:")
        for k, v in mc["final_return_percentiles"].items():
            self.stdout.write(f"    {k:>4}: {v:+.2f}%")

        self.stdout.write(
            f"\n  Mean max DD:  {mc['mean_max_dd_pct']:+.2f}% ± {mc['std_max_dd_pct']:.2f}%"
        )
        self.stdout.write(
            f"  Mean return:  {mc['mean_final_return_pct']:+.2f}% ± {mc['std_final_return_pct']:.2f}%"
        )
        self.stdout.write("")

    # ------------------------------------------------------------------
    def _write_json(self, path: str, obs: dict, mc: dict, opts: dict):
        out = {
            "params": {
                "days": opts["days"],
                "symbol": opts["symbol"] or "ALL",
                "sims": opts["sims"],
                "horizon": opts["horizon"] or obs["n_trades"],
                "ruin_threshold_pct": opts["ruin_threshold"],
                "seed": opts["seed"],
            },
            "observed": obs,
            "monte_carlo": mc,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        self.stdout.write(self.style.SUCCESS(f"Results saved → {p}"))
