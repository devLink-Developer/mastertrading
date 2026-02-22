"""
Show dynamic allocator weights — Bayesian rolling performance adjustment.

Displays base (static) weights, per-module rolling stats, and what the
dynamic weights would be (or are, if enabled).

Usage:
    python manage.py show_dynamic_weights                 # last 7 days
    python manage.py show_dynamic_weights --days 14
    python manage.py show_dynamic_weights --json reports/weights.json
"""
from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from signals.allocator import (
    MODULE_ORDER,
    _module_rolling_stats,
    default_weight_map,
    dynamic_weight_map,
)


class Command(BaseCommand):
    help = "Display static vs Bayesian-adjusted allocator weights with rolling stats"

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=None, help="Rolling window (default: from settings)")
        parser.add_argument("--json", type=str, default="", help="Export to JSON file")

    def handle(self, *args, **options):
        days = options["days"] or int(getattr(settings, "ALLOCATOR_DYNAMIC_WINDOW_DAYS", 7))
        alpha = float(getattr(settings, "ALLOCATOR_DYNAMIC_ALPHA_PRIOR", 2.0))
        beta = float(getattr(settings, "ALLOCATOR_DYNAMIC_BETA_PRIOR", 2.0))
        min_mult = float(getattr(settings, "ALLOCATOR_DYNAMIC_MIN_MULT", 0.5))
        max_mult = float(getattr(settings, "ALLOCATOR_DYNAMIC_MAX_MULT", 2.0))
        min_trades = int(getattr(settings, "ALLOCATOR_DYNAMIC_MIN_TRADES", 10))
        enabled = bool(getattr(settings, "ALLOCATOR_DYNAMIC_WEIGHTS_ENABLED", False))

        base = default_weight_map()
        stats = _module_rolling_stats(days=days)
        dyn = dynamic_weight_map(
            base_weights=base,
            days=days,
            alpha_prior=alpha,
            beta_prior=beta,
            min_mult=min_mult,
            max_mult=max_mult,
            min_trades=min_trades,
        )

        self.stdout.write("")
        self.stdout.write(f"  Dynamic weights {'ENABLED' if enabled else 'DISABLED (dry-run)'}")
        self.stdout.write(f"  Window: {days}d | Prior: Beta({alpha},{beta}) | Mult: [{min_mult}×,{max_mult}×] | Min trades: {min_trades}")
        self.stdout.write("")
        self.stdout.write(f"  {'Module':<10} {'Base':>7} {'Dynamic':>9} {'Mult':>7} {'Trades':>7} {'Wins':>6} {'WR':>7} {'PnL$':>10} {'Post':>7}")
        self.stdout.write("  " + "-" * 75)

        report: dict = {"config": {
            "enabled": enabled,
            "days": days,
            "alpha_prior": alpha,
            "beta_prior": beta,
            "min_mult": min_mult,
            "max_mult": max_mult,
            "min_trades": min_trades,
        }, "modules": {}}

        for mod in MODULE_ORDER:
            ms = stats.get(mod, {"wins": 0, "losses": 0, "n": 0, "pnl": 0.0})
            n = ms["n"]
            wins = ms["wins"]
            wr = (wins / n * 100) if n > 0 else 0.0
            pnl = ms["pnl"]
            b = base.get(mod, 0.0)
            d = dyn.get(mod, 0.0)
            mult = d / b if b > 0 else 1.0

            # Compute posterior
            if n >= min_trades:
                post = (alpha + wins) / (alpha + beta + wins + (n - wins))
            else:
                post = 0.5

            self.stdout.write(
                f"  {mod:<10} {b:>7.3f} {d:>9.3f} {mult:>6.2f}× {n:>7} {wins:>6} {wr:>6.1f}% {pnl:>+10.2f} {post:>7.3f}"
            )

            report["modules"][mod] = {
                "base_weight": round(b, 4),
                "dynamic_weight": round(d, 4),
                "multiplier": round(mult, 4),
                "trades": n,
                "wins": wins,
                "losses": n - wins,
                "win_rate": round(wr, 2),
                "pnl": round(pnl, 2),
                "posterior_mean": round(post, 4),
            }

        self.stdout.write("")

        # Summary direction
        for mod in MODULE_ORDER:
            m = report["modules"][mod]
            delta = m["dynamic_weight"] - m["base_weight"]
            if abs(delta) > 0.005:
                arrow = "↑" if delta > 0 else "↓"
                self.stdout.write(f"  {arrow} {mod}: {m['base_weight']:.3f} → {m['dynamic_weight']:.3f} ({delta:+.3f})")

        json_path = options.get("json", "")
        if json_path:
            p = Path(json_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(report, indent=2))
            self.stdout.write(f"\n  JSON → {p}")
