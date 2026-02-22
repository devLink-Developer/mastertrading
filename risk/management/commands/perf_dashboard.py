"""
Performance dashboard — per-module, per-symbol, and per-reason metrics
from live OperationReport data joined to Signal payload.

Usage:
    python manage.py perf_dashboard                     # last 30 days
    python manage.py perf_dashboard --days 60
    python manage.py perf_dashboard --symbol ETHUSDT
    python manage.py perf_dashboard --json reports/dashboard.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz

from execution.models import OperationReport
from signals.models import Signal


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _bucket_stats(trades: list[dict]) -> dict[str, Any]:
    """Compute aggregated stats for a bucket of trades."""
    if not trades:
        return {"n": 0}
    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n = len(pnls)
    wr = len(wins) / n if n else 0
    avg_w = sum(wins) / len(wins) if wins else 0
    avg_l = sum(losses) / len(losses) if losses else 0
    payoff = abs(avg_w / avg_l) if avg_l else 0
    expectancy = wr * avg_w + (1 - wr) * avg_l
    total = sum(pnls)

    # Profit factor (sum wins / abs sum losses)
    sw = sum(wins)
    sl_abs = abs(sum(losses))
    pf = sw / sl_abs if sl_abs else float("inf") if sw > 0 else 0

    # Avg duration (minutes)
    durations = [t.get("duration_min", 0) for t in trades if t.get("duration_min")]
    avg_dur = sum(durations) / len(durations) if durations else 0

    return {
        "n": n,
        "win_rate": round(wr, 4),
        "avg_win_pct": round(avg_w * 100, 4),
        "avg_loss_pct": round(avg_l * 100, 4),
        "payoff_ratio": round(payoff, 4),
        "expectancy_pct": round(expectancy * 100, 4),
        "profit_factor": round(pf, 4),
        "total_return_pct": round(total * 100, 4),
        "avg_duration_min": round(avg_dur, 1),
    }


def _cols(stats: dict, label: str) -> str:
    """Format a single row for the text table."""
    if stats["n"] == 0:
        return f"  {label:<20} {'---':>5}  {'':>7}  {'':>8}  {'':>8}  {'':>7}  {'':>7}  {'':>8}"
    return (
        f"  {label:<20} {stats['n']:>5}  "
        f"{stats['win_rate']:>6.1%}  "
        f"{stats['expectancy_pct']:>+7.3f}%  "
        f"{stats['profit_factor']:>7.2f}  "
        f"{stats['payoff_ratio']:>6.2f}  "
        f"{stats['total_return_pct']:>+6.2f}%  "
        f"{stats['avg_duration_min']:>7.0f}m"
    )


_HEADER = (
    f"  {'':20} {'#':>5}  {'WR':>7}  {'Expect':>8}  {'PF':>8}  "
    f"{'Payoff':>7}  {'Total':>7}  {'AvgDur':>8}"
)
_SEP = "  " + "-" * 82


class Command(BaseCommand):
    help = "Performance dashboard per module, symbol, reason, and direction"

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--symbol", type=str, default="")
        parser.add_argument("--json", type=str, default="")

    def handle(self, **opts):
        trades = self._load_trades(opts["days"], opts["symbol"])
        if not trades:
            self.stderr.write(self.style.ERROR("No trades found."))
            return

        # Enrich trades with module info from Signal payload
        self._enrich_with_signal(trades)

        result = {}

        # ---------- Overall ----------
        overall = _bucket_stats(trades)
        result["overall"] = overall

        self.stdout.write(self.style.NOTICE(
            f"\n{'='*90}\n  Performance Dashboard — {len(trades)} trades, "
            f"last {opts['days']}d"
            f"{(' (' + opts['symbol'] + ')') if opts['symbol'] else ''}"
            f"\n{'='*90}"
        ))
        self.stdout.write(_HEADER)
        self.stdout.write(_SEP)
        self.stdout.write(_cols(overall, "OVERALL"))

        # ---------- By Direction ----------
        by_dir = self._group_stats(trades, "side")
        result["by_direction"] = by_dir
        self.stdout.write(self.style.SUCCESS("\n  By Direction"))
        self.stdout.write(_HEADER)
        self.stdout.write(_SEP)
        for k in sorted(by_dir):
            self.stdout.write(_cols(by_dir[k], k))

        # ---------- By Symbol ----------
        by_sym = self._group_stats(trades, "symbol")
        result["by_symbol"] = by_sym
        self.stdout.write(self.style.SUCCESS("\n  By Symbol"))
        self.stdout.write(_HEADER)
        self.stdout.write(_SEP)
        for k in sorted(by_sym, key=lambda s: by_sym[s]["total_return_pct"], reverse=True):
            self.stdout.write(_cols(by_sym[k], k))

        # ---------- By Reason ----------
        by_reason = self._group_stats(trades, "reason")
        result["by_reason"] = by_reason
        self.stdout.write(self.style.SUCCESS("\n  By Close Reason"))
        self.stdout.write(_HEADER)
        self.stdout.write(_SEP)
        for k in sorted(by_reason, key=lambda r: by_reason[r]["n"], reverse=True):
            self.stdout.write(_cols(by_reason[k], k))

        # ---------- By Dominant Module ----------
        by_mod = self._group_stats(trades, "dominant_module")
        result["by_module"] = by_mod
        self.stdout.write(self.style.SUCCESS("\n  By Dominant Module (from signal)"))
        self.stdout.write(_HEADER)
        self.stdout.write(_SEP)
        for k in sorted(by_mod, key=lambda m: by_mod[m].get("total_return_pct", 0), reverse=True):
            self.stdout.write(_cols(by_mod[k], k))

        # ---------- Module contributions (all modules per trade) ----------
        mod_detail = self._module_contribution_stats(trades)
        result["by_module_contribution"] = mod_detail
        self.stdout.write(self.style.SUCCESS("\n  Module Contribution (all active modules)"))
        self.stdout.write(_HEADER)
        self.stdout.write(_SEP)
        for k in sorted(mod_detail, key=lambda m: mod_detail[m].get("total_return_pct", 0), reverse=True):
            self.stdout.write(_cols(mod_detail[k], k))

        self.stdout.write("")

        if opts["json"]:
            p = Path(opts["json"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(result, indent=2, default=str))
            self.stdout.write(self.style.SUCCESS(f"Saved → {p}"))

    # ------------------------------------------------------------------
    def _load_trades(self, days: int, symbol: str) -> list[dict]:
        cutoff = dj_tz.now() - timedelta(days=days)
        qs = OperationReport.objects.filter(closed_at__gte=cutoff).select_related("instrument")
        if symbol:
            qs = qs.filter(instrument__symbol__iexact=symbol)
        qs = qs.order_by("closed_at")

        trades = []
        for r in qs:
            dur = 0
            if r.opened_at and r.closed_at:
                dur = (r.closed_at - r.opened_at).total_seconds() / 60
            trades.append({
                "symbol": r.instrument.symbol,
                "side": r.side,
                "pnl_pct": _f(r.pnl_pct),
                "pnl_abs": _f(r.pnl_abs),
                "reason": r.reason,
                "close_sub_reason": r.close_sub_reason,
                "signal_id": r.signal_id,
                "outcome": r.outcome,
                "duration_min": dur,
                "opened_at": r.opened_at,
                "closed_at": r.closed_at,
                # Will be enriched
                "dominant_module": "unknown",
                "modules": [],
            })
        return trades

    def _enrich_with_signal(self, trades: list[dict]):
        """Join trades with their Signal payload to extract module info."""
        # Collect valid signal IDs
        sig_ids = set()
        for t in trades:
            if t["signal_id"]:
                try:
                    sig_ids.add(int(t["signal_id"]))
                except (ValueError, TypeError):
                    pass

        if not sig_ids:
            return

        signals = Signal.objects.filter(id__in=sig_ids).only("id", "payload_json", "strategy")
        sig_map = {s.id: s for s in signals}

        for t in trades:
            if not t["signal_id"]:
                continue
            try:
                sig = sig_map.get(int(t["signal_id"]))
            except (ValueError, TypeError):
                continue
            if not sig or not sig.payload_json:
                continue

            payload = sig.payload_json
            reasons = payload.get("reasons", {})
            contribs = reasons.get("module_contributions", [])

            if contribs:
                t["modules"] = [
                    {"name": c.get("module", "?"), "contribution": _f(c.get("contribution", 0))}
                    for c in contribs
                ]
                # Dominant = highest absolute contribution
                best = max(contribs, key=lambda c: abs(_f(c.get("contribution", 0))))
                t["dominant_module"] = best.get("module", "unknown")
            elif sig.strategy:
                t["dominant_module"] = sig.strategy

    @staticmethod
    def _group_stats(trades: list[dict], key: str) -> dict[str, dict]:
        groups: dict[str, list] = defaultdict(list)
        for t in trades:
            groups[t.get(key, "unknown")].append(t)
        return {k: _bucket_stats(v) for k, v in groups.items()}

    @staticmethod
    def _module_contribution_stats(trades: list[dict]) -> dict[str, dict]:
        """Stats for each module across ALL trades where it was active."""
        groups: dict[str, list] = defaultdict(list)
        for t in trades:
            for m in t.get("modules", []):
                if m["name"]:
                    groups[m["name"]].append(t)
        # Deduplicate: each trade counted once per module it appears in
        return {k: _bucket_stats(v) for k, v in groups.items()}
