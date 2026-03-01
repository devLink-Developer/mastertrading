"""
Performance dashboard for live OperationReport data.

Usage:
    python manage.py perf_dashboard                     # last 30 days
    python manage.py perf_dashboard --days 60
    python manage.py perf_dashboard --symbol ETHUSDT
    python manage.py perf_dashboard --json reports/dashboard.json
"""
from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz

from execution.models import OperationReport
from marketdata.models import Candle
from signals.models import Signal


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _finite_or_none(v: Any) -> float | None:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    q = max(0.0, min(1.0, float(q)))
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    idx = (len(xs) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return xs[lo]
    w = idx - lo
    return (xs[lo] * (1.0 - w)) + (xs[hi] * w)


def _fmt_opt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


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

    sw = sum(wins)
    sl_abs = abs(sum(losses))
    pf = sw / sl_abs if sl_abs else float("inf") if sw > 0 else 0

    durations = [t.get("duration_min", 0) for t in trades if t.get("duration_min")]
    avg_dur = sum(durations) / len(durations) if durations else 0

    mfe_capture_values = [
        x for x in (_finite_or_none(t.get("mfe_capture_ratio")) for t in trades) if x is not None
    ]
    mfe_capture_avg = (
        (sum(mfe_capture_values) / len(mfe_capture_values))
        if mfe_capture_values else None
    )
    mfe_capture_p50 = _percentile(mfe_capture_values, 0.50)
    mfe_capture_p75 = _percentile(mfe_capture_values, 0.75)

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
        "mfe_capture_avg": round(mfe_capture_avg, 4) if mfe_capture_avg is not None else None,
        "mfe_capture_p50": round(mfe_capture_p50, 4) if mfe_capture_p50 is not None else None,
        "mfe_capture_p75": round(mfe_capture_p75, 4) if mfe_capture_p75 is not None else None,
    }


def _cols(stats: dict, label: str) -> str:
    """Format a single row for the text table."""
    if stats["n"] == 0:
        return (
            f"  {label:<20} {'---':>5}  {'':>7}  {'':>8}  {'':>8}  "
            f"{'':>7}  {'':>7}  {'':>8}  {'':>8}"
        )
    return (
        f"  {label:<20} {stats['n']:>5}  "
        f"{stats['win_rate']:>6.1%}  "
        f"{stats['expectancy_pct']:>+7.3f}%  "
        f"{stats['profit_factor']:>7.2f}  "
        f"{stats['payoff_ratio']:>6.2f}  "
        f"{stats['total_return_pct']:>+6.2f}%  "
        f"{stats['avg_duration_min']:>7.0f}m  "
        f"{_fmt_opt(stats.get('mfe_capture_avg')):>8}"
    )


_HEADER = (
    f"  {'':20} {'#':>5}  {'WR':>7}  {'Expect':>8}  {'PF':>8}  "
    f"{'Payoff':>7}  {'Total':>7}  {'AvgDur':>8}  {'MFEcap':>8}"
)
_SEP = "  " + "-" * 94


class Command(BaseCommand):
    help = "Performance dashboard per module, symbol, reason, direction, and regime"

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--symbol", type=str, default="")
        parser.add_argument("--json", type=str, default="")

    def handle(self, **opts):
        trades = self._load_trades(opts["days"], opts["symbol"])
        if not trades:
            self.stderr.write(self.style.ERROR("No trades found."))
            return

        self._enrich_with_signal(trades)
        self._enrich_with_regime(trades)

        result: dict[str, Any] = {}

        overall = _bucket_stats(trades)
        result["overall"] = overall

        self.stdout.write(self.style.NOTICE(
            f"\n{'='*104}\n  Performance Dashboard - {len(trades)} trades, "
            f"last {opts['days']}d"
            f"{(' (' + opts['symbol'] + ')') if opts['symbol'] else ''}"
            f"\n{'='*104}"
        ))
        self.stdout.write(_HEADER)
        self.stdout.write(_SEP)
        self.stdout.write(_cols(overall, "OVERALL"))

        by_dir = self._group_stats(trades, "side")
        result["by_direction"] = by_dir
        self.stdout.write(self.style.SUCCESS("\n  By Direction"))
        self.stdout.write(_HEADER)
        self.stdout.write(_SEP)
        for k in sorted(by_dir):
            self.stdout.write(_cols(by_dir[k], k))

        by_sym = self._group_stats(trades, "symbol")
        result["by_symbol"] = by_sym
        self.stdout.write(self.style.SUCCESS("\n  By Symbol"))
        self.stdout.write(_HEADER)
        self.stdout.write(_SEP)
        for k in sorted(by_sym, key=lambda s: by_sym[s]["total_return_pct"], reverse=True):
            self.stdout.write(_cols(by_sym[k], k))

        by_regime = self._group_stats(trades, "regime")
        result["by_regime"] = by_regime
        self.stdout.write(self.style.SUCCESS("\n  By Regime"))
        self.stdout.write(_HEADER)
        self.stdout.write(_SEP)
        for k in sorted(by_regime):
            self.stdout.write(_cols(by_regime[k], k))

        by_reason = self._group_stats(trades, "reason")
        result["by_reason"] = by_reason
        self.stdout.write(self.style.SUCCESS("\n  By Close Reason"))
        self.stdout.write(_HEADER)
        self.stdout.write(_SEP)
        for k in sorted(by_reason, key=lambda r: by_reason[r]["n"], reverse=True):
            self.stdout.write(_cols(by_reason[k], k))

        by_mod = self._group_stats(trades, "dominant_module")
        result["by_module"] = by_mod
        self.stdout.write(self.style.SUCCESS("\n  By Dominant Module (from signal)"))
        self.stdout.write(_HEADER)
        self.stdout.write(_SEP)
        for k in sorted(by_mod, key=lambda m: by_mod[m].get("total_return_pct", 0), reverse=True):
            self.stdout.write(_cols(by_mod[k], k))

        mod_detail = self._module_contribution_stats(trades)
        result["by_module_contribution"] = mod_detail
        self.stdout.write(self.style.SUCCESS("\n  Module Contribution (all active modules)"))
        self.stdout.write(_HEADER)
        self.stdout.write(_SEP)
        for k in sorted(mod_detail, key=lambda m: mod_detail[m].get("total_return_pct", 0), reverse=True):
            self.stdout.write(_cols(mod_detail[k], k))

        by_sdr = self._group_stats_custom(
            trades,
            key_fn=lambda t: f"{t.get('symbol', 'unknown')}|{t.get('side', 'na')}|{t.get('regime', 'unknown')}",
        )
        result["by_symbol_direction_regime"] = by_sdr
        self.stdout.write(self.style.SUCCESS("\n  By Symbol|Direction|Regime"))
        self.stdout.write(_HEADER)
        self.stdout.write(_SEP)
        for k in sorted(by_sdr, key=lambda x: by_sdr[x]["n"], reverse=True):
            self.stdout.write(_cols(by_sdr[k], k))

        self.stdout.write(self.style.SUCCESS("\n  MFE Capture Ratio Summary"))
        self.stdout.write(
            f"  OVERALL avg={_fmt_opt(overall.get('mfe_capture_avg'))} "
            f"p50={_fmt_opt(overall.get('mfe_capture_p50'))} "
            f"p75={_fmt_opt(overall.get('mfe_capture_p75'))}"
        )
        for k in sorted(by_regime):
            stats = by_regime[k]
            self.stdout.write(
                f"  {k:<20} avg={_fmt_opt(stats.get('mfe_capture_avg'))} "
                f"p50={_fmt_opt(stats.get('mfe_capture_p50'))} "
                f"p75={_fmt_opt(stats.get('mfe_capture_p75'))}"
            )
        self.stdout.write("")

        if opts["json"]:
            p = Path(opts["json"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(f"Saved -> {p}"))

    def _load_trades(self, days: int, symbol: str) -> list[dict]:
        cutoff = dj_tz.now() - timedelta(days=days)
        qs = OperationReport.objects.filter(closed_at__gte=cutoff).select_related("instrument")
        if symbol:
            qs = qs.filter(instrument__symbol__iexact=symbol)
        qs = qs.order_by("closed_at")

        trades: list[dict[str, Any]] = []
        for r in qs:
            dur = 0.0
            if r.opened_at and r.closed_at:
                dur = (r.closed_at - r.opened_at).total_seconds() / 60.0
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
                "mfe_r": _finite_or_none(r.mfe_r),
                "mae_r": _finite_or_none(r.mae_r),
                "mfe_capture_ratio": _finite_or_none(r.mfe_capture_ratio),
                "dominant_module": "unknown",
                "modules": [],
                "regime": "unknown",
            })
        return trades

    def _enrich_with_signal(self, trades: list[dict]):
        """Join trades with Signal payload to extract module info."""
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
            if not sig or not isinstance(sig.payload_json, dict):
                if sig and sig.strategy:
                    t["dominant_module"] = sig.strategy
                continue

            payload = sig.payload_json
            reasons = payload.get("reasons", {})
            contribs = reasons.get("module_contributions", [])

            if contribs:
                t["modules"] = [
                    {"name": c.get("module", "?"), "contribution": _f(c.get("contribution", 0))}
                    for c in contribs
                ]
                best = max(contribs, key=lambda c: abs(_f(c.get("contribution", 0))))
                t["dominant_module"] = best.get("module", "unknown")
            elif sig.strategy:
                t["dominant_module"] = sig.strategy

    def _enrich_with_regime(self, trades: list[dict]) -> None:
        """Infer simple HTF regime (bull/bear/neutral) near trade close using 1h EMA20/EMA50."""
        if not trades:
            return
        symbols = sorted({str(t.get("symbol") or "").strip().upper() for t in trades if t.get("symbol")})
        closed_ts = [t.get("closed_at") for t in trades if t.get("closed_at") is not None]
        if not symbols or not closed_ts:
            return

        max_closed = max(closed_ts)
        rows = (
            Candle.objects.filter(
                timeframe="1h",
                instrument__symbol__in=symbols,
                ts__lte=max_closed,
            )
            .order_by("instrument__symbol", "ts")
            .values_list("instrument__symbol", "ts", "close")
        )
        by_symbol: dict[str, list[tuple[Any, float]]] = defaultdict(list)
        for sym, ts, close in rows:
            by_symbol[str(sym).upper()].append((ts, _f(close)))

        alpha20 = 2.0 / (20.0 + 1.0)
        alpha50 = 2.0 / (50.0 + 1.0)
        regime_lookup: dict[str, dict[str, list]] = {}

        for sym, series in by_symbol.items():
            if not series:
                continue
            ema20 = None
            ema50 = None
            ts_list: list[Any] = []
            regime_list: list[str] = []
            for ts, close in series:
                if ema20 is None:
                    ema20 = close
                    ema50 = close
                else:
                    ema20 = (close - ema20) * alpha20 + ema20
                    ema50 = (close - ema50) * alpha50 + ema50
                if ema20 > ema50 and close >= ema20:
                    regime = "bull"
                elif ema20 < ema50 and close <= ema20:
                    regime = "bear"
                else:
                    regime = "neutral"
                ts_list.append(ts)
                regime_list.append(regime)
            regime_lookup[sym] = {"ts": ts_list, "regime": regime_list}

        for t in trades:
            sym = str(t.get("symbol") or "").strip().upper()
            closed_at = t.get("closed_at")
            if not sym or not closed_at:
                t["regime"] = "unknown"
                continue
            lookup = regime_lookup.get(sym)
            if not lookup:
                t["regime"] = "unknown"
                continue
            ts_list = lookup["ts"]
            if not ts_list:
                t["regime"] = "unknown"
                continue
            idx = bisect_right(ts_list, closed_at) - 1
            if idx < 0:
                t["regime"] = "unknown"
                continue
            t["regime"] = lookup["regime"][idx]

    @staticmethod
    def _group_stats(trades: list[dict], key: str) -> dict[str, dict]:
        groups: dict[str, list] = defaultdict(list)
        for t in trades:
            groups[str(t.get(key, "unknown"))].append(t)
        return {k: _bucket_stats(v) for k, v in groups.items()}

    @staticmethod
    def _group_stats_custom(
        trades: list[dict],
        key_fn: Callable[[dict], str],
    ) -> dict[str, dict]:
        groups: dict[str, list] = defaultdict(list)
        for t in trades:
            key = str(key_fn(t) or "unknown")
            groups[key].append(t)
        return {k: _bucket_stats(v) for k, v in groups.items()}

    @staticmethod
    def _module_contribution_stats(trades: list[dict]) -> dict[str, dict]:
        """Stats for each module across all trades where it was active."""
        groups: dict[str, list] = defaultdict(list)
        for t in trades:
            for m in t.get("modules", []):
                name = str(m.get("name") or "").strip()
                if name:
                    groups[name].append(t)
        return {k: _bucket_stats(v) for k, v in groups.items()}
