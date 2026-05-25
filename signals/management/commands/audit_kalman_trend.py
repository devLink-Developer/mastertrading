from __future__ import annotations

import json
from bisect import bisect_left
from collections import defaultdict
from contextlib import contextmanager
from datetime import timedelta
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone as dj_tz

from core.models import Instrument
from execution.models import OperationReport
from marketdata.models import Candle
from signals.modules import trend as trend_module
from signals.sessions import get_current_session


DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,LINKUSDT,ENAUSDT"
DEFAULT_Q_GRID = "0.003,0.006,0.01,0.02,0.04"
DEFAULT_SLOPE_GRID = "0.0015,0.002,0.003,0.004,0.006"
DEFAULT_BOOST_GRID = "0,0.03,0.05"


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_float_grid(raw: str, *, name: str) -> list[float]:
    values: list[float] = []
    for part in str(raw or "").split(","):
        txt = part.strip()
        if not txt:
            continue
        try:
            values.append(float(txt))
        except ValueError as exc:
            raise CommandError(f"Invalid {name} value: {txt}") from exc
    if not values:
        raise CommandError(f"{name} cannot be empty")
    return values


def _parse_symbols(raw: str) -> list[str]:
    symbols = [s.strip().upper() for s in str(raw or "").split(",") if s.strip()]
    if not symbols:
        raise CommandError("--symbols cannot be empty")
    return symbols


@contextmanager
def _temporary_settings(**overrides):
    previous = {}
    for key, value in overrides.items():
        previous[key] = getattr(settings, key, None)
        setattr(settings, key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(settings, key, value)


def _candles_to_df(candles: list[Candle]) -> pd.DataFrame:
    rows = [
        {
            "ts": c.ts,
            "open": _f(c.open),
            "high": _f(c.high),
            "low": _f(c.low),
            "close": _f(c.close),
            "volume": _f(c.volume),
        }
        for c in candles
    ]
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).sort_values("ts")
    df = df.set_index(pd.DatetimeIndex(df.pop("ts")))
    return df


def _slice_until(df: pd.DataFrame, ts, lookback: int) -> pd.DataFrame:
    pos = int(df.index.searchsorted(pd.Timestamp(ts), side="right"))
    if pos <= 0:
        return df.iloc[0:0]
    start = max(0, pos - lookback)
    return df.iloc[start:pos]


def _future_close(df: pd.DataFrame, ts, minutes: int) -> float | None:
    target = pd.Timestamp(ts) + pd.Timedelta(minutes=int(minutes))
    pos = int(df.index.searchsorted(target, side="left"))
    if pos >= len(df):
        return None
    return _f(df.iloc[pos]["close"], default=0.0) or None


def _directional_return(direction: str, entry: float, future: float | None) -> float | None:
    if future is None or entry <= 0:
        return None
    sign = 1.0 if direction == "long" else -1.0
    return ((future - entry) / entry) * sign


def _empty_stats(q_ratio: float, slope_min: float, boost: float) -> dict[str, Any]:
    return {
        "q_ratio": q_ratio,
        "slope_min": slope_min,
        "boost": boost,
        "samples": 0,
        "emissions": 0,
        "kalman_fallback_count": 0,
        "kalman_boost_count": 0,
        "follow_30m_sum": 0.0,
        "follow_30m_samples": 0,
        "follow_60m_sum": 0.0,
        "follow_60m_samples": 0,
        "linked_trade_ids": set(),
        "linked_trade_pnl_abs": 0.0,
        "linked_trade_pnl_pct_sum": 0.0,
        "linked_trade_wins": 0,
        "by_symbol_session": defaultdict(
            lambda: {
                "samples": 0,
                "emissions": 0,
                "kalman_fallback_count": 0,
                "kalman_boost_count": 0,
                "follow_30m_sum": 0.0,
                "follow_30m_samples": 0,
                "follow_60m_sum": 0.0,
                "follow_60m_samples": 0,
                "linked_trades": 0,
                "linked_trade_pnl_abs": 0.0,
            }
        ),
    }


class Command(BaseCommand):
    help = "Read-only audit of Kalman trend parameter grids against historical candles and trades."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--symbols", type=str, default=DEFAULT_SYMBOLS)
        parser.add_argument("--ltf", type=str, default="1m")
        parser.add_argument("--htf", type=str, default="1h")
        parser.add_argument("--q-grid", type=str, default=DEFAULT_Q_GRID)
        parser.add_argument("--slope-grid", type=str, default=DEFAULT_SLOPE_GRID)
        parser.add_argument("--boost-grid", type=str, default=DEFAULT_BOOST_GRID)
        parser.add_argument(
            "--cadence-minutes",
            type=int,
            default=15,
            help="Sampling cadence for replay points. Use 5 for a denser audit.",
        )
        parser.add_argument("--lookback", type=int, default=0)
        parser.add_argument("--link-minutes", type=int, default=15)
        parser.add_argument("--top", type=int, default=10)
        parser.add_argument("--json", type=str, default="")

    def handle(self, *args, **opts):
        days = max(1, int(opts["days"]))
        symbols = _parse_symbols(opts["symbols"])
        q_grid = _parse_float_grid(opts["q_grid"], name="q-grid")
        slope_grid = _parse_float_grid(opts["slope_grid"], name="slope-grid")
        boost_grid = _parse_float_grid(opts["boost_grid"], name="boost-grid")
        cadence_minutes = max(1, int(opts["cadence_minutes"]))
        lookback = int(opts["lookback"]) or int(getattr(settings, "MODULE_LOOKBACK_BARS", 240))
        lookback = max(80, lookback)
        link_minutes = max(1, int(opts["link_minutes"]))
        top = max(1, int(opts["top"]))

        now = dj_tz.now()
        since = now - timedelta(days=days)
        warmup_since = since - timedelta(days=max(14, int((lookback / 24) + 3)))
        instruments = list(Instrument.objects.filter(symbol__in=symbols).order_by("symbol"))
        missing = sorted(set(symbols) - {inst.symbol for inst in instruments})
        if missing:
            raise CommandError(f"Instruments not found: {', '.join(missing)}")

        contexts = self._load_contexts(
            instruments=instruments,
            ltf=opts["ltf"],
            htf=opts["htf"],
            since=since,
            warmup_since=warmup_since,
            lookback=lookback,
            cadence_minutes=cadence_minutes,
            link_minutes=link_minutes,
        )
        if not contexts:
            raise CommandError("No usable candle contexts found for the requested filters.")

        results: list[dict[str, Any]] = []
        for q_ratio, slope_min, boost in product(q_grid, slope_grid, boost_grid):
            stats = _empty_stats(q_ratio, slope_min, boost)
            with _temporary_settings(
                MODULE_TREND_KALMAN_ENABLED=True,
                MODULE_TREND_KALMAN_Q_RATIO=q_ratio,
                MODULE_TREND_KALMAN_SLOPE_MIN=slope_min,
                MODULE_TREND_KALMAN_BOOST=boost,
            ):
                self._evaluate_combo(
                    stats=stats,
                    contexts=contexts,
                    lookback=lookback,
                    link_minutes=link_minutes,
                )
            results.append(self._finalize_stats(stats))

        results.sort(
            key=lambda row: (
                row["linked_trade_pnl_abs"],
                row["follow_60m_avg"],
                row["follow_30m_avg"],
                row["emissions"],
            ),
            reverse=True,
        )

        report = {
            "days": days,
            "symbols": symbols,
            "ltf": opts["ltf"],
            "htf": opts["htf"],
            "cadence_minutes": cadence_minutes,
            "lookback": lookback,
            "link_minutes": link_minutes,
            "combos": results,
        }
        self._print_report(report, top=top)

        if opts["json"]:
            path = Path(opts["json"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(f"\nJSON written to {path}"))

    def _load_contexts(
        self,
        *,
        instruments: list[Instrument],
        ltf: str,
        htf: str,
        since,
        warmup_since,
        lookback: int,
        cadence_minutes: int,
        link_minutes: int,
    ) -> list[dict[str, Any]]:
        contexts: list[dict[str, Any]] = []
        until_for_ops = dj_tz.now() + timedelta(minutes=link_minutes)
        for inst in instruments:
            ltf_df = _candles_to_df(
                list(
                    Candle.objects.filter(
                        instrument=inst,
                        timeframe=ltf,
                        ts__gte=warmup_since,
                    ).order_by("ts")
                )
            )
            htf_df = _candles_to_df(
                list(
                    Candle.objects.filter(
                        instrument=inst,
                        timeframe=htf,
                        ts__gte=warmup_since,
                    ).order_by("ts")
                )
            )
            if len(ltf_df) < 80 or len(htf_df) < 80:
                continue
            sample_times = self._sample_times(ltf_df, since=since, cadence_minutes=cadence_minutes)
            if not sample_times:
                continue
            contexts.append(
                {
                    "instrument": inst,
                    "symbol": inst.symbol,
                    "ltf_df": ltf_df,
                    "htf_df": htf_df,
                    "sample_times": sample_times,
                    "ops_by_side": self._load_ops_by_side(inst, since, until_for_ops),
                }
            )
        return contexts

    def _sample_times(self, df: pd.DataFrame, *, since, cadence_minutes: int) -> list[Any]:
        out: list[Any] = []
        next_ts = pd.Timestamp(since)
        step = pd.Timedelta(minutes=cadence_minutes)
        for ts in df.index:
            if ts < pd.Timestamp(since):
                continue
            if ts >= next_ts:
                out.append(ts.to_pydatetime())
                next_ts = ts + step
        return out

    def _load_ops_by_side(self, inst: Instrument, since, until) -> dict[str, list[dict[str, Any]]]:
        out = {"buy": [], "sell": []}
        qs = (
            OperationReport.objects.filter(
                instrument=inst,
                opened_at__gte=since,
                opened_at__lte=until,
            )
            .order_by("opened_at")
            .only("id", "side", "opened_at", "pnl_abs", "pnl_pct")
        )
        for op in qs:
            side = str(op.side or "").strip().lower()
            if side in out and op.opened_at is not None:
                out[side].append(
                    {
                        "id": op.id,
                        "opened_at": op.opened_at,
                        "pnl_abs": _f(op.pnl_abs),
                        "pnl_pct": _f(op.pnl_pct),
                    }
                )
        return out

    def _find_linked_op(
        self,
        ops: list[dict[str, Any]],
        ts,
        *,
        link_minutes: int,
    ) -> dict[str, Any] | None:
        if not ops:
            return None
        opened = [op["opened_at"] for op in ops]
        idx = bisect_left(opened, ts)
        if idx >= len(ops):
            return None
        op = ops[idx]
        if op["opened_at"] <= ts + timedelta(minutes=link_minutes):
            return op
        return None

    def _evaluate_combo(
        self,
        *,
        stats: dict[str, Any],
        contexts: list[dict[str, Any]],
        lookback: int,
        link_minutes: int,
    ) -> None:
        for ctx in contexts:
            symbol = ctx["symbol"]
            ltf_df = ctx["ltf_df"]
            htf_df = ctx["htf_df"]
            ops_by_side = ctx["ops_by_side"]
            for ts in ctx["sample_times"]:
                session = get_current_session(ts)
                bucket = stats["by_symbol_session"][f"{symbol}|{session}"]
                stats["samples"] += 1
                bucket["samples"] += 1

                df_ltf = _slice_until(ltf_df, ts, lookback)
                df_htf = _slice_until(htf_df, ts, lookback)
                if len(df_ltf) < 80 or len(df_htf) < 80:
                    continue
                result = trend_module.detect(df_ltf, df_htf, [], session, symbol=symbol)
                if not result:
                    continue

                direction = str(result.get("direction") or "").strip().lower()
                if direction not in {"long", "short"}:
                    continue
                stats["emissions"] += 1
                bucket["emissions"] += 1

                reasons = result.get("reasons") if isinstance(result.get("reasons"), dict) else {}
                if reasons.get("kalman_fallback"):
                    stats["kalman_fallback_count"] += 1
                    bucket["kalman_fallback_count"] += 1
                if _f(reasons.get("kalman_boost")) > 0:
                    stats["kalman_boost_count"] += 1
                    bucket["kalman_boost_count"] += 1

                entry = _f(df_ltf.iloc[-1]["close"])
                for minutes, sum_key, sample_key in (
                    (30, "follow_30m_sum", "follow_30m_samples"),
                    (60, "follow_60m_sum", "follow_60m_samples"),
                ):
                    ret = _directional_return(direction, entry, _future_close(ltf_df, ts, minutes))
                    if ret is None:
                        continue
                    stats[sum_key] += ret
                    stats[sample_key] += 1
                    bucket[sum_key] += ret
                    bucket[sample_key] += 1

                side = "buy" if direction == "long" else "sell"
                op = self._find_linked_op(
                    ops_by_side.get(side, []),
                    ts,
                    link_minutes=link_minutes,
                )
                if op and op["id"] not in stats["linked_trade_ids"]:
                    stats["linked_trade_ids"].add(op["id"])
                    stats["linked_trade_pnl_abs"] += op["pnl_abs"]
                    stats["linked_trade_pnl_pct_sum"] += op["pnl_pct"]
                    if op["pnl_abs"] > 0:
                        stats["linked_trade_wins"] += 1
                    bucket["linked_trades"] += 1
                    bucket["linked_trade_pnl_abs"] += op["pnl_abs"]

    def _finalize_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        linked_n = len(stats["linked_trade_ids"])
        by_bucket = []
        for bucket, vals in stats["by_symbol_session"].items():
            row = dict(vals)
            row["bucket"] = bucket
            row["follow_30m_avg"] = (
                row["follow_30m_sum"] / row["follow_30m_samples"]
                if row["follow_30m_samples"] else 0.0
            )
            row["follow_60m_avg"] = (
                row["follow_60m_sum"] / row["follow_60m_samples"]
                if row["follow_60m_samples"] else 0.0
            )
            row.pop("follow_30m_sum", None)
            row.pop("follow_60m_sum", None)
            by_bucket.append(row)
        by_bucket.sort(
            key=lambda row: (
                row["linked_trade_pnl_abs"],
                row["follow_60m_avg"],
                row["emissions"],
            ),
            reverse=True,
        )
        return {
            "q_ratio": stats["q_ratio"],
            "slope_min": stats["slope_min"],
            "boost": stats["boost"],
            "samples": stats["samples"],
            "emissions": stats["emissions"],
            "emission_rate": round(stats["emissions"] / stats["samples"], 6) if stats["samples"] else 0.0,
            "kalman_fallback_count": stats["kalman_fallback_count"],
            "kalman_boost_count": stats["kalman_boost_count"],
            "follow_30m_avg": (
                stats["follow_30m_sum"] / stats["follow_30m_samples"]
                if stats["follow_30m_samples"] else 0.0
            ),
            "follow_30m_samples": stats["follow_30m_samples"],
            "follow_60m_avg": (
                stats["follow_60m_sum"] / stats["follow_60m_samples"]
                if stats["follow_60m_samples"] else 0.0
            ),
            "follow_60m_samples": stats["follow_60m_samples"],
            "linked_trades": linked_n,
            "linked_trade_win_rate": round(stats["linked_trade_wins"] / linked_n, 6) if linked_n else 0.0,
            "linked_trade_pnl_abs": round(stats["linked_trade_pnl_abs"], 8),
            "linked_trade_pnl_pct_avg": (
                stats["linked_trade_pnl_pct_sum"] / linked_n if linked_n else 0.0
            ),
            "by_symbol_session": by_bucket,
        }

    def _print_report(self, report: dict[str, Any], *, top: int) -> None:
        self.stdout.write("")
        self.stdout.write(
            self.style.NOTICE(
                "Kalman trend audit | "
                f"days={report['days']} ltf={report['ltf']} htf={report['htf']} "
                f"cadence={report['cadence_minutes']}m lookback={report['lookback']}"
            )
        )
        self.stdout.write(
            "  "
            f"{'q':>7} {'slope':>8} {'boost':>7} {'samples':>8} {'emit':>7} "
            f"{'fb':>5} {'kboost':>7} {'f30':>9} {'f60':>9} "
            f"{'trades':>7} {'WR':>7} {'PnL$':>10}"
        )
        self.stdout.write("  " + "-" * 102)
        for row in report["combos"][:top]:
            self.stdout.write(
                "  "
                f"{row['q_ratio']:>7.4f} {row['slope_min']:>8.4f} {row['boost']:>7.3f} "
                f"{row['samples']:>8} {row['emissions']:>7} "
                f"{row['kalman_fallback_count']:>5} {row['kalman_boost_count']:>7} "
                f"{row['follow_30m_avg'] * 100:>+8.3f}% "
                f"{row['follow_60m_avg'] * 100:>+8.3f}% "
                f"{row['linked_trades']:>7} {row['linked_trade_win_rate']:>6.1%} "
                f"{row['linked_trade_pnl_abs']:>+10.4f}"
            )

        if report["combos"]:
            best = report["combos"][0]
            self.stdout.write("")
            self.stdout.write(self.style.SUCCESS("Best combo by linked PnL / follow-through"))
            self.stdout.write(
                f"  q={best['q_ratio']} slope={best['slope_min']} boost={best['boost']} "
                f"emissions={best['emissions']} linked_pnl={best['linked_trade_pnl_abs']:+.4f}"
            )
            self.stdout.write("  Top symbol/session buckets:")
            for bucket in best["by_symbol_session"][:top]:
                self.stdout.write(
                    "  "
                    f"{bucket['bucket']:<18} samples={bucket['samples']:>5} "
                    f"emit={bucket['emissions']:>4} fb={bucket['kalman_fallback_count']:>3} "
                    f"f30={bucket['follow_30m_avg'] * 100:+.3f}% "
                    f"f60={bucket['follow_60m_avg'] * 100:+.3f}% "
                    f"linked={bucket['linked_trades']:>3} "
                    f"pnl={bucket['linked_trade_pnl_abs']:+.4f}"
                )
