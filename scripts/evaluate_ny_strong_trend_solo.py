from __future__ import annotations

"""
Evaluate whether BTC/ETH should be allowed to trade with strong-trend solo
inside the NY session under contextual ADX thresholds.

This script runs a small set of controlled backtests and breaks results down
by entry session so we can answer a narrower question than a full optimization:
does relaxing strong-trend solo for BTC/ETH in NY improve edge, or just add
noise?

Usage:
  docker compose run --rm web python scripts/evaluate_ny_strong_trend_solo.py \
      --start 2026-02-01 --end 2026-02-23 \
      --out reports/ny_strong_trend_solo_20260201_20260223.json
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import django

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.conf import settings as django_settings  # noqa: E402

from backtest.engine import compute_metrics, run_backtest  # noqa: E402
from core.models import Instrument  # noqa: E402
from signals.sessions import get_current_session  # noqa: E402


WINDOWS = [
    ("full", "2026-02-01", "2026-02-23"),
    ("prior", "2026-02-01", "2026-02-12"),
    ("recent", "2026-02-12", "2026-02-23"),
]

CONFIGS = [
    {
        "name": "baseline_25_25",
        "overrides": {
            "ATR_MULT_TP": 1.6,
            "ATR_MULT_SL": 1.5,
            "MIN_SIGNAL_SCORE": 0.45,
            "EXECUTION_MIN_SIGNAL_SCORE": 0.45,
            "ALLOCATOR_MIN_MODULES_ACTIVE": 2,
            "ALLOCATOR_STRONG_TREND_SOLO_ENABLED": True,
            "ALLOCATOR_STRONG_TREND_ADX_MIN": 25.0,
            "ALLOCATOR_STRONG_TREND_ADX_MIN_BY_CONTEXT": {},
            "ALLOCATOR_STRONG_TREND_CONFIDENCE_MIN": 0.80,
            "ALLOCATOR_STRONG_TREND_SOLO_DISABLED_SESSIONS": {"ny_open"},
        },
    },
    {
        "name": "btc18_eth19_ny",
        "overrides": {
            "ATR_MULT_TP": 1.6,
            "ATR_MULT_SL": 1.5,
            "MIN_SIGNAL_SCORE": 0.45,
            "EXECUTION_MIN_SIGNAL_SCORE": 0.45,
            "ALLOCATOR_MIN_MODULES_ACTIVE": 2,
            "ALLOCATOR_STRONG_TREND_SOLO_ENABLED": True,
            "ALLOCATOR_STRONG_TREND_ADX_MIN": 25.0,
            "ALLOCATOR_STRONG_TREND_ADX_MIN_BY_CONTEXT": {
                "BTCUSDT:ny": 18.0,
                "ETHUSDT:ny": 19.0,
            },
            "ALLOCATOR_STRONG_TREND_CONFIDENCE_MIN": 0.80,
            "ALLOCATOR_STRONG_TREND_SOLO_DISABLED_SESSIONS": {"ny_open"},
        },
    },
    {
        "name": "btc17_eth18_ny",
        "overrides": {
            "ATR_MULT_TP": 1.6,
            "ATR_MULT_SL": 1.5,
            "MIN_SIGNAL_SCORE": 0.45,
            "EXECUTION_MIN_SIGNAL_SCORE": 0.45,
            "ALLOCATOR_MIN_MODULES_ACTIVE": 2,
            "ALLOCATOR_STRONG_TREND_SOLO_ENABLED": True,
            "ALLOCATOR_STRONG_TREND_ADX_MIN": 25.0,
            "ALLOCATOR_STRONG_TREND_ADX_MIN_BY_CONTEXT": {
                "BTCUSDT:ny": 17.0,
                "ETHUSDT:ny": 18.0,
            },
            "ALLOCATOR_STRONG_TREND_CONFIDENCE_MIN": 0.80,
            "ALLOCATOR_STRONG_TREND_SOLO_DISABLED_SESSIONS": {"ny_open"},
        },
    },
    {
        "name": "btc16_eth17_ny",
        "overrides": {
            "ATR_MULT_TP": 1.6,
            "ATR_MULT_SL": 1.5,
            "MIN_SIGNAL_SCORE": 0.45,
            "EXECUTION_MIN_SIGNAL_SCORE": 0.45,
            "ALLOCATOR_MIN_MODULES_ACTIVE": 2,
            "ALLOCATOR_STRONG_TREND_SOLO_ENABLED": True,
            "ALLOCATOR_STRONG_TREND_ADX_MIN": 25.0,
            "ALLOCATOR_STRONG_TREND_ADX_MIN_BY_CONTEXT": {
                "BTCUSDT:ny": 16.0,
                "ETHUSDT:ny": 17.0,
            },
            "ALLOCATOR_STRONG_TREND_CONFIDENCE_MIN": 0.80,
            "ALLOCATOR_STRONG_TREND_SOLO_DISABLED_SESSIONS": {"ny_open"},
        },
    },
    {
        "name": "eth19_no_opp_carry_conf90",
        "overrides": {
            "ATR_MULT_TP": 1.6,
            "ATR_MULT_SL": 1.5,
            "MIN_SIGNAL_SCORE": 0.45,
            "EXECUTION_MIN_SIGNAL_SCORE": 0.45,
            "ALLOCATOR_MIN_MODULES_ACTIVE": 2,
            "ALLOCATOR_STRONG_TREND_SOLO_ENABLED": True,
            "ALLOCATOR_STRONG_TREND_ADX_MIN": 25.0,
            "ALLOCATOR_STRONG_TREND_ADX_MIN_BY_CONTEXT": {
                "ETHUSDT:ny": 19.0,
            },
            "ALLOCATOR_STRONG_TREND_CONFIDENCE_MIN": 0.90,
            "ALLOCATOR_STRONG_TREND_SOLO_REQUIRES_NO_OPPOSING_CARRY": True,
            "ALLOCATOR_STRONG_TREND_SOLO_DISABLED_SESSIONS": {"ny_open"},
        },
    },
    {
        "name": "btc18_eth19_no_opp_carry_conf90",
        "overrides": {
            "ATR_MULT_TP": 1.6,
            "ATR_MULT_SL": 1.5,
            "MIN_SIGNAL_SCORE": 0.45,
            "EXECUTION_MIN_SIGNAL_SCORE": 0.45,
            "ALLOCATOR_MIN_MODULES_ACTIVE": 2,
            "ALLOCATOR_STRONG_TREND_SOLO_ENABLED": True,
            "ALLOCATOR_STRONG_TREND_ADX_MIN": 25.0,
            "ALLOCATOR_STRONG_TREND_ADX_MIN_BY_CONTEXT": {
                "BTCUSDT:ny": 18.0,
                "ETHUSDT:ny": 19.0,
            },
            "ALLOCATOR_STRONG_TREND_CONFIDENCE_MIN": 0.90,
            "ALLOCATOR_STRONG_TREND_SOLO_REQUIRES_NO_OPPOSING_CARRY": True,
            "ALLOCATOR_STRONG_TREND_SOLO_DISABLED_SESSIONS": {"ny_open"},
        },
    },
]


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


def _parse_date(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _trade_session_metrics(trades: list, session_name: str, initial_equity: float) -> dict:
    selected = [t for t in trades if get_current_session(t.entry_ts) == session_name]
    metrics = compute_metrics(selected, initial_equity)
    metrics["session"] = session_name
    metrics["trade_count"] = len(selected)
    metrics["symbols"] = {}
    for symbol in ("BTCUSDT", "ETHUSDT"):
        symbol_trades = [
            t for t in selected if getattr(t, "instrument_symbol", None) == symbol
        ]
        if not symbol_trades:
            continue
        symbol_metrics = compute_metrics(symbol_trades, initial_equity)
        symbol_metrics["trade_count"] = len(symbol_trades)
        metrics["symbols"][symbol] = symbol_metrics
    return metrics


def _attach_symbols(trades: list, instruments: list[Instrument]) -> None:
    inst_map = {inst.id: inst.symbol for inst in instruments}
    for trade in trades:
        trade.instrument_symbol = inst_map.get(trade.instrument_id, "?")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-02-01", help="YYYY-MM-DD")
    parser.add_argument("--end", default="2026-02-23", help="YYYY-MM-DD")
    parser.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT",
        help="Comma-separated symbols for the focused run",
    )
    parser.add_argument("--equity", type=float, default=1000.0)
    parser.add_argument("--ltf", default="5m")
    parser.add_argument("--htf", default="4h")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    base_start = _parse_date(args.start)
    base_end = _parse_date(args.end)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    instruments = list(Instrument.objects.filter(symbol__in=symbols))
    inst_map = {inst.symbol: inst for inst in instruments}
    missing = [sym for sym in symbols if sym not in inst_map]
    if missing:
        raise SystemExit(f"Missing instruments: {missing}")

    report = {
        "requested_start": base_start.isoformat(),
        "requested_end": base_end.isoformat(),
        "symbols": symbols,
        "configs": [c["name"] for c in CONFIGS],
        "windows": [],
    }

    for window_name, raw_start, raw_end in WINDOWS:
        start = max(base_start, _parse_date(raw_start))
        end = min(base_end, _parse_date(raw_end))
        if start >= end:
            continue

        window_row = {
            "name": window_name,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "results": [],
        }

        for config in CONFIGS:
            with _temporary_settings(**config["overrides"]):
                trades, metrics = run_backtest(
                    instruments=instruments,
                    start=start,
                    end=end,
                    initial_equity=float(args.equity),
                    ltf=args.ltf,
                    htf=args.htf,
                    trailing_stop=True,
                    verbose=False,
                )
            _attach_symbols(trades, instruments)
            window_row["results"].append(
                {
                    "config": config["name"],
                    "overrides": config["overrides"],
                    "overall": metrics,
                    "ny": _trade_session_metrics(trades, "ny", float(args.equity)),
                    "ny_open": _trade_session_metrics(
                        trades, "ny_open", float(args.equity)
                    ),
                }
            )
        report["windows"].append(window_row)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"Saved -> {out_path}")
    else:
        print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
