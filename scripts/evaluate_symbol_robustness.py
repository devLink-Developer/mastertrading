from __future__ import annotations

"""
Compare a small set of fixed backtest configurations per symbol over the
maximum available 5m history inside a requested date window.

This is intentionally cheaper than full walk-forward optimization. It is meant
to answer a narrower question: whether a candidate global configuration keeps
helping (or hurting) when evaluated symbol-by-symbol over a broader sample.

Usage:
  docker compose run --rm web python scripts/evaluate_symbol_robustness.py \
      --start 2026-01-01 --end 2026-02-23 \
      --symbols BTCUSDT,ETHUSDT,SOLUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,XRPUSDT \
      --out reports/symbol_robustness_20260101_20260223.json
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
from django.db.models import Count, Max, Min  # noqa: E402

from backtest.engine import run_backtest  # noqa: E402
from core.models import Instrument  # noqa: E402
from marketdata.models import Candle  # noqa: E402


CONFIGS = [
    {
        "name": "baseline_tp18_sl15_score045",
        "overrides": {
            "ATR_MULT_TP": 1.8,
            "ATR_MULT_SL": 1.5,
            "MIN_SIGNAL_SCORE": 0.45,
            "EXECUTION_MIN_SIGNAL_SCORE": 0.45,
        },
    },
    {
        "name": "candidate_tp16_sl15_score045",
        "overrides": {
            "ATR_MULT_TP": 1.6,
            "ATR_MULT_SL": 1.5,
            "MIN_SIGNAL_SCORE": 0.45,
            "EXECUTION_MIN_SIGNAL_SCORE": 0.45,
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


def _coverage(symbol: str, start: datetime, end: datetime) -> dict:
    inst = Instrument.objects.get(symbol=symbol)
    qs = Candle.objects.filter(
        instrument=inst,
        timeframe="5m",
        ts__gte=start,
        ts__lte=end,
    )
    agg = qs.aggregate(min_ts=Min("ts"), max_ts=Max("ts"), n=Count("id"))
    min_ts = agg["min_ts"]
    max_ts = agg["max_ts"]
    n = int(agg["n"] or 0)
    expected = 0
    coverage_ratio = None
    if min_ts and max_ts:
        expected = int(((max_ts - min_ts).total_seconds() / 300)) + 1
        coverage_ratio = round(n / expected, 4) if expected > 0 else None
    return {
        "min_ts": min_ts,
        "max_ts": max_ts,
        "count": n,
        "expected": expected,
        "coverage_ratio": coverage_ratio,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--symbols", default="", help="Comma-separated symbols")
    p.add_argument("--equity", type=float, default=1000.0)
    p.add_argument("--ltf", default="5m")
    p.add_argument("--htf", default="4h")
    p.add_argument("--out", default="")
    args = p.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if start >= end:
        raise SystemExit("start must be before end")

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        instruments = list(Instrument.objects.filter(symbol__in=symbols))
    else:
        instruments = list(Instrument.objects.filter(enabled=True))
        symbols = [i.symbol for i in instruments]

    inst_map = {i.symbol: i for i in instruments}
    missing = [s for s in symbols if s not in inst_map]
    if missing:
        raise SystemExit(f"Missing instruments: {missing}")

    report = {
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "configs": CONFIGS,
        "symbols": [],
    }

    for symbol in symbols:
        inst = inst_map[symbol]
        cov = _coverage(symbol, start, end)
        actual_start = max(start, cov["min_ts"]) if cov["min_ts"] else None
        actual_end = min(end, cov["max_ts"]) if cov["max_ts"] else None
        days = (
            round((actual_end - actual_start).total_seconds() / 86400, 2)
            if actual_start and actual_end
            else 0.0
        )

        row = {
            "symbol": symbol,
            "coverage": {
                "min_ts": cov["min_ts"].isoformat() if cov["min_ts"] else None,
                "max_ts": cov["max_ts"].isoformat() if cov["max_ts"] else None,
                "count": cov["count"],
                "expected": cov["expected"],
                "coverage_ratio": cov["coverage_ratio"],
                "actual_start": actual_start.isoformat() if actual_start else None,
                "actual_end": actual_end.isoformat() if actual_end else None,
                "days": days,
            },
            "results": [],
        }

        if not actual_start or not actual_end or actual_start >= actual_end:
            row["status"] = "no_data"
            report["symbols"].append(row)
            continue

        for config in CONFIGS:
            with _temporary_settings(**config["overrides"]):
                _, metrics = run_backtest(
                    instruments=[inst],
                    start=actual_start,
                    end=actual_end,
                    initial_equity=float(args.equity),
                    ltf=args.ltf,
                    htf=args.htf,
                    trailing_stop=True,
                    verbose=False,
                )
            row["results"].append(
                {
                    "name": config["name"],
                    "metrics": metrics,
                }
            )

        row["status"] = "ok"
        report["symbols"].append(row)

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
