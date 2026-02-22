from __future__ import annotations

"""
Analyze whether losing trades were ever in profit (max favorable excursion).

This uses Candle high/low between trade open and close to estimate:
- MFE (max favorable excursion): best unrealized profit during the trade
- MAE (max adverse excursion): worst unrealized drawdown during the trade

Usage (inside docker container recommended):
  python scripts/analyze_losers_mfe.py --days 30 --timeframe 1m
"""

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import django

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.db.models import Count, Max, Min  # noqa: E402
from django.utils import timezone as dj_tz  # noqa: E402

from execution.models import OperationReport, Order  # noqa: E402
from marketdata.models import Candle  # noqa: E402


@dataclass(frozen=True)
class LossMFE:
    report_id: int
    symbol: str
    side: str
    reason: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    open_ts: datetime
    close_ts: datetime
    open_src: str
    duration_min: float
    candle_tf: str
    candle_cnt: int
    max_fav_pct: float
    max_fav_usdt: float
    max_adv_pct: float
    max_r_at_sl: Optional[float]


def _infer_opened_at(rep: OperationReport, max_open_gap: timedelta) -> tuple[Optional[datetime], str]:
    if rep.opened_at:
        return rep.opened_at, "op_report.opened_at"

    if rep.correlation_id:
        o = (
            Order.objects.filter(correlation_id=rep.correlation_id)
            .order_by("-opened_at")
            .first()
        )
        if o and o.opened_at and (rep.closed_at - o.opened_at) <= max_open_gap:
            return o.opened_at, "order.by_correlation_id"

    # Fallback: last entry order for that instrument/side before the close.
    o = (
        Order.objects.filter(instrument=rep.instrument, side=rep.side, opened_at__lte=rep.closed_at)
        .order_by("-opened_at")
        .first()
    )
    if o and o.opened_at and (rep.closed_at - o.opened_at) <= max_open_gap:
        return o.opened_at, "order.latest_before_close"

    return None, "missing"


def _mfe_mae_from_candles(
    rep: OperationReport,
    open_ts: datetime,
    timeframe: str,
    fallback_timeframe: str | None,
) -> tuple[str, int, Optional[float], Optional[float]]:
    """
    Returns (tf_used, candle_cnt, max_high, min_low).
    """
    def _agg(tf: str):
        qs = Candle.objects.filter(
            instrument=rep.instrument,
            timeframe=tf,
            ts__gte=open_ts,
            ts__lte=rep.closed_at,
        )
        return qs.aggregate(cnt=Count("id"), max_high=Max("high"), min_low=Min("low"))

    tf_used = timeframe
    a = _agg(timeframe)
    cnt = int(a["cnt"] or 0)
    if cnt == 0 and fallback_timeframe and fallback_timeframe != timeframe:
        tf_used = fallback_timeframe
        a = _agg(fallback_timeframe)
        cnt = int(a["cnt"] or 0)

    max_high = float(a["max_high"]) if a["max_high"] is not None else None
    min_low = float(a["min_low"]) if a["min_low"] is not None else None
    return tf_used, cnt, max_high, min_low


def _pct_str(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.2f}%"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30, help="Lookback window in days (default: 30).")
    p.add_argument("--timeframe", type=str, default="1m", help="Candle timeframe (default: 1m).")
    p.add_argument("--fallback-timeframe", type=str, default="5m", help="Fallback timeframe if missing (default: 5m).")
    p.add_argument(
        "--mode",
        type=str,
        default="live",
        help="Filter by OperationReport.mode (default: live). Use 'any' for no filter.",
    )
    p.add_argument("--top", type=int, default=10, help="Show top N losses by MFE (default: 10).")
    p.add_argument(
        "--max-open-gap-hours",
        type=float,
        default=48.0,
        help="Max hours between inferred open and close (default: 48).",
    )
    args = p.parse_args()

    now = dj_tz.now()
    start = now - timedelta(days=max(1, int(args.days)))
    mode = (args.mode or "").strip().lower()
    max_open_gap = timedelta(hours=float(args.max_open_gap_hours))
    fallback_tf = (args.fallback_timeframe or "").strip() or None

    losses = (
        OperationReport.objects.select_related("instrument")
        .filter(outcome=OperationReport.Outcome.LOSS, closed_at__gte=start, closed_at__lte=now)
        .order_by("-closed_at")
    )
    if mode and mode != "any":
        losses = losses.filter(mode=mode)

    total = losses.count()
    if total == 0:
        print(f"No losing trades found in the last {args.days} day(s).")
        return 0

    rows: list[LossMFE] = []
    missing_open = 0
    missing_candles = 0

    for rep in losses:
        open_ts, open_src = _infer_opened_at(rep, max_open_gap=max_open_gap)
        if not open_ts:
            missing_open += 1
            continue

        tf_used, candle_cnt, max_high, min_low = _mfe_mae_from_candles(
            rep,
            open_ts=open_ts,
            timeframe=args.timeframe,
            fallback_timeframe=fallback_tf,
        )
        if candle_cnt <= 0 or max_high is None or min_low is None:
            missing_candles += 1
            continue

        entry = float(rep.entry_price or 0.0)
        if entry <= 0:
            continue

        is_long = rep.side == "buy"
        if is_long:
            max_fav_pct = (max_high - entry) / entry
            max_adv_pct = (entry - min_low) / entry
        else:
            max_fav_pct = (entry - min_low) / entry
            max_adv_pct = (max_high - entry) / entry

        notional = float(rep.notional_usdt or 0.0)
        max_fav_usdt = max_fav_pct * notional if notional > 0 else 0.0

        # If it closed by SL, approximate "R" using the realized loss pct.
        max_r_at_sl = None
        pnl_pct = float(rep.pnl_pct or 0.0)
        if rep.reason == "sl" and pnl_pct < 0:
            sl_pct = abs(pnl_pct)
            if sl_pct > 0:
                max_r_at_sl = max_fav_pct / sl_pct

        dur_min = (rep.closed_at - open_ts).total_seconds() / 60.0
        rows.append(
            LossMFE(
                report_id=int(rep.id),
                symbol=rep.instrument.symbol,
                side=rep.side,
                reason=rep.reason,
                entry_price=float(rep.entry_price),
                exit_price=float(rep.exit_price),
                pnl_pct=pnl_pct,
                open_ts=open_ts,
                close_ts=rep.closed_at,
                open_src=open_src,
                duration_min=dur_min,
                candle_tf=tf_used,
                candle_cnt=int(candle_cnt),
                max_fav_pct=max_fav_pct,
                max_fav_usdt=max_fav_usdt,
                max_adv_pct=max_adv_pct,
                max_r_at_sl=max_r_at_sl,
            )
        )

    print("")
    print("Losing trades: unrealized profit check (MFE)")
    print(f"Window: {start:%Y-%m-%d} -> {now:%Y-%m-%d %H:%M} (server time)")
    if mode and mode != "any":
        print(f"Mode: {mode}")
    print(f"Losses found: {total}")
    print(f"Usable (open_ts + candles): {len(rows)}")
    if missing_open:
        print(f"Skipped (missing open time): {missing_open}")
    if missing_candles:
        print(f"Skipped (missing candle data): {missing_candles}")

    if not rows:
        return 0

    ever_green = sum(1 for r in rows if r.max_fav_pct > 0)
    print(f"Ever in profit (MFE>0): {ever_green}/{len(rows)} ({ever_green / len(rows) * 100:.1f}%)")

    thresholds = [0.001, 0.002, 0.005, 0.01]  # 0.1%, 0.2%, 0.5%, 1.0%
    for t in thresholds:
        c = sum(1 for r in rows if r.max_fav_pct >= t)
        print(f"MFE >= {t*100:.1f}%: {c}/{len(rows)} ({c / len(rows) * 100:.1f}%)")

    # R thresholds (only where SL is inferable)
    r_rows = [r for r in rows if r.max_r_at_sl is not None]
    if r_rows:
        for r_thr in [0.5, 1.0]:
            c = sum(1 for r in r_rows if (r.max_r_at_sl or 0.0) >= r_thr)
            print(f"MFE >= {r_thr:.1f}R (SL-inferred only): {c}/{len(r_rows)} ({c / len(r_rows) * 100:.1f}%)")

    print("")
    print(f"Top {min(args.top, len(rows))} losses by MFE (best unrealized profit before turning red):")
    top = sorted(rows, key=lambda r: r.max_fav_pct, reverse=True)[: max(1, int(args.top))]
    for r in top:
        r_mult = f"{r.max_r_at_sl:.2f}R" if r.max_r_at_sl is not None else "n/a"
        print(
            f"- #{r.report_id} {r.symbol} {r.side} reason={r.reason} "
            f"pnl={r.pnl_pct*100:.2f}% dur={r.duration_min:.1f}m "
            f"MFE={_pct_str(r.max_fav_pct)} (~${r.max_fav_usdt:.2f}) "
            f"MAE={_pct_str(r.max_adv_pct)} maxR={r_mult} "
            f"open_src={r.open_src} candles={r.candle_cnt}@{r.candle_tf}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
