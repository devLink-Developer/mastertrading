"""
Management command: python manage.py backfill_candles

Downloads historical candle data from the exchange for backtesting.
Uses CCXT to fetch OHLCV data in bulk.
"""
from __future__ import annotations

import logging
import math
import re
import time
from datetime import datetime, timezone, timedelta

from django.core.management.base import BaseCommand, CommandError

from adapters import get_default_adapter
from core.models import Instrument
from marketdata.models import Candle

logger = logging.getLogger(__name__)

# CCXT timeframe → max candles per request (conservative)
TF_LIMIT = {
    # BingX rejects limits > 1440 on minute-level candle endpoints.
    "1m": 1440,
    "5m": 1440,
    "15m": 1000,
    "1h": 500,
    "4h": 500,
    "1d": 365,
}

# Timeframe → timedelta per bar
TF_DELTA = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}

_BINGX_RETRY_AT_RE = re.compile(r"retry after time:\s*(\d+)", re.IGNORECASE)


def _parse_retry_after_seconds(exc: Exception, default_sleep: float) -> float:
    msg = str(exc or "")
    m = _BINGX_RETRY_AT_RE.search(msg)
    if not m:
        return max(float(default_sleep), 1.0)
    try:
        raw = int(m.group(1))
    except Exception:
        return max(float(default_sleep), 1.0)
    retry_at_ms = raw if raw > 10_000_000_000 else raw * 1000
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    wait_s = math.ceil(max(0, retry_at_ms - now_ms) / 1000.0) + 1.0
    return max(wait_s, float(default_sleep), 1.0)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return (
        "109429" in msg
        or "rate limit" in msg
        or "too many requests" in msg
        or "retry after time" in msg
    )


def _resolve_ccxt_symbol(adapter, inst: Instrument) -> str | None:
    candidates = [
        inst.symbol,
        f"{inst.base}/{inst.quote}:USDT",
        f"{inst.base}/USDT:USDT",
    ]
    client = getattr(adapter, "client", None)
    markets = getattr(client, "markets", None) or {}
    for sym_candidate in candidates:
        mapped = (
            adapter._map_symbol(sym_candidate)
            if hasattr(adapter, "_map_symbol")
            else sym_candidate
        )
        if mapped in markets:
            return mapped
        if sym_candidate in markets:
            return sym_candidate
    if hasattr(adapter, "_map_symbol"):
        return adapter._map_symbol(inst.symbol)
    return candidates[0]


class Command(BaseCommand):
    help = "Backfill historical candle data from the exchange for backtesting."

    def add_arguments(self, parser):
        parser.add_argument(
            "--start",
            type=str,
            required=True,
            help="Start date YYYY-MM-DD",
        )
        parser.add_argument(
            "--end",
            type=str,
            default=None,
            help="End date YYYY-MM-DD (default: now)",
        )
        parser.add_argument(
            "--symbols",
            type=str,
            default="",
            help="Comma-separated instrument symbols (default: all enabled)",
        )
        parser.add_argument(
            "--timeframes",
            type=str,
            default="5m,4h",
            help="Comma-separated timeframes to fetch (default: 5m,4h)",
        )
        parser.add_argument(
            "--sleep",
            type=float,
            default=1.0,
            help="Sleep between API calls in seconds (default: 1.0)",
        )
        parser.add_argument(
            "--max-retries",
            type=int,
            default=8,
            help="Max retries for rate-limited requests before skipping the chunk (default: 8)",
        )

    def handle(self, *args, **options):
        try:
            start = datetime.strptime(options["start"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            raise CommandError(f"Invalid start date: {options['start']}")

        if options["end"]:
            try:
                end = datetime.strptime(options["end"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                raise CommandError(f"Invalid end date: {options['end']}")
        else:
            end = datetime.now(timezone.utc)

        if options["symbols"]:
            symbols = [s.strip().upper() for s in options["symbols"].split(",")]
            instruments = list(Instrument.objects.filter(symbol__in=symbols))
        else:
            instruments = list(Instrument.objects.filter(enabled=True))

        if not instruments:
            raise CommandError("No instruments found.")

        timeframes = [tf.strip() for tf in options["timeframes"].split(",")]
        sleep_sec = options["sleep"]
        max_retries = max(1, int(options["max_retries"]))

        adapter = get_default_adapter()

        self.stdout.write(self.style.HTTP_INFO(f"Backfilling candles: {start:%Y-%m-%d} → {end:%Y-%m-%d}"))
        self.stdout.write(f"  Instruments: {', '.join(i.symbol for i in instruments)}")
        self.stdout.write(f"  Timeframes : {', '.join(timeframes)}")

        total_saved = 0

        for inst in instruments:
            ccxt_symbol = _resolve_ccxt_symbol(adapter, inst)
            if not ccxt_symbol:
                self.stdout.write(self.style.WARNING(f"  Could not resolve symbol for {inst.symbol}, skipping"))
                continue

            for tf in timeframes:
                limit = TF_LIMIT.get(tf, 500)
                delta = TF_DELTA.get(tf, timedelta(hours=1))
                cursor = start
                saved_tf = 0

                self.stdout.write(f"\n  {inst.symbol} / {tf}:")

                while cursor < end:
                    since_ms = int(cursor.timestamp() * 1000)
                    ohlcv = None
                    for attempt in range(1, max_retries + 1):
                        try:
                            ohlcv = adapter.fetch_ohlcv(ccxt_symbol, tf, since=since_ms, limit=limit)
                            break
                        except Exception as exc:
                            if _is_rate_limit_error(exc) and attempt < max_retries:
                                wait_s = _parse_retry_after_seconds(exc, sleep_sec)
                                self.stdout.write(
                                    self.style.WARNING(
                                        f"    Rate limit at {cursor} (attempt {attempt}/{max_retries}); "
                                        f"sleeping {wait_s:.1f}s and retrying same chunk"
                                    )
                                )
                                time.sleep(wait_s)
                                continue
                            self.stdout.write(self.style.WARNING(f"    API error at {cursor}: {exc}"))
                            ohlcv = None
                            break

                    if ohlcv is None:
                        cursor += delta * limit
                        time.sleep(sleep_sec)
                        continue

                    if not ohlcv:
                        break

                    batch = []
                    for row in ohlcv:
                        ts = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc)
                        if ts > end:
                            break
                        batch.append(Candle(
                            instrument=inst,
                            timeframe=tf,
                            ts=ts,
                            open=row[1],
                            high=row[2],
                            low=row[3],
                            close=row[4],
                            volume=row[5] or 0,
                        ))

                    if batch:
                        # Use update_or_create equivalent via bulk
                        Candle.objects.bulk_create(
                            batch,
                            update_conflicts=True,
                            unique_fields=["instrument", "timeframe", "ts"],
                            update_fields=["open", "high", "low", "close", "volume"],
                        )
                        saved_tf += len(batch)

                    # Move cursor past the last candle we received
                    last_ts = datetime.fromtimestamp(ohlcv[-1][0] / 1000, tz=timezone.utc)
                    cursor = last_ts + delta

                    self.stdout.write(f"    {last_ts:%Y-%m-%d %H:%M} — {len(batch)} candles", ending="\r")
                    self.stdout.flush()

                    time.sleep(sleep_sec)

                self.stdout.write(f"    → {saved_tf:,} candles saved for {inst.symbol}/{tf}         ")
                total_saved += saved_tf

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(f"  ✓ Total: {total_saved:,} candles saved"))
