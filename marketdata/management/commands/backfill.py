from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from adapters import get_default_adapter
from core.models import Instrument
from marketdata.models import Candle


TF_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def _parse_utc_datetime(raw: str) -> datetime:
    val = (raw or "").strip()
    if not val:
        raise CommandError("Empty datetime value.")
    # Try YYYY-MM-DD first for convenience
    if len(val) == 10:
        try:
            return datetime.strptime(val, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError as exc:  # pragma: no cover - input validation
            raise CommandError(f"Invalid date '{raw}'. Use YYYY-MM-DD or ISO format.") from exc
    # Fallback ISO parsing
    try:
        dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CommandError(f"Invalid datetime '{raw}'. Use YYYY-MM-DD or ISO format.") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_timeframes(raw: str) -> list[str]:
    vals = [x.strip() for x in (raw or "").split(",") if x.strip()]
    if not vals:
        raise CommandError("No timeframes provided.")
    invalid = [tf for tf in vals if tf not in TF_SECONDS]
    if invalid:
        raise CommandError(f"Unsupported timeframe(s): {invalid}. Allowed: {sorted(TF_SECONDS.keys())}")
    return vals


def _parse_symbols(raw: str) -> list[str]:
    return [x.strip().upper() for x in (raw or "").split(",") if x.strip()]


class Command(BaseCommand):
    help = "Backfill OHLCV history with pagination (supports long ranges like 4 years)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--start",
            type=str,
            required=True,
            help="UTC start date/time (YYYY-MM-DD or ISO). Example: 2022-01-01",
        )
        parser.add_argument(
            "--end",
            type=str,
            default="",
            help="UTC end date/time (YYYY-MM-DD or ISO). Default: now",
        )
        parser.add_argument(
            "--timeframes",
            type=str,
            default="1h,4h",
            help="Comma-separated timeframes. Default: 1h,4h",
        )
        parser.add_argument(
            "--symbols",
            type=str,
            default="",
            help="Comma-separated symbols. Default: all enabled instruments",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=500,
            help="Candles per API page (default: 500)",
        )
        parser.add_argument(
            "--pause-ms",
            type=int,
            default=150,
            help="Sleep between API pages in milliseconds (default: 150)",
        )
        parser.add_argument(
            "--max-pages",
            type=int,
            default=0,
            help="Safety cap of pages per symbol/timeframe (0 = no cap)",
        )

    def handle(self, *args, **options):
        adapter = get_default_adapter()

        start_dt = _parse_utc_datetime(options["start"])
        end_dt = _parse_utc_datetime(options["end"]) if options["end"] else datetime.now(timezone.utc)
        if start_dt >= end_dt:
            raise CommandError("--start must be earlier than --end.")

        tf_list = _parse_timeframes(options["timeframes"])
        req_symbols = _parse_symbols(options["symbols"])
        limit = int(options["limit"])
        if limit <= 0:
            raise CommandError("--limit must be > 0.")
        pause_seconds = max(0.0, int(options["pause_ms"])) / 1000.0
        max_pages = max(0, int(options["max_pages"]))

        instruments_qs = Instrument.objects.filter(enabled=True)
        if req_symbols:
            instruments_qs = instruments_qs.filter(symbol__in=req_symbols)
        instruments = list(instruments_qs.order_by("symbol"))
        if not instruments:
            raise CommandError("No instruments found for backfill.")

        self.stdout.write(self.style.HTTP_INFO("=" * 72))
        self.stdout.write(self.style.HTTP_INFO("Backfill OHLCV (paginated)"))
        self.stdout.write(self.style.HTTP_INFO("=" * 72))
        self.stdout.write(f"Range      : {start_dt.isoformat()} -> {end_dt.isoformat()}")
        self.stdout.write(f"Timeframes : {', '.join(tf_list)}")
        self.stdout.write(f"Instruments: {', '.join(i.symbol for i in instruments)}")
        self.stdout.write(f"Page limit : {limit}")
        self.stdout.write(f"Pause      : {pause_seconds:.3f}s")
        self.stdout.write(f"Max pages  : {max_pages if max_pages else 'unlimited'}")
        self.stdout.write(self.style.HTTP_INFO("-" * 72))

        total_rows = 0
        total_pages = 0

        for instrument in instruments:
            self.stdout.write(f"\n{instrument.symbol}:")
            for tf in tf_list:
                tf_sec = TF_SECONDS[tf]
                cursor_ms = int(start_dt.timestamp() * 1000)
                end_ms = int(end_dt.timestamp() * 1000)
                pages = 0
                rows_written = 0
                last_seen_ts: int | None = None

                while cursor_ms <= end_ms:
                    if max_pages and pages >= max_pages:
                        self.stdout.write(
                            self.style.WARNING(
                                f"  {tf}: reached --max-pages={max_pages}, stopping early."
                            )
                        )
                        break

                    try:
                        ohlcvs = adapter.fetch_ohlcv(
                            instrument.symbol,
                            timeframe=tf,
                            since=cursor_ms,
                            limit=limit,
                        )
                    except Exception as exc:
                        self.stdout.write(self.style.WARNING(f"  {tf}: fetch failed: {exc}"))
                        break

                    pages += 1
                    total_pages += 1

                    if not ohlcvs:
                        # If start is older than listing date, some exchanges return empty
                        # pages until cursor reaches available history. Move forward and retry.
                        next_cursor = cursor_ms + (tf_sec * 1000 * limit)
                        if next_cursor <= cursor_ms or next_cursor > end_ms:
                            break
                        cursor_ms = next_cursor
                        if pause_seconds > 0:
                            time.sleep(pause_seconds)
                        continue

                    candles: list[Candle] = []
                    max_ts_in_page: int | None = None
                    for ts, o, h, l, c, v in ohlcvs:
                        ts_i = int(ts)
                        if ts_i > end_ms:
                            continue
                        if max_ts_in_page is None or ts_i > max_ts_in_page:
                            max_ts_in_page = ts_i
                        candles.append(
                            Candle(
                                instrument=instrument,
                                timeframe=tf,
                                ts=datetime.fromtimestamp(ts_i / 1000, tz=timezone.utc),
                                open=o,
                                high=h,
                                low=l,
                                close=c,
                                volume=v,
                            )
                        )

                    if candles:
                        Candle.objects.bulk_create(
                            candles,
                            update_conflicts=True,
                            unique_fields=["instrument", "timeframe", "ts"],
                            update_fields=["open", "high", "low", "close", "volume"],
                            batch_size=1000,
                        )
                        rows_written += len(candles)
                        total_rows += len(candles)

                    # Stop if page has no usable data in requested range.
                    if max_ts_in_page is None:
                        break

                    # Prevent infinite loops on repeated pages from exchange.
                    if last_seen_ts is not None and max_ts_in_page <= last_seen_ts:
                        self.stdout.write(
                            self.style.WARNING(
                                f"  {tf}: repeated/old page detected at ts={max_ts_in_page}, stopping."
                            )
                        )
                        break
                    last_seen_ts = max_ts_in_page

                    # Advance cursor by exactly one candle to avoid duplicates/loops.
                    cursor_ms = max_ts_in_page + (tf_sec * 1000)
                    if cursor_ms > end_ms:
                        break

                    if pause_seconds > 0:
                        time.sleep(pause_seconds)

                self.stdout.write(f"  {tf}: pages={pages} rows_upserted={rows_written}")

        self.stdout.write(self.style.HTTP_INFO("-" * 72))
        self.stdout.write(self.style.SUCCESS(f"Done. Total pages={total_pages} candles_upserted={total_rows}"))
