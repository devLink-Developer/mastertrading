from __future__ import annotations

import logging
import math
import time
import uuid
from datetime import datetime, timezone, timedelta

import redis
from celery import shared_task
from django.conf import settings
from django.db.models import Max

from adapters import get_default_adapter, get_default_adapter_signature
from core.models import Instrument
from .models import Candle, FundingRate

logger = logging.getLogger(__name__)


_ADAPTER = None
_ADAPTER_SIG = None


def _adapter():
    global _ADAPTER, _ADAPTER_SIG
    signature = get_default_adapter_signature()
    if _ADAPTER is None or _ADAPTER_SIG != signature:
        _ADAPTER = get_default_adapter()
        _ADAPTER_SIG = signature
        logger.info("marketdata adapter reloaded (%s)", signature.split("|")[0])
    return _ADAPTER


def _redis_client():
    try:
        return redis.from_url(settings.CELERY_BROKER_URL)
    except Exception:
        return None


def _marketdata_dispatch_lock_key() -> str:
    return "lock:marketdata:dispatcher"


def _marketdata_instrument_lock_key(instrument_id: int) -> str:
    return f"lock:marketdata:instrument:{int(instrument_id)}"


def _acquire_marketdata_lock(lock_key: str, ttl_seconds: int) -> tuple[redis.Redis | None, str]:
    client = _redis_client()
    if client is None:
        return None, ""
    token = uuid.uuid4().hex
    try:
        acquired = bool(client.set(lock_key, token, nx=True, ex=max(1, int(ttl_seconds or 1))))
        return client, (token if acquired else "")
    except Exception:
        return None, ""


def _release_marketdata_lock(client: redis.Redis | None, lock_key: str, token: str) -> None:
    if client is None or not token:
        return
    try:
        current = client.get(lock_key)
        if isinstance(current, bytes):
            current = current.decode("utf-8", errors="ignore")
        if current and current == token:
            client.delete(lock_key)
    except Exception:
        return


def _has_active_marketdata_fetches(client: redis.Redis | None, instrument_ids: list[int]) -> bool:
    if client is None or not instrument_ids:
        return False
    try:
        keys = [_marketdata_instrument_lock_key(inst_id) for inst_id in instrument_ids]
        return any(val is not None for val in client.mget(keys))
    except Exception:
        return False


def _tf_to_seconds(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400
    return 60


def _ohlcv_fetch_params(
    latest_ts: datetime | None,
    timeframe: str,
    now_utc: datetime,
    *,
    incremental_limit: int = 20,
    full_limit: int = 200,
) -> tuple[int | None, int, bool]:
    """
    Choose incremental vs catch-up fetch params.

    If a symbol is many candles behind, asking for `since=old_ts` with a tiny limit can
    trap the worker fetching the same stale chunk forever. In that case switch to a
    tail fetch (`since=None`) so the DB snaps back to the latest market window.
    """
    if latest_ts is None:
        return None, full_limit, False

    tf_seconds = _tf_to_seconds(timeframe)
    lag_seconds = max(0.0, (now_utc - latest_ts).total_seconds())
    lag_candles = int(math.ceil(lag_seconds / max(tf_seconds, 1)))
    if lag_candles > max(3, incremental_limit - 4):
        catchup_limit = min(full_limit, max(incremental_limit, lag_candles + 5))
        return None, catchup_limit, True

    since_dt = latest_ts - timedelta(seconds=tf_seconds * 3)
    return int(since_dt.timestamp() * 1000), incremental_limit, False


# ---------------------------------------------------------------------------
# Dispatcher: fan-out one sub-task per enabled instrument
# ---------------------------------------------------------------------------
@shared_task(expires=55)
def fetch_ohlcv_and_funding():
    """Fan-out: dispatches one fetch_instrument_data task per enabled instrument.
    Uses Redis locks so beat does not pile duplicate instrument fetches during slow exchange windows."""
    client = _redis_client()
    lock_key = _marketdata_dispatch_lock_key()
    ids = list(Instrument.objects.filter(enabled=True).values_list("id", flat=True))
    if client:
        if _has_active_marketdata_fetches(client, ids):
            logger.info("fetch_ohlcv_and_funding skipped - instrument fetch still active")
            return
        if not client.set(
            lock_key,
            "1",
            nx=True,
            ex=max(5, int(getattr(settings, "MARKETDATA_DISPATCH_LOCK_SECONDS", 50) or 50)),
        ):
            logger.info("fetch_ohlcv_and_funding skipped - previous dispatch still active")
            return
    logger.info("fetch_ohlcv_and_funding dispatching %d instrument tasks", len(ids))
    for inst_id in ids:
        fetch_instrument_data.delay(inst_id)


# ---------------------------------------------------------------------------
# Per-instrument worker: OHLCV (bulk), mark price, funding
# ---------------------------------------------------------------------------
@shared_task(autoretry_for=(Exception,), retry_backoff=True, max_retries=2, expires=55)
def fetch_instrument_data(instrument_id: int):
    """Fetch OHLCV + mark price + funding for a single instrument."""
    t0 = time.monotonic()
    lock_key = _marketdata_instrument_lock_key(instrument_id)
    lock_client, lock_token = _acquire_marketdata_lock(
        lock_key,
        int(getattr(settings, "MARKETDATA_INSTRUMENT_LOCK_SECONDS", 330) or 330),
    )
    if lock_client is not None and not lock_token:
        logger.info("fetch_instrument_data skipped - instrument_id=%s lock active", instrument_id)
        return

    try:
        try:
            instrument = Instrument.objects.get(id=instrument_id)
        except Instrument.DoesNotExist:
            logger.warning("Instrument id=%s not found, skipping", instrument_id)
            return

        adapter = _adapter()
        tf_list = getattr(settings, "DEFAULT_TIMEFRAMES", ["1m", "5m", "15m", "1h", "4h", "1d"])
        symbol = instrument.symbol
        now_utc = datetime.now(tz=timezone.utc)

        latest_by_tf = dict(
            Candle.objects.filter(instrument=instrument, timeframe__in=tf_list)
            .values("timeframe")
            .annotate(max_ts=Max("ts"))
            .values_list("timeframe", "max_ts")
        )

        # ---- OHLCV (bulk upsert) ----
        for tf in tf_list:
            try:
                latest_ts = latest_by_tf.get(tf)
                since_ms, limit, catchup_mode = _ohlcv_fetch_params(
                    latest_ts,
                    tf,
                    now_utc,
                )
                if catchup_mode:
                    lag_seconds = max(0.0, (now_utc - latest_ts).total_seconds()) if latest_ts else 0.0
                    logger.info(
                        "fetch_ohlcv catch-up mode %s %s lag=%.0fs limit=%d",
                        symbol,
                        tf,
                        lag_seconds,
                        limit,
                    )
                ohlcvs = adapter.fetch_ohlcv(symbol, timeframe=tf, since=since_ms, limit=limit)
                candles = [
                    Candle(
                        instrument=instrument,
                        timeframe=tf,
                        ts=datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                        open=o, high=h, low=l, close=c, volume=v,
                    )
                    for ts, o, h, l, c, v in ohlcvs
                ]
                if candles:
                    Candle.objects.bulk_create(
                        candles,
                        update_conflicts=True,
                        unique_fields=["instrument", "timeframe", "ts"],
                        update_fields=["open", "high", "low", "close", "volume"],
                    )
            except Exception as exc:
                logger.warning("Failed fetch_ohlcv %s %s: %s", symbol, tf, exc)

        # ---- Mark price (persist on latest 1m candle) ----
        try:
            ticker = adapter.fetch_ticker(symbol)
            mark = ticker.get("info", {}).get("markPrice") or ticker.get("mark")
            if mark:
                latest = Candle.objects.filter(
                    instrument=instrument, timeframe="1m"
                ).order_by("-ts").first()
                if latest and latest.mark_price is None:
                    latest.mark_price = mark
                    latest.save(update_fields=["mark_price"])
        except Exception as exc:
            logger.debug("mark_price fetch failed %s: %s", symbol, exc)

        # ---- Funding Rate ----
        try:
            min_interval = int(getattr(settings, "FUNDING_FETCH_MIN_INTERVAL_SECONDS", 300))
            latest_funding_ts = (
                FundingRate.objects.filter(instrument=instrument)
                .order_by("-ts")
                .values_list("ts", flat=True)
                .first()
            )
            should_fetch_funding = (
                latest_funding_ts is None
                or (now_utc - latest_funding_ts).total_seconds() >= min_interval
            )
            if should_fetch_funding:
                fr = adapter.fetch_funding_rate_current(symbol)
                if fr and fr.get("fundingRate") is not None:
                    ts_val = fr.get("timestamp") or fr.get("datetime")
                    if isinstance(ts_val, str):
                        from django.utils.dateparse import parse_datetime
                        ts_dt = parse_datetime(ts_val) or now_utc
                    elif isinstance(ts_val, (int, float)):
                        ts_dt = datetime.fromtimestamp(ts_val / 1000, tz=timezone.utc)
                    else:
                        ts_dt = now_utc
                    FundingRate.objects.update_or_create(
                        instrument=instrument,
                        ts=ts_dt,
                        defaults={"rate": fr["fundingRate"]},
                    )
        except Exception as exc:
            logger.warning("Failed fetch_funding %s: %s", symbol, exc)

        elapsed = time.monotonic() - t0
        logger.info("fetch_instrument_data %s completed in %.1fs", symbol, elapsed)
    finally:
        _release_marketdata_lock(lock_client, lock_key, lock_token)



