"""
Startup health check for MasterTrading services.
Run after docker compose up:
docker compose exec web python scripts/startup_check.py
"""

import os
import sys

import django


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.conf import settings
from django.utils import timezone as dj_tz

from core.exchange_runtime import extract_balance_values, get_runtime_exchange_context
from core.models import Instrument
from marketdata.models import Candle
from risk.notifications import send_telegram


def check_instruments():
    instruments = list(Instrument.objects.filter(enabled=True).values_list("symbol", flat=True))
    assert len(instruments) >= 1, "No enabled instruments found!"
    return instruments


def check_market_data(instruments):
    stale = []
    for symbol in instruments:
        latest = (
            Candle.objects.filter(instrument__symbol=symbol, timeframe="1m")
            .order_by("-ts")
            .values_list("ts", flat=True)
            .first()
        )
        if latest is None:
            stale.append(f"{symbol}: NO DATA")
            continue
        age_minutes = (dj_tz.now() - latest).total_seconds() / 60
        if age_minutes > 10:
            stale.append(f"{symbol}: {age_minutes:.0f}m old")
    return stale


def check_exchange():
    from adapters import get_default_adapter

    adapter = get_default_adapter()
    balance = adapter.fetch_balance()
    runtime_ctx = get_runtime_exchange_context()
    balance_assets = list(runtime_ctx.get("balance_assets") or ["USDT"])
    free_balance, equity_balance, balance_asset = extract_balance_values(balance, balance_assets)
    return free_balance, equity_balance, balance_asset, runtime_ctx


def check_redis():
    import redis

    client = redis.from_url(settings.CELERY_BROKER_URL)
    return client.ping()


def check_db():
    from django.db import connection

    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
    return True


def _build_startup_message(
    status: str,
    instruments: list[str],
    stale: list[str],
    errors: list[str],
    free_balance: float,
    equity_balance: float,
    balance_asset: str,
    runtime_ctx: dict,
) -> str:
    lines = [
        "MasterTrading startup check",
        f"Status: {status}",
        f"Equity: {equity_balance:.2f} {balance_asset}",
        f"Free: {free_balance:.2f} {balance_asset}",
        f"Instruments: {', '.join(instruments)}",
        f"Mode: {settings.MODE}",
        f"Env: {runtime_ctx.get('label', 'N/A')}",
    ]
    if stale:
        lines.append(f"Stale data: {', '.join(stale)}")
    if errors:
        lines.append("Errors:")
        lines.extend(f"- {err}" for err in errors)
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("MASTERTRADING STARTUP CHECK")
    print("=" * 60)
    errors = []
    runtime_ctx = get_runtime_exchange_context()
    balance_asset = str(runtime_ctx.get("primary_asset") or "USDT")
    free_balance = 0.0
    equity_balance = 0.0

    try:
        check_db()
        print("[OK] PostgreSQL connected")
    except Exception as exc:
        errors.append(f"DB: {exc}")
        print(f"[FAIL] PostgreSQL: {exc}")

    try:
        check_redis()
        print("[OK] Redis connected")
    except Exception as exc:
        errors.append(f"Redis: {exc}")
        print(f"[FAIL] Redis: {exc}")

    try:
        instruments = check_instruments()
        print(f"[OK] Instruments: {', '.join(instruments)}")
    except Exception as exc:
        errors.append(f"Instruments: {exc}")
        print(f"[FAIL] Instruments: {exc}")
        instruments = []

    stale = check_market_data(instruments)
    if stale:
        for item in stale:
            print(f"[WARN] Market data: {item}")
    else:
        print("[OK] Market data is fresh")

    try:
        free_balance, equity_balance, balance_asset, runtime_ctx = check_exchange()
        print(
            f"[OK] Exchange: equity={equity_balance:.2f} {balance_asset}, "
            f"free={free_balance:.2f} {balance_asset}"
        )
        if free_balance < settings.MIN_EQUITY_USDT:
            errors.append(
                f"Low free balance: {free_balance:.2f} {balance_asset} < MIN {settings.MIN_EQUITY_USDT:.2f}"
            )
    except Exception as exc:
        errors.append(f"Exchange: {exc}")
        print(f"[FAIL] Exchange: {exc}")

    print("\n--- Config ---")
    print(f"MODE={settings.MODE}, TRADING_ENABLED={settings.TRADING_ENABLED}")
    print(
        f"EXCHANGE={runtime_ctx.get('service', settings.EXCHANGE)}, "
        f"SANDBOX={runtime_ctx.get('sandbox', False)}"
    )
    print(f"RISK_PER_TRADE={settings.RISK_PER_TRADE_PCT * 100:.1f}%")
    print(
        f"DAILY_DD_LIMIT={settings.DAILY_DD_LIMIT * 100:.0f}%, "
        f"WEEKLY_DD_LIMIT={settings.WEEKLY_DD_LIMIT * 100:.0f}%"
    )
    print(f"MAX_EFF_LEVERAGE={settings.MAX_EFF_LEVERAGE}")
    print(
        f"TRAILING_STOP={settings.TRAILING_STOP_ENABLED}, "
        f"ACTIVATION_R={settings.TRAILING_STOP_ACTIVATION_R}"
    )
    print(
        f"COOLDOWN={settings.SIGNAL_COOLDOWN_MINUTES}min (BTC/SOL), "
        f"{settings.PER_INSTRUMENT_COOLDOWN}"
    )
    print(f"TELEGRAM_ENABLED={settings.TELEGRAM_ENABLED}")
    print(f"DEBUG={settings.DEBUG}")

    status = "OK" if not errors else f"ERRORS: {len(errors)}"
    message = _build_startup_message(
        status=status,
        instruments=instruments,
        stale=stale,
        errors=errors,
        free_balance=free_balance,
        equity_balance=equity_balance,
        balance_asset=balance_asset,
        runtime_ctx=runtime_ctx,
    )
    sent = send_telegram(message, parse_mode=None)
    if sent:
        print("\n[OK] Telegram notification sent")
    else:
        print("\n[WARN] Telegram notification failed (token/chatid configured?)")

    print("\n" + "=" * 60)
    if errors:
        print(f"RESULT: {len(errors)} ERROR(S) - FIX BEFORE TRADING")
        sys.exit(1)
    print("RESULT: ALL SYSTEMS GO")
    sys.exit(0)


if __name__ == "__main__":
    main()
