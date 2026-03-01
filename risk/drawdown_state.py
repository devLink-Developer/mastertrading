from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any

import redis
from django.conf import settings
from django.db import transaction
from django.utils import timezone as dj_tz

from risk.models import DrawdownBaseline


def _redis_client():
    try:
        return redis.from_url(settings.CELERY_BROKER_URL)
    except Exception:
        return None


def _to_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return default


def _is_valid_equity(value: Decimal) -> bool:
    return bool(value.is_finite() and value > 0)


def _cache_key(risk_ns: str, period_type: str, period_key: str) -> str:
    return f"risk:dd_baseline:{risk_ns}:{period_type}:{period_key}"


def _write_cache(row: DrawdownBaseline) -> None:
    client = _redis_client()
    if client is None:
        return
    payload = {
        "id": row.id,
        "risk_namespace": row.risk_namespace,
        "period_type": row.period_type,
        "period_key": row.period_key,
        "start_equity": str(row.start_equity),
        "last_equity": str(row.last_equity),
        "last_dd": str(row.last_dd),
        "last_emitted_dd": str(row.last_emitted_dd) if row.last_emitted_dd is not None else "",
    }
    ttl = max(300, int(getattr(settings, "DAILY_TRADE_COUNT_TTL_SECONDS", 90000)))
    try:
        client.setex(
            _cache_key(row.risk_namespace, row.period_type, row.period_key),
            ttl,
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        )
    except Exception:
        return


def get_or_init_baseline(
    risk_ns: str,
    period_type: str,
    period_key: str,
    equity: float,
) -> DrawdownBaseline | None:
    eq = _to_decimal(equity)
    if not _is_valid_equity(eq):
        return None
    ns = str(risk_ns or "global").strip() or "global"
    p_type = str(period_type or DrawdownBaseline.PeriodType.DAILY).strip().lower()
    if p_type not in {DrawdownBaseline.PeriodType.DAILY, DrawdownBaseline.PeriodType.WEEKLY}:
        p_type = DrawdownBaseline.PeriodType.DAILY
    p_key = str(period_key or "").strip()
    if not p_key:
        p_key = dj_tz.now().date().isoformat() if p_type == DrawdownBaseline.PeriodType.DAILY else "unknown"

    with transaction.atomic():
        row, created = DrawdownBaseline.objects.select_for_update().get_or_create(
            risk_namespace=ns,
            period_type=p_type,
            period_key=p_key,
            defaults={
                "start_equity": eq,
                "last_equity": eq,
                "last_dd": Decimal("0"),
            },
        )
        if created:
            _write_cache(row)
            return row
        changed = False
        if not _is_valid_equity(_to_decimal(row.start_equity)):
            row.start_equity = eq
            changed = True
        if not _is_valid_equity(_to_decimal(row.last_equity)):
            row.last_equity = eq
            changed = True
        if changed:
            row.save(update_fields=["start_equity", "last_equity", "updated_at"])
        _write_cache(row)
        return row


def update_baseline(
    baseline: DrawdownBaseline,
    *,
    equity: float,
    dd: float,
    mark_emitted: bool = False,
) -> DrawdownBaseline:
    eq = _to_decimal(equity)
    dd_dec = _to_decimal(dd)
    fields = ["last_equity", "last_dd", "updated_at"]
    baseline.last_equity = eq
    baseline.last_dd = dd_dec
    if mark_emitted:
        baseline.last_emitted_dd = dd_dec
        fields.append("last_emitted_dd")
    baseline.save(update_fields=fields)
    _write_cache(baseline)
    return baseline


def compute_drawdown(
    risk_ns: str,
    period_type: str,
    period_key: str,
    equity: float,
) -> tuple[DrawdownBaseline | None, float]:
    baseline = get_or_init_baseline(risk_ns, period_type, period_key, equity)
    if baseline is None:
        return None, 0.0
    start = _to_decimal(baseline.start_equity)
    eq = _to_decimal(equity)
    if not _is_valid_equity(start) or not _is_valid_equity(eq):
        return baseline, 0.0
    dd = float((eq - start) / start)
    if not (dd == dd and abs(dd) != float("inf")):
        dd = 0.0
    update_baseline(baseline, equity=float(eq), dd=dd, mark_emitted=False)
    return baseline, dd


def should_emit_drawdown_event(
    baseline: DrawdownBaseline | None,
    dd: float,
    *,
    min_delta: float = 0.01,
) -> bool:
    if baseline is None:
        return True
    last = baseline.last_emitted_dd
    if last is None:
        return True
    try:
        return abs(float(dd) - float(last)) >= max(0.0, float(min_delta))
    except Exception:
        return True


def mark_drawdown_event_emitted(
    baseline: DrawdownBaseline | None,
    dd: float,
) -> None:
    if baseline is None:
        return
    update_baseline(baseline, equity=float(_to_decimal(baseline.last_equity)), dd=dd, mark_emitted=True)
