from __future__ import annotations

import json
from typing import Any

import redis
from django.conf import settings

from .models import StrategyConfig


RUNTIME_OVERRIDES_VERSION = "runtime_cfg_v1"
RUNTIME_OVERRIDES_CACHE_KEY = f"signals:{RUNTIME_OVERRIDES_VERSION}"
RUNTIME_OVERRIDES_CACHE_TTL_SECONDS = 30


def _redis_client():
    try:
        return redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    except Exception:
        return None


def _normalize_payload_value(row: StrategyConfig) -> Any:
    payload = row.params_json if isinstance(row.params_json, dict) else {}
    if "value" in payload:
        return payload.get("value")
    return row.enabled


def _load_rows_from_db() -> dict[str, Any]:
    rows = StrategyConfig.objects.filter(
        version=RUNTIME_OVERRIDES_VERSION,
        enabled=True,
    ).only("name", "enabled", "params_json")
    out: dict[str, Any] = {}
    for row in rows:
        key = str(row.name or "").strip()
        if not key:
            continue
        out[key] = _normalize_payload_value(row)
    return out


def get_runtime_overrides() -> dict[str, Any]:
    client = _redis_client()
    if client is not None:
        try:
            cached = client.get(RUNTIME_OVERRIDES_CACHE_KEY)
            if cached:
                payload = json.loads(cached)
                if isinstance(payload, dict):
                    return payload
        except Exception:
            pass

    try:
        data = _load_rows_from_db()
    except Exception:
        return {}

    if client is not None:
        try:
            client.setex(
                RUNTIME_OVERRIDES_CACHE_KEY,
                RUNTIME_OVERRIDES_CACHE_TTL_SECONDS,
                json.dumps(data, ensure_ascii=True, separators=(",", ":")),
            )
        except Exception:
            pass
    return data


def invalidate_runtime_overrides_cache() -> None:
    client = _redis_client()
    if client is None:
        return
    try:
        client.delete(RUNTIME_OVERRIDES_CACHE_KEY)
    except Exception:
        pass


def get_runtime_override(name: str, default: Any = None) -> Any:
    key = str(name or "").strip()
    if not key:
        return default
    overrides = get_runtime_overrides()
    if key not in overrides:
        return default
    return overrides.get(key, default)


def get_runtime_dict(name: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    val = get_runtime_override(name, default if default is not None else {})
    if isinstance(val, dict):
        return dict(val)
    if isinstance(val, str):
        try:
            payload = json.loads(val)
        except Exception:
            return dict(default or {})
        if isinstance(payload, dict):
            return dict(payload)
    return dict(default or {})


def get_runtime_bool(name: str, default: bool) -> bool:
    val = get_runtime_override(name, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    txt = str(val or "").strip().lower()
    if txt in {"1", "true", "yes", "on"}:
        return True
    if txt in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def get_runtime_int(name: str, default: int, *, minimum: int | None = None) -> int:
    try:
        val = int(get_runtime_override(name, default))
    except Exception:
        val = int(default)
    if minimum is not None:
        val = max(int(minimum), val)
    return val


def get_runtime_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        val = float(get_runtime_override(name, default))
    except Exception:
        val = float(default)
    if minimum is not None:
        val = max(float(minimum), val)
    if maximum is not None:
        val = min(float(maximum), val)
    return val


def get_runtime_str_list(name: str, default: set[str] | list[str] | tuple[str, ...]) -> set[str]:
    val = get_runtime_override(name, default)
    if isinstance(val, str):
        items = [part.strip() for part in val.split(",")]
    elif isinstance(val, (list, tuple, set)):
        items = [str(part).strip() for part in val]
    else:
        items = [str(part).strip() for part in default]
    return {item.lower() for item in items if item}
