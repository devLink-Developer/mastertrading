from __future__ import annotations

import os
from typing import Any

from django.apps import apps
from django.db import OperationalError, ProgrammingError


SUPPORTED_SERVICES = {"kucoin", "binance", "bingx"}


def _normalize_service(raw: str | None) -> str:
    svc = (raw or "").strip().lower()
    return svc if svc in SUPPORTED_SERVICES else "kucoin"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


def _db_credentials_for(service: str) -> dict[str, Any] | None:
    if not apps.ready:
        return None
    try:
        from core.models import ExchangeCredential

        row = ExchangeCredential.objects.filter(service=service).first()
        if not row:
            return None
        return {
            "service": row.service,
            "api_key": row.api_key or "",
            "api_secret": row.api_secret or "",
            "api_passphrase": row.api_passphrase or "",
            "sandbox": bool(row.sandbox),
            "margin_mode": row.margin_mode or "cross",
            "leverage": int(row.leverage or 3),
            "active": bool(row.active),
            "updated_at": row.updated_at.isoformat(),
            "source": "db",
        }
    except (OperationalError, ProgrammingError):
        return None
    except Exception:
        return None


def _env_credentials_for(service: str) -> dict[str, Any]:
    if service == "binance":
        return {
            "service": "binance",
            "api_key": os.getenv("BINANCE_API_KEY", ""),
            "api_secret": os.getenv("BINANCE_API_SECRET", ""),
            "api_passphrase": "",
            "sandbox": _env_bool("BINANCE_TESTNET", True),
            "margin_mode": os.getenv("BINANCE_MARGIN_MODE", "cross"),
            "leverage": int(os.getenv("BINANCE_LEVERAGE", "3")),
            "active": True,
            "updated_at": "env",
            "source": "env",
        }
    if service == "bingx":
        return {
            "service": "bingx",
            "api_key": os.getenv("BINGX_API_KEY", ""),
            "api_secret": os.getenv("BINGX_API_SECRET", ""),
            "api_passphrase": os.getenv("BINGX_API_PASSPHRASE", ""),
            "sandbox": _env_bool("BINGX_SANDBOX", False),
            "margin_mode": os.getenv("BINGX_MARGIN_MODE", "cross"),
            "leverage": int(os.getenv("BINGX_LEVERAGE", "3")),
            "active": True,
            "updated_at": "env",
            "source": "env",
        }
    return {
        "service": "kucoin",
        "api_key": os.getenv("KUCOIN_API_KEY", ""),
        "api_secret": os.getenv("KUCOIN_API_SECRET", ""),
        "api_passphrase": os.getenv("KUCOIN_API_PASSPHRASE", ""),
        "sandbox": _env_bool("KUCOIN_SANDBOX", True),
        "margin_mode": os.getenv("KUCOIN_MARGIN_MODE", "cross"),
        "leverage": int(os.getenv("KUCOIN_LEVERAGE", "3")),
        "active": True,
        "updated_at": "env",
        "source": "env",
    }


def get_active_service(default_env: str | None = None) -> str:
    fallback = _normalize_service(default_env or os.getenv("EXCHANGE", "kucoin"))
    if not apps.ready:
        return fallback
    try:
        from core.models import ExchangeCredential

        row = (
            ExchangeCredential.objects.filter(active=True)
            .order_by("-updated_at")
            .first()
        )
        if row and row.service:
            return _normalize_service(row.service)
    except (OperationalError, ProgrammingError):
        return fallback
    except Exception:
        return fallback
    return fallback


def get_exchange_credentials(service: str) -> dict[str, Any]:
    normalized = _normalize_service(service)
    db_data = _db_credentials_for(normalized)
    if db_data is not None:
        return db_data
    return _env_credentials_for(normalized)


def get_default_adapter_signature(default_env: str | None = None) -> str:
    service = get_active_service(default_env=default_env)
    cfg = get_exchange_credentials(service)
    source = str(cfg.get("source", "unknown"))
    updated_at = str(cfg.get("updated_at", "na"))
    sandbox = "1" if bool(cfg.get("sandbox")) else "0"
    margin_mode = str(cfg.get("margin_mode", "cross"))
    leverage = str(cfg.get("leverage", "3"))
    return f"{service}|{source}|{updated_at}|{sandbox}|{margin_mode}|{leverage}"
