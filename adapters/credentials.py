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


def _env_optional_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    val = raw.strip().lower()
    if val == "":
        return None
    return val in {"1", "true", "yes", "on"}


def _env_account_alias() -> str:
    return (os.getenv("EXCHANGE_ACCOUNT_ALIAS") or "").strip().lower()


def _env_account_owner() -> str:
    return (os.getenv("EXCHANGE_ACCOUNT_OWNER") or "").strip()


def _serialize_db_row(row: Any) -> dict[str, Any]:
    owner_username = ""
    try:
        owner_obj = getattr(row, "owner", None)
        owner_username = str(getattr(owner_obj, "username", "") or "")
    except Exception:
        owner_username = ""
    return {
        "credential_id": int(row.id),
        "service": row.service,
        "name_alias": str(row.name_alias or ""),
        "owner_username": owner_username,
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


def _select_db_credential_row(service: str | None, allow_cross_service: bool) -> Any | None:
    """
    Select a DB credential row honoring optional account selectors:
    - EXCHANGE_ACCOUNT_ALIAS
    - EXCHANGE_ACCOUNT_OWNER
    - EXCHANGE_ACCOUNT_SANDBOX
    """
    if not apps.ready:
        return None
    try:
        from core.models import ExchangeCredential

        qs = ExchangeCredential.objects.all()
        alias = _env_account_alias()
        owner_username = _env_account_owner()
        sandbox_hint = _env_optional_bool("EXCHANGE_ACCOUNT_SANDBOX")

        if alias:
            qs = qs.filter(name_alias__iexact=alias)
        if owner_username:
            qs = qs.filter(owner__username=owner_username)
        if sandbox_hint is not None:
            qs = qs.filter(sandbox=sandbox_hint)

        if service:
            svc_qs = qs.filter(service=service)
            row = svc_qs.filter(active=True).order_by("-updated_at").first()
            if row:
                return row
            row = svc_qs.order_by("-updated_at").first()
            if row:
                return row
            if not allow_cross_service:
                return None

        row = qs.filter(active=True).order_by("-updated_at").first()
        if row:
            return row
        return qs.order_by("-updated_at").first()
    except (OperationalError, ProgrammingError):
        return None
    except Exception:
        return None


def _db_credentials_for(service: str) -> dict[str, Any] | None:
    if not apps.ready:
        return None
    try:
        row = _select_db_credential_row(service=service, allow_cross_service=False)
        if not row:
            return None
        return _serialize_db_row(row)
    except (OperationalError, ProgrammingError):
        return None
    except Exception:
        return None


def _env_credentials_for(service: str) -> dict[str, Any]:
    if service == "binance":
        return {
            "credential_id": "",
            "service": "binance",
            "name_alias": "",
            "owner_username": "",
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
            "credential_id": "",
            "service": "bingx",
            "name_alias": "",
            "owner_username": "",
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
        "credential_id": "",
        "service": "kucoin",
        "name_alias": "",
        "owner_username": "",
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
        row = _select_db_credential_row(service=fallback, allow_cross_service=True)
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
    alias = str(cfg.get("name_alias", ""))
    owner = str(cfg.get("owner_username", ""))
    credential_id = str(cfg.get("credential_id", ""))
    return f"{service}|{source}|{updated_at}|{sandbox}|{margin_mode}|{leverage}|{alias}|{owner}|{credential_id}"
