from __future__ import annotations

from typing import Any, Sequence

from django.conf import settings

from adapters.credentials import get_active_service, get_exchange_credentials


def _to_float(val: Any) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def get_runtime_exchange_context(default_env: str | None = None) -> dict[str, Any]:
    service = get_active_service(default_env=default_env)
    cfg = get_exchange_credentials(service)
    sandbox = bool(cfg.get("sandbox"))
    account_alias = str(cfg.get("name_alias", "") or "")
    owner_username = str(cfg.get("owner_username", "") or "")
    mode = str(getattr(settings, "MODE", "live") or "live")

    if service == "bingx" and sandbox:
        assets = ["VST", "USDT", "USDC"]
    elif service == "bingx":
        assets = ["USDT", "USDC", "VST"]
    elif service == "binance":
        assets = ["USDT", "BUSD", "USDC"]
    else:
        assets = ["USDT", "USDC", "BUSD"]

    env = "demo" if sandbox else "live"
    account_ns = account_alias or "default"
    label_parts: list[str] = []
    if account_alias:
        label_parts.append(account_alias)
    if owner_username:
        label_parts.append(owner_username)
    label_suffix = f" [{' / '.join(label_parts)}]" if label_parts else ""
    return {
        "service": service,
        "account_alias": account_alias,
        "owner_username": owner_username,
        "sandbox": sandbox,
        "env": env,
        "mode": mode,
        "balance_assets": assets,
        "primary_asset": assets[0],
        "risk_namespace": f"{service}:{env}:{mode}:{account_ns}",
        "label": f"{service.upper()} {'DEMO' if sandbox else 'LIVE'}{label_suffix}",
    }


def _has_asset(balance: dict[str, Any], asset: str) -> bool:
    if asset in balance:
        return True
    free = balance.get("free")
    if isinstance(free, dict) and asset in free:
        return True
    total = balance.get("total")
    if isinstance(total, dict) and asset in total:
        return True
    return False


def _bucket_amount(balance: dict[str, Any], bucket: str, asset: str) -> float:
    bucket_data = balance.get(bucket)
    if isinstance(bucket_data, dict):
        val = bucket_data.get(asset)
        if val is not None:
            return _to_float(val)
    asset_data = balance.get(asset)
    if isinstance(asset_data, dict):
        val = asset_data.get(bucket)
        if val is not None:
            return _to_float(val)
    return 0.0


def extract_balance_values(
    balance: dict[str, Any] | None,
    candidate_assets: Sequence[str],
) -> tuple[float, float, str]:
    if not isinstance(balance, dict):
        asset = candidate_assets[0] if candidate_assets else "USDT"
        return 0.0, 0.0, asset

    for asset in candidate_assets:
        free = _bucket_amount(balance, "free", asset)
        equity = _bucket_amount(balance, "total", asset)
        if equity <= 0 and free > 0:
            equity = free
        if free > 0 or equity > 0:
            return free, equity, asset

    for asset in candidate_assets:
        if _has_asset(balance, asset):
            free = _bucket_amount(balance, "free", asset)
            equity = _bucket_amount(balance, "total", asset)
            if equity <= 0 and free > 0:
                equity = free
            return free, equity, asset

    info = balance.get("info")
    if isinstance(info, dict):
        account_equity = _to_float(info.get("accountEquity"))
        if account_equity > 0:
            asset = candidate_assets[0] if candidate_assets else "USDT"
            return account_equity, account_equity, asset

    asset = candidate_assets[0] if candidate_assets else "USDT"
    return 0.0, 0.0, asset
