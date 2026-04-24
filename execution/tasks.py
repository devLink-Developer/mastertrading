import logging
import math
import re
import uuid
import hashlib
import json
from pathlib import Path
from io import StringIO
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from datetime import datetime, timezone, timedelta
from statistics import median
from typing import Any

from celery import shared_task
from django.db import transaction
from django.db.models import OuterRef, Subquery, Q
from django.conf import settings
from django.core.management import call_command
from django.utils import timezone as dj_tz

import redis

from adapters import get_default_adapter, get_default_adapter_signature
from core.ai_feedback import record_ai_feedback_event
from core.models import Instrument, AiFeedbackEvent
from core.exchange_runtime import (
    extract_balance_values,
    get_runtime_exchange_context,
)
from signals.sessions import (
    get_current_session,
    get_session_risk_mult,
    get_session_score_min,
    get_weekday_name,
    get_weekday_risk_mult,
    get_weekday_score_offset,
    is_dead_session,
)
from signals.direction_policy import (
    get_direction_mode,
    is_direction_allowed,
)
from signals.feature_flags import FEATURE_KEYS, resolve_runtime_flags
from signals.regime_mtf import (
    build_symbol_regime_snapshot,
    consolidate_lead_state,
    recommended_bias as _mtf_recommended_bias,
)
from signals.models import Signal
from signals.runtime_overrides import (
    get_runtime_bool,
    get_runtime_float,
    get_runtime_int,
    get_runtime_override,
    get_runtime_str_list,
)
from marketdata.models import Candle
from risk.models import RiskEvent
from risk.drawdown_state import (
    compute_drawdown as _compute_drawdown_state,
    mark_drawdown_event_emitted as _mark_drawdown_event_emitted,
    should_emit_drawdown_event as _should_emit_drawdown_event,
)
from risk.notifications import (
    notify_kill_switch, notify_trade_opened, notify_trade_closed,
    notify_risk_event, notify_error,
)
from execution.ml_entry_filter import load_model, predict_entry_success_probability
from execution.ai_entry_gate import evaluate_ai_entry_gate
from execution.ai_exit_gate import evaluate_ai_exit_gate
from execution.risk_policy import (
    max_daily_trades_for_adx as _shared_max_daily_trades_for_adx,
    volatility_adjusted_risk as _shared_volatility_adjusted_risk,
)
from .models import Order, OperationReport, BalanceSnapshot, Position

logger = logging.getLogger(__name__)
_EXEC_ADAPTER = None
_EXEC_ADAPTER_SIG = None


def _adapter():
    global _EXEC_ADAPTER, _EXEC_ADAPTER_SIG
    signature = get_default_adapter_signature()
    if _EXEC_ADAPTER is None or _EXEC_ADAPTER_SIG != signature:
        _EXEC_ADAPTER = get_default_adapter()
        _EXEC_ADAPTER_SIG = signature
        logger.info("execution adapter reloaded (%s)", signature.split("|")[0])
    return _EXEC_ADAPTER


def _to_float(val) -> float:
    """
    Defensive conversion to float to avoid mixing Decimal/float in math operations.
    Falls back to 0.0 when value is missing or invalid.
    """
    try:
        # Decimal -> float, str/None handled by except
        return float(val)
    except (TypeError, ValueError, InvalidOperation):
        return 0.0


def _is_valid_equity_value(equity: float) -> bool:
    return math.isfinite(equity) and equity > 0


def _extract_fee_usdt(order_resp: dict[str, Any] | None) -> float:
    """
    Best-effort fee extraction from CCXT order payloads.
    Returns a non-negative fee in quote currency (USDT/VST for our futures flow).
    """
    if not isinstance(order_resp, dict):
        return 0.0

    fees_list = order_resp.get("fees")
    if isinstance(fees_list, list):
        total = 0.0
        for item in fees_list:
            if isinstance(item, dict):
                total += _to_float(item.get("cost") or item.get("fee"))
            else:
                total += _to_float(item)
        if total > 0:
            return total

    fee_obj = order_resp.get("fee")
    if isinstance(fee_obj, dict):
        cost = _to_float(fee_obj.get("cost") or fee_obj.get("fee"))
        if cost > 0:
            return cost
    elif fee_obj is not None:
        cost = _to_float(fee_obj)
        if cost > 0:
            return cost

    info = order_resp.get("info")
    if isinstance(info, dict):
        for key in ("totalFee", "cumFee", "commission", "execFee", "fee"):
            cost = _to_float(info.get(key))
            if cost > 0:
                return cost

    return 0.0


def _trade_notional_usdt(qty: float, price: float, contract_size: float = 1.0) -> float:
    cs = _to_float(contract_size)
    if cs <= 0:
        cs = 1.0
    return abs(_to_float(qty) * _to_float(price) * cs)


def _estimate_order_fee_usdt(notional_usdt: float) -> float:
    if not bool(getattr(settings, "ORDER_FEE_FALLBACK_ENABLED", True)):
        return 0.0
    fee_pct = max(0.0, _to_float(getattr(settings, "ORDER_FEE_FALLBACK_TAKER_PCT", 0.0005)))
    if fee_pct <= 0:
        return 0.0
    return max(0.0, _to_float(notional_usdt) * fee_pct)


def _resolve_order_fee_usdt(order_resp: dict[str, Any] | None, notional_usdt: float) -> float:
    fee = _extract_fee_usdt(order_resp)
    if fee > 0:
        return fee
    return _estimate_order_fee_usdt(notional_usdt)


def _lookup_entry_order_fee_usdt(
    inst: Instrument,
    side: str,
    correlation_id: str,
    opened_at: datetime | None,
) -> float:
    corr = str(correlation_id or "").strip()
    if not corr:
        return 0.0
    try:
        qs = Order.objects.filter(
            instrument=inst,
            side=side,
            correlation_id=corr,
            reduce_only=False,
        ).only("fee_usdt", "opened_at")
        if opened_at is not None:
            qs = qs.filter(opened_at=opened_at)
        order = qs.order_by("-opened_at", "-id").first()
        if order is None:
            return 0.0
        return max(0.0, _to_float(order.fee_usdt))
    except Exception:
        return 0.0


def _report_pnl_for_notification(
    report: OperationReport | None,
    pnl_pct_fallback: float,
    pnl_abs_fallback: float,
) -> tuple[float, float]:
    if report is None:
        return pnl_pct_fallback, pnl_abs_fallback
    try:
        return _to_float(report.pnl_pct), _to_float(report.pnl_abs)
    except Exception:
        return pnl_pct_fallback, pnl_abs_fallback


def _is_no_position_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    patterns = (
        "no position to close",
        "position does not exist",
        "position not exist",
        "position not found",
        "insufficient position",
        "position size is 0",
        "position size is zero",
        "reduceonly order can not be filled",
        "reduceonly order cannot be filled",
    )
    return any(p in msg for p in patterns)


def _is_insufficient_margin_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    patterns = (
        "insufficient margin",
        "margin insufficient",
        "not enough margin",
        "insufficient funds",
        "insufficient balance",
        "available balance is not enough",
        "risk rate",
    )
    return any(p in msg for p in patterns)


def _norm_symbol(sym: str) -> str:
    """
    Normaliza sÃƒÆ’Ã‚Â­mbolos provenientes de diferentes exchanges/ccxt
    para que BTCUSDT, BTC/USDT, BTC/USDT:USDT o XBTUSDTM coincidan.
    """
    if not sym:
        return ""
    s = sym.replace(" ", "")
    s = s.replace("XBT", "BTC")  # KuCoin usa XBT
    if s.endswith("USDTM"):
        base = s[:-5]
        return f"{base}USDT"
    if s.endswith(":USDT"):
        base = s.split("/")[0]
        return f"{base}USDT"
    if s.endswith("/USDT"):
        base = s.split("/")[0]
        return f"{base}USDT"
    return s


def _safe_correlation_id(raw: str, max_len: int = 64) -> str:
    txt = str(raw or "").strip()
    if not txt:
        return ""
    return txt[:max_len]


def _ai_entry_reject_cooldown_enabled() -> bool:
    return bool(getattr(settings, "AI_ENTRY_GATE_REJECT_COOLDOWN_ENABLED", True))


def _ai_entry_reject_cooldown_seconds() -> int:
    return max(
        30,
        int(getattr(settings, "AI_ENTRY_GATE_REJECT_COOLDOWN_SECONDS", 900) or 900),
    )


def _ai_entry_cache_token(raw: str, max_len: int = 64) -> str:
    txt = str(raw or "").strip().lower()
    txt = re.sub(r"[^a-z0-9_.:-]+", "_", txt)
    txt = txt[:max_len].strip("_")
    return txt or "na"


def _ai_entry_reject_cache_key(
    *,
    account_alias: str,
    account_service: str,
    symbol: str,
    strategy_name: str,
    signal_direction: str,
) -> str:
    return (
        "ai:entry:reject:"
        f"{_ai_entry_cache_token(account_alias)}:"
        f"{_ai_entry_cache_token(account_service)}:"
        f"{_ai_entry_cache_token(symbol.upper())}:"
        f"{_ai_entry_cache_token(strategy_name)}:"
        f"{_ai_entry_cache_token(signal_direction)}"
    )


def _round_or_none(value: Any, ndigits: int) -> float | None:
    if value is None:
        return None
    val = _to_float(value)
    if not math.isfinite(val):
        return None
    return round(val, ndigits)


def _bucket_or_none(value: Any, step: float, ndigits: int) -> float | None:
    if value is None:
        return None
    val = _to_float(value)
    if not math.isfinite(val):
        return None
    bucket_step = _to_float(step)
    if not math.isfinite(bucket_step) or bucket_step <= 0:
        return round(val, ndigits)
    bucketed = round(val / bucket_step) * bucket_step
    return round(bucketed, ndigits)


def _ai_entry_market_fingerprint(
    *,
    symbol: str,
    strategy_name: str,
    signal_direction: str,
    session_name: str,
    sig_score: float,
    atr: float | None,
    spread_bps: float | None,
    sl_pct: float,
    sig_payload: dict[str, Any] | None,
    coarse: bool = False,
) -> str:
    payload = sig_payload if isinstance(sig_payload, dict) else {}
    reasons = payload.get("reasons") if isinstance(payload.get("reasons"), dict) else {}
    module_rows_in = reasons.get("module_rows") if isinstance(reasons, dict) else []
    module_rows_out: list[list[Any]] = []
    if isinstance(module_rows_in, list):
        for row in module_rows_in[:8]:
            if not isinstance(row, dict):
                continue
            module_rows_out.append(
                [
                    str(row.get("module") or "").strip().lower(),
                    str(row.get("direction") or "").strip().lower()[:1],
                    (
                        _bucket_or_none(row.get("confidence"), 0.05, 2)
                        if coarse
                        else _round_or_none(row.get("confidence"), 3)
                    ),
                    (
                        _bucket_or_none(row.get("raw_score"), 0.05, 2)
                        if coarse
                        else _round_or_none(row.get("raw_score"), 3)
                    ),
                ]
            )

    compact = {
        "sym": str(symbol or "").strip().upper(),
        "st": str(strategy_name or "").strip().lower(),
        "dir": str(signal_direction or "").strip().lower(),
        "ses": str(session_name or "").strip().lower(),
        "sc": (
            _bucket_or_none(sig_score, 0.02, 2)
            if coarse
            else _round_or_none(sig_score, 4)
        ),
        "atr": (
            _bucket_or_none(atr, 0.0005, 4)
            if coarse
            else _round_or_none(atr, 5)
        ),
        "spr": (
            _bucket_or_none(spread_bps, 0.5, 1)
            if coarse
            else _round_or_none(spread_bps, 2)
        ),
        "sl": (
            _bucket_or_none(sl_pct, 0.0005, 4)
            if coarse
            else _round_or_none(sl_pct, 5)
        ),
        "ns": (
            _bucket_or_none(reasons.get("net_score"), 0.02, 2)
            if (coarse and isinstance(reasons, dict))
            else (_round_or_none(reasons.get("net_score"), 4) if isinstance(reasons, dict) else None)
        ),
        "rb": (
            _bucket_or_none(payload.get("risk_budget_pct"), 0.0005, 4)
            if coarse
            else _round_or_none(payload.get("risk_budget_pct"), 6)
        ),
        "er": str(payload.get("entry_reason") or "").strip().lower()[:64] if payload else "",
        "mr": module_rows_out,
    }
    raw = json.dumps(compact, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def _ai_entry_should_suppress_retry(
    *,
    account_alias: str,
    account_service: str,
    symbol: str,
    strategy_name: str,
    signal_direction: str,
    market_fingerprint: str,
    market_fingerprint_coarse: str = "",
) -> tuple[bool, str]:
    if not _ai_entry_reject_cooldown_enabled():
        return False, ""
    client = _redis_client()
    if client is None:
        return False, ""
    key = _ai_entry_reject_cache_key(
        account_alias=account_alias,
        account_service=account_service,
        symbol=symbol,
        strategy_name=strategy_name,
        signal_direction=signal_direction,
    )
    try:
        raw = client.get(key)
    except Exception:
        return False, ""
    if raw is None:
        return False, ""

    cached = _decode_redis_value(raw).strip()
    if not cached:
        return False, ""
    cached_fp = cached
    cached_fp_coarse = ""
    cached_reason = ""
    if cached.startswith("{"):
        try:
            obj = json.loads(cached)
            if isinstance(obj, dict):
                cached_fp = str(obj.get("fp") or "").strip()
                cached_fp_coarse = str(obj.get("fp_coarse") or "").strip()
                cached_reason = str(obj.get("reason") or "").strip()[:160]
        except Exception:
            pass
    matches_exact = bool(cached_fp and cached_fp == market_fingerprint)
    matches_coarse = bool(
        cached_fp_coarse
        and market_fingerprint_coarse
        and cached_fp_coarse == market_fingerprint_coarse
    )
    if matches_exact or matches_coarse:
        # Keep cooldown alive while the same market pattern keeps firing,
        # preventing repeated AI calls in unchanged conditions.
        try:
            client.expire(key, _ai_entry_reject_cooldown_seconds())
        except Exception:
            pass
        return True, cached_reason or "ai_reject_cached"
    return False, ""


def _ai_entry_should_suppress_retry_from_feedback(
    *,
    account_alias: str,
    account_service: str,
    symbol: str,
    strategy_name: str,
    signal_direction: str,
    session_name: str,
    sig_score: float,
    spread_bps: float | None,
    market_fingerprint: str,
    market_fingerprint_coarse: str,
) -> tuple[bool, str]:
    """
    DB fallback dedup for AI denials.
    Useful when Redis cache is unavailable/empty after restarts and to avoid
    repeated token spend on effectively unchanged conditions.
    """
    if not _ai_entry_reject_cooldown_enabled():
        return False, ""

    cutoff = dj_tz.now() - timedelta(seconds=_ai_entry_reject_cooldown_seconds())
    alias = str(account_alias or "").strip()
    service = str(account_service or "").strip().lower()
    sym = str(symbol or "").strip().upper()
    strategy = str(strategy_name or "").strip().lower()
    direction = str(signal_direction or "").strip().lower()
    session = str(session_name or "").strip().lower()
    score_now = _to_float(sig_score)
    spread_now = _round_or_none(spread_bps, 3)

    try:
        recent = (
            AiFeedbackEvent.objects.filter(
                created_at__gte=cutoff,
                event_type="ai_gate_decision",
                allow=False,
                account_alias=alias,
                account_service=service,
                symbol=sym,
                strategy=strategy,
            )
            .order_by("-created_at")[:10]
        )
    except Exception:
        return False, ""

    for row in recent:
        payload = row.payload_json if isinstance(row.payload_json, dict) else {}
        prev_dir = str(payload.get("direction") or "").strip().lower()
        if prev_dir and direction and prev_dir != direction:
            continue

        prev_fp = str(payload.get("market_fp") or "").strip()
        prev_fp_coarse = str(payload.get("market_fp_coarse") or "").strip()
        if prev_fp and market_fingerprint and prev_fp == market_fingerprint:
            return True, str(row.reason or "").strip()[:160] or "ai_reject_cached"
        if (
            prev_fp_coarse
            and market_fingerprint_coarse
            and prev_fp_coarse == market_fingerprint_coarse
        ):
            return True, str(row.reason or "").strip()[:160] or "ai_reject_cached"

        # Legacy fallback for older rows without explicit fp payload.
        prev_session = str(payload.get("session") or "").strip().lower()
        if prev_session and session and prev_session != session:
            continue
        prev_score = _to_float(payload.get("sig_score"))
        prev_spread = _round_or_none(payload.get("spread_bps"), 3)
        if abs(prev_score - score_now) > 0.02:
            continue
        if spread_now is not None and prev_spread is not None and abs(prev_spread - spread_now) > 0.50:
            continue
        return True, str(row.reason or "").strip()[:160] or "ai_reject_cached"

    return False, ""


def _ai_entry_mark_rejected(
    *,
    account_alias: str,
    account_service: str,
    symbol: str,
    strategy_name: str,
    signal_direction: str,
    market_fingerprint: str,
    market_fingerprint_coarse: str = "",
    reason: str,
) -> None:
    if not _ai_entry_reject_cooldown_enabled():
        return
    client = _redis_client()
    if client is None:
        return
    key = _ai_entry_reject_cache_key(
        account_alias=account_alias,
        account_service=account_service,
        symbol=symbol,
        strategy_name=strategy_name,
        signal_direction=signal_direction,
    )
    value = json.dumps(
        {
            "fp": str(market_fingerprint or "").strip(),
            "fp_coarse": str(market_fingerprint_coarse or "").strip(),
            "reason": str(reason or "").strip()[:160],
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
    try:
        client.set(key, value, ex=_ai_entry_reject_cooldown_seconds())
    except Exception as exc:
        logger.warning("AI reject cache write failed key=%s err=%s", key, exc)
        return


def _ai_entry_clear_reject_cache(
    *,
    account_alias: str,
    account_service: str,
    symbol: str,
    strategy_name: str,
    signal_direction: str,
) -> None:
    if not _ai_entry_reject_cooldown_enabled():
        return
    client = _redis_client()
    if client is None:
        return
    key = _ai_entry_reject_cache_key(
        account_alias=account_alias,
        account_service=account_service,
        symbol=symbol,
        strategy_name=strategy_name,
        signal_direction=signal_direction,
    )
    try:
        client.delete(key)
    except Exception:
        return


def _strategy_module_name(strategy_name: str) -> str:
    strategy_txt = str(strategy_name or "").strip().lower()
    if strategy_txt.startswith("alloc_"):
        return "allocator"
    if strategy_txt.startswith("smc_"):
        return "smc"
    if strategy_txt.startswith("mod_"):
        parts = strategy_txt.split("_")
        if len(parts) >= 3:
            return parts[1]
    return ""


def _strategy_is_microvol(strategy_name: str) -> bool:
    return _strategy_module_name(strategy_name) == "microvol"


def _macro_high_impact_allows_entry(
    *,
    strategy_name: str,
    symbol: str,
) -> bool:
    if not _strategy_is_microvol(strategy_name):
        return False
    if not bool(getattr(settings, "MACRO_HIGH_IMPACT_ALLOW_MICROVOL", False)):
        return False
    allowed_symbols = set(
        getattr(settings, "MACRO_HIGH_IMPACT_ALLOW_MICROVOL_SYMBOLS", set()) or set()
    )
    symbol_norm = str(symbol or "").strip().upper()
    return (not allowed_symbols) or (symbol_norm in allowed_symbols)


def _microvol_exit_profile(strategy_name: str) -> dict[str, float | bool | str]:
    if not _strategy_is_microvol(strategy_name):
        return {"active": False, "profile": ""}
    return {
        "active": True,
        "profile": "microvol",
        "tp_mult": float(getattr(settings, "MODULE_MICROVOL_TP_MULT", 0.55) or 0.55),
        "partial_r_mult": float(getattr(settings, "MODULE_MICROVOL_PARTIAL_R_MULT", 0.60) or 0.60),
        "trail_r_mult": float(getattr(settings, "MODULE_MICROVOL_TRAIL_R_MULT", 0.45) or 0.45),
        "breakeven_r_mult": float(getattr(settings, "MODULE_MICROVOL_BREAKEVEN_R_MULT", 0.50) or 0.50),
        "lockin_mult": float(getattr(settings, "MODULE_MICROVOL_LOCKIN_MULT", 1.25) or 1.25),
    }


def _signal_active_modules(payload: dict[str, Any] | None, strategy_name: str = "") -> list[str]:
    """
    Extract active module ids from signal payload.
    For allocator signals this reads reasons.module_rows[].module.
    """
    modules: list[str] = []
    if isinstance(payload, dict):
        reasons = payload.get("reasons")
        if isinstance(reasons, dict):
            rows = reasons.get("module_rows")
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    module_name = str(row.get("module") or "").strip().lower()
                    if module_name:
                        modules.append(module_name)
            if not modules:
                active_raw = reasons.get("active_modules")
                if isinstance(active_raw, list):
                    for item in active_raw:
                        module_name = str(item or "").strip().lower()
                        if module_name:
                            modules.append(module_name)

    if not modules:
        inferred = _strategy_module_name(strategy_name)
        if inferred:
            modules.append(inferred)

    # Stable de-duplication.
    deduped: list[str] = []
    seen: set[str] = set()
    for module_name in modules:
        if module_name in seen:
            continue
        seen.add(module_name)
        deduped.append(module_name)
    return deduped


def _signal_entry_reason(payload: dict[str, Any] | None, strategy_name: str = "") -> str:
    """Build a concise human-readable reason for opening a position."""
    strategy_txt = str(strategy_name or "").strip().lower()
    sections: list[str] = []
    module_fragments: list[str] = []

    reasons = {}
    if isinstance(payload, dict):
        reasons = payload.get("reasons") or {}

    if isinstance(reasons, dict):
        module_rows = reasons.get("module_rows")
        if isinstance(module_rows, list):
            for row in module_rows[:4]:
                if not isinstance(row, dict):
                    continue
                module_name = str(row.get("module") or "").strip().lower()
                direction = str(row.get("direction") or "").strip().lower()
                confidence_raw = row.get("confidence")
                if confidence_raw is None:
                    confidence_raw = row.get("raw_score")
                confidence = _to_float(confidence_raw)

                fragment = module_name
                if direction in {"long", "short"}:
                    fragment += f" {direction}"
                if confidence > 0:
                    fragment += f" ({confidence:.2f})"
                if fragment:
                    module_fragments.append(fragment)

    if not module_fragments:
        fallback_modules = _signal_active_modules(payload, strategy_name)
        if fallback_modules:
            module_fragments = fallback_modules

    if module_fragments:
        sections.append(f"confluencia: {', '.join(module_fragments)}")

    risk_budget_pct = 0.0
    if isinstance(payload, dict):
        risk_budget_pct = _to_float(payload.get("risk_budget_pct", 0.0))
    if risk_budget_pct > 0:
        sections.append(f"risk_budget={risk_budget_pct:.3%}")

    net_score = 0.0
    if isinstance(reasons, dict):
        net_score = _to_float(reasons.get("net_score", 0.0))
    if net_score > 0:
        sections.append(f"net_score={net_score:.3f}")

    if strategy_txt:
        sections.insert(0, f"signal={strategy_txt}")

    return " | ".join(sections)


def _position_root_correlation(inst: Instrument, side: str) -> str:
    """
    Best-effort root correlation id for an active position side.
    For new flows we persist parent_correlation_id on every entry/add.
    """
    last_order = (
        Order.objects.filter(
            instrument=inst,
            side=side,
            status=Order.OrderStatus.FILLED,
            opened_at__isnull=False,
        )
        .order_by("-opened_at", "-id")
        .first()
    )
    if not last_order:
        return ""
    root = _safe_correlation_id(last_order.parent_correlation_id or last_order.correlation_id)
    return root


def _count_pyramid_adds(inst: Instrument, side: str, root_correlation_id: str) -> int:
    root = _safe_correlation_id(root_correlation_id)
    if not root:
        return 0
    return int(
        Order.objects.filter(
            instrument=inst,
            side=side,
            status=Order.OrderStatus.FILLED,
            parent_correlation_id=root,
            opened_at__isnull=False,
        )
        .exclude(correlation_id=root)
        .count()
    )


def _last_pyramid_add_opened_at(inst: Instrument, side: str, root_correlation_id: str):
    root = _safe_correlation_id(root_correlation_id)
    if not root:
        return None
    return (
        Order.objects.filter(
            instrument=inst,
            side=side,
            status=Order.OrderStatus.FILLED,
            parent_correlation_id=root,
            opened_at__isnull=False,
        )
        .exclude(correlation_id=root)
        .order_by("-opened_at")
        .values_list("opened_at", flat=True)
        .first()
    )


def _current_position(adapter, symbol: str, positions: list | None = None):
    """Devuelve (qty, entry_price) de la posiciÃƒÆ’Ã‚Â³n abierta para el sÃƒÆ’Ã‚Â­mbolo normalizado."""
    target = _norm_symbol(symbol)
    if positions is None:
        try:
            # KuCoin ignora a veces el filtro por sÃƒÆ’Ã‚Â­mbolo, preferimos traer todo y filtrar local.
            positions = adapter.fetch_positions()
        except Exception:
            positions = []
    for pos in positions or []:
        sym_pos = _norm_symbol(pos.get("symbol"))
        if sym_pos != target:
            continue
        qty = _to_float(pos.get("contracts") or pos.get("size") or 0)
        side = pos.get("side")  # 'long' o 'short'
        if side == "short":
            qty = -qty
        entry = _to_float(pos.get("entryPrice") or pos.get("averagePrice") or 0)
        opened_ms = pos.get("openingTimestamp")
        opened_at = datetime.fromtimestamp(opened_ms / 1000, tz=timezone.utc) if opened_ms else None
        return qty, entry, opened_at
    return 0.0, 0.0, None


def _spread_bps(ticker) -> float | None:
    bid = _to_float(ticker.get("bid"))
    ask = _to_float(ticker.get("ask"))
    if bid <= 0 or ask <= 0:
        return None
    mid = (ask + bid) / 2
    if mid <= 0:
        return None
    return ((ask - bid) / mid) * 10000  # ratio -> bps


def _max_spread_bps(symbol: str, atr_pct: float | None = None) -> float:
    base_bps = settings.PER_INSTRUMENT_MAX_SPREAD_BPS.get(symbol, settings.MAX_SPREAD_BPS)
    if not settings.SPREAD_DYNAMIC_BY_ATR_ENABLED or atr_pct is None or atr_pct <= 0:
        return float(base_bps)
    atr_bps = atr_pct * 10000  # 1% ATR == 100 bps
    dynamic_cap = base_bps + (atr_bps * settings.SPREAD_ATR_RELAX_FACTOR)
    return float(min(dynamic_cap, settings.MAX_DYNAMIC_SPREAD_BPS))


def _effective_leverage(adapter, equity: float, positions: list | None = None) -> tuple[float, float]:
    if equity <= 0:
        return 0.0, 0.0
    if positions is None:
        try:
            positions = adapter.fetch_positions()
        except Exception:
            positions = []
    notional = 0.0
    for pos in positions or []:
        size = _to_float(pos.get("contracts") or pos.get("size") or 0)
        mark = _to_float(pos.get("markPrice") or pos.get("info", {}).get("markPrice") or 0)
        contract_sz = _to_float(pos.get("contractSize") or pos.get("info", {}).get("multiplier") or 1.0)
        if size and mark:
            notional += abs(size * mark * (contract_sz if contract_sz else 1.0))
    return (notional / equity if equity else 0.0, notional)


def _redis_client():
    try:
        return redis.from_url(settings.CELERY_BROKER_URL)
    except Exception:
        return None


def _decode_redis_value(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except Exception:
            return str(raw)
    return str(raw)


def _acquire_task_lock(lock_key: str, ttl_seconds: int) -> tuple[Any, str]:
    """
    Acquire a Redis lock for long-running Celery tasks.
    Returns (client, token):
    - client is None when Redis is unavailable (fail-open).
    - token is empty when lock is already held.
    """
    client = _redis_client()
    if client is None:
        return None, ""
    token = uuid.uuid4().hex
    ttl = max(1, int(ttl_seconds or 1))
    try:
        acquired = bool(client.set(lock_key, token, nx=True, ex=ttl))
        return client, (token if acquired else "")
    except Exception:
        return None, ""


def _release_task_lock(client: Any, lock_key: str, token: str) -> None:
    """Best-effort lock release. Deletes only if token still matches."""
    if client is None or not token:
        return
    try:
        current = _decode_redis_value(client.get(lock_key))
        if current and current == token:
            client.delete(lock_key)
    except Exception:
        return


def _queue_ml_retrain_after_operation(symbol: str, mode: str, reason: str) -> None:
    """Best-effort enqueue of ML retraining after a trade close."""
    if not bool(getattr(settings, "ML_ENTRY_FILTER_RETRAIN_ON_OPERATION_ENABLED", False)):
        return
    if not bool(getattr(settings, "ML_ENTRY_FILTER_ENABLED", False)):
        return
    if not bool(getattr(settings, "ML_ENTRY_FILTER_AUTO_TRAIN_ENABLED", False)):
        return

    min_interval = max(
        0,
        int(getattr(settings, "ML_ENTRY_FILTER_RETRAIN_ON_OPERATION_MIN_INTERVAL_SECONDS", 0) or 0),
    )
    throttle_key = "ml:entry_filter:retrain:last_trigger_at"
    now_ts = int(dj_tz.now().timestamp())
    client = _redis_client()
    if client is None:
        logger.warning(
            "ML retrain-on-operation skipped: Redis client unavailable symbol=%s mode=%s reason=%s",
            symbol,
            mode,
            reason,
        )
        return

    try:
        client.ping()
    except Exception as exc:
        logger.warning(
            "ML retrain-on-operation skipped: Redis unreachable symbol=%s mode=%s reason=%s error=%s",
            symbol,
            mode,
            reason,
            exc,
        )
        return

    if min_interval > 0:
        try:
            raw = client.get(throttle_key)
            last_ts = int(float(raw)) if raw is not None else 0
            elapsed = now_ts - last_ts
            if last_ts > 0 and elapsed < min_interval:
                logger.info(
                    "ML retrain-on-operation throttled: symbol=%s mode=%s reason=%s elapsed=%ss interval=%ss",
                    symbol,
                    mode,
                    reason,
                    elapsed,
                    min_interval,
                )
                return
        except Exception:
            pass

    try:
        retrain_entry_filter_model.delay()
        try:
            client.set(throttle_key, now_ts)
        except Exception:
            pass
        logger.info(
            "ML retrain-on-operation queued: symbol=%s mode=%s reason=%s",
            symbol,
            mode,
            reason,
        )
    except Exception as exc:
        logger.warning(
            "ML retrain-on-operation enqueue failed: symbol=%s mode=%s reason=%s error=%s",
            symbol,
            mode,
            reason,
            exc,
        )


def _ml_strategy_tag(strategy_name: str) -> str:
    raw = str(strategy_name or "").strip().lower()
    cleaned = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")
    return cleaned.upper()


def _ml_entry_filter_min_prob(default: float, symbol: str = "", strategy_name: str = "") -> float:
    """Resolve ML min-probability from Redis override (symbol/strategy/global), fallback to settings."""
    fallback = max(0.0, min(float(default or 0.0), 1.0))
    client = _redis_client()
    if client is None:
        return fallback
    symbol_txt = str(symbol or "").strip().upper()
    strategy_tag = _ml_strategy_tag(strategy_name)
    keys = []
    if symbol_txt and strategy_tag:
        keys.append(f"ml:entry_filter:min_prob:{symbol_txt}:{strategy_tag}")
    if symbol_txt:
        keys.append(f"ml:entry_filter:min_prob:{symbol_txt}")
    if strategy_tag:
        keys.append(f"ml:entry_filter:min_prob:strategy:{strategy_tag}")
    keys.append("ml:entry_filter:min_prob")
    for key in keys:
        try:
            raw = client.get(key)
            if raw is None:
                continue
            val = float(raw)
            if not math.isfinite(val):
                continue
            return max(0.0, min(val, 1.0))
        except Exception:
            continue
    return fallback


def _ml_entry_filter_model_path(symbol: str, strategy_name: str = "") -> str:
    """
    Resolve model path with precedence:
    1) per-symbol+strategy
    2) per-symbol
    3) per-strategy
    4) global
    """
    global_path = str(getattr(settings, "ML_ENTRY_FILTER_MODEL_PATH", "") or "").strip()
    symbol_txt = str(symbol or "").strip().upper()
    strategy_tag = _ml_strategy_tag(strategy_name)
    per_symbol_enabled = bool(getattr(settings, "ML_ENTRY_FILTER_PER_SYMBOL_ENABLED", False))
    per_strategy_enabled = bool(getattr(settings, "ML_ENTRY_FILTER_PER_STRATEGY_ENABLED", False))
    per_symbol_fallback_global = bool(getattr(settings, "ML_ENTRY_FILTER_PER_SYMBOL_FALLBACK_GLOBAL", True))
    per_strategy_fallback_global = bool(getattr(settings, "ML_ENTRY_FILTER_PER_STRATEGY_FALLBACK_GLOBAL", True))

    model_dir = str(getattr(settings, "ML_ENTRY_FILTER_MODEL_DIR", "") or "").strip()
    if not model_dir:
        return global_path
    model_dir_path = Path(model_dir)

    per_symbol_strategy_path = (
        str(model_dir_path / f"entry_filter_model_{symbol_txt}_{strategy_tag}.json")
        if symbol_txt and strategy_tag
        else ""
    )
    per_symbol_path = str(model_dir_path / f"entry_filter_model_{symbol_txt}.json") if symbol_txt else ""
    per_strategy_path = (
        str(model_dir_path / f"entry_filter_model_strategy_{strategy_tag}.json")
        if strategy_tag
        else ""
    )

    if per_symbol_enabled and per_strategy_enabled and per_symbol_strategy_path and Path(per_symbol_strategy_path).exists():
        return per_symbol_strategy_path
    if per_symbol_enabled and per_symbol_path and Path(per_symbol_path).exists():
        return per_symbol_path
    if per_strategy_enabled and per_strategy_path and Path(per_strategy_path).exists():
        return per_strategy_path

    if per_symbol_enabled and per_symbol_path and not per_symbol_fallback_global:
        if per_strategy_enabled and per_symbol_strategy_path:
            return per_symbol_strategy_path
        return per_symbol_path
    if per_strategy_enabled and per_strategy_path and not per_strategy_fallback_global:
        return per_strategy_path
    return global_path


def _check_drawdown(equity: float, risk_ns: str = "global") -> tuple[bool, float, dict[str, Any]]:
    """
    Compute daily drawdown using DB baseline as source of truth.
    Redis is used as read-through cache by the drawdown_state service.
    Returns (allowed, dd, meta).
    """
    if not _is_valid_equity_value(equity):
        logger.warning("Skipping daily DD check: invalid equity=%.8f ns=%s", equity, risk_ns)
        return True, 0.0, {"emit_event": False}
    period_key = dj_tz.now().date().isoformat()
    baseline, dd = _compute_drawdown_state(
        risk_ns=risk_ns,
        period_type="daily",
        period_key=period_key,
        equity=equity,
    )
    if not math.isfinite(dd):
        return True, 0.0, {"emit_event": False}
    breach = dd <= -settings.DAILY_DD_LIMIT
    emit_event = breach and _should_emit_drawdown_event(baseline, dd, min_delta=0.01)
    return (
        (not breach),
        dd,
        {
            "emit_event": emit_event,
            "baseline": baseline,
            "period_type": "daily",
            "period_key": period_key,
        },
    )


def _check_weekly_drawdown(equity: float, risk_ns: str = "global") -> tuple[bool, float, dict[str, Any]]:
    """
    Compute weekly drawdown using DB baseline as source of truth.
    Returns (allowed, dd, meta).
    """
    if not _is_valid_equity_value(equity):
        logger.warning("Skipping weekly DD check: invalid equity=%.8f ns=%s", equity, risk_ns)
        return True, 0.0, {"emit_event": False}
    today = dj_tz.now().date()
    iso_year, iso_week, _ = today.isocalendar()
    period_key = f"{iso_year}-W{iso_week:02d}"
    baseline, dd = _compute_drawdown_state(
        risk_ns=risk_ns,
        period_type="weekly",
        period_key=period_key,
        equity=equity,
    )
    if not math.isfinite(dd):
        return True, 0.0, {"emit_event": False}
    breach = dd <= -settings.WEEKLY_DD_LIMIT
    emit_event = breach and _should_emit_drawdown_event(baseline, dd, min_delta=0.01)
    return (
        (not breach),
        dd,
        {
            "emit_event": emit_event,
            "baseline": baseline,
            "period_type": "weekly",
            "period_key": period_key,
        },
    )


def _check_data_staleness(instrument: Instrument) -> bool:
    """Check if market data is fresh enough to trade. Returns True if data is OK."""
    latest = (
        Candle.objects.filter(instrument=instrument, timeframe="1m")
        .order_by("-ts")
        .values_list("ts", flat=True)
        .first()
    )
    if latest is None:
        return False
    age = (dj_tz.now() - latest).total_seconds()
    return age <= settings.DATA_STALE_SECONDS


def _track_consecutive_errors(symbol: str, success: bool) -> int:
    """
    Track consecutive order errors in Redis.
    Returns current consecutive error count.
    """
    client = _redis_client()
    if client is None:
        return 0
    key = f"risk:consec_errors:{symbol}"
    if success:
        client.set(key, 0)
        return 0
    count = client.incr(key)
    client.expire(key, 3600)  # expire in 1 hour
    return int(count)


def _create_risk_event(
    kind: str,
    severity: str,
    instrument=None,
    details: dict = None,
    risk_ns: str = "global",
):
    """Create a RiskEvent record and send notification (throttled + deduplicated)."""
    try:
        symbol = instrument.symbol if instrument else "global"
        details_obj = details or {}
        dedup_seconds = max(
            1,
            int(getattr(settings, "RISK_EVENT_DEDUP_SECONDS", 300) or 300),
        )
        bucket = int(dj_tz.now().timestamp()) // dedup_seconds
        # Stable fingerprint by namespace/kind/symbol/window to avoid event spam.
        fp_raw = f"{risk_ns}|{kind}|{symbol}|{bucket}"
        fp = hashlib.sha1(fp_raw.encode("utf-8", errors="ignore")).hexdigest()
        dedup_key = f"risk:event:dedup:{fp}"
        client = _redis_client()
        if client is not None:
            allow_emit = bool(client.set(dedup_key, "1", nx=True, ex=dedup_seconds))
            if not allow_emit:
                return

        RiskEvent.objects.create(
            instrument=instrument,
            kind=kind,
            severity=severity,
            details_json=details_obj,
        )
        notify_details = _format_risk_event_notification_details(
            instrument=instrument,
            details=details_obj,
        )
        # Throttle Telegram: only notify once per kind+instrument every 30 min
        throttle_key = f"risk:notified:{risk_ns}:{kind}:{symbol}"
        if client and client.set(throttle_key, "1", nx=True, ex=1800):
            notify_risk_event(kind, severity, notify_details)
        elif client is None:
            notify_risk_event(kind, severity, notify_details)
        # else: throttled - skip Telegram
    except Exception as exc:
        logger.warning("Failed to create risk event: %s", exc)


def _record_min_qty_risk_guard_event(
    instrument: Instrument,
    *,
    qty: float,
    risk_qty: float,
    min_qty: float,
    actual_risk_amount: float,
    target_risk_amount: float,
    risk_multiplier: float,
    stop_distance_pct: float,
) -> None:
    """Persist a deduplicated DB event for min-qty over-risk blocks without Telegram noise."""
    try:
        symbol = instrument.symbol
        dedup_seconds = max(
            1,
            int(getattr(settings, "RISK_EVENT_DEDUP_SECONDS", 300) or 300),
        )
        bucket = int(dj_tz.now().timestamp()) // dedup_seconds
        fp_raw = f"min_qty_risk_guard|{symbol}|{bucket}"
        fp = hashlib.sha1(fp_raw.encode("utf-8", errors="ignore")).hexdigest()
        dedup_key = f"risk:event:quiet:{fp}"
        client = _redis_client()
        if client is not None:
            allow_emit = bool(client.set(dedup_key, "1", nx=True, ex=dedup_seconds))
            if not allow_emit:
                return
        RiskEvent.objects.create(
            instrument=instrument,
            kind="min_qty_risk_guard",
            severity=RiskEvent.Severity.WARN,
            details_json={
                "qty": round(float(qty), 10),
                "risk_qty": round(float(risk_qty), 10),
                "min_qty": round(float(min_qty), 10),
                "risk_actual": round(float(actual_risk_amount), 8),
                "risk_target": round(float(target_risk_amount), 8),
                "risk_mult": round(float(risk_multiplier), 4),
                "stop_pct": round(float(stop_distance_pct), 6),
            },
        )
    except Exception as exc:
        logger.warning("Failed to record min-qty risk guard event: %s", exc)


def _format_risk_event_notification_details(instrument=None, details: dict | None = None) -> str:
    details_obj = details or {}
    lines: list[str] = []
    symbol = ""
    try:
        symbol = str(getattr(instrument, "symbol", "") or details_obj.get("symbol") or "").strip()
    except Exception:
        symbol = ""
    if symbol and symbol.lower() != "global":
        lines.append(f"Symbol: {symbol}")

    if isinstance(details_obj, dict):
        latest_ts = details_obj.get("latest_ts")
        age_seconds = details_obj.get("age_seconds")
        if latest_ts:
            lines.append(f"Latest 1m: {latest_ts}")
        if age_seconds is not None:
            try:
                lines.append(f"Age: {int(float(age_seconds))}s")
            except Exception:
                lines.append(f"Age: {age_seconds}")
        for key in sorted(details_obj):
            if key in {"symbol", "latest_ts", "age_seconds"}:
                continue
            value = details_obj.get(key)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=True, sort_keys=True)
            lines.append(f"{key}: {value}")
    elif details_obj:
        lines.append(str(details_obj))
    return "\n".join(lines)


def _track_data_staleness_transition(
    instrument: Instrument,
    latest_ts,
    *,
    now_ts=None,
    risk_ns: str = "global",
) -> tuple[bool, bool, dict]:
    """
    Track stale-data transitions per symbol/stack.

    Returns:
      (is_stale, should_emit_event, details)
    """
    now_ts = now_ts or dj_tz.now()
    age_seconds = None
    if latest_ts is not None:
        try:
            age_seconds = max(0, int((now_ts - latest_ts).total_seconds()))
        except Exception:
            age_seconds = None
    is_stale = latest_ts is None or (age_seconds is not None and age_seconds > settings.DATA_STALE_SECONDS)
    symbol = str(getattr(instrument, "symbol", "") or "global").strip() or "global"
    state_key = f"risk:data_stale:state:{risk_ns}:{symbol}"
    state_ttl = max(int(getattr(settings, "DATA_STALE_SECONDS", 300) or 300) * 12, 3600)

    if not is_stale:
        client = _redis_client()
        if client is not None and client.delete(state_key):
            logger.info("Market data recovered for %s", symbol)
        return False, False, {}

    details = {
        "symbol": symbol,
        "latest_ts": latest_ts.isoformat() if latest_ts is not None else None,
        "age_seconds": age_seconds,
    }
    client = _redis_client()
    if client is not None:
        should_emit = bool(client.set(state_key, "1", nx=True, ex=state_ttl))
        if not should_emit:
            client.expire(state_key, state_ttl)
        return True, should_emit, details

    recent_window_seconds = max(
        int(getattr(settings, "DATA_STALE_SECONDS", 300) or 300),
        int(getattr(settings, "RISK_EVENT_DEDUP_SECONDS", 300) or 300),
        300,
    ) * 2
    recent_cutoff = now_ts - timedelta(seconds=recent_window_seconds)
    recent_exists = RiskEvent.objects.filter(
        instrument=instrument,
        kind="data_stale",
        ts__gte=recent_cutoff,
    ).exists()
    return True, not recent_exists, details

def _atr_pct(instrument: Instrument, period: int = 14, tf: str = "5m"):
    qs = (
        Candle.objects.filter(instrument=instrument, timeframe=tf)
        .order_by("-ts")[: period + 1]
        .values("high", "low", "close")
    )
    # Normalize all numeric values to float up-front to avoid float/Decimal mixups
    rows = [
        {
            "high": _to_float(r["high"]),
            "low": _to_float(r["low"]),
            "close": _to_float(r["close"]),
        }
        for r in qs
    ]
    if len(rows) < period + 1:
        return None
    rows = rows[::-1]  # chronological
    trs = []
    prev_close = rows[0]["close"]
    for row in rows[1:]:
        high = row["high"]
        low = row["low"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        prev_close = row["close"]
    atr = sum(trs[-period:]) / period
    last_close = rows[-1]["close"]
    if last_close == 0:
        return None
    return atr / last_close


def _volume_activity_ratio(instrument: Instrument, tf: str = "5m", lookback: int = 48) -> float | None:
    """
    Compare current candle volume vs median recent volume.
    Returns `current_volume / median_volume` or None when data is insufficient.
    """
    bars = max(10, int(lookback or 48))
    qs = (
        Candle.objects.filter(instrument=instrument, timeframe=tf)
        .order_by("-ts")[: bars + 1]
        .values_list("volume", flat=True)
    )
    vols = [_to_float(v) for v in qs][::-1]  # chronological
    if len(vols) < bars + 1:
        return None
    current_vol = vols[-1]
    baseline = [v for v in vols[:-1] if v > 0]
    if current_vol <= 0 or len(baseline) < max(5, bars // 2):
        return None
    baseline_median = _to_float(median(baseline))
    if baseline_median <= 0:
        return None
    return current_vol / baseline_median


def _volume_gate_min_ratio(session_name: str | None = None) -> float:
    base = max(0.0, float(getattr(settings, "ENTRY_VOLUME_FILTER_MIN_RATIO", 0.75) or 0.75))
    session = str(session_name or "").strip().lower()
    if not session:
        return base
    raw_map = getattr(settings, "ENTRY_VOLUME_FILTER_MIN_RATIO_BY_SESSION", {})
    if not isinstance(raw_map, dict):
        return base
    session_val = raw_map.get(session)
    if session_val is None:
        return base
    try:
        return max(0.0, float(session_val))
    except (TypeError, ValueError):
        return base


def _volume_gate_allowed(
    instrument: Instrument,
    session_name: str | None = None,
) -> tuple[bool, float | None]:
    """
    Execution-time volume quality gate.
    Returns (allowed, ratio). ratio is None when data is insufficient.
    """
    if not bool(getattr(settings, "ENTRY_VOLUME_FILTER_ENABLED", False)):
        return True, None
    tf = str(getattr(settings, "ENTRY_VOLUME_FILTER_TIMEFRAME", "5m") or "5m").strip().lower()
    lookback = max(10, int(getattr(settings, "ENTRY_VOLUME_FILTER_LOOKBACK", 48) or 48))
    min_ratio = _volume_gate_min_ratio(session_name=session_name)
    ratio = _volume_activity_ratio(instrument, tf=tf, lookback=lookback)
    if ratio is None:
        return bool(getattr(settings, "ENTRY_VOLUME_FILTER_FAIL_OPEN", True)), None
    return ratio >= min_ratio, ratio


def _compute_stop_distance(instrument: Instrument, side: str, entry_price: float) -> float | None:
    """
    DEPRECATED: This previously returned only ATR-based stop distance, which could
    diverge from the actual SL we place (because _compute_tp_sl_prices() enforces
    MIN_SL_PCT). Kept for backward-compatibility; prefer using sl_pct from
    _compute_tp_sl_prices() for both SL placement and risk sizing.

    Returns None when ATR is unavailable.
    """
    atr_pct = _atr_pct(instrument)
    if atr_pct is None:
        return None

    # Match runtime SL logic: incorporate STOP_LOSS_PCT and MIN_SL_PCT floors.
    sl_pct = settings.STOP_LOSS_PCT
    sl_pct = max(sl_pct, atr_pct * settings.ATR_MULT_SL)
    min_sl = float(getattr(settings, "MIN_SL_PCT", 0.0) or 0.0)
    if min_sl > 0:
        sl_pct = max(sl_pct, min_sl)
    return float(sl_pct)


def _risk_based_qty(
    equity: float,
    entry_price: float,
    stop_distance_pct: float,
    contract_size: float = 1.0,
    leverage: float = 1.0,
    risk_pct: float | None = None,
) -> float:
    """
    Calculate position size based on risk per trade:
    size = (risk_pct * equity) / (stop_distance * entry_price * contract_size)

    Returns number of contracts (float, caller should ceil/floor as needed).
    """
    if stop_distance_pct <= 0 or entry_price <= 0:
        return 0.0
    effective_risk = risk_pct if risk_pct is not None else settings.RISK_PER_TRADE_PCT
    risk_amount = effective_risk * equity
    stop_distance_abs = stop_distance_pct * entry_price
    qty = risk_amount / (stop_distance_abs * contract_size)
    return qty


def _actual_stop_risk_amount(
    *,
    qty: float,
    entry_price: float,
    stop_distance_pct: float,
    contract_size: float = 1.0,
) -> float:
    if qty <= 0 or entry_price <= 0 or stop_distance_pct <= 0 or contract_size <= 0:
        return 0.0
    stop_distance_abs = stop_distance_pct * entry_price
    return abs(qty) * stop_distance_abs * contract_size


def _min_qty_risk_guard(
    *,
    qty: float,
    risk_qty: float,
    min_qty: float,
    entry_price: float,
    stop_distance_pct: float,
    contract_size: float,
    target_risk_amount: float,
) -> tuple[bool, float, float]:
    """
    Blocks entries when exchange-imposed minimum quantity forces realized stop risk
    far above the intended risk budget.

    Returns (blocked, actual_risk_amount, risk_multiplier).
    """
    if not bool(getattr(settings, "MIN_QTY_RISK_GUARD_ENABLED", True)):
        return False, 0.0, 0.0
    if min_qty <= 0 or qty <= 0 or risk_qty <= 0 or target_risk_amount <= 0:
        return False, 0.0, 0.0
    if qty <= risk_qty + 1e-12:
        return False, 0.0, 0.0

    actual_risk_amount = _actual_stop_risk_amount(
        qty=qty,
        entry_price=entry_price,
        stop_distance_pct=stop_distance_pct,
        contract_size=contract_size,
    )
    if actual_risk_amount <= 0:
        return False, 0.0, 0.0
    risk_multiplier = actual_risk_amount / target_risk_amount
    max_multiplier = float(getattr(settings, "MIN_QTY_RISK_MULTIPLIER_MAX", 3.0) or 3.0)
    return risk_multiplier > max_multiplier, actual_risk_amount, risk_multiplier


def _min_qty_absolute_risk_cap_allows(
    *,
    actual_risk_amount: float,
    equity_usdt: float,
) -> tuple[bool, float, float]:
    """
    Optional escape hatch for tiny accounts / low-budget allocator signals.

    The normal min-qty guard compares actual stop risk against the intended
    signal budget. When the signal budget is deliberately tiny, the exchange
    minimum can look like a huge multiplier while still being small in absolute
    account-risk terms. This helper only allows that case when explicitly
    enabled and the actual risk remains under a configured equity cap.
    """
    enabled = get_runtime_bool(
        "MIN_QTY_RISK_ABSOLUTE_CAP_ENABLED",
        bool(getattr(settings, "MIN_QTY_RISK_ABSOLUTE_CAP_ENABLED", False)),
    )
    if not enabled:
        return False, 0.0, 0.0
    equity = max(0.0, float(equity_usdt or 0.0))
    actual = max(0.0, float(actual_risk_amount or 0.0))
    if equity <= 0 or actual <= 0:
        return False, 0.0, 0.0
    cap_pct = get_runtime_float(
        "MIN_QTY_RISK_ABSOLUTE_CAP_PCT",
        float(
            getattr(
                settings,
                "MIN_QTY_RISK_ABSOLUTE_CAP_PCT",
                getattr(settings, "RISK_PER_TRADE_PCT", 0.003),
            )
            or 0.0
        ),
        minimum=0.0,
        maximum=0.05,
    )
    if cap_pct <= 0:
        return False, actual / equity, cap_pct
    actual_pct = actual / equity
    return actual_pct <= cap_pct, actual_pct, cap_pct


def _order_is_reduce_only(order_payload: dict | None) -> bool:
    if not isinstance(order_payload, dict):
        return False
    info = order_payload.get("info") or {}
    raw = info.get("reduceOnly", order_payload.get("reduceOnly"))
    if isinstance(raw, bool):
        return raw
    return str(raw or "").strip().lower() in {"true", "1", "yes"}


def _min_qty_dynamic_allowlist_state(risk_multiplier: float) -> str:
    block_mult = max(
        1.0,
        float(
            getattr(
                settings,
                "MIN_QTY_DYNAMIC_ALLOWLIST_BLOCK_MULTIPLIER",
                getattr(settings, "MIN_QTY_RISK_MULTIPLIER_MAX", 3.0),
            )
            or getattr(settings, "MIN_QTY_RISK_MULTIPLIER_MAX", 3.0)
        ),
    )
    watch_mult = max(
        1.0,
        min(
            block_mult,
            float(getattr(settings, "MIN_QTY_DYNAMIC_ALLOWLIST_WATCH_MULTIPLIER", 2.0) or 2.0),
        ),
    )
    mult = max(0.0, float(risk_multiplier or 0.0))
    if mult >= block_mult:
        return "blocked"
    if mult >= watch_mult:
        return "watch"
    return "tradable"


def _confidence_adjusted_entry_leverage(
    *,
    base_leverage: float,
    strategy_name: str,
    sig_score: float,
    ml_prob: float | None,
    ml_enabled: bool,
    signal_direction: str = "",
) -> tuple[float, str]:
    """
    Optional leverage boost for high-confidence entries.
    Returns (entry_leverage, reason).
    """
    base = max(1.0, float(base_leverage or 1.0))
    if not bool(getattr(settings, "CONFIDENCE_LEVERAGE_BOOST_ENABLED", False)):
        return base, "disabled"
    if signal_direction == "long" and not bool(getattr(settings, "CONFIDENCE_LEVERAGE_LONG_ENABLED", False)):
        return base, "long_boost_disabled"
    strategy_txt = str(strategy_name or "").strip().lower()
    is_allocator = strategy_txt.startswith("alloc_")
    is_microvol = _strategy_is_microvol(strategy_txt)
    if bool(getattr(settings, "CONFIDENCE_LEVERAGE_ONLY_ALLOCATOR", True)):
        allow_microvol = bool(getattr(settings, "CONFIDENCE_LEVERAGE_ALLOW_MICROVOL", True))
        if not is_allocator and not (allow_microvol and is_microvol):
            return base, "non_allocator"

    score_threshold = float(getattr(settings, "CONFIDENCE_LEVERAGE_SCORE_THRESHOLD", 0.90) or 0.90)
    if is_microvol:
        score_threshold = float(
            getattr(settings, "CONFIDENCE_LEVERAGE_MICROVOL_SCORE_THRESHOLD", score_threshold)
            or score_threshold
        )
    ml_threshold = float(getattr(settings, "CONFIDENCE_LEVERAGE_ML_PROB_THRESHOLD", 0.70) or 0.70)
    require_both = bool(getattr(settings, "CONFIDENCE_LEVERAGE_REQUIRE_BOTH", False))

    score_ok = float(sig_score or 0.0) >= score_threshold
    ml_ok = bool(ml_enabled and ml_prob is not None and float(ml_prob) >= ml_threshold)
    qualifies = (score_ok and ml_ok) if require_both else (score_ok or ml_ok)
    if not qualifies:
        return base, "below_threshold"

    mult = max(1.0, float(getattr(settings, "CONFIDENCE_LEVERAGE_MULT", 1.30) or 1.30))
    max_lev = max(base, float(getattr(settings, "CONFIDENCE_LEVERAGE_MAX", base) or base))
    boosted = max(base, min(base * mult, max_lev))
    return boosted, ("score+ml" if (score_ok and ml_ok) else ("score" if score_ok else "ml"))


def _ensure_entry_leverage(adapter, symbol: str, target_leverage: float) -> tuple[bool, str]:
    """
    Best-effort per-symbol leverage setter before opening an order.
    Uses adapter internals for ccxt set_leverage compatibility by exchange.
    """
    try:
        client = getattr(adapter, "client", None)
        if client is None or not hasattr(client, "set_leverage"):
            return False, "unsupported"

        target = int(max(1, round(float(target_leverage or 1.0))))
        mapped = symbol
        try:
            if hasattr(adapter, "_map_symbol"):
                mapped = str(adapter._map_symbol(symbol))
        except Exception:
            mapped = symbol

        cache = getattr(adapter, "_symbol_leverage_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            try:
                setattr(adapter, "_symbol_leverage_cache", cache)
            except Exception:
                cache = {}

        prev = cache.get(mapped)
        if prev == target:
            return True, "cached"

        adapter_name = str(adapter.__class__.__name__ or "").strip().lower()
        if "bingx" in adapter_name:
            try:
                client.set_leverage(target, mapped, {"side": "BOTH"})
            except Exception:
                client.set_leverage(target, mapped, {"side": "LONG"})
                client.set_leverage(target, mapped, {"side": "SHORT"})
        elif "kucoin" in adapter_name:
            margin_mode = str(getattr(adapter, "margin_mode", "cross") or "cross")
            client.set_leverage(target, mapped, {"marginMode": margin_mode})
        else:
            client.set_leverage(target, mapped)

        cache[mapped] = target
        lev_set = getattr(adapter, "_leverage_set_symbols", None)
        if isinstance(lev_set, set):
            lev_set.add(mapped)
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _volatility_adjusted_risk(symbol: str, atr_pct: float | None, base_risk: float) -> float:
    """Backward-compatible wrapper to shared risk sizing policy."""
    return _shared_volatility_adjusted_risk(symbol, atr_pct, base_risk)


def _tp_sl_gate_pnl_pct(pnl_pct_gross: float) -> tuple[float, float]:
    """
    Return (pnl_pct_for_gate, fee_pct_estimate).

    The TP/SL gate can use a conservative net estimate by discounting
    an expected roundtrip fee from the live (gross) pnl%.
    """
    fee_pct_estimate = 0.0
    if bool(getattr(settings, "TP_SL_FEE_ADJUST_ENABLED", True)):
        fee_pct_estimate = max(
            0.0,
            float(getattr(settings, "TP_SL_ESTIMATED_ROUNDTRIP_FEE_PCT", 0.0010) or 0.0010),
        )
    return pnl_pct_gross - fee_pct_estimate, fee_pct_estimate


# ---------------------------------------------------------------------------
# Daily trade count limiter (risk-management skill: 95% success rate)
# "Trade count inversely correlates with performance in flat markets"
# ---------------------------------------------------------------------------

def _daily_trade_count_key() -> str:
    """Redis key for today's trade count."""
    return f"risk:daily_trades:{dj_tz.now().strftime('%Y-%m-%d')}"


def _get_daily_trade_count() -> int:
    """Return number of new entries opened today."""
    client = _redis_client()
    if client is None:
        return 0
    try:
        raw = client.get(_daily_trade_count_key())
        return int(raw) if raw is not None else 0
    except Exception:
        return 0


def _increment_daily_trade_count() -> None:
    """Increment today's trade counter after opening a new position."""
    client = _redis_client()
    if client is None:
        return
    try:
        key = _daily_trade_count_key()
        ttl_seconds = max(
            60,
            int(getattr(settings, "DAILY_TRADE_COUNT_TTL_SECONDS", 86400 + 3600) or (86400 + 3600)),
        )
        pipe = client.pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl_seconds)
        pipe.execute()
    except Exception:
        pass


def _max_daily_trades_for_adx(htf_adx: float | None) -> int:
    """Backward-compatible wrapper to shared ADX trade throttle policy."""
    return _shared_max_daily_trades_for_adx(htf_adx)


# ---------------------------------------------------------------------------
# Time-based stale position cleanup (risk skill: 82-88% success rate)
# "Close positions near breakeven to free margin for higher-conviction trades"
# ---------------------------------------------------------------------------

def _should_close_stale_position(
    pos_opened_at: datetime | None,
    pnl_pct: float,
) -> bool:
    """
    Return True if position is old and stuck near breakeven.
    Frees margin for higher-conviction trades.
    """
    if not getattr(settings, "STALE_POSITION_ENABLED", False):
        return False
    if pos_opened_at is None:
        return False

    max_hours = int(getattr(settings, "STALE_POSITION_MAX_HOURS", 12))
    pnl_band = float(getattr(settings, "STALE_POSITION_PNL_BAND_PCT", 0.005))

    age_hours = (dj_tz.now() - pos_opened_at).total_seconds() / 3600
    if age_hours < max_hours:
        return False

    # Position is old Ã¢â‚¬â€ close if PnL is within the breakeven band
    return -pnl_band <= pnl_pct <= pnl_band


def _is_tick_size_mode(precision_mode: Any) -> bool:
    txt = str(precision_mode or "").strip().lower()
    if txt in {"tick_size", "ticksize"}:
        return True
    try:
        # CCXT uses numeric precision modes; 4 maps to TICK_SIZE.
        return int(precision_mode) == 4
    except Exception:
        return False


def _market_min_qty(
    market: dict | None,
    fallback: float = 0.0,
    *,
    precision_mode: Any = None,
    last_price: float | None = None,
    contract_size: float = 1.0,
) -> float:
    if not isinstance(market, dict):
        return max(0.0, _to_float(fallback))
    limits = {}
    limit_min = 0.0
    cost_min = 0.0
    try:
        limits = market.get("limits") or {}
        amount_limits = limits.get("amount") or {}
        limit_min = _to_float(amount_limits.get("min"))
        cost_limits = limits.get("cost") or {}
        cost_min = _to_float(cost_limits.get("min"))
    except Exception:
        limit_min = 0.0
        cost_min = 0.0
    precision_step = _market_amount_step(market, precision_mode=precision_mode)
    cost_based_min = 0.0
    price_val = _to_float(last_price)
    contract_val = max(_to_float(contract_size), 1e-12)
    if cost_min > 0 and price_val > 0 and contract_val > 0:
        cost_based_min = cost_min / (price_val * contract_val)
    discovered_min = max(0.0, limit_min, precision_step, cost_based_min)
    if discovered_min > 0:
        return discovered_min
    return max(0.0, _to_float(fallback))


def _market_amount_step(
    market: dict | None,
    *,
    precision_mode: Any = None,
) -> float:
    if not isinstance(market, dict):
        return 0.0
    precision_step = 0.0
    try:
        precision = market.get("precision") or {}
        amount_raw = precision.get("amount")
        if amount_raw is not None:
            amount_precision = _to_float(amount_raw)
            if math.isfinite(amount_precision) and amount_precision >= 0:
                if _is_tick_size_mode(precision_mode) and amount_precision > 0:
                    # In tick-size mode, precision.amount is already the amount step.
                    precision_step = amount_precision
                elif float(amount_precision).is_integer():
                    # Decimal places mode (0 -> min step 1, 1 -> 0.1, ...).
                    precision_step = 10 ** (-int(amount_precision))
                elif amount_precision > 0:
                    # Some exchanges expose amount step directly.
                    precision_step = amount_precision
    except Exception:
        precision_step = 0.0
    return max(0.0, precision_step)


def _floor_to_step(value: float, step: float) -> float:
    if step <= 0 or not math.isfinite(value) or value <= 0:
        return 0.0
    try:
        d_value = Decimal(str(value))
        d_step = Decimal(str(step))
        if d_step <= 0:
            return 0.0
        units = (d_value / d_step).to_integral_value(rounding=ROUND_FLOOR)
        return _to_float(units * d_step)
    except Exception:
        return 0.0


def _ceil_to_step(value: float, step: float) -> float:
    if step <= 0 or not math.isfinite(value) or value <= 0:
        return 0.0
    try:
        d_value = Decimal(str(value))
        d_step = Decimal(str(step))
        if d_step <= 0:
            return 0.0
        units = (d_value / d_step).to_integral_value(rounding=ROUND_CEILING)
        return _to_float(units * d_step)
    except Exception:
        return 0.0


def _precision_step_from_error(exc: Exception) -> float:
    msg = str(exc or "")
    match = re.search(r"minimum amount precision of\s*([0-9]*\.?[0-9]+)", msg, re.IGNORECASE)
    if not match:
        return 0.0
    return max(0.0, _to_float(match.group(1)))


def _minimum_order_amount_from_error(exc: Exception) -> float:
    msg = str(exc or "")
    patterns = (
        r"minimum order amount is\s*([0-9]*\.?[0-9]+)",
        r"minimum amount is\s*([0-9]*\.?[0-9]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, msg, re.IGNORECASE)
        if match:
            return max(0.0, _to_float(match.group(1)))
    return 0.0


def _align_min_order_qty(
    adapter,
    symbol: str,
    min_qty: float,
    *,
    market: dict | None = None,
    precision_mode: Any = None,
) -> float:
    min_val = _to_float(min_qty)
    if min_val <= 0:
        return 0.0
    step = _market_amount_step(market, precision_mode=precision_mode)
    aligned = _ceil_to_step(min_val, step) if step > 0 else min_val
    if not math.isfinite(aligned) or aligned <= 0:
        return 0.0
    mapped = symbol
    try:
        mapped = adapter._map_symbol(symbol)
    except Exception:
        mapped = symbol
    try:
        normalized = _to_float(adapter.client.amount_to_precision(mapped, aligned))
        if math.isfinite(normalized) and normalized > 0 and normalized + 1e-12 >= min_val:
            return normalized
    except Exception:
        pass
    return aligned


def _normalize_order_qty(adapter, symbol: str, qty: float) -> float:
    """
    Normalize quantity using exchange precision rules.
    Keeps sizing conservative by relying on exchange precision formatting.
    """
    if not math.isfinite(qty) or qty <= 0:
        return 0.0
    mapped = symbol
    try:
        mapped = adapter._map_symbol(symbol)
    except Exception:
        mapped = symbol
    try:
        normalized = _to_float(adapter.client.amount_to_precision(mapped, qty))
    except Exception as exc:
        # Some BingX symbols reject sub-integer amounts (e.g. "minimum amount precision of 1").
        # Never return raw qty here; derive a safe floor from market metadata/error text.
        step = 0.0
        try:
            market = adapter.client.market(mapped)
            precision_mode = getattr(adapter.client, "precisionMode", None)
            step = _market_min_qty(market, fallback=0.0, precision_mode=precision_mode)
        except Exception:
            step = 0.0
        step = max(step, _precision_step_from_error(exc))
        normalized = _floor_to_step(qty, step) if step > 0 else 0.0
    if not math.isfinite(normalized) or normalized <= 0:
        return 0.0
    return normalized


def _is_macro_high_impact_window(
    now_utc: datetime | None = None,
    *,
    session_name: str | None = None,
) -> tuple[bool, dict]:
    enabled = bool(getattr(settings, "MACRO_HIGH_IMPACT_FILTER_ENABLED", False))
    session_txt = str(session_name or "").strip().lower()
    if not enabled:
        return False, {"enabled": False, "session": session_txt}

    ts = now_utc or dj_tz.now()
    hour = int(ts.hour)
    weekday = int(ts.weekday())  # Monday=0
    hours = set(getattr(settings, "MACRO_HIGH_IMPACT_UTC_HOURS", set()) or set())
    weekdays = set(getattr(settings, "MACRO_HIGH_IMPACT_WEEKDAYS", set()) or set())
    sessions = set(getattr(settings, "MACRO_HIGH_IMPACT_SESSIONS", set()) or set())

    hour_ok = (not hours) or (hour in hours)
    weekday_ok = (not weekdays) or (weekday in weekdays)
    session_ok = (not sessions) or (session_txt in sessions)
    active = bool(hour_ok and weekday_ok and session_ok)

    return active, {
        "enabled": enabled,
        "hour_utc": hour,
        "weekday": weekday,
        "session": session_txt,
        "hour_ok": hour_ok,
        "weekday_ok": weekday_ok,
        "session_ok": session_ok,
        "active": active,
    }


def _tactical_exit_profile(side: str, recommended_bias: str | None) -> dict[str, float | bool | str]:
    bias = str(recommended_bias or "").strip().lower()
    side_txt = str(side or "").strip().lower()
    active = (
        (side_txt == "buy" and bias == "tactical_long")
        or (side_txt == "sell" and bias == "tactical_short")
    )
    if not bool(getattr(settings, "TACTICAL_EXIT_PROFILE_ENABLED", True)):
        active = False
    return {
        "active": active,
        "bias": bias,
        "tp_mult": float(getattr(settings, "TACTICAL_EXIT_TP_MULT", 0.75) or 0.75),
        "partial_r_mult": float(getattr(settings, "TACTICAL_EXIT_PARTIAL_R_MULT", 0.85) or 0.85),
        "trail_r_mult": float(getattr(settings, "TACTICAL_EXIT_TRAIL_R_MULT", 0.75) or 0.75),
        "breakeven_r_mult": float(getattr(settings, "TACTICAL_EXIT_BREAKEVEN_R_MULT", 0.75) or 0.75),
        "lockin_mult": float(getattr(settings, "TACTICAL_EXIT_LOCKIN_MULT", 1.15) or 1.15),
    }


def _resolve_regime_label(symbol: str) -> str:
    """Return HMM regime label for *symbol* or empty string on miss."""
    try:
        from signals.regime import get_cached_regime
        info = get_cached_regime(symbol)
        return (info or {}).get("label", "")
    except Exception:
        return ""


def _compute_tp_sl_prices(
    side: str,
    entry_price: float,
    atr_pct: float | None,
    recommended_bias: str | None = "",
    strategy_name: str = "",
    regime_label: str = "",
):
    """
    Compute TP and SL prices based on ATR or defaults.
    Returns (tp_price, sl_price, tp_pct, sl_pct).
    Applies a minimum SL floor (MIN_SL_PCT) to avoid micro stop-outs from noise.
    """
    tp_pct = settings.TAKE_PROFIT_PCT
    sl_pct = settings.STOP_LOSS_PCT
    min_sl = getattr(settings, 'MIN_SL_PCT', 0.008)
    tp_mult_default = float(getattr(settings, "ATR_MULT_TP", 0.0) or 0.0)
    tp_mult_long = float(getattr(settings, "ATR_MULT_TP_LONG", tp_mult_default) or tp_mult_default)
    tp_mult_short = float(getattr(settings, "ATR_MULT_TP_SHORT", tp_mult_default) or tp_mult_default)
    tp_mult_side = tp_mult_long if side == "buy" else tp_mult_short
    if atr_pct:
        tp_pct = max(tp_pct, atr_pct * tp_mult_side)
        sl_pct = max(sl_pct, atr_pct * settings.ATR_MULT_SL)

    # Global TP tuning: allow slightly closer TP to secure gains earlier.
    tp_mult = float(getattr(settings, "TAKE_PROFIT_DYNAMIC_MULT", 1.0) or 1.0)
    tp_mult = max(0.1, min(tp_mult, 2.0))
    tp_floor = float(getattr(settings, "TAKE_PROFIT_MIN_PCT", 0.0) or 0.0)
    tp_floor = max(0.0, tp_floor)
    tp_pct = max(tp_floor, tp_pct * tp_mult)

    # ── Regime-adaptive TP/SL ──
    if regime_label and getattr(settings, "REGIME_DYNAMIC_TPSL_ENABLED", True):
        if regime_label == "trending":
            _r_tp = float(getattr(settings, "REGIME_TP_TRENDING_MULT", 1.15) or 1.15)
            tp_pct *= max(1.0, min(_r_tp, 1.50))
        elif regime_label == "choppy":
            _r_tp = float(getattr(settings, "REGIME_TP_CHOPPY_MULT", 0.85) or 0.85)
            _r_sl = float(getattr(settings, "REGIME_SL_CHOPPY_MULT", 1.10) or 1.10)
            tp_pct *= max(0.50, min(_r_tp, 1.0))
            sl_pct *= max(1.0, min(_r_sl, 1.30))

    # Fast-exit mode for high-volatility phases: reduce TP distance to secure gains sooner.
    if (
        bool(getattr(settings, "VOL_FAST_EXIT_ENABLED", False))
        and atr_pct is not None
        and atr_pct >= float(getattr(settings, "VOL_FAST_EXIT_ATR_PCT", 0.012) or 0.012)
    ):
        tp_mult = float(getattr(settings, "VOL_FAST_EXIT_TP_MULT", 0.75) or 0.75)
        min_tp = float(getattr(settings, "VOL_FAST_EXIT_MIN_TP_PCT", 0.006) or 0.006)
        tp_pct = max(min_tp, tp_pct * max(0.1, min(tp_mult, 1.0)))

    tactical_profile = _tactical_exit_profile(side, recommended_bias)
    microvol_profile = _microvol_exit_profile(strategy_name)
    if bool(tactical_profile.get("active")):
        tp_pct = max(
            tp_floor,
            tp_pct * max(0.1, min(float(tactical_profile.get("tp_mult", 0.75) or 0.75), 1.0)),
        )
    if bool(microvol_profile.get("active")):
        tp_pct = max(
            tp_floor,
            tp_pct * max(0.1, min(float(microvol_profile.get("tp_mult", 0.55) or 0.55), 1.0)),
        )

    # Enforce absolute minimum SL to prevent noise-driven stop-outs
    sl_pct = max(sl_pct, min_sl)

    if side == "buy":
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    else:
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

    return tp_price, sl_price, tp_pct, sl_pct


def _place_sl_order(adapter, symbol: str, side: str, qty: float, sl_price: float) -> str | None:
    """
    Place a stop-market order on the exchange as a protective SL.
    Returns exchange order ID or None on failure.

    Uses CCXT's unified ``triggerPrice`` param so the call works on both
    Binance Futures (``stop_market`` type) **and** KuCoin Futures (which
    only accepts ``market``/``limit`` types with a trigger).

    For a SL:
      - long  position ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ close with sell when price drops  ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ stop="down"
      - short position ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ close with buy  when price rises  ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ stop="up"
    """
    close_side = "sell" if side == "buy" else "buy"
    # Determine trigger direction: price moving *against* the position
    stop_direction = "down" if side == "buy" else "up"
    try:
        resp = adapter.create_order(
            symbol,
            close_side,
            "market",               # unified type; trigger makes it a stop order
            qty,
            price=None,
            params={
                "triggerPrice": sl_price,
                "stop": stop_direction,          # KuCoin-specific direction hint
                "reduceOnly": True,
            },
        )
        order_id = resp.get("id") or resp.get("orderId", "")
        logger.info("SL stop-order placed %s side=%s qty=%s stopPrice=%.4f id=%s", symbol, close_side, qty, sl_price, order_id)
        return order_id
    except Exception as exc:
        logger.warning("Failed to place SL stop-order %s: %s (will use polling fallback)", symbol, exc)
        return None


def _extract_trigger_price(order_payload: dict | None) -> float:
    if not isinstance(order_payload, dict):
        return 0.0
    info = order_payload.get("info") or {}
    return _to_float(
        order_payload.get("triggerPrice")
        or order_payload.get("stopPrice")
        or info.get("stopPrice")
        or info.get("triggerPrice")
    )


def _has_sl_stop_order(adapter, symbol: str, position_side: str) -> tuple[bool, float | None, list]:
    """
    Check if there is an active stop/trigger order on the exchange that
    would close this position (i.e. a protective SL).
    Uses fetch_open_stop_orders (separate endpoint on KuCoin Futures).
    Returns (exists, stop_price, stop_orders).
    """
    try:
        stop_orders = adapter.fetch_open_stop_orders(symbol)
    except Exception as exc:
        logger.debug("fetch_open_stop_orders failed for %s: %s", symbol, exc)
        return True, None, []  # assume exists ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ avoid placing duplicates on API error

    if stop_orders:
        logger.debug("Found %d stop order(s) for %s", len(stop_orders), symbol)
        # Extract trigger price from the first stop order
        stop_price = None
        for so in stop_orders:
            sp = _extract_trigger_price(so)
            if sp and sp > 0:
                stop_price = sp
                break
        return True, stop_price, stop_orders
    return False, None, []


def _cancel_stop_orders(adapter, symbol: str, stop_orders: list):
    """Cancel all existing stop orders for a symbol."""
    for so in stop_orders:
        order_id = so.get("id") or so.get("orderId") or (so.get("info") or {}).get("orderId")
        if order_id:
            try:
                # Try with stop param first (KuCoin), fallback to plain cancel
                try:
                    adapter.client.cancel_order(
                        order_id,
                        adapter._map_symbol(symbol),
                        params={"stop": True},
                    )
                except Exception:
                    adapter.cancel_order(order_id, symbol)
                logger.info("Cancelled old SL stop-order %s for %s", order_id, symbol)
            except Exception as exc:
                logger.warning("Failed to cancel stop-order %s for %s: %s", order_id, symbol, exc)


def _cleanup_orphan_reduce_only_orders(
    adapter,
    symbol: str,
    *,
    force: bool = False,
) -> list[str]:
    """
    Cancel stale reduce-only orders for a symbol when there is no open position.

    This is especially important on BingX, where protective trigger orders can
    remain open after the position has already been closed by TP/kill/manual
    logic, leaving stale reduceOnly orders visible in the account UI.
    """
    if not bool(getattr(settings, "ORPHAN_REDUCE_ONLY_CLEANUP_ENABLED", True)):
        return []

    interval_seconds = max(
        60,
        int(getattr(settings, "ORPHAN_REDUCE_ONLY_CLEANUP_INTERVAL_SECONDS", 600) or 600),
    )
    client = _redis_client()
    if client is not None and not force:
        try:
            cleanup_key = f"orphan_reduce_only_cleanup:{_norm_symbol(symbol)}"
            acquired = client.set(cleanup_key, "1", nx=True, ex=interval_seconds)
            if not acquired:
                return []
        except Exception:
            pass

    try:
        orders = adapter.fetch_open_orders(symbol) or []
    except Exception as exc:
        logger.debug("Orphan reduce-only cleanup fetch failed for %s: %s", symbol, exc)
        return []

    cancelled: list[str] = []
    for order in orders:
        if not _order_is_reduce_only(order):
            continue
        order_id = order.get("id") or order.get("orderId") or (order.get("info") or {}).get("orderId")
        if not order_id:
            continue
        try:
            adapter.cancel_order(order_id, symbol)
            cancelled.append(str(order_id))
        except Exception as exc:
            logger.warning("Failed to cancel orphan reduce-only order %s for %s: %s", order_id, symbol, exc)
    if cancelled:
        logger.warning(
            "Cancelled %d orphan reduce-only order(s) for %s: %s",
            len(cancelled),
            symbol,
            ",".join(cancelled),
        )
    return cancelled


def _reconcile_sl(
    adapter, symbol: str, side: str, current_qty: float,
    entry_price: float, atr_pct: float | None, opened_at=None,
):
    """
    Verify that a protective SL stop-order exists on the exchange
    AND that it is wide enough (>= current MIN_SL_PCT).
    If missing or too tight, cancel old + place new.
    Called every cycle for each open position.
    """
    _, sl_price, _, sl_pct = _compute_tp_sl_prices(side, entry_price, atr_pct)
    exists, current_stop_price, stop_orders = _has_sl_stop_order(adapter, symbol, side)

    if exists and stop_orders:
        stop_prices = [sp for sp in (_extract_trigger_price(so) for so in stop_orders) if sp > 0]
        if side == "buy":
            valid_stop_prices = [sp for sp in stop_prices if sp < entry_price]
            selected_stop = max(valid_stop_prices) if valid_stop_prices else 0.0
            current_sl_distance = ((entry_price - selected_stop) / entry_price) if selected_stop else 0.0
        else:
            valid_stop_prices = [sp for sp in stop_prices if sp > entry_price]
            selected_stop = min(valid_stop_prices) if valid_stop_prices else 0.0
            current_sl_distance = ((selected_stop - entry_price) / entry_price) if selected_stop else 0.0

        expected_sl_distance = float(sl_pct or 0.0)
        has_duplicate_stops = len(stop_prices) > 1
        invalid_for_side = selected_stop <= 0
        sl_too_tight_mult = max(
            0.0,
            float(getattr(settings, "SL_RECONCILE_TOO_TIGHT_MULT", 0.80) or 0.80),
        )
        sl_too_wide_mult = max(
            1.0,
            float(getattr(settings, "SL_RECONCILE_TOO_WIDE_MULT", 2.00) or 2.00),
        )
        too_tight = (
            expected_sl_distance > 0
            and current_sl_distance > 0
            and current_sl_distance < expected_sl_distance * sl_too_tight_mult
        )
        too_wide = (
            expected_sl_distance > 0
            and current_sl_distance > expected_sl_distance * sl_too_wide_mult
        )
        if invalid_for_side or too_tight or too_wide or has_duplicate_stops:
            logger.warning(
                "SL misaligned for %s: selected=%.4f current=%.4f%% expected=%.4f%% duplicates=%s; replacing",
                symbol,
                selected_stop,
                current_sl_distance * 100,
                expected_sl_distance * 100,
                has_duplicate_stops,
            )
            _cancel_stop_orders(adapter, symbol, stop_orders)
            placed = _place_sl_order(adapter, symbol, side, abs(current_qty), sl_price)
            if placed is not None:
                _remember_protective_stop_price(symbol, opened_at, entry_price, sl_price)
        elif selected_stop > 0:
            _remember_protective_stop_price(symbol, opened_at, entry_price, selected_stop)
        return  # SL exists and is aligned enough or was replaced

    if exists:
        return  # SL exists but couldn't read price ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â leave it alone

    # SL missing entirely
    logger.warning(
        "SL stop-order missing for %s (side=%s entry=%.4f); placing at %.4f",
        symbol, side, entry_price, sl_price,
    )
    placed = _place_sl_order(adapter, symbol, side, abs(current_qty), sl_price)
    if placed is not None:
        _remember_protective_stop_price(symbol, opened_at, entry_price, sl_price)


def _position_state_key(symbol: str, opened_at, entry_price: float) -> str:
    """
    Build a per-position key for Redis state (HWM, partial-close, etc).
    Prefer exchange-provided opened_at; fallback to entry_price.
    """
    safe_symbol = _norm_symbol(symbol)
    if opened_at:
        try:
            return f"{safe_symbol}:{int(opened_at.timestamp())}"
        except Exception:
            pass
    return f"{safe_symbol}:{round(float(entry_price or 0.0), 6)}"


def _trail_state_ttl_seconds() -> int:
    return max(
        60,
        int(getattr(settings, "TRAILING_STATE_TTL_SECONDS", 172800) or 172800),
    )


def _trail_stop_price_key(state_key: str) -> str:
    return f"trail:stop_price:{state_key}"


def _remember_protective_stop_price(symbol: str, opened_at, entry_price: float, stop_price: float) -> None:
    stop_price_val = max(0.0, float(stop_price or 0.0))
    if stop_price_val <= 0 or entry_price <= 0:
        return
    client = _redis_client()
    if client is None:
        return
    state_key = _position_state_key(symbol, opened_at, entry_price)
    try:
        client.set(
            _trail_stop_price_key(state_key),
            str(stop_price_val),
            ex=_trail_state_ttl_seconds(),
        )
    except Exception:
        pass


def _load_protective_stop_price(symbol: str, opened_at, entry_price: float) -> float:
    if entry_price <= 0:
        return 0.0
    client = _redis_client()
    if client is None:
        return 0.0
    state_key = _position_state_key(symbol, opened_at, entry_price)
    try:
        return max(0.0, _to_float(client.get(_trail_stop_price_key(state_key))))
    except Exception:
        return 0.0


def _origin_signal_id_from_correlation(correlation_id: str) -> int | None:
    text = str(correlation_id or "").strip()
    if not text:
        return None
    base = text.split(":", 1)[0]
    match = re.match(r"^(\d+)-", base)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _position_origin_signal(inst: Instrument, pos_side: str) -> Signal | None:
    order = (
        Order.objects.filter(
            instrument=inst,
            side=pos_side,
            status=Order.OrderStatus.FILLED,
            reduce_only=False,
            opened_at__isnull=False,
        )
        .order_by("-opened_at", "-id")
        .first()
    )
    if not order:
        return None
    for token in (getattr(order, "correlation_id", ""), getattr(order, "parent_correlation_id", "")):
        sig_id = _origin_signal_id_from_correlation(token)
        if not sig_id:
            continue
        sig = Signal.objects.filter(id=sig_id).first()
        if sig:
            return sig
    return None


def _position_origin_refs(inst: Instrument, pos_side: str, origin_signal: Signal | None, sig: Signal | None) -> tuple[str, str]:
    """
    Resolve stable operation attribution for an already-open position.

    signal_id should point to the entry/origin signal when available.
    correlation_id should point to the root position correlation, not the
    latest cycle signal, so close reports remain tied to the position origin.
    """
    origin_signal_id = str(getattr(origin_signal, "id", "") or (str(getattr(sig, "id", "")) if sig else ""))
    root_corr = _position_root_correlation(inst, pos_side)
    if root_corr:
        return origin_signal_id, root_corr
    if origin_signal_id:
        return origin_signal_id, _safe_correlation_id(f"{origin_signal_id}-{inst.symbol}")
    if sig:
        return "", _safe_correlation_id(f"{sig.id}-{inst.symbol}")
    return "", ""


def _trail_sl_from_hwm(entry_price: float, max_fav_pct: float, lock_in_pct: float, is_long: bool) -> float:
    """
    Trailing SL derived from the max favorable move (HWM).
    lock_in_pct=0.5 keeps half of the peak profit as a dynamic stop.
    """
    if entry_price <= 0:
        return 0.0
    lock = max(0.0, min(float(lock_in_pct), 1.0))
    if max_fav_pct <= 0:
        return entry_price
    if is_long:
        return entry_price * (1 + max_fav_pct * lock)
    return entry_price * (1 - max_fav_pct * lock)


def _tp_progress_exit_reason_against_position(side: str, recommended_bias: str | None) -> bool:
    side_txt = str(side or "").strip().lower()
    bias = str(recommended_bias or "").strip().lower()
    if side_txt == "buy":
        return bias in {"short_bias", "tactical_short"}
    if side_txt == "sell":
        return bias in {"long_bias", "tactical_long"}
    return False


def _evaluate_tp_progress_exit(
    *,
    symbol: str,
    side: str,
    entry_price: float,
    last_price: float,
    tp_pct: float,
    sl_pct: float,
    opened_at=None,
    signal_direction: str = "",
    recommended_bias: str | None = "",
    strategy_name: str = "",
) -> tuple[bool, str, dict[str, float | str | int | bool | None]]:
    meta: dict[str, float | str | int | bool | None] = {
        "symbol": symbol,
        "progress": 0.0,
        "max_fav": 0.0,
        "giveback_ratio": 0.0,
        "r_multiple": 0.0,
        "score": 0,
        "signal_direction": str(signal_direction or "").strip().lower(),
        "bias": str(recommended_bias or "").strip().lower(),
        "age_min": None,
    }
    if not bool(getattr(settings, "TP_PROGRESS_EARLY_EXIT_ENABLED", True)):
        return False, "tp_progress_exit_disabled", meta
    if entry_price <= 0 or last_price <= 0 or tp_pct <= 0 or sl_pct <= 0:
        return False, "tp_progress_exit_invalid", meta

    side_txt = str(side or "").strip().lower()
    if side_txt not in {"buy", "sell"}:
        return False, "tp_progress_exit_invalid_side", meta

    pos_direction = "long" if side_txt == "buy" else "short"
    pnl_pct_gross = ((last_price - entry_price) / entry_price) * (1 if side_txt == "buy" else -1)
    pnl_pct_gate, _ = _tp_sl_gate_pnl_pct(pnl_pct_gross)
    if pnl_pct_gate <= 0:
        return False, "tp_progress_exit_non_positive", meta

    progress = max(0.0, pnl_pct_gate / tp_pct) if tp_pct > 0 else 0.0
    r_multiple = pnl_pct_gate / sl_pct if sl_pct > 0 else 0.0
    meta["progress"] = progress
    meta["r_multiple"] = r_multiple

    min_progress = max(0.50, min(0.95, float(getattr(settings, "TP_PROGRESS_EARLY_EXIT_MIN_PROGRESS", 0.70) or 0.70)))
    min_r = max(0.0, float(getattr(settings, "TP_PROGRESS_EARLY_EXIT_MIN_R", 0.8) or 0.8))
    if progress < min_progress or r_multiple < min_r:
        return False, "tp_progress_exit_not_ready", meta

    state_key = _position_state_key(symbol, opened_at, entry_price)
    max_fav = max(pnl_pct_gate, 0.0)
    client = _redis_client()
    if client is not None:
        try:
            raw_fav = client.get(f"trail:max_fav:{state_key}")
            max_fav = max(max_fav, max(0.0, _to_float(raw_fav)))
        except Exception:
            pass
    giveback = max(0.0, max_fav - pnl_pct_gate)
    giveback_ratio = (giveback / max_fav) if max_fav > 0 else 0.0
    meta["max_fav"] = max_fav
    meta["giveback_ratio"] = giveback_ratio

    age_min = None
    if opened_at:
        try:
            age_min = max(0.0, (dj_tz.now() - opened_at).total_seconds() / 60.0)
        except Exception:
            age_min = None
    meta["age_min"] = age_min

    force_progress = max(min_progress, min(0.99, float(getattr(settings, "TP_PROGRESS_EARLY_EXIT_FORCE_PROGRESS", 0.90) or 0.90)))
    force_giveback = max(0.05, min(0.95, float(getattr(settings, "TP_PROGRESS_EARLY_EXIT_FORCE_GIVEBACK_RATIO", 0.18) or 0.18)))
    if progress >= force_progress and giveback_ratio >= force_giveback:
        meta["score"] = 99
        return True, "force_giveback", meta

    close_score = 0
    reasons: list[str] = []

    max_giveback_ratio = max(0.05, min(0.95, float(getattr(settings, "TP_PROGRESS_EARLY_EXIT_MAX_GIVEBACK_RATIO", 0.25) or 0.25)))
    if giveback_ratio >= max_giveback_ratio:
        close_score += 1
        reasons.append("giveback")

    sig_dir = str(signal_direction or "").strip().lower()
    if sig_dir in {"long", "short", "flat"} and sig_dir != pos_direction:
        close_score += 2
        reasons.append("signal_mismatch")

    if _tp_progress_exit_reason_against_position(side_txt, recommended_bias):
        close_score += 1
        reasons.append("bias_opposed")

    if _strategy_is_microvol(strategy_name) and age_min is not None:
        max_hold_min = max(1, int(getattr(settings, "MODULE_MICROVOL_MAX_HOLD_MINUTES", 18) or 18))
        age_ratio = max(0.10, min(1.0, float(getattr(settings, "TP_PROGRESS_EARLY_EXIT_MICROVOL_AGE_RATIO", 0.50) or 0.50)))
        microvol_age_trigger = max_hold_min * age_ratio
        if age_min >= microvol_age_trigger and giveback_ratio >= max(0.05, max_giveback_ratio * 0.5):
            close_score += 1
            reasons.append("microvol_age")

    meta["score"] = close_score
    if close_score >= max(1, int(getattr(settings, "TP_PROGRESS_EARLY_EXIT_CLOSE_SCORE", 2) or 2)):
        return True, ",".join(reasons)[:160] or "tp_progress_exit", meta
    return False, "tp_progress_exit_hold", meta


def _check_trailing_stop(
    adapter, symbol: str, side: str, current_qty: float,
    entry_price: float, last_price: float, sl_pct: float,
    opened_at=None,
    contract_size: float = 1.0,
    atr_pct: float | None = None,
    recommended_bias: str | None = "",
    strategy_name: str = "",
) -> tuple[bool, float]:
    """
    Profit protection:
    - Partial close at R multiple (best effort; skips tiny positions).
    - Trailing SL based on max favorable excursion (HWM). Updates exchange-side SL to lock profits.

    Returns (closed_by_trailing, close_fee_usdt) if force-closed at market in this cycle.
    """
    if not settings.TRAILING_STOP_ENABLED:
        return False, 0.0
    if entry_price <= 0 or last_price <= 0 or sl_pct <= 0:
        return False, 0.0

    is_long = current_qty > 0
    pnl_pct = (last_price - entry_price) / entry_price * (1 if is_long else -1)
    r_multiple = pnl_pct / sl_pct if sl_pct > 0 else 0.0
    state_key = _position_state_key(symbol, opened_at, entry_price)

    client = _redis_client()
    trail_state_ttl = _trail_state_ttl_seconds()
    sl_min_move_pct = max(
        0.0,
        float(getattr(settings, "TRAILING_SL_MIN_MOVE_PCT", 0.0002) or 0.0002),
    )
    tactical_profile = _tactical_exit_profile(side, recommended_bias)
    microvol_profile = _microvol_exit_profile(strategy_name)

    partial_r_trigger = float(getattr(settings, "PARTIAL_CLOSE_AT_R", 1.0) or 1.0)
    trail_activation_r = float(getattr(settings, "TRAILING_STOP_ACTIVATION_R", 2.5) or 2.5)
    if bool(getattr(settings, "TRAILING_ADAPTIVE_ENABLED", True)) and atr_pct is not None:
        atr_gate = float(
            getattr(settings, "TRAILING_ACTIVATION_ATR_THRESHOLD", getattr(settings, "VOL_RISK_HIGH_ATR_PCT", 0.015))
            or getattr(settings, "VOL_RISK_HIGH_ATR_PCT", 0.015)
        )
        low_vol_r = float(getattr(settings, "TRAILING_ACTIVATION_R_LOWVOL", 2.5) or 2.5)
        high_vol_r = float(getattr(settings, "TRAILING_ACTIVATION_R_HIGHVOL", 1.5) or 1.5)
        trail_activation_r = high_vol_r if atr_pct >= atr_gate else low_vol_r
    if (
        bool(getattr(settings, "VOL_FAST_EXIT_ENABLED", False))
        and atr_pct is not None
        and atr_pct >= float(getattr(settings, "VOL_FAST_EXIT_ATR_PCT", 0.012) or 0.012)
    ):
        partial_mult = float(getattr(settings, "VOL_FAST_EXIT_PARTIAL_R_MULT", 0.80) or 0.80)
        trail_mult = float(getattr(settings, "VOL_FAST_EXIT_TRAIL_R_MULT", 0.75) or 0.75)
        partial_r_trigger *= max(0.1, min(partial_mult, 1.0))
        trail_activation_r *= max(0.1, min(trail_mult, 1.0))
    if bool(tactical_profile.get("active")):
        partial_r_trigger *= max(0.1, min(float(tactical_profile.get("partial_r_mult", 0.85) or 0.85), 1.0))
        trail_activation_r *= max(0.1, min(float(tactical_profile.get("trail_r_mult", 0.75) or 0.75), 1.0))
    if bool(microvol_profile.get("active")):
        partial_r_trigger *= max(0.1, min(float(microvol_profile.get("partial_r_mult", 0.60) or 0.60), 1.0))
        trail_activation_r *= max(0.1, min(float(microvol_profile.get("trail_r_mult", 0.45) or 0.45), 1.0))

    # Partial close at PARTIAL_CLOSE_AT_R (supports fractional positions/contracts).
    if r_multiple >= partial_r_trigger:
        total_abs = abs(_to_float(current_qty))
        if total_abs > 0:
            partial_key = f"trail:partial_done:{state_key}"
            already_done = client.get(partial_key) if client else None
            if not already_done:
                close_pct = max(
                    0.0,
                    min(float(getattr(settings, "PARTIAL_CLOSE_PCT", 0.5) or 0.5), 1.0),
                )
                min_remaining_qty = max(
                    0.0,
                    float(getattr(settings, "PARTIAL_CLOSE_MIN_REMAINING_QTY", 0.0) or 0.0),
                )
                min_qty = 0.0
                try:
                    market = adapter.client.market(adapter._map_symbol(symbol))
                    precision_mode = getattr(adapter.client, "precisionMode", None)
                    min_qty = _market_min_qty(
                        market,
                        fallback=0.0,
                        precision_mode=precision_mode,
                        last_price=last_price,
                        contract_size=contract_size,
                    )
                    min_qty = _align_min_order_qty(
                        adapter,
                        symbol,
                        min_qty,
                        market=market,
                        precision_mode=precision_mode,
                    )
                except Exception:
                    min_qty = 0.0
                min_remaining = max(min_remaining_qty, min_qty)

                close_qty = total_abs * close_pct
                max_close_qty = total_abs - min_remaining
                if max_close_qty > 0:
                    close_qty = min(close_qty, max_close_qty)
                else:
                    close_qty = 0.0

                close_qty = _normalize_order_qty(adapter, symbol, close_qty)
                if min_qty > 0 and close_qty < min_qty:
                    close_qty = 0.0
                if close_qty >= total_abs:
                    keep_qty = min_remaining if min_remaining > 0 else 0.0
                    close_qty = _normalize_order_qty(
                        adapter,
                        symbol,
                        max(total_abs - keep_qty, 0.0),
                    )

                if close_qty > 0 and close_qty < total_abs:
                    close_side = "sell" if is_long else "buy"
                    try:
                        adapter.create_order(symbol, close_side, "market", close_qty, params={"reduceOnly": True})
                        logger.info("Partial close %s qty=%s at R=%.2f", symbol, close_qty, r_multiple)
                        if client:
                            client.set(partial_key, "1", ex=trail_state_ttl)
                    except Exception as exc:
                        logger.warning("Partial close failed %s: %s", symbol, exc)

    # Track favorable/adverse excursions in Redis so exits can be measured ex-post.
    max_fav = max(pnl_pct, 0.0)
    max_adv = max(-pnl_pct, 0.0)
    if client:
        hwm_key = f"trail:max_fav:{state_key}"
        adv_key = f"trail:max_adv:{state_key}"
        sl_key = f"trail:sl_pct:{state_key}"
        try:
            prev = client.get(hwm_key)
            if prev is not None:
                try:
                    max_fav = max(max_fav, float(prev))
                except Exception:
                    pass
            prev_adv = client.get(adv_key)
            if prev_adv is not None:
                try:
                    max_adv = max(max_adv, float(prev_adv))
                except Exception:
                    pass
            client.set(hwm_key, str(max_fav), ex=trail_state_ttl)
            client.set(adv_key, str(max_adv), ex=trail_state_ttl)
            client.set(sl_key, str(max(0.0, float(sl_pct))), ex=trail_state_ttl)
        except Exception:
            pass

    max_r = (max_fav / sl_pct) if sl_pct > 0 else 0.0

    # Breakeven SL: once we have X R in favor, move exchange SL to entry (or slight buffer).
    if getattr(settings, "BREAKEVEN_STOP_ENABLED", True) and max_fav > 0:
        be_window_min = int(getattr(settings, "BREAKEVEN_WINDOW_MINUTES", 0) or 0)
        be_age_min = None
        if opened_at:
            try:
                be_age_min = (dj_tz.now() - opened_at).total_seconds() / 60.0
            except Exception:
                be_age_min = None
        be_allowed = True
        if be_window_min > 0:
            be_allowed = False
            if be_age_min is not None and be_age_min <= be_window_min:
                be_allowed = True

        if be_allowed:
            be_at_r = float(getattr(settings, "BREAKEVEN_STOP_AT_R", 1.0) or 0.0)
            if bool(tactical_profile.get("active")):
                be_at_r *= max(0.1, min(float(tactical_profile.get("breakeven_r_mult", 0.75) or 0.75), 1.0))
            if bool(microvol_profile.get("active")):
                be_at_r *= max(0.1, min(float(microvol_profile.get("breakeven_r_mult", 0.50) or 0.50), 1.0))
            if be_at_r > 0 and max_r >= be_at_r:
                be_offset = float(getattr(settings, "BREAKEVEN_STOP_OFFSET_PCT", 0.0) or 0.0)
                be_offset = max(0.0, be_offset)
                be_price = entry_price * (1 + be_offset) if is_long else entry_price * (1 - be_offset)

                # Basic sanity: ensure BE stop is on the correct side of the market to avoid immediate triggers.
                if (is_long and be_price < last_price) or ((not is_long) and be_price > last_price):
                    try:
                        exists, current_stop_price, stop_orders = _has_sl_stop_order(adapter, symbol, side)
                        cur = float(current_stop_price or 0.0)
                        min_move_pct = sl_min_move_pct

                        if not exists:
                            placed = _place_sl_order(adapter, symbol, side, abs(current_qty), be_price)
                            if placed is not None:
                                _remember_protective_stop_price(symbol, opened_at, entry_price, be_price)
                            logger.info(
                                "BE SL placed %s side=%s stop=%.4f (R=%.2f age=%s window=%s)",
                                symbol,
                                side,
                                be_price,
                                max_r,
                                f"{be_age_min:.1f}min" if be_age_min is not None else "n/a",
                                be_window_min if be_window_min > 0 else "off",
                            )
                        elif cur > 0 and stop_orders:
                            should_update = False
                            if is_long:
                                should_update = be_price > cur * (1 + min_move_pct)
                            else:
                                should_update = be_price < cur * (1 - min_move_pct)
                            if should_update:
                                _cancel_stop_orders(adapter, symbol, stop_orders)
                                placed = _place_sl_order(adapter, symbol, side, abs(current_qty), be_price)
                                if placed is not None:
                                    _remember_protective_stop_price(symbol, opened_at, entry_price, be_price)
                                logger.info(
                                    "BE SL updated %s side=%s stop=%.4f prev=%.4f (R=%.2f age=%s window=%s)",
                                    symbol, side, be_price, cur, max_r,
                                    f"{be_age_min:.1f}min" if be_age_min is not None else "n/a",
                                    be_window_min if be_window_min > 0 else "off",
                                )
                            elif cur > 0:
                                _remember_protective_stop_price(symbol, opened_at, entry_price, cur)
                    except Exception as exc:
                        logger.debug("BE SL update failed for %s: %s", symbol, exc)

    # Trailing activation gate (based on HWM, not current price)
    if max_fav <= 0:
        return False, 0.0
    if max_r < trail_activation_r:
        return False, 0.0

    lock_in = float(getattr(settings, "TRAILING_STOP_LOCK_IN_PCT", 0.5) or 0.5)
    if bool(getattr(settings, "TRAILING_ADAPTIVE_ENABLED", True)) and atr_pct is not None:
        lock_min = float(getattr(settings, "TRAILING_LOCKIN_MIN", 0.4) or 0.4)
        lock_max = float(getattr(settings, "TRAILING_LOCKIN_MAX", 0.7) or 0.7)
        lock_slope = float(getattr(settings, "TRAILING_LOCKIN_SLOPE", 15.0) or 15.0)
        dyn_lock = lock_min + (atr_pct * lock_slope)
        lock_in = max(lock_min, min(lock_max, dyn_lock))
    if bool(tactical_profile.get("active")):
        lock_in = max(0.0, min(1.0, lock_in * float(tactical_profile.get("lockin_mult", 1.15) or 1.15)))
    if bool(microvol_profile.get("active")):
        lock_in = max(0.0, min(1.0, lock_in * float(microvol_profile.get("lockin_mult", 1.25) or 1.25)))
    trail_sl = _trail_sl_from_hwm(entry_price, max_fav, lock_in, is_long)

    # If trailing already violated, force-close as fallback (exchange SL should also trigger).
    if (is_long and last_price <= trail_sl) or ((not is_long) and last_price >= trail_sl):
        close_side = "sell" if is_long else "buy"
        close_qty = abs(current_qty)
        try:
            exists, _, stop_orders = _has_sl_stop_order(adapter, symbol, side)
            if stop_orders:
                _cancel_stop_orders(adapter, symbol, stop_orders)
        except Exception:
            pass
        try:
            close_resp = adapter.create_order(symbol, close_side, "market", close_qty, params={"reduceOnly": True})
            close_fee = _resolve_order_fee_usdt(
                close_resp,
                _trade_notional_usdt(close_qty, last_price, contract_size),
            )
            logger.info(
                "Trailing stop hit %s (%s) trail_sl=%.4f last=%.4f hwm=%.4f%% lock=%.2f",
                symbol,
                "long" if is_long else "short",
                trail_sl,
                last_price,
                max_fav * 100,
                max(0.0, min(lock_in, 1.0)),
            )
            return True, close_fee
        except Exception as exc:
            if _is_no_position_error(exc):
                logger.info("Trailing close skipped %s: no open position on exchange (%s)", symbol, exc)
                return False, 0.0
            logger.warning("Trailing stop close failed %s: %s", symbol, exc)
            return False, 0.0

    # Tighten exchange-side SL to the trailing level when it's more protective.
    # This makes the bot resilient if the worker is slow/offline.
    try:
        exists, current_stop_price, stop_orders = _has_sl_stop_order(adapter, symbol, side)
        cur = float(current_stop_price or 0.0)
        min_move_pct = sl_min_move_pct

        if not exists:
            placed = _place_sl_order(adapter, symbol, side, abs(current_qty), trail_sl)
            if placed is not None:
                _remember_protective_stop_price(symbol, opened_at, entry_price, trail_sl)
            logger.info("Trailing SL placed %s side=%s stop=%.4f (hwm=%.4f%%)", symbol, side, trail_sl, max_fav * 100)
        elif cur > 0 and stop_orders:
            should_update = False
            if is_long:
                should_update = trail_sl > cur * (1 + min_move_pct) and trail_sl < last_price
            else:
                should_update = trail_sl < cur * (1 - min_move_pct) and trail_sl > last_price
            if should_update:
                _cancel_stop_orders(adapter, symbol, stop_orders)
                placed = _place_sl_order(adapter, symbol, side, abs(current_qty), trail_sl)
                if placed is not None:
                    _remember_protective_stop_price(symbol, opened_at, entry_price, trail_sl)
                logger.info(
                    "Trailing SL updated %s side=%s stop=%.4f prev=%.4f (hwm=%.4f%%)",
                    symbol, side, trail_sl, cur, max_fav * 100,
                )
            else:
                _remember_protective_stop_price(symbol, opened_at, entry_price, cur)
    except Exception as exc:
        logger.debug("Trailing SL update failed for %s: %s", symbol, exc)
        return False, 0.0

    return False, 0.0

@shared_task
def _log_operation(
    inst: Instrument,
    side: str,
    qty: float,
    entry_price: float,
    exit_price: float,
    reason: str,
    signal_id: str,
    correlation_id: str,
    leverage: float,
    equity_before: float | None = None,
    equity_after: float | None = None,
    fee_usdt: float = 0.0,
    opened_at=None,
    contract_size: float = 1.0,
    close_sub_reason: str = "",
):
    # contract_size converts exchange contracts to real asset units
    # e.g. KuCoin BTC = 0.001 BTC per contract
    pnl_abs_gross = (exit_price - entry_price) * qty * contract_size
    # qty is positive for buy, negative for sell; adjust sign for shorts
    outcome_side = 1 if side == "buy" else -1
    pnl_abs_gross *= outcome_side
    notional = abs(qty * entry_price * contract_size)
    close_fee_val = max(0.0, _to_float(fee_usdt))
    entry_fee_val = _lookup_entry_order_fee_usdt(inst, side, correlation_id, opened_at)
    fee_val = close_fee_val + entry_fee_val
    pnl_abs = pnl_abs_gross - fee_val
    pnl_pct_gross = ((exit_price - entry_price) / entry_price) * outcome_side if entry_price else 0.0
    fee_pct = (fee_val / notional) if notional > 0 else 0.0
    pnl_pct = pnl_pct_gross - fee_pct
    margin_used = notional / leverage if leverage else notional
    outcome = OperationReport.Outcome.BE
    if pnl_abs > 0:
        outcome = OperationReport.Outcome.WIN
    elif pnl_abs < 0:
        outcome = OperationReport.Outcome.LOSS

    # Small exchange closes classified as near_breakeven should not count as operational losses.
    if (
        reason == "exchange_close"
        and str(close_sub_reason or "").strip().lower() == "near_breakeven"
        and pnl_pct < 0
        and abs(pnl_pct)
        <= float(getattr(settings, "NEAR_BREAKEVEN_LOSS_TO_BE_PCT", 0.0015) or 0.0015)
    ):
        outcome = OperationReport.Outcome.BE

    mfe_r = None
    mae_r = None
    mfe_capture_ratio = None
    state_key = _position_state_key(inst.symbol, opened_at, entry_price)
    try:
        client = _redis_client()
        max_fav = 0.0
        max_adv = 0.0
        sl_ref = 0.0
        if client is not None:
            try:
                raw_fav = client.get(f"trail:max_fav:{state_key}")
                raw_adv = client.get(f"trail:max_adv:{state_key}")
                raw_sl = client.get(f"trail:sl_pct:{state_key}")
                max_fav = max(0.0, _to_float(raw_fav))
                max_adv = max(0.0, _to_float(raw_adv))
                sl_ref = max(0.0, _to_float(raw_sl))
            except Exception:
                max_fav = 0.0
                max_adv = 0.0
                sl_ref = 0.0
        if sl_ref <= 0:
            sl_ref = max(
                _to_float(getattr(settings, "STOP_LOSS_PCT", 0.0)),
                _to_float(getattr(settings, "MIN_SL_PCT", 0.0)),
                1e-6,
            )
        if sl_ref > 0:
            mfe_r = max_fav / sl_ref if max_fav > 0 else 0.0
            mae_r = max_adv / sl_ref if max_adv > 0 else 0.0
            realized_r = pnl_pct / sl_ref
            if mfe_r and mfe_r > 0:
                mfe_capture_ratio = realized_r / mfe_r
    except Exception:
        mfe_r = None
        mae_r = None
        mfe_capture_ratio = None

    regime_snapshot = _operation_regime_snapshot(inst)

    existing_op = None
    identity_qs = OperationReport.objects.filter(
        instrument=inst,
        side=side,
        mode=settings.MODE,
    )
    if correlation_id:
        identity_qs = identity_qs.filter(correlation_id=correlation_id)
    elif signal_id:
        identity_qs = identity_qs.filter(signal_id=str(signal_id or ""))
    if opened_at is not None:
        identity_qs = identity_qs.filter(opened_at=opened_at)
    else:
        identity_qs = identity_qs.filter(opened_at__isnull=True)
    existing_op = identity_qs.order_by("-closed_at", "-id").first()

    incoming_priority = _operation_reason_priority(reason, close_sub_reason)
    existing_priority = (
        _operation_reason_priority(existing_op.reason, existing_op.close_sub_reason)
        if existing_op is not None else -1
    )

    if existing_op is not None and incoming_priority < existing_priority:
        logger.info(
            "Skipping lower-priority OperationReport update for %s corr=%s existing=%s:%s incoming=%s:%s",
            inst.symbol,
            correlation_id or "",
            existing_op.reason,
            existing_op.close_sub_reason,
            reason,
            close_sub_reason,
        )
        return existing_op

    report_payload = {
        "qty": qty,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_abs": pnl_abs,
        "pnl_pct": pnl_pct,
        "notional_usdt": notional,
        "margin_used_usdt": margin_used,
        "fee_usdt": fee_val,
        "leverage": leverage or 0,
        "equity_before": equity_before,
        "equity_after": equity_after,
        "mode": settings.MODE,
        "opened_at": opened_at,
        "outcome": outcome,
        "reason": reason,
        "close_sub_reason": close_sub_reason,
        "signal_id": str(signal_id or ""),
        "correlation_id": correlation_id or "",
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "mfe_capture_ratio": mfe_capture_ratio,
        "monthly_regime": str(regime_snapshot.get("monthly_regime", "") or ""),
        "weekly_regime": str(regime_snapshot.get("weekly_regime", "") or ""),
        "daily_regime": str(regime_snapshot.get("daily_regime", "") or ""),
        "btc_lead_state": str(regime_snapshot.get("btc_lead_state", "") or ""),
        "recommended_bias": str(regime_snapshot.get("recommended_bias", "") or ""),
        "closed_at": dj_tz.now(),
    }

    if existing_op is None:
        OperationReport.objects.create(
            instrument=inst,
            side=side,
            **report_payload,
        )
    else:
        for field_name, value in report_payload.items():
            setattr(existing_op, field_name, value)
        existing_op.save()
    try:
        client = _redis_client()
        if client is not None:
            for key in (
                f"trail:max_fav:{state_key}",
                f"trail:max_adv:{state_key}",
                f"trail:sl_pct:{state_key}",
                _trail_stop_price_key(state_key),
                f"trail:partial_done:{state_key}",
            ):
                client.delete(key)
    except Exception:
        pass
    _queue_ml_retrain_after_operation(inst.symbol, settings.MODE, reason)


def _operation_reason_priority(reason_text: str, sub_reason_text: str) -> int:
    reason_norm = str(reason_text or "").strip().lower()
    sub_reason_norm = str(sub_reason_text or "").strip().lower()
    if reason_norm == "exchange_close":
        if sub_reason_norm == "unknown":
            return 10
        if sub_reason_norm == "near_breakeven":
            return 20
        if sub_reason_norm in {"exchange_stop", "exchange_tp_limit", "likely_liquidation"}:
            return 30
        if sub_reason_norm == "bot_close_missed":
            return 5
        return 15
    if reason_norm in {
        "tp",
        "sl",
        "trailing_stop",
        "signal_flip",
        "stale_cleanup",
        "microvol_timeout",
        "uptrend_short_kill",
        "downtrend_long_kill",
        "tp_progress_exit",
        "ai_tp_early_exit",
    }:
        return 100
    return 80


def _log_balance(equity: float, free: float, eff_lev: float, notional: float, note: str = ""):
    try:
        BalanceSnapshot.objects.create(
            equity_usdt=equity,
            free_usdt=free,
            notional_usdt=notional,
            eff_leverage=eff_lev,
            note=note,
        )
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to log balance snapshot: %s", exc)


def _mark_position_closed(inst: Instrument):
    """
    Best-effort local DB update after we close a position via the bot.
    This prevents the next _sync_positions() pass from logging a duplicate
    "exchange_close" operation for the same instrument.
    """
    try:
        Position.objects.filter(instrument=inst, is_open=True).update(
            is_open=False,
            qty=0,
            notional_usdt=0,
            margin_used_usdt=0,
            unrealized_pnl=0,
            pnl_pct=0,
        )
    except Exception:
        pass


def _classify_exchange_close(
    adapter,
    symbol: str,
    pos_side: str,
    entry_price: float,
    liq_price_est: float,
    exit_price: float = 0.0,
    sl_pct_hint: float | None = None,
    tp_pct_hint: float | None = None,
    protective_stop_price_hint: float | None = None,
) -> str:
    """
    Best-effort classification of WHY a position disappeared from the exchange.
    Returns one of: 'exchange_stop', 'likely_liquidation', 'exchange_tp_limit',
    'bot_close_missed', 'near_breakeven', 'unknown'.

    Uses a 3-layer approach:
      1. Check DB for recent bot-initiated close (race-condition dedup).
      2. Check exchange order history for stop/limit fills.
      3. PnL-based heuristic: match loss vs SL range, gain vs TP range, etc.
    """
    recent_bot_close_minutes = max(
        1,
        int(getattr(settings, "EXCHANGE_CLOSE_RECENT_BOT_CLOSE_MINUTES", 5) or 5),
    )
    try:
        # 1. Check if a recent OperationReport (non-exchange_close) already covers this
        recent_bot_close = OperationReport.objects.filter(
            instrument__symbol__iexact=symbol.replace("/USDT:USDT", "USDT"),
            closed_at__gte=dj_tz.now() - timedelta(minutes=recent_bot_close_minutes),
        ).exclude(reason="exchange_close").exists()
        if recent_bot_close:
            return "bot_close_missed"
    except Exception:
        pass

    try:
        # 2. Check recently closed orders on the exchange
        since_ms = int((dj_tz.now() - timedelta(hours=1)).timestamp() * 1000)
        closed_orders = adapter.fetch_closed_orders(symbol, since=since_ms, limit=10)
        for order in reversed(closed_orders or []):
            order_type = (order.get("type") or "").lower()
            order_info = order.get("info") or {}
            order_stop = order_info.get("stop") or order_info.get("stopOrderType") or ""
            order_status = (order.get("status") or "").lower()
            if order_status not in ("closed", "filled"):
                continue
            # KuCoin stop order types: stop, stop_market, etc.
            if order_stop or "stop" in order_type:
                return "exchange_stop"
            if order_type == "limit":
                return "exchange_tp_limit"
    except Exception as exc:
        logger.debug("_classify_exchange_close: fetch_closed_orders failed for %s: %s", symbol, exc)

    # 3. Check if price was near liquidation estimate
    if liq_price_est > 0 and entry_price > 0:
        try:
            ticker = adapter.fetch_ticker(symbol)
            last_price = _to_float(ticker.get("last") or 0)
            if last_price > 0:
                dist_to_liq = abs(last_price - liq_price_est) / entry_price
                if dist_to_liq < 0.005:  # within 0.5% of liq price
                    return "likely_liquidation"
        except Exception:
            pass

    # 4. PnL-based heuristic fallback (when exchange API didn't give a clear answer)
    if entry_price > 0 and exit_price > 0:
        side_sign = 1 if pos_side == "buy" else -1
        pnl_pct = ((exit_price - entry_price) / entry_price) * side_sign

        sl_ref = max(
            _to_float(sl_pct_hint),
            _to_float(getattr(settings, "STOP_LOSS_PCT", 0.0)),
            _to_float(getattr(settings, "MIN_SL_PCT", 0.0)),
            1e-6,
        )
        tp_ref = max(
            _to_float(tp_pct_hint),
            _to_float(getattr(settings, "TAKE_PROFIT_PCT", sl_ref)),
            sl_ref,
        )
        stop_scale = max(
            0.05,
            _to_float(getattr(settings, "EXCHANGE_CLOSE_CLASSIFY_STOP_SCALE", 0.35)),
        )
        tp_scale = max(
            0.05,
            _to_float(getattr(settings, "EXCHANGE_CLOSE_CLASSIFY_TP_SCALE", 0.35)),
        )
        min_band = max(
            0.0,
            _to_float(getattr(settings, "EXCHANGE_CLOSE_CLASSIFY_MIN_BAND_PCT", 0.0015)),
        )
        breakeven_scale = max(
            0.0,
            _to_float(getattr(settings, "EXCHANGE_CLOSE_CLASSIFY_BREAKEVEN_SCALE", 0.20)),
        )

        stop_trigger = -max(min_band, sl_ref * stop_scale)
        tp_trigger = max(min_band, tp_ref * tp_scale)
        breakeven_band = max(min_band, sl_ref * breakeven_scale)

        protective_stop = max(0.0, _to_float(protective_stop_price_hint))
        if protective_stop > 0:
            stop_gap_pct = abs(protective_stop - entry_price) / entry_price
            stop_fill_band_pct = max(min_band, stop_gap_pct * max(stop_scale, 0.25))
            if pos_side == "buy":
                hit_protective_stop = exit_price <= protective_stop * (1 + stop_fill_band_pct)
            else:
                hit_protective_stop = exit_price >= protective_stop * (1 - stop_fill_band_pct)
            if hit_protective_stop:
                if -breakeven_band <= pnl_pct <= breakeven_band:
                    return "near_breakeven"
                return "exchange_stop"

        if pnl_pct <= stop_trigger:
            return "exchange_stop"
        if pnl_pct >= tp_trigger:
            return "exchange_tp_limit"
        if -breakeven_band <= pnl_pct <= breakeven_band:
            return "near_breakeven"
        return "unknown"

    return "unknown"


def _sync_positions(
    adapter,
    positions: list | None = None,
    balance_assets: list[str] | None = None,
):
    """
    Pull open positions from the exchange and reflect them in Position model.
    Runs each cycle, even if no new signals, so admin always shows live state.
    """
    equity_total = 0.0
    candidate_assets = balance_assets or list(
        get_runtime_exchange_context().get("balance_assets") or ["USDT"]
    )
    try:
        bal = adapter.fetch_balance()
        _, equity_total, _ = extract_balance_values(bal, candidate_assets)
    except Exception:
        equity_total = 0.0
    if positions is None:
        try:
            positions = adapter.fetch_positions()
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not sync positions: %s", exc)
            return

    inst_map = {}
    for inst in Instrument.objects.all():
        sym = inst.symbol
        inst_map[_norm_symbol(sym)] = inst
        if sym.endswith("USDT"):
            base = sym[:-4]
            inst_map[_norm_symbol(f"{base}/USDT:USDT")] = inst
            inst_map[_norm_symbol(f"{base}/USDT")] = inst

    touched = set()

    for pos in positions or []:
        qty = _to_float(pos.get("contracts") or pos.get("size") or 0)
        if qty == 0:
            continue
        side_val = (pos.get("side") or pos.get("positionSide") or "").lower()
        if side_val == "short":
            qty = -qty
        sym = pos.get("symbol")
        inst = inst_map.get(_norm_symbol(sym))
        if not inst:
            continue
        mark = _to_float(pos.get("markPrice") or pos.get("info", {}).get("markPrice") or 0)
        entry = _to_float(pos.get("entryPrice") or pos.get("averagePrice") or 0)
        contract_sz = _to_float(pos.get("contractSize") or pos.get("info", {}).get("multiplier") or 1.0)
        notional = abs(qty * mark * (contract_sz if contract_sz else 1.0))
        lev = _to_float(pos.get("leverage") or pos.get("info", {}).get("leverage") or 0)
        margin = notional / lev if lev else notional
        u_pnl = _to_float(pos.get("unrealisedPnl") or pos.get("unrealizedPnl") or pos.get("info", {}).get("unrealisedPnl") or 0)
        pnl_pct = ((mark - entry) / entry) * (1 if qty > 0 else -1) if entry else 0
        opened_ms = pos.get("openingTimestamp")
        opened_at = datetime.fromtimestamp(opened_ms / 1000, tz=timezone.utc) if opened_ms else None

        # Liquidation price estimate (cross): solve equity + qty*cs*(liq-entry) = MM
        mmr = _to_float(pos.get("maintenanceMarginRate") or pos.get("info", {}).get("maintenanceMarginRate") or 0)
        if mmr == 0:
            try:
                market = adapter.client.market(adapter._map_symbol(sym))  # type: ignore[attr-defined]
                mmr = _to_float(market.get("maintenanceMarginRate"))
            except Exception:
                mmr = 0.005
        maint_extra = _to_float(pos.get("maintenanceMargin") or pos.get("info", {}).get("maintenanceMargin") or 0)
        mm = notional * (mmr if mmr else 0.005) + maint_extra
        liq_est = 0.0
        denom = qty * (contract_sz if contract_sz else 1.0)
        if denom != 0 and entry:
            liq_est = entry + (mm - (equity_total if equity_total else margin)) / denom
            if liq_est < 0:
                liq_est = 0.0

        Position.objects.update_or_create(
            instrument=inst,
            defaults={
                "qty": Decimal(str(qty)),
                "avg_price": Decimal(str(entry or 0)),
                "last_price": Decimal(str(mark or 0)),
                "notional_usdt": Decimal(str(notional)),
                "margin_used_usdt": Decimal(str(margin)),
                "unrealized_pnl": Decimal(str(u_pnl)),
                "pnl_pct": Decimal(str(pnl_pct)),
                "leverage_eff": Decimal(str(lev)),
                "side": Position.Side.LONG if qty >= 0 else Position.Side.SHORT,
                "is_open": True,
                "mode": settings.MODE,
                "opened_at": opened_at,
                "liq_price_est": Decimal(str(liq_est)),
            },
        )
        touched.add(inst.id)

    cleanup_candidates = list(
        Instrument.objects.filter(enabled=True).exclude(id__in=touched).values_list("symbol", flat=True)
    )
    for symbol in cleanup_candidates:
        _cleanup_orphan_reduce_only_orders(adapter, symbol)

    # Mark others as closed ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â detect transitions and notify
    newly_closed = Position.objects.filter(is_open=True).exclude(instrument_id__in=touched)
    for pos in newly_closed:
        try:
            entry = _to_float(pos.avg_price)
            last = _to_float(pos.last_price) or entry
            qty_abs = abs(_to_float(pos.qty))
            trade_side = "buy" if _to_float(pos.qty) > 0 else "sell"
            pnl_pct_val = _to_float(pos.pnl_pct)
            leverage_val = _to_float(pos.leverage_eff)

            # Best-effort: attach correlation_id + opened_at from the last entry Order.
            # Many exchanges don't return openingTimestamp reliably, so DB Position.opened_at
            # may be null; using Order makes post-trade analysis (MFE/MAE) much easier.
            corr_id = ""
            sig_id = ""
            opened_at = pos.opened_at
            try:
                entry_order = (
                    Order.objects.filter(
                        instrument=pos.instrument,
                        side=trade_side,
                        status=Order.OrderStatus.FILLED,
                        opened_at__isnull=False,
                    )
                    .order_by("-opened_at")
                    .first()
                )
                if entry_order and entry_order.opened_at and (dj_tz.now() - entry_order.opened_at) <= timedelta(hours=48):
                    opened_at = opened_at or entry_order.opened_at
                    corr_id = entry_order.correlation_id or ""
                    head = corr_id.split("-", 1)[0] if corr_id else ""
                    if head.isdigit():
                        sig_id = head
            except Exception:
                pass

            strategy_name_for_notify = ""
            active_modules_for_notify: list[str] = []
            if sig_id and str(sig_id).isdigit():
                try:
                    sig_obj = Signal.objects.filter(id=int(sig_id)).only("strategy", "payload_json").first()
                    if sig_obj:
                        strategy_name_for_notify = str(sig_obj.strategy or "")
                        payload_for_notify = sig_obj.payload_json if isinstance(sig_obj.payload_json, dict) else {}
                        active_modules_for_notify = _signal_active_modules(
                            payload_for_notify,
                            strategy_name_for_notify,
                        )
                except Exception:
                    pass

            # Resolve contract_size for this instrument
            cs = 1.0
            sym = pos.instrument.symbol
            for sym_try in (sym, f"{sym[:-4]}/USDT:USDT" if sym.endswith("USDT") else None):
                if not sym_try:
                    continue
                try:
                    mkt = adapter.client.market(adapter._map_symbol(sym_try))
                    cs = _to_float(mkt.get("contractSize") or 1.0)
                    break
                except Exception:
                    continue

            # Compute PnL absoluto with contract_size
            pnl_abs_val = (last - entry) * qty_abs * cs * (1 if trade_side == "buy" else -1) if entry else 0.0
            # Duration
            dur_min = 0.0
            if opened_at:
                dur_min = (dj_tz.now() - opened_at).total_seconds() / 60

            # Dedup: if we already logged a close for this instrument very recently,
            # don't emit a second "exchange_close" report.
            recent_close_minutes = max(
                1,
                int(getattr(settings, "EXCHANGE_CLOSE_DEDUP_MINUTES", 3) or 3),
            )
            recent_close = OperationReport.objects.filter(
                instrument=pos.instrument,
                side=trade_side,
                closed_at__gte=dj_tz.now() - timedelta(minutes=recent_close_minutes),
            ).exists()
            if recent_close:
                logger.info("Skipping exchange_close log for %s: recent OperationReport exists", pos.instrument.symbol)
                continue

            # Classify sub-reason for exchange_close
            liq_est = _to_float(pos.liq_price_est) if hasattr(pos, "liq_price_est") else 0.0
            atr_for_close = _atr_pct(pos.instrument)
            _, _, tp_pct_hint, sl_pct_hint = _compute_tp_sl_prices(
                trade_side,
                entry,
                atr_pct=atr_for_close,
            )
            protective_stop_price_hint = _load_protective_stop_price(sym, opened_at, entry)
            sub_reason = _classify_exchange_close(
                adapter,
                sym,
                trade_side,
                entry,
                liq_est,
                exit_price=last,
                sl_pct_hint=sl_pct_hint,
                tp_pct_hint=tp_pct_hint,
                protective_stop_price_hint=protective_stop_price_hint,
            )
            logger.info(
                "exchange_close sub-reason for %s: %s (entry=%.4f exit=%.4f liq_est=%.4f stop_hint=%.4f)",
                sym, sub_reason, entry, last, liq_est, protective_stop_price_hint,
            )

            if sub_reason == "bot_close_missed":
                logger.info("Skipping duplicate exchange_close log for %s (bot_close_missed)", sym)
                pos.is_open = False
                pos.save()
                continue

            # Log operation
            _log_operation(
                pos.instrument, trade_side, qty_abs, entry, last,
                reason="exchange_close",
                signal_id=sig_id,
                correlation_id=corr_id,
                leverage=leverage_val,
                equity_before=equity_total or None,
                equity_after=None,
                fee_usdt=0.0,
                opened_at=opened_at,
                contract_size=cs,
                close_sub_reason=sub_reason,
            )
            notify_trade_closed(
                pos.instrument.symbol, "exchange_close", pnl_pct_val,
                pnl_abs=pnl_abs_val, entry_price=entry,
                exit_price=last, qty=qty_abs,
                equity_before=equity_total,
                duration_min=dur_min,
                side=trade_side, leverage=leverage_val,
                strategy_name=strategy_name_for_notify,
                active_modules=active_modules_for_notify,
            )
            logger.info(
                "Position closed on exchange: %s side=%s pnl=%.4f pnl_usdt=%.4f cs=%.4f",
                pos.instrument.symbol, trade_side, pnl_pct_val, pnl_abs_val, cs,
            )
        except Exception as exc:
            logger.warning("Failed to notify closed position %s: %s", pos.instrument.symbol, exc)

    newly_closed.update(
        is_open=False,
        qty=0,
        notional_usdt=0,
        margin_used_usdt=0,
        unrealized_pnl=0,
        pnl_pct=0,
    )


@shared_task
def retrain_entry_filter_model():
    """Retrain global + per-symbol ML entry filter models and apply thresholds via Redis."""
    if not bool(getattr(settings, "ML_ENTRY_FILTER_AUTO_TRAIN_ENABLED", False)):
        return "ml-entry-filter:auto-train-disabled"

    days = max(1, int(getattr(settings, "ML_ENTRY_FILTER_TRAIN_DAYS", 21) or 21))
    train_source = str(getattr(settings, "ML_ENTRY_FILTER_TRAIN_SOURCE", "mixed") or "mixed").strip().lower()
    if train_source not in {"live", "backtest", "mixed"}:
        train_source = "mixed"
    backtest_days = max(1, int(getattr(settings, "ML_ENTRY_FILTER_TRAIN_BACKTEST_DAYS", 180) or 180))
    min_samples = max(20, int(getattr(settings, "ML_ENTRY_FILTER_TRAIN_MIN_SAMPLES", 120) or 120))
    per_symbol_min_samples = max(
        6,
        int(getattr(settings, "ML_ENTRY_FILTER_PER_SYMBOL_MIN_SAMPLES", 50) or 50),
    )
    per_strategy_min_samples = max(
        6,
        int(getattr(settings, "ML_ENTRY_FILTER_PER_STRATEGY_MIN_SAMPLES", 80) or 80),
    )
    per_symbol_strategy_min_samples = max(
        6,
        int(getattr(settings, "ML_ENTRY_FILTER_PER_SYMBOL_STRATEGY_MIN_SAMPLES", 40) or 40),
    )
    epochs = max(100, int(getattr(settings, "ML_ENTRY_FILTER_TRAIN_EPOCHS", 1200) or 1200))
    learning_rate = max(1e-5, float(getattr(settings, "ML_ENTRY_FILTER_TRAIN_LR", 0.05) or 0.05))
    l2 = max(0.0, float(getattr(settings, "ML_ENTRY_FILTER_TRAIN_L2", 0.001) or 0.001))
    model_path = str(getattr(settings, "ML_ENTRY_FILTER_MODEL_PATH", "") or "").strip()
    if not model_path:
        logger.warning("ML entry filter retrain skipped: empty model path")
        return "ml-entry-filter:no-model-path"

    cmd_output = StringIO()
    try:
        call_command(
            "train_entry_filter_ml",
            source=train_source,
            days=days,
            backtest_days=backtest_days,
            min_samples=min_samples,
            epochs=epochs,
            lr=learning_rate,
            l2=l2,
            output=model_path,
            stdout=cmd_output,
        )
    except Exception as exc:
        logger.warning("ML entry filter retrain failed: %s", exc)
        return f"ml-entry-filter:train-failed:{exc}"

    global_threshold = max(
        0.0,
        min(float(getattr(settings, "ML_ENTRY_FILTER_MIN_PROB", 0.52) or 0.52), 1.0),
    )
    try:
        model = load_model(model_path)
        global_threshold = max(
            0.0,
            min(
                float(
                    (model.get("threshold_analysis") or {}).get(
                        "recommended_threshold",
                        global_threshold,
                    )
                ),
                1.0,
            ),
        )
    except Exception as exc:
        logger.warning("ML entry filter retrain: could not read trained model: %s", exc)

    client = _redis_client()
    applied_global_threshold = None
    applied_symbol_thresholds: dict[str, float] = {}
    trained_symbol_thresholds: dict[str, float] = {}
    applied_strategy_thresholds: dict[str, float] = {}
    trained_strategy_thresholds: dict[str, float] = {}
    applied_symbol_strategy_thresholds: dict[str, float] = {}
    trained_symbol_strategy_thresholds: dict[str, float] = {}
    if bool(getattr(settings, "ML_ENTRY_FILTER_AUTO_APPLY_THRESHOLD", True)):
        if client is not None:
            try:
                client.set("ml:entry_filter:min_prob", global_threshold)
                client.set("ml:entry_filter:last_retrain_at", dj_tz.now().isoformat())
                applied_global_threshold = global_threshold
            except Exception as exc:
                logger.warning("ML entry filter threshold apply failed: %s", exc)

    per_symbol_enabled = bool(getattr(settings, "ML_ENTRY_FILTER_PER_SYMBOL_ENABLED", False))
    per_symbol_list = [str(s).upper() for s in getattr(settings, "ML_ENTRY_FILTER_PER_SYMBOLS", []) if str(s).strip()]
    per_strategy_enabled = bool(getattr(settings, "ML_ENTRY_FILTER_PER_STRATEGY_ENABLED", False))
    per_strategy_list = [
        str(s).strip().lower()
        for s in getattr(settings, "ML_ENTRY_FILTER_PER_STRATEGIES", [])
        if str(s).strip()
    ]
    per_symbol_dir = str(getattr(settings, "ML_ENTRY_FILTER_MODEL_DIR", "") or "").strip()
    if per_symbol_enabled and per_symbol_list and per_symbol_dir:
        Path(per_symbol_dir).mkdir(parents=True, exist_ok=True)
        for symbol in per_symbol_list:
            symbol_output = str(Path(per_symbol_dir) / f"entry_filter_model_{symbol}.json")
            symbol_cmd_output = StringIO()
            try:
                call_command(
                    "train_entry_filter_ml",
                    source=train_source,
                    days=days,
                    backtest_days=backtest_days,
                    min_samples=per_symbol_min_samples,
                    epochs=epochs,
                    lr=learning_rate,
                    l2=l2,
                    output=symbol_output,
                    symbols=symbol,
                    stdout=symbol_cmd_output,
                )
            except Exception as exc:
                logger.warning("ML entry filter per-symbol retrain failed for %s: %s", symbol, exc)
                continue

            symbol_threshold = global_threshold
            try:
                symbol_model = load_model(symbol_output)
                symbol_threshold = max(
                    0.0,
                    min(
                        float(
                            (symbol_model.get("threshold_analysis") or {}).get(
                                "recommended_threshold",
                                symbol_threshold,
                            )
                        ),
                        1.0,
                    ),
                )
            except Exception as exc:
                logger.warning("ML entry filter per-symbol model read failed for %s: %s", symbol, exc)
            trained_symbol_thresholds[symbol] = symbol_threshold

            if bool(getattr(settings, "ML_ENTRY_FILTER_AUTO_APPLY_THRESHOLD", True)) and client is not None:
                try:
                    client.set(f"ml:entry_filter:min_prob:{symbol}", symbol_threshold)
                    applied_symbol_thresholds[symbol] = symbol_threshold
                except Exception as exc:
                    logger.warning("ML entry filter per-symbol threshold apply failed for %s: %s", symbol, exc)

            symbol_output_text = symbol_cmd_output.getvalue().strip()
            if symbol_output_text:
                logger.info("ML entry filter per-symbol retrain output [%s]:\n%s", symbol, symbol_output_text)

    if per_strategy_enabled and per_strategy_list and per_symbol_dir:
        Path(per_symbol_dir).mkdir(parents=True, exist_ok=True)
        for strategy_name in per_strategy_list:
            strategy_tag = _ml_strategy_tag(strategy_name)
            if not strategy_tag:
                continue
            strategy_output = str(Path(per_symbol_dir) / f"entry_filter_model_strategy_{strategy_tag}.json")
            strategy_cmd_output = StringIO()
            try:
                call_command(
                    "train_entry_filter_ml",
                    source=train_source,
                    days=days,
                    backtest_days=backtest_days,
                    min_samples=per_strategy_min_samples,
                    epochs=epochs,
                    lr=learning_rate,
                    l2=l2,
                    output=strategy_output,
                    strategies=strategy_name,
                    stdout=strategy_cmd_output,
                )
            except Exception as exc:
                logger.warning(
                    "ML entry filter per-strategy retrain failed for %s: %s",
                    strategy_name,
                    exc,
                )
                continue

            strategy_threshold = global_threshold
            try:
                strategy_model = load_model(strategy_output)
                strategy_threshold = max(
                    0.0,
                    min(
                        float(
                            (strategy_model.get("threshold_analysis") or {}).get(
                                "recommended_threshold",
                                strategy_threshold,
                            )
                        ),
                        1.0,
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "ML entry filter per-strategy model read failed for %s: %s",
                    strategy_name,
                    exc,
                )
            trained_strategy_thresholds[strategy_tag] = strategy_threshold
            if bool(getattr(settings, "ML_ENTRY_FILTER_AUTO_APPLY_THRESHOLD", True)) and client is not None:
                try:
                    client.set(f"ml:entry_filter:min_prob:strategy:{strategy_tag}", strategy_threshold)
                    applied_strategy_thresholds[strategy_tag] = strategy_threshold
                except Exception as exc:
                    logger.warning(
                        "ML entry filter per-strategy threshold apply failed for %s: %s",
                        strategy_name,
                        exc,
                    )

            strategy_output_text = strategy_cmd_output.getvalue().strip()
            if strategy_output_text:
                logger.info(
                    "ML entry filter per-strategy retrain output [%s]:\n%s",
                    strategy_name,
                    strategy_output_text,
                )

            if per_symbol_enabled and per_symbol_list:
                for symbol in per_symbol_list:
                    symbol_strategy_output = str(
                        Path(per_symbol_dir) / f"entry_filter_model_{symbol}_{strategy_tag}.json"
                    )
                    symbol_strategy_cmd_output = StringIO()
                    try:
                        call_command(
                            "train_entry_filter_ml",
                            source=train_source,
                            days=days,
                            backtest_days=backtest_days,
                            min_samples=per_symbol_strategy_min_samples,
                            epochs=epochs,
                            lr=learning_rate,
                            l2=l2,
                            output=symbol_strategy_output,
                            symbols=symbol,
                            strategies=strategy_name,
                            stdout=symbol_strategy_cmd_output,
                        )
                    except Exception as exc:
                        logger.warning(
                            "ML entry filter per-symbol+strategy retrain failed for %s/%s: %s",
                            symbol,
                            strategy_name,
                            exc,
                        )
                        continue

                    symbol_strategy_threshold = strategy_threshold
                    try:
                        symbol_strategy_model = load_model(symbol_strategy_output)
                        symbol_strategy_threshold = max(
                            0.0,
                            min(
                                float(
                                    (symbol_strategy_model.get("threshold_analysis") or {}).get(
                                        "recommended_threshold",
                                        symbol_strategy_threshold,
                                    )
                                ),
                                1.0,
                            ),
                        )
                    except Exception as exc:
                        logger.warning(
                            "ML entry filter per-symbol+strategy model read failed for %s/%s: %s",
                            symbol,
                            strategy_name,
                            exc,
                        )
                    key = f"{symbol}:{strategy_tag}"
                    trained_symbol_strategy_thresholds[key] = symbol_strategy_threshold
                    if bool(getattr(settings, "ML_ENTRY_FILTER_AUTO_APPLY_THRESHOLD", True)) and client is not None:
                        try:
                            client.set(
                                f"ml:entry_filter:min_prob:{symbol}:{strategy_tag}",
                                symbol_strategy_threshold,
                            )
                            applied_symbol_strategy_thresholds[key] = symbol_strategy_threshold
                        except Exception as exc:
                            logger.warning(
                                "ML entry filter per-symbol+strategy threshold apply failed for %s/%s: %s",
                                symbol,
                                strategy_name,
                                exc,
                            )

                    symbol_strategy_output_text = symbol_strategy_cmd_output.getvalue().strip()
                    if symbol_strategy_output_text:
                        logger.info(
                            "ML entry filter per-symbol+strategy retrain output [%s/%s]:\n%s",
                            symbol,
                            strategy_name,
                            symbol_strategy_output_text,
                        )

    output_text = cmd_output.getvalue().strip()
    if output_text:
        logger.info("ML entry filter retrain output:\n%s", output_text)
    logger.info(
        "ML entry filter retrained: global=%.3f applied_global=%s per_symbol=%s per_strategy=%s per_symbol_strategy=%s model=%s",
        global_threshold,
        f"{applied_global_threshold:.3f}" if applied_global_threshold is not None else "n/a",
        {
            k: round(v, 3)
            for k, v in (applied_symbol_thresholds if applied_symbol_thresholds else trained_symbol_thresholds).items()
        },
        {
            k: round(v, 3)
            for k, v in (applied_strategy_thresholds if applied_strategy_thresholds else trained_strategy_thresholds).items()
        },
        {
            k: round(v, 3)
            for k, v in (
                applied_symbol_strategy_thresholds
                if applied_symbol_strategy_thresholds
                else trained_symbol_strategy_thresholds
            ).items()
        },
        model_path,
    )
    if applied_global_threshold is not None:
        return (
            f"ml-entry-filter:trained-and-applied:"
            f"global={applied_global_threshold:.3f}:symbols={len(applied_symbol_thresholds)}:"
            f"strategies={len(applied_strategy_thresholds)}:"
            f"symbol_strategies={len(applied_symbol_strategy_thresholds)}"
        )
    return (
        f"ml-entry-filter:trained:global={global_threshold:.3f}:"
        f"symbols={len(trained_symbol_thresholds)}:"
        f"strategies={len(trained_strategy_thresholds)}:"
        f"symbol_strategies={len(trained_symbol_strategy_thresholds)}"
    )


def _evaluate_balance_and_guardrails(
    adapter,
    runtime_ctx: dict[str, Any],
    risk_ns: str,
    positions_snapshot: list[dict[str, Any]] | list[Any] | None = None,
) -> tuple[bool, float, float, float, float, float]:
    """
    Evaluate account-level guardrails and return cycle context:
    (can_open, free_usdt, equity_usdt, eff_lev, total_notional, leverage)
    """
    can_open = True
    free_usdt = 0.0
    equity_usdt = 0.0
    eff_lev = 0.0
    total_notional = 0.0
    leverage = float(getattr(adapter, "leverage", None) or settings.MAX_EFF_LEVERAGE)
    env_label = str(runtime_ctx.get("label") or "EXCHANGE")
    balance_assets = list(runtime_ctx.get("balance_assets") or ["USDT"])
    balance_asset = str(runtime_ctx.get("primary_asset") or "USDT")

    try:
        bal = adapter.fetch_balance()
        free_usdt, equity_usdt, balance_asset = extract_balance_values(
            bal,
            balance_assets,
        )

        if free_usdt < settings.MIN_EQUITY_USDT:
            logger.warning(
                "Insufficient balance (%s): free %s=%s < MIN=%s",
                env_label,
                balance_asset,
                free_usdt,
                settings.MIN_EQUITY_USDT,
            )
            can_open = False

        allowed_dd, dd, dd_meta = _check_drawdown(equity_usdt, risk_ns=risk_ns)
        if not allowed_dd:
            logger.warning("Daily DD limit hit dd=%.4f; blocking new trades", dd)
            _client = _redis_client()
            _period_key = str(dd_meta.get("period_key") or dj_tz.now().date().isoformat())
            _ks_key = f"risk:ks_notified:daily_dd:{risk_ns}:{_period_key}"
            if (dd_meta.get("emit_event", True)) and (_client is None or _client.set(_ks_key, "1", nx=True, ex=86400)):
                _create_risk_event(
                    "daily_dd_limit",
                    "high",
                    details={"dd": dd},
                    risk_ns=risk_ns,
                )
                _mark_drawdown_event_emitted(dd_meta.get("baseline"), dd)
                notify_kill_switch(f"Daily DD limit: {dd:.4f}")
            can_open = False

        allowed_wdd, wdd, wdd_meta = _check_weekly_drawdown(equity_usdt, risk_ns=risk_ns)
        if not allowed_wdd:
            logger.warning("Weekly DD limit hit wdd=%.4f; blocking new trades", wdd)
            _client = _redis_client()
            _period_key = str(wdd_meta.get("period_key") or dj_tz.now().date().isoformat())
            _ks_key = f"risk:ks_notified:weekly_dd:{risk_ns}:{_period_key}"
            if (wdd_meta.get("emit_event", True)) and (_client is None or _client.set(_ks_key, "1", nx=True, ex=86400)):
                _create_risk_event(
                    "weekly_dd_limit",
                    "high",
                    details={"wdd": wdd},
                    risk_ns=risk_ns,
                )
                _mark_drawdown_event_emitted(wdd_meta.get("baseline"), wdd)
                notify_kill_switch(f"Weekly DD limit: {wdd:.4f}")
            can_open = False

        eff_lev, total_notional = _effective_leverage(
            adapter,
            equity_usdt,
            positions=positions_snapshot or [],
        )
        if eff_lev > settings.MAX_EFF_LEVERAGE:
            logger.warning("Eff leverage %.2f > MAX %.2f; blocking new trades", eff_lev, settings.MAX_EFF_LEVERAGE)
            can_open = False

        _log_balance(
            equity_usdt,
            free_usdt,
            eff_lev,
            total_notional,
            note=f"cycle:{runtime_ctx.get('service')}:{runtime_ctx.get('env')}:{balance_asset}",
        )
    except Exception as exc:
        logger.warning("Balance check failed (manage-only): %s", exc)
        notify_error(f"Balance check failed: {exc}")
        record_ai_feedback_event(
            event_type="balance_check_error",
            level="error",
            account_alias=str(runtime_ctx.get("account_alias") or ""),
            account_service=str(runtime_ctx.get("service") or ""),
            reason=str(exc)[:255],
            payload={
                "risk_ns": risk_ns,
                "env": str(runtime_ctx.get("env") or ""),
            },
        )
        can_open = False

    return can_open, free_usdt, equity_usdt, eff_lev, total_notional, leverage


def _apply_circuit_breaker_gate(
    can_open: bool,
    equity_usdt: float,
    risk_ns: str,
) -> bool:
    from risk.models import CircuitBreakerConfig

    try:
        cb = CircuitBreakerConfig.get()
        if cb.enabled and equity_usdt > 0:
            if equity_usdt > cb.peak_equity:
                cb.peak_equity = equity_usdt
                cb.save(update_fields=["peak_equity"])

            if cb.is_tripped:
                if cb.tripped_at and cb.cooldown_minutes_after_trigger:
                    elapsed = (dj_tz.now() - cb.tripped_at).total_seconds() / 60
                    if elapsed < cb.cooldown_minutes_after_trigger:
                        logger.warning(
                            "Circuit breaker still tripped (%.0f min left): %s",
                            cb.cooldown_minutes_after_trigger - elapsed,
                            cb.trip_reason,
                        )
                        can_open = False
                    else:
                        logger.info("Circuit breaker cooldown expired, resetting")
                        cb.reset()
                else:
                    can_open = False

            if not cb.is_tripped and equity_usdt > 0:
                day_key = dj_tz.now().date().isoformat()
                client = _redis_client()
                if client:
                    start_raw = client.get(f"risk:equity_start:{risk_ns}:{day_key}")
                    if start_raw:
                        day_start = float(start_raw)
                        daily_dd_pct = (day_start - equity_usdt) / day_start * 100 if day_start > 0 else 0
                        if daily_dd_pct >= cb.max_daily_dd_pct:
                            cb.is_tripped = True
                            cb.tripped_at = dj_tz.now()
                            cb.trip_reason = f"Daily DD {daily_dd_pct:.1f}% >= {cb.max_daily_dd_pct}%"
                            cb.save(update_fields=["is_tripped", "tripped_at", "trip_reason"])
                            _create_risk_event(
                                "circuit_breaker_daily_dd",
                                "critical",
                                details={"dd_pct": daily_dd_pct},
                                risk_ns=risk_ns,
                            )
                            notify_kill_switch(cb.trip_reason)
                            can_open = False

            if not cb.is_tripped and cb.peak_equity > 0:
                total_dd_pct = (cb.peak_equity - equity_usdt) / cb.peak_equity * 100
                if total_dd_pct >= cb.max_total_dd_pct:
                    cb.is_tripped = True
                    cb.tripped_at = dj_tz.now()
                    cb.trip_reason = (
                        f"Total DD {total_dd_pct:.1f}% >= {cb.max_total_dd_pct}% "
                        f"(peak={cb.peak_equity:.2f})"
                    )
                    cb.save(update_fields=["is_tripped", "tripped_at", "trip_reason"])
                    _create_risk_event(
                        "circuit_breaker_total_dd",
                        "critical",
                        details={"dd_pct": total_dd_pct, "peak": cb.peak_equity},
                        risk_ns=risk_ns,
                    )
                    notify_kill_switch(cb.trip_reason)
                    can_open = False

            if not cb.is_tripped and cb.max_consecutive_losses > 0:
                window_hours = max(
                    0.0,
                    float(
                        getattr(settings, "CIRCUIT_BREAKER_CONSECUTIVE_LOSS_WINDOW_HOURS", 24.0)
                        or 24.0
                    ),
                )
                recent_ops_qs = OperationReport.objects.order_by("-closed_at")
                if window_hours > 0:
                    recent_ops_qs = recent_ops_qs.filter(
                        closed_at__gte=dj_tz.now() - timedelta(hours=window_hours)
                    )
                recent_ops = list(
                    recent_ops_qs[:cb.max_consecutive_losses].values_list("outcome", flat=True)
                )
                if len(recent_ops) >= cb.max_consecutive_losses and all(o == "loss" for o in recent_ops):
                    cb.is_tripped = True
                    cb.tripped_at = dj_tz.now()
                    if window_hours > 0:
                        cb.trip_reason = (
                            f"{cb.max_consecutive_losses} consecutive losses in {window_hours:.0f}h"
                        )
                    else:
                        cb.trip_reason = f"{cb.max_consecutive_losses} consecutive losses"
                    cb.save(update_fields=["is_tripped", "tripped_at", "trip_reason"])
                    _create_risk_event(
                        "circuit_breaker_consec_losses",
                        "critical",
                        details={"count": cb.max_consecutive_losses},
                        risk_ns=risk_ns,
                    )
                    notify_kill_switch(cb.trip_reason)
                    can_open = False
    except Exception as exc:
        logger.warning("Circuit breaker check failed: %s", exc)
    return can_open


def _load_enabled_instruments_and_latest_signals(
    use_allocator_signals: bool,
) -> tuple[list[Instrument], dict[int, Signal], datetime]:
    latest_signal_qs = Signal.objects.filter(instrument_id=OuterRef("pk"))
    if use_allocator_signals:
        latest_signal_qs = latest_signal_qs.filter(
            Q(strategy__startswith="alloc_") | Q(strategy__startswith="mod_microvol_")
        )
    enabled_instruments = list(
        Instrument.objects.filter(enabled=True).annotate(
            latest_1m_ts=Subquery(
                Candle.objects.filter(
                    instrument_id=OuterRef("pk"),
                    timeframe="1m",
                )
                .order_by("-ts")
                .values("ts")[:1]
            ),
            latest_signal_id=Subquery(
                latest_signal_qs
                .order_by("-ts", "-id")
                .values("id")[:1]
            ),
        )
    )
    signal_ids = [
        inst.latest_signal_id
        for inst in enabled_instruments
        if getattr(inst, "latest_signal_id", None)
    ]
    latest_signals = Signal.objects.in_bulk(signal_ids) if signal_ids else {}
    return enabled_instruments, latest_signals, dj_tz.now()


def _regime_adx_min_for_symbol_session(
    symbol: str,
    session_name: str,
    default_min: float,
) -> float:
    """
    Resolve ADX gate threshold for a symbol/session with layered fallbacks:
      1) SYMBOL:session
      2) SYMBOL:*
      3) *:session
      4) *:*
      5) default_min
    """
    sym = str(symbol or "").strip().upper()
    session = str(session_name or "").strip().lower()
    resolved = max(0.0, float(default_min or 0.0))
    raw_overrides = getattr(settings, "MARKET_REGIME_ADX_MIN_BY_CONTEXT", {}) or {}
    if not isinstance(raw_overrides, dict):
        return resolved
    for key in (f"{sym}:{session}", f"{sym}:*", f"*:{session}", "*:*"):
        if key not in raw_overrides:
            continue
        try:
            return max(0.0, float(raw_overrides[key]))
        except Exception:
            continue
    return resolved


def _compute_regime_adx_gate(
    enabled_instruments: list[Instrument],
    current_session: str,
) -> tuple[dict[str, float], set[str], float, float | None, dict[str, str], dict[str, float]]:
    _regime_adx_by_symbol: dict[str, float] = {}
    _regime_bias_by_symbol: dict[str, str] = {}
    _regime_adx_min = float(getattr(settings, "MARKET_REGIME_ADX_MIN", 0))
    _regime_adx_min_by_symbol: dict[str, float] = {}
    try:
        from signals.modules.common import compute_adx as _compute_adx
        import pandas as _pd
        for _ri in enabled_instruments:
            try:
                _htf_rows = list(
                    Candle.objects.filter(instrument=_ri, timeframe="1h")
                    .order_by("-ts")[:100]
                    .values("ts", "open", "high", "low", "close", "volume")
                )
                if len(_htf_rows) >= 30:
                    _htf_df = _pd.DataFrame(_htf_rows).sort_values("ts")
                    for _col in ("open", "high", "low", "close", "volume"):
                        _htf_df[_col] = _htf_df[_col].astype(float)
                    _htf_df.set_index("ts", inplace=True)
                    _regime_adx_by_symbol[_ri.symbol] = _compute_adx(_htf_df, period=14)
                    if len(_htf_df) >= 50:
                        _ema20 = _htf_df["close"].ewm(span=20, adjust=False).mean().iloc[-1]
                        _ema50 = _htf_df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
                        _last = float(_htf_df["close"].iloc[-1])
                        if _ema20 > _ema50 and _last >= _ema20:
                            _regime_bias_by_symbol[_ri.symbol] = "bull"
                        elif _ema20 < _ema50 and _last <= _ema20:
                            _regime_bias_by_symbol[_ri.symbol] = "bear"
                        else:
                            _regime_bias_by_symbol[_ri.symbol] = "neutral"
                    else:
                        _regime_bias_by_symbol[_ri.symbol] = "neutral"
            except Exception:
                pass
    except Exception:
        pass
    _market_regime_adx = _regime_adx_by_symbol.get("BTCUSDT")
    if _regime_adx_by_symbol:
        _adx_summary = ", ".join(
            f"{s}={v:.1f}" for s, v in sorted(_regime_adx_by_symbol.items())
        )
        logger.info("Market regime ADX (1h per-instrument): %s", _adx_summary)
    _regime_blocked_symbols: set[str] = set()
    if _regime_adx_min > 0 or bool(getattr(settings, "MARKET_REGIME_ADX_MIN_BY_CONTEXT", {})):
        for _sym, _adx_val in _regime_adx_by_symbol.items():
            _effective_min = _regime_adx_min_for_symbol_session(
                _sym,
                current_session,
                _regime_adx_min,
            )
            _regime_adx_min_by_symbol[_sym] = _effective_min
            if _effective_min > 0 and _adx_val < _effective_min:
                _regime_blocked_symbols.add(_sym)
        if _regime_blocked_symbols:
            logger.warning(
                "Market regime gate ACTIVE (per-instrument/session=%s): blocked: %s",
                current_session,
                ", ".join(
                    f"{s}(adx={_regime_adx_by_symbol[s]:.1f}<min={_regime_adx_min_by_symbol.get(s, _regime_adx_min):.1f})"
                    for s in sorted(_regime_blocked_symbols)
                ),
            )
    return (
        _regime_adx_by_symbol,
        _regime_blocked_symbols,
        _regime_adx_min,
        _market_regime_adx,
        _regime_bias_by_symbol,
        _regime_adx_min_by_symbol,
    )


def _compute_mtf_regime_context(
    enabled_instruments: list[Instrument],
) -> tuple[dict[str, dict[str, Any]], str, str]:
    snapshots: dict[str, dict[str, Any]] = {}
    if not bool(getattr(settings, "MTF_REGIME_ENABLED", True)):
        return snapshots, "transition", "balanced"

    for inst in enabled_instruments:
        try:
            snapshots[inst.symbol] = build_symbol_regime_snapshot(inst)
        except Exception as exc:
            logger.debug("MTF regime snapshot failed for %s: %s", inst.symbol, exc)

    btc_snapshot = snapshots.get("BTCUSDT")
    if btc_snapshot is None:
        try:
            btc_inst = next((x for x in enabled_instruments if x.symbol == "BTCUSDT"), None)
            if btc_inst is None:
                btc_inst = Instrument.objects.filter(symbol="BTCUSDT").first()
            if btc_inst is not None:
                btc_snapshot = build_symbol_regime_snapshot(btc_inst)
                snapshots["BTCUSDT"] = btc_snapshot
        except Exception:
            btc_snapshot = None

    if not isinstance(btc_snapshot, dict):
        return snapshots, "transition", "balanced"

    btc_lead_state = consolidate_lead_state(
        btc_snapshot.get("monthly_regime", "transition"),
        btc_snapshot.get("weekly_regime", "transition"),
        btc_snapshot.get("daily_regime", "transition"),
    )
    btc_recommended_bias = _mtf_recommended_bias(
        btc_snapshot.get("monthly_regime", "transition"),
        btc_snapshot.get("weekly_regime", "transition"),
        btc_snapshot.get("daily_regime", "transition"),
        lead_state=btc_lead_state,
    )
    logger.info(
        "MTF regime BTC context: monthly=%s weekly=%s daily=%s lead=%s bias=%s",
        btc_snapshot.get("monthly_regime", "transition"),
        btc_snapshot.get("weekly_regime", "transition"),
        btc_snapshot.get("daily_regime", "transition"),
        btc_lead_state,
        btc_recommended_bias,
    )
    return snapshots, btc_lead_state, btc_recommended_bias


def _regime_directional_risk_mult(
    symbol: str,
    signal_direction: str,
    regime_bias: str,
) -> tuple[float, bool, str]:
    """
    Returns (multiplier, blocked, reason) for directional risk adjustment by HTF bias.
    """
    if not bool(getattr(settings, "REGIME_DIRECTIONAL_PENALTY_ENABLED", True)):
        return 1.0, False, "disabled"
    sym = str(symbol or "").strip().upper()
    direction = str(signal_direction or "").strip().lower()
    bias = str(regime_bias or "neutral").strip().lower()
    if bias == "bear" and direction == "long":
        if sym == "BTCUSDT" and bool(getattr(settings, "BTC_BEAR_LONG_BLOCK_ENABLED", False)):
            return 0.0, True, "btc_bear_long_block"
        penalty = max(0.0, min(float(getattr(settings, "REGIME_BEAR_LONG_PENALTY", 0.15) or 0.15), 0.95))
        return (1.0 - penalty), False, "bear_long_penalty"
    if bias == "bull" and direction == "short":
        if get_runtime_bool(
            "REGIME_BULL_SHORT_BLOCK_ENABLED",
            bool(getattr(settings, "REGIME_BULL_SHORT_BLOCK_ENABLED", False)),
        ):
            return 0.0, True, "bull_short_block"
        penalty = max(0.0, min(float(getattr(settings, "REGIME_BULL_SHORT_PENALTY", 0.10) or 0.10), 0.95))
        return (1.0 - penalty), False, "bull_short_penalty"
    return 1.0, False, "neutral"


def _btc_lead_directional_risk_mult(
    symbol: str,
    signal_direction: str,
    btc_lead_state: str,
    btc_recommended_bias: str,
) -> tuple[float, bool, str]:
    if not get_runtime_bool(
        "BTC_LEAD_FILTER_ENABLED",
        bool(getattr(settings, "BTC_LEAD_FILTER_ENABLED", False)),
    ):
        return 1.0, False, "disabled"
    sym = str(symbol or "").strip().upper()
    if not sym or sym == "BTCUSDT":
        return 1.0, False, "self"

    direction = str(signal_direction or "").strip().lower()
    lead_state = str(btc_lead_state or "transition").strip().lower()
    rec_bias = str(btc_recommended_bias or "balanced").strip().lower()
    penalty = max(0.0, min(float(getattr(settings, "BTC_LEAD_ALT_RISK_PENALTY", 0.15) or 0.15), 0.95))
    hard_block = bool(getattr(settings, "BTC_LEAD_HARD_BLOCK_ENABLED", False))

    if direction == "long" and lead_state in {"bear_confirmed", "bear_weak"}:
        tactical_ok = rec_bias == "tactical_long"
        if hard_block and lead_state == "bear_confirmed" and not tactical_ok:
            return 0.0, True, "btc_lead_bear_alt_long_block"
        scale = 0.5 if tactical_ok or lead_state == "bear_weak" else 1.0
        return max(0.0, 1.0 - penalty * scale), False, "btc_lead_bear_alt_long_penalty"

    if direction == "short" and lead_state in {"bull_confirmed", "bull_weak"}:
        tactical_ok = rec_bias == "tactical_short"
        if hard_block and lead_state == "bull_confirmed" and not tactical_ok:
            return 0.0, True, "btc_lead_bull_alt_short_block"
        scale = 0.5 if tactical_ok or lead_state == "bull_weak" else 1.0
        return max(0.0, 1.0 - penalty * scale), False, "btc_lead_bull_alt_short_penalty"

    return 1.0, False, "neutral"


def _runtime_lower_set(name: str, default: set[str] | list[str] | tuple[str, ...]) -> set[str]:
    val = get_runtime_override(name, default)
    if isinstance(val, str):
        items = [part.strip().lower() for part in val.split(",") if part.strip()]
    elif isinstance(val, (list, tuple, set)):
        items = [str(part).strip().lower() for part in val if str(part).strip()]
    else:
        items = [str(part).strip().lower() for part in default if str(part).strip()]
    return {item for item in items if item}


def _ny_open_weak_long_precheck(
    *,
    strategy_name: str,
    signal_direction: str,
    current_session: str,
    btc_lead_state: str,
    btc_recommended_bias: str,
) -> tuple[bool, str]:
    if _strategy_is_microvol(strategy_name):
        return True, "microvol_exempt"
    if str(current_session or "").strip().lower() != "ny_open":
        return True, "n/a"
    if str(signal_direction or "").strip().lower() != "long":
        return True, "n/a"
    if not get_runtime_bool(
        "NY_OPEN_WEAK_LONG_BLOCK_ENABLED",
        bool(getattr(settings, "NY_OPEN_WEAK_LONG_BLOCK_ENABLED", False)),
    ):
        return True, "disabled"

    lead_state = str(btc_lead_state or "transition").strip().lower()
    rec_bias = str(btc_recommended_bias or "balanced").strip().lower()
    blocked_leads = _runtime_lower_set(
        "NY_OPEN_WEAK_LONG_BLOCK_LEAD_STATES",
        getattr(settings, "NY_OPEN_WEAK_LONG_BLOCK_LEAD_STATES", {"transition"}),
    )
    blocked_biases = _runtime_lower_set(
        "NY_OPEN_WEAK_LONG_BLOCK_RECOMMENDED_BIASES",
        getattr(settings, "NY_OPEN_WEAK_LONG_BLOCK_RECOMMENDED_BIASES", {"balanced"}),
    )
    if lead_state in blocked_leads and rec_bias in blocked_biases:
        return False, f"ny_open_weak_long_context:{lead_state}:{rec_bias}"
    return True, "ok"


def _weak_long_bear_weak_precheck(
    *,
    strategy_name: str,
    signal_direction: str,
    current_session: str = "",
    monthly_regime: str,
    daily_regime: str,
    btc_lead_state: str,
    btc_recommended_bias: str,
    sig_score: float = 0.0,
    symbol_adx_1h: float = 0.0,
    trend_context_direction: str = "",
    trend_context_is_strong: bool = False,
) -> tuple[bool, str]:
    if _strategy_is_microvol(strategy_name):
        return True, "microvol_exempt"
    if str(signal_direction or "").strip().lower() != "long":
        return True, "n/a"
    if not get_runtime_bool(
        "WEAK_LONG_BEAR_WEAK_BLOCK_ENABLED",
        bool(getattr(settings, "WEAK_LONG_BEAR_WEAK_BLOCK_ENABLED", False)),
    ):
        return True, "disabled"

    month = str(monthly_regime or "").strip().lower()
    day = str(daily_regime or "").strip().lower()
    lead_state = str(btc_lead_state or "transition").strip().lower()
    rec_bias = str(btc_recommended_bias or "balanced").strip().lower()
    blocked_months = _runtime_lower_set(
        "WEAK_LONG_BEAR_WEAK_BLOCK_MONTHLY_REGIMES",
        getattr(settings, "WEAK_LONG_BEAR_WEAK_BLOCK_MONTHLY_REGIMES", {"bear_confirmed"}),
    )
    blocked_days = _runtime_lower_set(
        "WEAK_LONG_BEAR_WEAK_BLOCK_DAILY_REGIMES",
        getattr(settings, "WEAK_LONG_BEAR_WEAK_BLOCK_DAILY_REGIMES", {"bear_weak"}),
    )
    blocked_leads = _runtime_lower_set(
        "WEAK_LONG_BEAR_WEAK_BLOCK_LEAD_STATES",
        getattr(settings, "WEAK_LONG_BEAR_WEAK_BLOCK_LEAD_STATES", {"transition"}),
    )
    blocked_biases = _runtime_lower_set(
        "WEAK_LONG_BEAR_WEAK_BLOCK_RECOMMENDED_BIASES",
        getattr(settings, "WEAK_LONG_BEAR_WEAK_BLOCK_RECOMMENDED_BIASES", {"balanced"}),
    )
    if (
        month in blocked_months
        and day in blocked_days
        and lead_state in blocked_leads
        and rec_bias in blocked_biases
    ):
        transition_relax_enabled = get_runtime_bool(
            "WEAK_LONG_TRANSITION_STRONG_TREND_RELAX_ENABLED",
            bool(getattr(settings, "WEAK_LONG_TRANSITION_STRONG_TREND_RELAX_ENABLED", False)),
        )
        transition_relax_sessions = _runtime_lower_set(
            "WEAK_LONG_TRANSITION_STRONG_TREND_ALLOWED_SESSIONS",
            getattr(
                settings,
                "WEAK_LONG_TRANSITION_STRONG_TREND_ALLOWED_SESSIONS",
                {"london", "overlap", "ny_open", "ny"},
            ),
        )
        transition_relax_min_score = get_runtime_float(
            "WEAK_LONG_TRANSITION_STRONG_TREND_MIN_SCORE",
            float(getattr(settings, "WEAK_LONG_TRANSITION_STRONG_TREND_MIN_SCORE", 0.68) or 0.68),
            minimum=0.0,
            maximum=1.0,
        )
        transition_relax_min_adx = get_runtime_float(
            "WEAK_LONG_TRANSITION_STRONG_TREND_MIN_ADX",
            float(getattr(settings, "WEAK_LONG_TRANSITION_STRONG_TREND_MIN_ADX", 24.0) or 24.0),
            minimum=0.0,
        )
        session_name = str(current_session or "").strip().lower()
        score_now = max(0.0, float(sig_score or 0.0))
        trend_dir = str(trend_context_direction or "").strip().lower()
        if (
            transition_relax_enabled
            and day == "transition"
            and session_name in transition_relax_sessions
            and trend_dir == "long"
            and bool(trend_context_is_strong)
            and score_now >= transition_relax_min_score
            and symbol_adx_1h >= transition_relax_min_adx > 0
        ):
            return True, (
                f"transition_strong_trend_ok:{session_name}:{score_now:.3f}:{symbol_adx_1h:.1f}"
            )
        adx_override_enabled = get_runtime_bool(
            "WEAK_LONG_BEAR_WEAK_ADX_OVERRIDE_ENABLED",
            bool(getattr(settings, "WEAK_LONG_BEAR_WEAK_ADX_OVERRIDE_ENABLED", False)),
        )
        adx_override_min = float(
            getattr(settings, "WEAK_LONG_BEAR_WEAK_ADX_OVERRIDE_MIN", 35.0)
        )
        if (
            adx_override_enabled
            and
            symbol_adx_1h >= adx_override_min > 0
            and trend_dir == "long"
            and bool(trend_context_is_strong)
        ):
            return True, (
                f"adx_override:{symbol_adx_1h:.1f}>={adx_override_min:.0f}"
                f":{trend_dir}:strong"
            )
        return False, f"weak_long_bear_weak:{month}:{day}:{lead_state}:{rec_bias}"
    return True, "ok"


def _asia_weak_short_precheck(
    *,
    strategy_name: str,
    signal_direction: str,
    current_session: str,
    btc_lead_state: str,
    btc_recommended_bias: str,
    trend_context_direction: str = "",
    trend_context_is_strong: bool = False,
    trend_context_confidence: float = 0.0,
    trend_context_adx_htf: float = 0.0,
) -> tuple[bool, str]:
    if _strategy_is_microvol(strategy_name):
        return True, "microvol_exempt"
    if str(current_session or "").strip().lower() != "asia":
        return True, "n/a"
    if str(signal_direction or "").strip().lower() != "short":
        return True, "n/a"
    if not get_runtime_bool(
        "ASIA_WEAK_SHORT_BLOCK_ENABLED",
        bool(getattr(settings, "ASIA_WEAK_SHORT_BLOCK_ENABLED", False)),
    ):
        return True, "disabled"

    lead_state = str(btc_lead_state or "transition").strip().lower()
    rec_bias = str(btc_recommended_bias or "balanced").strip().lower()
    blocked_leads = _runtime_lower_set(
        "ASIA_WEAK_SHORT_BLOCK_LEAD_STATES",
        getattr(settings, "ASIA_WEAK_SHORT_BLOCK_LEAD_STATES", {"transition"}),
    )
    blocked_biases = _runtime_lower_set(
        "ASIA_WEAK_SHORT_BLOCK_RECOMMENDED_BIASES",
        getattr(settings, "ASIA_WEAK_SHORT_BLOCK_RECOMMENDED_BIASES", {"balanced"}),
    )
    trend_dir = str(trend_context_direction or "").strip().lower()
    if lead_state in blocked_leads and rec_bias in blocked_biases:
        if trend_dir == "short" and bool(trend_context_is_strong):
            return True, "strong_short_trend_ok"
        relaxed_enabled = get_runtime_bool(
            "ASIA_WEAK_SHORT_RELAXED_TREND_ENABLED",
            bool(getattr(settings, "ASIA_WEAK_SHORT_RELAXED_TREND_ENABLED", False)),
        )
        relaxed_conf_min = get_runtime_float(
            "ASIA_WEAK_SHORT_RELAXED_TREND_CONF_MIN",
            float(getattr(settings, "ASIA_WEAK_SHORT_RELAXED_TREND_CONF_MIN", 0.34) or 0.34),
            minimum=0.0,
        )
        relaxed_adx_min = get_runtime_float(
            "ASIA_WEAK_SHORT_RELAXED_TREND_ADX_MIN",
            float(getattr(settings, "ASIA_WEAK_SHORT_RELAXED_TREND_ADX_MIN", 19.5) or 19.5),
            minimum=0.0,
        )
        trend_conf = max(0.0, float(trend_context_confidence or 0.0))
        trend_adx = max(0.0, float(trend_context_adx_htf or 0.0))
        if (
            relaxed_enabled
            and trend_dir == "short"
            and trend_conf >= relaxed_conf_min
            and trend_adx >= relaxed_adx_min
        ):
            return True, f"relaxed_short_trend_ok:{trend_conf:.3f}:{trend_adx:.1f}"
        return False, f"asia_weak_short:{lead_state}:{rec_bias}:{trend_dir or 'none'}"
    return True, "ok"


def _weak_short_transition_precheck(
    *,
    strategy_name: str,
    signal_direction: str,
    current_session: str,
    daily_regime: str,
    btc_lead_state: str,
    btc_recommended_bias: str,
    trend_context_direction: str = "",
    trend_context_is_strong: bool = False,
) -> tuple[bool, str]:
    if _strategy_is_microvol(strategy_name):
        return True, "microvol_exempt"
    if str(signal_direction or "").strip().lower() != "short":
        return True, "n/a"
    if not get_runtime_bool(
        "WEAK_SHORT_TRANSITION_BLOCK_ENABLED",
        bool(getattr(settings, "WEAK_SHORT_TRANSITION_BLOCK_ENABLED", False)),
    ):
        return True, "disabled"

    session = str(current_session or "").strip().lower()
    allowed_sessions = _runtime_lower_set(
        "WEAK_SHORT_TRANSITION_BLOCK_SESSIONS",
        getattr(settings, "WEAK_SHORT_TRANSITION_BLOCK_SESSIONS", {"london", "overlap", "ny"}),
    )
    if session not in allowed_sessions:
        return True, "n/a"

    day = str(daily_regime or "").strip().lower()
    lead_state = str(btc_lead_state or "transition").strip().lower()
    rec_bias = str(btc_recommended_bias or "balanced").strip().lower()
    blocked_days = _runtime_lower_set(
        "WEAK_SHORT_TRANSITION_BLOCK_DAILY_REGIMES",
        getattr(settings, "WEAK_SHORT_TRANSITION_BLOCK_DAILY_REGIMES", {"bear_weak"}),
    )
    blocked_leads = _runtime_lower_set(
        "WEAK_SHORT_TRANSITION_BLOCK_LEAD_STATES",
        getattr(settings, "WEAK_SHORT_TRANSITION_BLOCK_LEAD_STATES", {"transition"}),
    )
    blocked_biases = _runtime_lower_set(
        "WEAK_SHORT_TRANSITION_BLOCK_RECOMMENDED_BIASES",
        getattr(settings, "WEAK_SHORT_TRANSITION_BLOCK_RECOMMENDED_BIASES", {"balanced"}),
    )
    trend_dir = str(trend_context_direction or "").strip().lower()
    if day in blocked_days and lead_state in blocked_leads and rec_bias in blocked_biases:
        if trend_dir == "short" and bool(trend_context_is_strong):
            return True, "strong_short_trend_ok"
        return False, f"weak_short_transition:{session}:{day}:{lead_state}:{rec_bias}:{trend_dir or 'none'}"
    return True, "ok"


def _long_bias_short_precheck(
    *,
    strategy_name: str,
    signal_direction: str,
    btc_recommended_bias: str,
    sig_score: float,
    sig_payload: dict[str, Any] | None,
) -> tuple[bool, str]:
    if _strategy_is_microvol(strategy_name):
        return True, "microvol_exempt"
    if str(signal_direction or "").strip().lower() != "short":
        return True, "n/a"
    if not get_runtime_bool(
        "LONG_BIAS_SHORT_BLOCK_ENABLED",
        bool(getattr(settings, "LONG_BIAS_SHORT_BLOCK_ENABLED", False)),
    ):
        return True, "disabled"

    rec_bias = str(btc_recommended_bias or "balanced").strip().lower()
    blocked_biases = _runtime_lower_set(
        "LONG_BIAS_SHORT_BLOCK_RECOMMENDED_BIASES",
        getattr(settings, "LONG_BIAS_SHORT_BLOCK_RECOMMENDED_BIASES", {"long_bias"}),
    )
    if rec_bias not in blocked_biases:
        return True, "n/a"

    score = max(0.0, _to_float(sig_score))
    min_score = get_runtime_float(
        "LONG_BIAS_SHORT_BLOCK_COUNTERTREND_MIN_SCORE",
        float(getattr(settings, "LONG_BIAS_SHORT_BLOCK_COUNTERTREND_MIN_SCORE", 0.90) or 0.90),
        minimum=0.0,
        maximum=1.0,
    )
    allowed_modules = get_runtime_str_list(
        "LONG_BIAS_SHORT_BLOCK_ALLOWED_MODULES",
        getattr(settings, "LONG_BIAS_SHORT_BLOCK_ALLOWED_MODULES", {"meanrev", "smc"}),
    )
    needed = get_runtime_int(
        "LONG_BIAS_SHORT_BLOCK_MIN_ALLOWED_MODULES",
        int(getattr(settings, "LONG_BIAS_SHORT_BLOCK_MIN_ALLOWED_MODULES", 1) or 1),
        minimum=1,
    )
    active_modules = set(_signal_active_modules(sig_payload if isinstance(sig_payload, dict) else {}, strategy_name))
    matched_modules = active_modules & allowed_modules
    if score >= min_score and len(matched_modules) >= needed:
        return True, (
            f"countertrend_short_ok:{score:.3f}:"
            f"{','.join(sorted(matched_modules)) or 'none'}"
        )
    return False, (
        f"long_bias_short_block:{rec_bias}:{score:.3f}:"
        f"{','.join(sorted(active_modules)) or 'none'}"
    )


def _symbol_health_precheck(inst: Instrument) -> tuple[bool, str]:
    if not get_runtime_bool(
        "SYMBOL_HEALTH_GUARD_ENABLED",
        bool(getattr(settings, "SYMBOL_HEALTH_GUARD_ENABLED", False)),
    ):
        return True, "disabled"

    symbol = str(getattr(inst, "symbol", "") or "").strip().upper()
    exempt_symbols = get_runtime_str_list(
        "SYMBOL_HEALTH_GUARD_EXEMPT_SYMBOLS",
        getattr(settings, "SYMBOL_HEALTH_GUARD_EXEMPT_SYMBOLS", set()),
    )
    if symbol.lower() in exempt_symbols:
        return True, "exempt"

    lookback_days = get_runtime_int(
        "SYMBOL_HEALTH_GUARD_LOOKBACK_DAYS",
        int(getattr(settings, "SYMBOL_HEALTH_GUARD_LOOKBACK_DAYS", 14) or 14),
        minimum=1,
    )
    min_trades = get_runtime_int(
        "SYMBOL_HEALTH_GUARD_MIN_TRADES",
        int(getattr(settings, "SYMBOL_HEALTH_GUARD_MIN_TRADES", 8) or 8),
        minimum=1,
    )
    min_pf = get_runtime_float(
        "SYMBOL_HEALTH_GUARD_MIN_PROFIT_FACTOR",
        float(getattr(settings, "SYMBOL_HEALTH_GUARD_MIN_PROFIT_FACTOR", 0.90) or 0.90),
        minimum=0.0,
    )
    min_expectancy = get_runtime_float(
        "SYMBOL_HEALTH_GUARD_MIN_EXPECTANCY_USDT",
        float(getattr(settings, "SYMBOL_HEALTH_GUARD_MIN_EXPECTANCY_USDT", 0.0) or 0.0),
    )

    cutoff = dj_tz.now() - timedelta(days=lookback_days)
    reports = list(
        OperationReport.objects.filter(
            instrument=inst,
            closed_at__gte=cutoff,
        )
        .order_by("-closed_at")
        .values_list("pnl_abs", flat=True)
    )
    sample_size = len(reports)
    if sample_size < min_trades:
        return True, f"insufficient_sample:{sample_size}<{min_trades}"

    pnls = [_to_float(pnl) for pnl in reports]
    gross_win = sum(pnl for pnl in pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
    profit_factor = (
        float("inf")
        if gross_win > 0 and gross_loss <= 0
        else (gross_win / gross_loss if gross_loss > 0 else 0.0)
    )
    expectancy = sum(pnls) / sample_size if sample_size else 0.0
    total_pnl = sum(pnls)

    if profit_factor < min_pf or expectancy < min_expectancy:
        return False, (
            f"symbol_health:{symbol}:n={sample_size}:pf={profit_factor:.3f}:"
            f"expect={expectancy:.5f}:pnl={total_pnl:.5f}"
        )
    return True, (
        f"healthy:{symbol}:n={sample_size}:pf={profit_factor:.3f}:"
        f"expect={expectancy:.5f}:pnl={total_pnl:.5f}"
    )


def _dead_session_strong_trend_breakout_override(
    *,
    strategy_name: str,
    signal_direction: str,
    current_session: str,
    sig_score: float,
    sig_payload: dict[str, Any] | None,
) -> tuple[bool, str, float | None, float | None]:
    if str(current_session or "").strip().lower() != "dead":
        return False, "n/a", None, None
    if str(strategy_name or "").strip().lower() != "alloc_long":
        return False, "strategy_not_allowed", None, None
    if str(signal_direction or "").strip().lower() != "long":
        return False, "n/a", None, None
    if not get_runtime_bool(
        "DEAD_SESSION_STRONG_TREND_BREAKOUT_ENABLED",
        bool(getattr(settings, "DEAD_SESSION_STRONG_TREND_BREAKOUT_ENABLED", False)),
    ):
        return False, "disabled", None, None

    payload = sig_payload if isinstance(sig_payload, dict) else {}
    reasons = payload.get("reasons") if isinstance(payload.get("reasons"), dict) else {}
    trend_ctx = reasons.get("trend_context") if isinstance(reasons.get("trend_context"), dict) else {}
    trend_dir = str(trend_ctx.get("direction", "")).strip().lower()
    trend_is_strong = bool(trend_ctx.get("is_strong"))
    adx_htf = _to_float(trend_ctx.get("adx_htf", 0.0))
    trend_conf = _to_float(trend_ctx.get("confidence", 0.0))
    min_score = get_runtime_float(
        "DEAD_SESSION_STRONG_TREND_BREAKOUT_MIN_SCORE",
        float(getattr(settings, "DEAD_SESSION_STRONG_TREND_BREAKOUT_MIN_SCORE", 0.70) or 0.70),
        minimum=0.0,
        maximum=1.0,
    )
    min_adx = get_runtime_float(
        "DEAD_SESSION_STRONG_TREND_BREAKOUT_MIN_ADX",
        float(getattr(settings, "DEAD_SESSION_STRONG_TREND_BREAKOUT_MIN_ADX", 40.0) or 40.0),
        minimum=0.0,
    )
    min_trend_conf = get_runtime_float(
        "DEAD_SESSION_STRONG_TREND_BREAKOUT_MIN_TREND_CONF",
        float(getattr(settings, "DEAD_SESSION_STRONG_TREND_BREAKOUT_MIN_TREND_CONF", 0.80) or 0.80),
        minimum=0.0,
        maximum=1.0,
    )
    if _to_float(sig_score) < min_score:
        return False, f"low_score:{_to_float(sig_score):.3f}<{min_score:.3f}", None, None
    if trend_dir != "long" or not trend_is_strong:
        return False, f"trend_not_strong:{trend_dir or 'none'}:{trend_is_strong}", None, None
    if adx_htf < min_adx:
        return False, f"adx_low:{adx_htf:.1f}<{min_adx:.1f}", None, None
    if trend_conf < min_trend_conf:
        return False, f"trend_conf_low:{trend_conf:.3f}<{min_trend_conf:.3f}", None, None

    module_rows = reasons.get("module_rows") if isinstance(reasons.get("module_rows"), list) else []
    saw_trend = False
    for row in module_rows:
        if not isinstance(row, dict):
            continue
        module = str(row.get("module", "")).strip().lower()
        direction = str(row.get("direction", "")).strip().lower()
        if module == "trend":
            saw_trend = True
        if direction and direction != "long":
            return False, f"opposed_module:{module}:{direction}", None, None
    if not saw_trend:
        return False, "trend_module_missing", None, None

    score_override = min_score
    risk_mult = get_runtime_float(
        "DEAD_SESSION_STRONG_TREND_BREAKOUT_RISK_MULT",
        float(getattr(settings, "DEAD_SESSION_STRONG_TREND_BREAKOUT_RISK_MULT", 0.35) or 0.35),
        minimum=0.0,
        maximum=1.0,
    )
    return True, "dead_strong_trend_breakout", score_override, risk_mult


def _post_tp_alt_reentry_quality_precheck(
    *,
    inst: Instrument,
    strategy_name: str,
    signal_direction: str,
    sig_payload: dict[str, Any] | None,
) -> tuple[bool, str]:
    if _strategy_is_microvol(strategy_name):
        return True, "microvol_exempt"
    if str(strategy_name or "").strip().lower() != "alloc_long":
        return True, "n/a"
    if str(signal_direction or "").strip().lower() != "long":
        return True, "n/a"
    if not get_runtime_bool(
        "POST_TP_ALT_REENTRY_QUALITY_GATE_ENABLED",
        bool(getattr(settings, "POST_TP_ALT_REENTRY_QUALITY_GATE_ENABLED", False)),
    ):
        return True, "disabled"

    tier_map = getattr(settings, "INSTRUMENT_TIER_MAP", {}) or {}
    symbol_tier = str(tier_map.get(str(inst.symbol or "").strip().upper(), "")).strip().lower()
    gated_tiers = get_runtime_str_list(
        "POST_TP_ALT_REENTRY_QUALITY_TIERS",
        getattr(settings, "POST_TP_ALT_REENTRY_QUALITY_TIERS", {"alt"}),
    )
    if not symbol_tier or symbol_tier not in gated_tiers:
        return True, "n/a"

    window_minutes = get_runtime_int(
        "POST_TP_ALT_REENTRY_QUALITY_WINDOW_MINUTES",
        int(getattr(settings, "POST_TP_ALT_REENTRY_QUALITY_WINDOW_MINUTES", 120) or 120),
        minimum=1,
    )
    last_same_side_close = (
        OperationReport.objects.filter(
            instrument=inst,
            side="buy",
            closed_at__isnull=False,
        )
        .only("closed_at", "reason", "outcome")
        .order_by("-closed_at")
        .first()
    )
    if last_same_side_close is None or last_same_side_close.closed_at is None:
        return True, "no_recent_close"
    if str(last_same_side_close.reason or "").strip().lower() != "tp":
        return True, f"last_close_not_tp:{str(last_same_side_close.reason or 'n/a').strip().lower()}"
    if str(last_same_side_close.outcome or "").strip().lower() != str(OperationReport.Outcome.WIN):
        return True, f"last_tp_not_win:{str(last_same_side_close.outcome or 'n/a').strip().lower()}"
    elapsed_min = (dj_tz.now() - last_same_side_close.closed_at).total_seconds() / 60.0
    if elapsed_min > float(window_minutes):
        return True, f"tp_outside_window:{elapsed_min:.1f}>{window_minutes}"

    payload = sig_payload if isinstance(sig_payload, dict) else {}
    reasons = payload.get("reasons") if isinstance(payload.get("reasons"), dict) else {}
    module_rows = reasons.get("module_rows") if isinstance(reasons.get("module_rows"), list) else []
    trend_row = None
    for row in module_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("module", "")).strip().lower() != "trend":
            continue
        if str(row.get("direction", "")).strip().lower() != "long":
            continue
        trend_row = row
        break
    if trend_row is None:
        return True, "no_trend_row_fail_open"

    min_trend_conf = get_runtime_float(
        "POST_TP_ALT_REENTRY_MIN_TREND_CONFIDENCE",
        float(getattr(settings, "POST_TP_ALT_REENTRY_MIN_TREND_CONFIDENCE", 0.94) or 0.94),
        minimum=0.0,
        maximum=1.0,
    )
    trend_conf = max(0.0, _to_float(trend_row.get("confidence", 0.0)))
    if trend_conf < min_trend_conf:
        return False, f"post_tp_alt_reentry_trend_conf_low:{trend_conf:.3f}<{min_trend_conf:.3f}"

    trend_reasons = trend_row.get("reasons") if isinstance(trend_row.get("reasons"), dict) else {}
    min_volume_ratio = get_runtime_float(
        "POST_TP_ALT_REENTRY_MIN_TREND_VOLUME_RATIO",
        float(getattr(settings, "POST_TP_ALT_REENTRY_MIN_TREND_VOLUME_RATIO", 0.45) or 0.45),
        minimum=0.0,
    )
    trend_volume_ratio = max(0.0, _to_float(trend_reasons.get("volume_ratio", 0.0)))
    if trend_volume_ratio > 0 and trend_volume_ratio < min_volume_ratio:
        return False, (
            f"post_tp_alt_reentry_trend_volume_low:{trend_volume_ratio:.3f}<{min_volume_ratio:.3f}"
        )

    max_volume_penalty = get_runtime_float(
        "POST_TP_ALT_REENTRY_MAX_TREND_VOLUME_PENALTY",
        float(getattr(settings, "POST_TP_ALT_REENTRY_MAX_TREND_VOLUME_PENALTY", 0.06) or 0.06),
        minimum=0.0,
        maximum=1.0,
    )
    trend_volume_penalty = max(0.0, _to_float(trend_reasons.get("volume_penalty", 0.0)))
    if trend_volume_penalty > max_volume_penalty:
        return False, (
            "post_tp_alt_reentry_trend_volume_penalty_high:"
            f"{trend_volume_penalty:.3f}>{max_volume_penalty:.3f}"
        )

    min_net_score = get_runtime_float(
        "POST_TP_ALT_REENTRY_MIN_NET_SCORE",
        float(getattr(settings, "POST_TP_ALT_REENTRY_MIN_NET_SCORE", 0.23) or 0.23),
        minimum=0.0,
    )
    net_score = max(0.0, _to_float(payload.get("net_score", 0.0)))
    if net_score < min_net_score:
        return False, f"post_tp_alt_reentry_net_score_low:{net_score:.3f}<{min_net_score:.3f}"

    return True, (
        "post_tp_alt_reentry_ok:"
        f"{elapsed_min:.1f}m:"
        f"net={net_score:.3f}:"
        f"trend_conf={trend_conf:.3f}:"
        f"vol_ratio={trend_volume_ratio:.3f}:"
        f"vol_pen={trend_volume_penalty:.3f}"
    )


def _corr_guard_positions_snapshot(
    base_positions: list[dict] | None,
    pending_entries: list[dict] | None = None,
) -> list[dict]:
    """Merge exchange positions with same-cycle pending entries for corr guard checks."""
    merged = list(base_positions or [])
    for entry in pending_entries or []:
        side_txt = str(entry.get("side") or "").strip().lower()
        if side_txt not in {"long", "short"}:
            continue
        merged.append(
            {
                "symbol": str(entry.get("symbol") or ""),
                "side": side_txt,
                "contracts": float(entry.get("contracts") or 1.0),
            }
        )
    return merged


def _cross_symbol_correlation_guard(
    *,
    inst: Instrument,
    signal_direction: str,
    positions_snapshot: list[dict],
    equity_usdt: float,
) -> tuple[float, str]:
    """Return (risk_mult, reason) based on same-direction open positions.

    When multiple correlated crypto assets are open in the same direction,
    effective portfolio risk is higher than the sum of independent bets.
    This guard reduces risk for new entries that add to directional
    concentration.

    Returns:
        risk_mult: 1.0 (no reduction) down to the configured floor.
        reason: human-readable explanation or empty string.
    """
    if not bool(getattr(settings, "CROSS_SYMBOL_CORR_GUARD_ENABLED", True)):
        return 1.0, ""

    max_same_direction = max(1, int(
        getattr(settings, "CROSS_SYMBOL_CORR_GUARD_MAX_SAME_DIR", 4) or 4
    ))
    risk_reduction_per_pos = max(0.0, min(0.5,
        float(getattr(settings, "CROSS_SYMBOL_CORR_GUARD_RISK_REDUCTION", 0.15) or 0.15),
    ))
    risk_floor = max(0.10, min(1.0,
        float(getattr(settings, "CROSS_SYMBOL_CORR_GUARD_RISK_FLOOR", 0.40) or 0.40),
    ))

    direction = str(signal_direction or "").strip().lower()
    if direction not in ("long", "short"):
        return 1.0, ""

    same_dir_count = 0
    for pos in (positions_snapshot or []):
        pos_symbol = str(pos.get("symbol") or pos.get("info", {}).get("symbol") or "")
        # Skip self
        if inst.symbol and inst.symbol.upper() in pos_symbol.upper():
            continue
        pos_side_raw = str(pos.get("side") or "").strip().lower()
        pos_qty = float(pos.get("contracts") or pos.get("contractSize") or 0)
        if pos_qty == 0:
            continue
        # Determine position direction
        if pos_side_raw == "long" or (pos_side_raw == "" and pos_qty > 0):
            pos_dir = "long"
        elif pos_side_raw == "short" or (pos_side_raw == "" and pos_qty < 0):
            pos_dir = "short"
        else:
            continue
        if pos_dir == direction:
            same_dir_count += 1

    if same_dir_count == 0:
        return 1.0, ""

    if same_dir_count >= max_same_direction:
        return 0.0, f"max_same_dir:{same_dir_count}>={max_same_direction}"

    mult = max(risk_floor, 1.0 - (same_dir_count * risk_reduction_per_pos))
    reason = f"corr_guard:{same_dir_count}_same_{direction},risk_mult={mult:.2f}"
    return mult, reason


# ---------------------------------------------------------------------------
# Flat-signal timeout — close positions when signal stays flat too long
# ---------------------------------------------------------------------------

def _flat_signal_since_key(symbol: str) -> str:
    return f"flat_signal_since:{symbol}"


def _track_flat_signal(symbol: str) -> float:
    """Record the first time we see a flat signal for *symbol*.

    Returns seconds since the first flat observation, or 0 on error/first call.
    """
    client = _redis_client()
    if client is None:
        return 0.0
    key = _flat_signal_since_key(symbol)
    now_ts = dj_tz.now().timestamp()
    try:
        existing = client.get(key)
        if existing:
            return max(0.0, now_ts - float(existing))
        client.set(key, str(now_ts), ex=86400)
        return 0.0
    except Exception:
        return 0.0


def _clear_flat_signal(symbol: str) -> None:
    """Clear flat signal tracker when the signal is no longer flat."""
    client = _redis_client()
    if client is None:
        return
    try:
        client.delete(_flat_signal_since_key(symbol))
    except Exception:
        pass


def _grid_structural_sl_tp_hints(
    *,
    strategy_name: str,
    sig_payload: dict[str, Any] | None,
    side: str,
    entry_price: float,
) -> tuple[float, float, float]:
    """Return validated grid SL/TP hints plus stop distance pct for sizing."""
    if "grid" not in str(strategy_name or "").strip().lower():
        return 0.0, 0.0, 0.0
    if not isinstance(sig_payload, dict):
        return 0.0, 0.0, 0.0
    if entry_price <= 0:
        return 0.0, 0.0, 0.0

    sl_hint = _to_float(sig_payload.get("sl_price_hint"))
    tp_hint = _to_float(sig_payload.get("tp_price_hint"))
    side_txt = str(side or "").strip().lower()

    if side_txt == "buy":
        if sl_hint >= entry_price:
            sl_hint = 0.0
        if 0 < tp_hint <= entry_price:
            tp_hint = 0.0
    elif side_txt == "sell":
        if 0 < sl_hint <= entry_price:
            sl_hint = 0.0
        if tp_hint >= entry_price:
            tp_hint = 0.0

    stop_dist_pct = abs(entry_price - sl_hint) / entry_price if sl_hint > 0 else 0.0
    return sl_hint, tp_hint, stop_dist_pct


# ---------------------------------------------------------------------------
# Symbol heat guard — progressive risk scaling based on recent performance
# ---------------------------------------------------------------------------

def _symbol_heat_guard(symbol: str, side: str = "") -> tuple[float, str]:
    """Return (risk_mult, reason) based on recent trade outcomes for *symbol*.

    Instead of hard-blocking entries on a losing symbol, this progressively
    reduces risk so the bot can still participate when the symbol recovers
    while limiting damage during losing streaks.

    Returns:
        risk_mult: 1.0 (no reduction) down to the configured floor.
        reason: human-readable explanation or empty string.
    """
    if not bool(getattr(settings, "SYMBOL_HEAT_GUARD_ENABLED", False)):
        return 1.0, ""

    window = max(3, int(getattr(settings, "SYMBOL_HEAT_GUARD_WINDOW", 7) or 7))
    wr_neutral = max(0.10, min(1.0,
        float(getattr(settings, "SYMBOL_HEAT_GUARD_WR_NEUTRAL", 0.50) or 0.50),
    ))
    wr_floor = max(0.0, min(wr_neutral,
        float(getattr(settings, "SYMBOL_HEAT_GUARD_WR_FLOOR", 0.25) or 0.25),
    ))
    min_risk_mult = max(0.10, min(1.0,
        float(getattr(settings, "SYMBOL_HEAT_GUARD_MIN_RISK_MULT", 0.35) or 0.35),
    ))

    from execution.models import OperationReport
    from core.models import Instrument

    try:
        inst = Instrument.objects.filter(symbol=symbol, enabled=True).first()
        if inst is None:
            return 1.0, ""

        op_qs = OperationReport.objects.filter(instrument=inst)
        side_txt = str(side or "").strip().lower()
        if side_txt in {"buy", "sell"}:
            op_qs = op_qs.filter(side=side_txt)
        recent = list(op_qs.order_by("-closed_at").values_list("pnl_abs", flat=True)[:window])
    except Exception:
        return 1.0, ""

    if len(recent) < max(3, int(getattr(settings, "SYMBOL_HEAT_GUARD_MIN_TRADES", 3) or 3)):
        return 1.0, ""

    wins = sum(1 for pnl in recent if float(pnl) > 0)
    wr = wins / len(recent)

    if wr >= wr_neutral:
        return 1.0, ""

    if wr <= wr_floor:
        return min_risk_mult, (
            f"heat_guard:{symbol}:{side_txt or 'all'},wr={wr:.0%}({wins}/{len(recent)}),mult={min_risk_mult:.2f}"
        )

    # Linear interpolation between wr_floor → wr_neutral
    ratio = (wr - wr_floor) / (wr_neutral - wr_floor)
    mult = min_risk_mult + ratio * (1.0 - min_risk_mult)
    mult = round(max(min_risk_mult, min(1.0, mult)), 4)
    return mult, (
        f"heat_guard:{symbol}:{side_txt or 'all'},wr={wr:.0%}({wins}/{len(recent)}),mult={mult:.2f}"
    )


def _volume_delta_check(
    symbol: str,
    signal_direction: str,
) -> tuple[bool, float, str]:
    """Volume-delta imbalance confirmation for entries.

    Uses recent 1-minute candle buy/sell pressure as order-flow proxy.
    Returns (ok, score_bonus, reason).
    """
    if not bool(getattr(settings, "VOLUME_DELTA_GATE_ENABLED", True)):
        return True, 0.0, ""

    direction = str(signal_direction or "").strip().lower()
    if direction not in ("long", "short"):
        return True, 0.0, ""

    try:
        from signals.modules.common import volume_delta_imbalance
        from marketdata.models import Candle
        from core.models import Instrument as Inst

        inst = Inst.objects.filter(symbol=symbol).first()
        if not inst:
            return True, 0.0, ""

        lookback = max(10, int(getattr(settings, "VOLUME_DELTA_LOOKBACK", 20) or 20))
        qs = Candle.objects.filter(instrument=inst, timeframe="1m").order_by("-ts")[:lookback]
        if qs.count() < lookback:
            return True, 0.0, ""

        import pandas as pd
        rows = list(qs.values("open", "high", "low", "close", "volume"))
        rows.reverse()
        df = pd.DataFrame(rows)
        info = volume_delta_imbalance(df, lookback=lookback)
        if not info:
            return True, 0.0, ""

        imb = float(info.get("imbalance", 0))
        min_imb = float(getattr(settings, "VOLUME_DELTA_MIN_IMBALANCE", 0.10) or 0.10)

        # Check alignment: long needs positive imbalance, short needs negative
        aligned = (direction == "long" and imb > 0) or (direction == "short" and imb < 0)
        opposed = (direction == "long" and imb < -min_imb) or (direction == "short" and imb > min_imb)

        if opposed and bool(getattr(settings, "VOLUME_DELTA_BLOCK_OPPOSED", False)):
            return False, 0.0, f"vol_delta_opposed:{imb:.3f}_vs_{direction}"

        bonus = 0.0
        if aligned and abs(imb) >= min_imb:
            bonus = min(0.05, abs(imb) * 0.10)

        return True, bonus, f"vol_delta:{imb:.3f}"
    except Exception:
        return True, 0.0, ""


def _operation_regime_snapshot(inst: Instrument) -> dict[str, str]:
    snapshot = {
        "monthly_regime": "",
        "weekly_regime": "",
        "daily_regime": "",
        "btc_lead_state": "",
        "recommended_bias": "",
    }
    if not bool(getattr(settings, "MTF_REGIME_ENABLED", True)):
        return snapshot

    try:
        symbol_snapshot = build_symbol_regime_snapshot(inst)
        snapshot["monthly_regime"] = str(symbol_snapshot.get("monthly_regime", "") or "")
        snapshot["weekly_regime"] = str(symbol_snapshot.get("weekly_regime", "") or "")
        snapshot["daily_regime"] = str(symbol_snapshot.get("daily_regime", "") or "")
        if inst.symbol == "BTCUSDT":
            btc_snapshot = symbol_snapshot
        else:
            btc_inst = Instrument.objects.filter(symbol="BTCUSDT").first()
            btc_snapshot = build_symbol_regime_snapshot(btc_inst) if btc_inst is not None else None
        if isinstance(btc_snapshot, dict):
            lead_state = consolidate_lead_state(
                btc_snapshot.get("monthly_regime", "transition"),
                btc_snapshot.get("weekly_regime", "transition"),
                btc_snapshot.get("daily_regime", "transition"),
            )
            snapshot["btc_lead_state"] = lead_state
            snapshot["recommended_bias"] = _mtf_recommended_bias(
                btc_snapshot.get("monthly_regime", "transition"),
                btc_snapshot.get("weekly_regime", "transition"),
                btc_snapshot.get("daily_regime", "transition"),
                lead_state=lead_state,
            )
    except Exception as exc:
        logger.debug("Operation regime snapshot failed for %s: %s", getattr(inst, "symbol", "n/a"), exc)
    return snapshot


def _bull_short_retrace_precheck(
    *,
    symbol: str,
    strategy_name: str,
    signal_direction: str,
    regime_bias: str,
    sig_score: float,
    sig_payload: dict[str, Any] | None,
) -> tuple[bool, str]:
    """
    Pre-AI deterministic guard:
    In bull regime, only allow shorts when retracement probability is high enough
    by score + module evidence. This reduces repeated low-quality AI calls.
    """
    if not get_runtime_bool(
        "REGIME_BULL_SHORT_RETRACE_STRICT_ENABLED",
        bool(getattr(settings, "REGIME_BULL_SHORT_RETRACE_STRICT_ENABLED", True)),
    ):
        return True, "disabled"
    direction = str(signal_direction or "").strip().lower()
    bias = str(regime_bias or "neutral").strip().lower()
    if direction != "short" or bias != "bull":
        return True, "n/a"

    score = _to_float(sig_score)
    min_score = max(
        0.0,
        min(1.0, _to_float(getattr(settings, "REGIME_BULL_SHORT_RETRACE_MIN_SCORE", 0.88))),
    )
    if score < min_score:
        return False, f"bull_short_low_retrace_score:{score:.3f}<{min_score:.3f}"

    active_modules = set(_signal_active_modules(sig_payload if isinstance(sig_payload, dict) else {}, strategy_name))
    allowed_set = get_runtime_str_list(
        "REGIME_BULL_SHORT_RETRACE_ALLOWED_MODULES",
        getattr(settings, "REGIME_BULL_SHORT_RETRACE_ALLOWED_MODULES", {"meanrev", "smc", "carry"}),
    )
    if not allowed_set:
        allowed_set = {"meanrev", "smc", "carry"}
    needed = get_runtime_int(
        "REGIME_BULL_SHORT_RETRACE_MIN_ALLOWED_MODULES",
        int(getattr(settings, "REGIME_BULL_SHORT_RETRACE_MIN_ALLOWED_MODULES", 1) or 1),
        minimum=1,
    )
    matched = active_modules & allowed_set
    if len(matched) < needed:
        return False, f"bull_short_low_retrace_modules:{','.join(sorted(active_modules)) or 'none'}"
    return True, "ok"


def _resolve_signal_direction(strategy_name: str) -> tuple[str, str]:
    strategy_name = str(strategy_name or "").strip().lower()
    signal_direction = "flat"
    side = ""
    if strategy_name.startswith("alloc_"):
        alloc_direction = strategy_name.replace("alloc_", "", 1)
        if alloc_direction in {"long", "short"}:
            signal_direction = alloc_direction
            side = "buy" if alloc_direction == "long" else "sell"
    else:
        if "long" in strategy_name:
            signal_direction = "long"
            side = "buy"
        elif "short" in strategy_name:
            signal_direction = "short"
            side = "sell"
    return signal_direction, side


def _resolve_market_snapshot(
    adapter,
    inst_symbol: str,
    atr_pct: float | None,
) -> tuple[str, float | None, dict[str, Any] | None, float | None, float, Any]:
    symbol = inst_symbol
    alt_symbol = None
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        alt_symbol = f"{base}/USDT:USDT"

    max_spread_bps = _max_spread_bps(inst_symbol, atr_pct)
    last_price = None
    ticker_used = None
    spread_bps_selected = None
    contract_size = 1.0
    market_info = None

    for sym in (symbol, alt_symbol):
        if not sym:
            continue
        try:
            ticker = adapter.fetch_ticker(sym)
            spread_bps = _spread_bps(ticker)
            if spread_bps is not None and spread_bps > max_spread_bps:
                logger.info(
                    "Spread too wide on %s: %.1f bps > %.1f bps cap",
                    sym,
                    spread_bps,
                    max_spread_bps,
                )
                continue
            last_price = ticker.get("last")
            try:
                market = adapter.client.market(adapter._map_symbol(sym))  # type: ignore[attr-defined]
                contract_size = float(market.get("contractSize") or 1.0)
                market_info = market
            except Exception:
                contract_size = 1.0
                market_info = None
            if last_price:
                last_price = float(last_price)
                symbol = sym
                ticker_used = ticker
                spread_bps_selected = spread_bps
                break
        except Exception:
            continue

    return symbol, last_price, ticker_used, spread_bps_selected, contract_size, market_info


def _attempt_entry_open(
    *,
    adapter,
    inst: Instrument,
    sig: Signal,
    sig_payload: dict[str, Any],
    strategy_name: str,
    side: str,
    signal_direction: str,
    direction_allowed: bool,
    signal_expired: bool,
    can_open: bool,
    macro_active: bool,
    macro_context: dict[str, Any],
    macro_block_entries: bool,
    macro_risk_mult: float,
    regime_blocked_symbols: set[str],
    regime_adx_by_symbol: dict[str, float],
    regime_adx_min_by_symbol: dict[str, float],
    regime_bias_by_symbol: dict[str, str],
    regime_adx_min: float,
    market_regime_adx: float | None,
    mtf_symbol_snapshot: dict[str, Any],
    btc_lead_state: str,
    btc_recommended_bias: str,
    allow_scale_entry: bool,
    scale_parent_correlation: str,
    scale_add_index: int,
    session_policy_enabled: bool,
    session_dead_zone_block: bool,
    current_session: str,
    session_min_score: float,
    session_risk_mult: float,
    ml_entry_filter_enabled: bool,
    ml_entry_filter_default_min_prob: float,
    ml_entry_filter_fail_open: bool,
    use_allocator_signals: bool,
    symbol: str,
    last_price: float,
    contract_size: float,
    market_info: Any,
    atr: float | None,
    sl_pct: float,
    spread_bps_selected: float | None,
    free_usdt: float,
    equity_usdt: float,
    leverage: float,
    total_notional: float,
    cycle_notional_added: float,
    account_ai_enabled: bool,
    account_ai_config_id: int | None,
    account_owner_id: int | None,
    account_alias: str,
    account_service: str,
    positions_snapshot: list[dict] | None = None,
) -> tuple[int, float]:
    if not can_open:
        return 0, 0.0

    if signal_expired:
        return 0, 0.0
    if signal_direction not in {"long", "short"}:
        return 0, 0.0
    if not direction_allowed:
        return 0, 0.0

    entry_reason = _signal_entry_reason(sig_payload, strategy_name)
    macro_override_allowed = bool(
        macro_active
        and macro_block_entries
        and _macro_high_impact_allows_entry(strategy_name=strategy_name, symbol=inst.symbol)
    )
    if macro_active and macro_block_entries and not macro_override_allowed:
        logger.info(
            "Macro high-impact window blocked entry on %s (session=%s hour=%s weekday=%s)",
            inst.symbol,
            macro_context.get("session"),
            macro_context.get("hour_utc"),
            macro_context.get("weekday"),
        )
        return 0, 0.0
    if macro_override_allowed:
        logger.info(
            "Macro high-impact window allowing microvol entry on %s with reduced risk "
            "(session=%s hour=%s weekday=%s)",
            inst.symbol,
            macro_context.get("session"),
            macro_context.get("hour_utc"),
            macro_context.get("weekday"),
        )

    if inst.symbol in regime_blocked_symbols and not allow_scale_entry:
        effective_regime_adx_min = float(regime_adx_min_by_symbol.get(inst.symbol, regime_adx_min))
        logger.info(
            "Market regime gate blocked entry on %s: 1h ADX=%.1f < %.1f",
            inst.symbol,
            regime_adx_by_symbol.get(inst.symbol, 0),
            effective_regime_adx_min,
        )
        return 0, 0.0

    if not allow_scale_entry:
        htf_adx_for_limit = regime_adx_by_symbol.get(inst.symbol, market_regime_adx)
        daily_limit = _max_daily_trades_for_adx(htf_adx_for_limit)
        daily_count = _get_daily_trade_count()
        if daily_count >= daily_limit:
            logger.info(
                "Daily trade limit reached: %d/%d (adx=%.1f) Ã¢â‚¬â€ blocking entry on %s",
                daily_count,
                daily_limit,
                htf_adx_for_limit or 0,
                inst.symbol,
            )
            return 0, 0.0

    sig_score = _to_float(getattr(sig, "score", 0.0))
    if session_policy_enabled and session_dead_zone_block and is_dead_session(current_session):
        (
            dead_override_allowed,
            dead_override_reason,
            dead_override_score_min,
            dead_override_risk_mult,
        ) = _dead_session_strong_trend_breakout_override(
            strategy_name=strategy_name,
            signal_direction=signal_direction,
            current_session=current_session,
            sig_score=sig_score,
            sig_payload=sig_payload if isinstance(sig_payload, dict) else {},
        )
        if not dead_override_allowed:
            logger.info(
                "Session dead zone active, skipping new entry for %s (session=%s reason=%s)",
                inst.symbol,
                current_session,
                dead_override_reason,
            )
            return 0, 0.0
        if dead_override_score_min is not None:
            session_min_score = float(dead_override_score_min)
        if dead_override_risk_mult is not None:
            session_risk_mult = float(dead_override_risk_mult)
        logger.info(
            "Dead-session breakout override allowing entry on %s: reason=%s score_min=%.3f risk_mult=%.3f",
            inst.symbol,
            dead_override_reason,
            float(session_min_score),
            float(session_risk_mult),
        )

    exec_min_score = session_min_score if session_policy_enabled else settings.EXECUTION_MIN_SIGNAL_SCORE
    if sig_score < exec_min_score:
        logger.info(
            "Signal score too low for execution on %s: %.3f < %.3f (session=%s)",
            inst.symbol,
            sig_score,
            exec_min_score,
            current_session if session_policy_enabled else "n/a",
        )
        return 0, 0.0

    regime_bias = str(regime_bias_by_symbol.get(inst.symbol, "neutral") or "neutral").strip().lower()
    retrace_ok, retrace_reason = _bull_short_retrace_precheck(
        symbol=inst.symbol,
        strategy_name=strategy_name,
        signal_direction=signal_direction,
        regime_bias=regime_bias,
        sig_score=sig_score,
        sig_payload=sig_payload if isinstance(sig_payload, dict) else {},
    )
    if not retrace_ok:
        logger.info(
            "Pre-AI bull short retrace gate blocked %s: reason=%s",
            inst.symbol,
            retrace_reason,
        )
        return 0, 0.0

    btc_lead_mult, btc_lead_blocked, btc_lead_reason = _btc_lead_directional_risk_mult(
        inst.symbol,
        signal_direction,
        btc_lead_state,
        btc_recommended_bias,
    )
    if btc_lead_blocked:
        logger.info(
            "BTC lead context blocked entry on %s: lead=%s bias=%s reason=%s",
            inst.symbol,
            btc_lead_state,
            btc_recommended_bias,
            btc_lead_reason,
        )
        return 0, 0.0

    ny_open_ok, ny_open_reason = _ny_open_weak_long_precheck(
        strategy_name=strategy_name,
        signal_direction=signal_direction,
        current_session=current_session,
        btc_lead_state=btc_lead_state,
        btc_recommended_bias=btc_recommended_bias,
    )
    if not ny_open_ok:
        logger.info(
            "NY open weak-long context blocked entry on %s: lead=%s bias=%s reason=%s",
            inst.symbol,
            btc_lead_state,
            btc_recommended_bias,
            ny_open_reason,
        )
        return 0, 0.0

    trend_context = (((sig_payload or {}).get("reasons", {}) or {}).get("trend_context", {}) or {})
    trend_context_direction = str(trend_context.get("direction", "")).strip().lower()
    trend_context_is_strong = bool(trend_context.get("is_strong"))
    trend_context_confidence = _to_float(trend_context.get("confidence", 0.0))
    trend_context_adx_htf = _to_float(trend_context.get("adx_htf", 0.0))

    weak_long_ok, weak_long_reason = _weak_long_bear_weak_precheck(
        strategy_name=strategy_name,
        signal_direction=signal_direction,
        current_session=current_session,
        monthly_regime=str(mtf_symbol_snapshot.get("monthly_regime", "") or ""),
        daily_regime=str(mtf_symbol_snapshot.get("daily_regime", "") or ""),
        btc_lead_state=btc_lead_state,
        btc_recommended_bias=btc_recommended_bias,
        sig_score=sig_score,
        symbol_adx_1h=regime_adx_by_symbol.get(inst.symbol, 0.0),
        trend_context_direction=trend_context_direction,
        trend_context_is_strong=trend_context_is_strong,
    )
    if not weak_long_ok:
        logger.info(
            "Weak-long bear-weak context blocked entry on %s: month=%s day=%s lead=%s bias=%s reason=%s",
            inst.symbol,
            mtf_symbol_snapshot.get("monthly_regime", ""),
            mtf_symbol_snapshot.get("daily_regime", ""),
            btc_lead_state,
            btc_recommended_bias,
            weak_long_reason,
        )
        return 0, 0.0

    weak_short_ok, weak_short_reason = _asia_weak_short_precheck(
        strategy_name=strategy_name,
        signal_direction=signal_direction,
        current_session=current_session,
        btc_lead_state=btc_lead_state,
        btc_recommended_bias=btc_recommended_bias,
        trend_context_direction=trend_context_direction,
        trend_context_is_strong=trend_context_is_strong,
        trend_context_confidence=trend_context_confidence,
        trend_context_adx_htf=trend_context_adx_htf,
    )
    if not weak_short_ok:
        logger.info(
            "Asia weak-short context blocked entry on %s: session=%s lead=%s bias=%s trend_dir=%s trend_strong=%s trend_conf=%.3f trend_adx=%.1f reason=%s",
            inst.symbol,
            current_session,
            btc_lead_state,
            btc_recommended_bias,
            trend_context_direction or "n/a",
            trend_context_is_strong,
            trend_context_confidence,
            trend_context_adx_htf,
            weak_short_reason,
        )
        return 0, 0.0

    weak_short_transition_ok, weak_short_transition_reason = _weak_short_transition_precheck(
        strategy_name=strategy_name,
        signal_direction=signal_direction,
        current_session=current_session,
        daily_regime=str(mtf_symbol_snapshot.get("daily_regime", "") or ""),
        btc_lead_state=btc_lead_state,
        btc_recommended_bias=btc_recommended_bias,
        trend_context_direction=trend_context_direction,
        trend_context_is_strong=trend_context_is_strong,
    )
    if not weak_short_transition_ok:
        logger.info(
            "Weak short transition context blocked entry on %s: session=%s day=%s lead=%s bias=%s trend_dir=%s trend_strong=%s reason=%s",
            inst.symbol,
            current_session,
            mtf_symbol_snapshot.get("daily_regime", ""),
            btc_lead_state,
            btc_recommended_bias,
            trend_context_direction or "n/a",
            trend_context_is_strong,
            weak_short_transition_reason,
        )
        return 0, 0.0

    long_bias_short_ok, long_bias_short_reason = _long_bias_short_precheck(
        strategy_name=strategy_name,
        signal_direction=signal_direction,
        btc_recommended_bias=btc_recommended_bias,
        sig_score=sig_score,
        sig_payload=sig_payload if isinstance(sig_payload, dict) else {},
    )
    if not long_bias_short_ok:
        logger.info(
            "Long-bias short context blocked entry on %s: bias=%s reason=%s",
            inst.symbol,
            btc_recommended_bias,
            long_bias_short_reason,
        )
        return 0, 0.0

    symbol_health_ok, symbol_health_reason = _symbol_health_precheck(inst)
    if not symbol_health_ok:
        logger.info(
            "Symbol health guard blocked entry on %s: reason=%s",
            inst.symbol,
            symbol_health_reason,
        )
        return 0, 0.0

    # --- Cross-symbol correlation guard ---
    corr_risk_mult, corr_reason = _cross_symbol_correlation_guard(
        inst=inst,
        signal_direction=signal_direction,
        positions_snapshot=positions_snapshot or [],
        equity_usdt=equity_usdt,
    )
    if corr_risk_mult <= 0.0:
        logger.info(
            "Cross-symbol correlation guard blocked entry on %s %s: %s",
            inst.symbol, signal_direction, corr_reason,
        )
        return 0, 0.0

    # --- Volume delta imbalance check ---
    vd_ok, vd_bonus, vd_reason = _volume_delta_check(
        symbol=inst.symbol,
        signal_direction=signal_direction,
    )
    if not vd_ok:
        logger.info(
            "Volume delta gate blocked entry on %s %s: %s",
            inst.symbol, signal_direction, vd_reason,
        )
        return 0, 0.0

    ml_prob = None
    if ml_entry_filter_enabled:
        ml_model_path = _ml_entry_filter_model_path(inst.symbol, strategy_name=strategy_name)
        ml_entry_filter_min_prob = _ml_entry_filter_min_prob(
            ml_entry_filter_default_min_prob,
            symbol=inst.symbol,
            strategy_name=strategy_name,
        )
        ml_prob = predict_entry_success_probability(
            strategy_name=strategy_name,
            symbol=inst.symbol,
            sig_score=sig_score,
            payload=sig_payload,
            model_path=ml_model_path,
            atr_pct=atr,
            spread_bps=spread_bps_selected,
        )
        if ml_prob is None:
            if not ml_entry_filter_fail_open:
                logger.info(
                    "ML entry filter blocked %s: model unavailable path=%s",
                    inst.symbol,
                    ml_model_path,
                )
                return 0, 0.0
            logger.info(
                "ML entry filter unavailable on %s (path=%s); fail-open",
                inst.symbol,
                ml_model_path,
            )
        elif ml_prob < ml_entry_filter_min_prob:
            logger.info(
                "ML entry filter blocked entry on %s: prob=%.3f < %.3f strategy=%s",
                inst.symbol,
                ml_prob,
                ml_entry_filter_min_prob,
                strategy_name,
            )
            return 0, 0.0

    base_cooldown_min = settings.PER_INSTRUMENT_COOLDOWN.get(
        inst.symbol,
        settings.SIGNAL_COOLDOWN_MINUTES,
    )
    if _strategy_is_microvol(strategy_name):
        base_cooldown_min = max(
            0,
            int(getattr(settings, "MODULE_MICROVOL_COOLDOWN_MINUTES", base_cooldown_min) or base_cooldown_min),
        )
    sl_cooldown_min = int(getattr(settings, "SIGNAL_COOLDOWN_AFTER_SL_MINUTES", base_cooldown_min))
    if base_cooldown_min > 0 and not allow_scale_entry:
        last_order = (
            Order.objects.filter(
                instrument=inst,
                status=Order.OrderStatus.FILLED,
                opened_at__isnull=False,
            )
            .order_by("-opened_at")
            .values_list("opened_at", flat=True)
            .first()
        )
        last_op = (
            OperationReport.objects.filter(
                instrument=inst,
                closed_at__isnull=False,
            )
            .order_by("-closed_at")
            .first()
        )
        last_close = last_op.closed_at if last_op else None
        last_close_reason = getattr(last_op, "reason", "") if last_op else ""
        anchor_dt = None
        anchor_src = ""
        if last_order and last_close:
            if last_close >= last_order:
                anchor_dt = last_close
                anchor_src = "close"
            else:
                anchor_dt = last_order
                anchor_src = "open"
        elif last_close:
            anchor_dt = last_close
            anchor_src = "close"
        elif last_order:
            anchor_dt = last_order
            anchor_src = "open"

        if anchor_dt:
            cooldown_min = base_cooldown_min
            if anchor_src == "close" and last_close_reason in ("sl", "stale_cleanup", "uptrend_short_kill"):
                cooldown_min = sl_cooldown_min
            elapsed_min = (dj_tz.now() - anchor_dt).total_seconds() / 60
            if elapsed_min < cooldown_min:
                logger.info(
                    "Cooldown active for %s: %.1f min elapsed < %d min required (anchor=%s reason=%s)",
                    inst.symbol,
                    elapsed_min,
                    cooldown_min,
                    anchor_src,
                    last_close_reason or "n/a",
                )
                return 0, 0.0

    post_tp_reentry_ok, post_tp_reentry_reason = _post_tp_alt_reentry_quality_precheck(
        inst=inst,
        strategy_name=strategy_name,
        signal_direction=signal_direction,
        sig_payload=sig_payload if isinstance(sig_payload, dict) else {},
    )
    if not post_tp_reentry_ok:
        logger.info(
            "Post-TP alt reentry quality gate blocked entry on %s: reason=%s",
            inst.symbol,
            post_tp_reentry_reason,
        )
        return 0, 0.0

    volume_allowed, volume_ratio = _volume_gate_allowed(
        inst,
        session_name=current_session,
    )
    if not volume_allowed:
        tf = str(getattr(settings, "ENTRY_VOLUME_FILTER_TIMEFRAME", "5m") or "5m").strip().lower()
        lookback = max(10, int(getattr(settings, "ENTRY_VOLUME_FILTER_LOOKBACK", 48) or 48))
        min_ratio = _volume_gate_min_ratio(current_session)
        if volume_ratio is None:
            logger.info(
                "Volume gate blocked %s: insufficient %s data (session=%s need ~%d bars)",
                inst.symbol,
                tf,
                current_session,
                lookback + 1,
            )
        else:
            logger.info(
                "Volume gate blocked %s: ratio=%.2f < %.2f (session=%s tf=%s lookback=%d)",
                inst.symbol,
                volume_ratio,
                min_ratio,
                current_session,
                tf,
                lookback,
            )
        return 0, 0.0

    ai_market_fp = _ai_entry_market_fingerprint(
        symbol=inst.symbol,
        strategy_name=strategy_name,
        signal_direction=signal_direction,
        session_name=current_session,
        sig_score=sig_score,
        atr=atr,
        spread_bps=spread_bps_selected,
        sl_pct=sl_pct,
        sig_payload=sig_payload if isinstance(sig_payload, dict) else {},
    )
    ai_market_fp_coarse = _ai_entry_market_fingerprint(
        symbol=inst.symbol,
        strategy_name=strategy_name,
        signal_direction=signal_direction,
        session_name=current_session,
        sig_score=sig_score,
        atr=atr,
        spread_bps=spread_bps_selected,
        sl_pct=sl_pct,
        sig_payload=sig_payload if isinstance(sig_payload, dict) else {},
        coarse=True,
    )
    ai_retry_suppressed, ai_retry_reason = _ai_entry_should_suppress_retry(
        account_alias=account_alias,
        account_service=account_service,
        symbol=inst.symbol,
        strategy_name=strategy_name,
        signal_direction=signal_direction,
        market_fingerprint=ai_market_fp,
        market_fingerprint_coarse=ai_market_fp_coarse,
    )
    if not ai_retry_suppressed:
        ai_retry_suppressed, ai_retry_reason = _ai_entry_should_suppress_retry_from_feedback(
            account_alias=account_alias,
            account_service=account_service,
            symbol=inst.symbol,
            strategy_name=strategy_name,
            signal_direction=signal_direction,
            session_name=current_session,
            sig_score=sig_score,
            spread_bps=spread_bps_selected,
            market_fingerprint=ai_market_fp,
            market_fingerprint_coarse=ai_market_fp_coarse,
        )
    if ai_retry_suppressed:
        logger.info(
            "AI gate retry suppressed on %s: unchanged market conditions (reason=%s)",
            inst.symbol,
            ai_retry_reason or "ai_reject_cached",
        )
        return 0, 0.0

    ai_risk_mult = 1.0
    ai_allow, ai_risk_mult, ai_reason, ai_meta = evaluate_ai_entry_gate(
        account_ai_enabled=account_ai_enabled,
        account_ai_config_id=account_ai_config_id,
        account_owner_id=account_owner_id,
        account_alias=account_alias,
        account_service=account_service,
        symbol=inst.symbol,
        strategy_name=strategy_name,
        signal_direction=signal_direction,
        sig_score=sig_score,
        atr=atr,
        spread_bps=spread_bps_selected,
        sl_pct=sl_pct,
        session_name=current_session,
        sig_payload=sig_payload if isinstance(sig_payload, dict) else {},
        market_fingerprint=ai_market_fp,
        market_fingerprint_coarse=ai_market_fp_coarse,
    )
    if not ai_allow:
        _ai_entry_mark_rejected(
            account_alias=account_alias,
            account_service=account_service,
            symbol=inst.symbol,
            strategy_name=strategy_name,
            signal_direction=signal_direction,
            market_fingerprint=ai_market_fp,
            market_fingerprint_coarse=ai_market_fp_coarse,
            reason=ai_reason,
        )
        logger.info(
            "AI gate blocked entry on %s: reason=%s cfg=%s",
            inst.symbol,
            ai_reason,
            ai_meta.get("cfg_alias", "n/a"),
        )
        return 0, 0.0
    _ai_entry_clear_reject_cache(
        account_alias=account_alias,
        account_service=account_service,
        symbol=inst.symbol,
        strategy_name=strategy_name,
        signal_direction=signal_direction,
    )
    if ai_risk_mult < 1.0:
        logger.info(
            "AI gate risk reduction on %s: risk_mult=%.3f reason=%s",
            inst.symbol,
            ai_risk_mult,
            ai_reason,
        )
    ai_risk_mult = max(0.0, min(float(ai_risk_mult), 1.0))
    regime_mult, regime_blocked, regime_reason = _regime_directional_risk_mult(
        inst.symbol,
        signal_direction,
        regime_bias,
    )
    if regime_blocked:
        logger.info(
            "Regime directional block on %s: bias=%s reason=%s",
            inst.symbol,
            regime_bias,
            regime_reason,
        )
        return 0, 0.0
    if regime_mult < 1.0:
        ai_risk_mult *= regime_mult
        logger.info(
            "Regime directional penalty on %s: bias=%s mult=%.3f reason=%s",
            inst.symbol,
            regime_bias,
            regime_mult,
            regime_reason,
        )
    if btc_lead_mult < 1.0:
        ai_risk_mult *= btc_lead_mult
        logger.info(
            "BTC lead penalty on %s: lead=%s bias=%s mult=%.3f reason=%s",
            inst.symbol,
            btc_lead_state,
            btc_recommended_bias,
            btc_lead_mult,
            btc_lead_reason,
        )
    ai_risk_mult = max(0.0, min(float(ai_risk_mult), 1.0))

    entry_leverage, lev_reason = _confidence_adjusted_entry_leverage(
        base_leverage=leverage,
        strategy_name=strategy_name,
        sig_score=sig_score,
        ml_prob=ml_prob,
        ml_enabled=ml_entry_filter_enabled,
        signal_direction=signal_direction,
    )
    if entry_leverage > float(leverage or 0):
        logger.info(
            "Confidence leverage boost on %s: %.2fx -> %.2fx (%s; sig=%.3f ml=%s)",
            inst.symbol,
            float(leverage or 0),
            entry_leverage,
            lev_reason,
            sig_score,
            f"{ml_prob:.3f}" if ml_prob is not None else "n/a",
        )

    inst_notional = 0.0
    try:
        pos_obj = Position.objects.filter(instrument=inst, is_open=True).first()
        if pos_obj:
            inst_notional = float(pos_obj.notional_usdt or 0)
    except Exception:
        pass
    max_inst_notional = equity_usdt * settings.MAX_EXPOSURE_PER_INSTRUMENT_PCT * entry_leverage
    if inst_notional >= max_inst_notional and max_inst_notional > 0:
        logger.info("Per-instrument exposure cap reached for %s (%.2f >= %.2f)", symbol, inst_notional, max_inst_notional)
        return 0, 0.0

    is_allocator_signal = use_allocator_signals and strategy_name.startswith("alloc_")
    if is_allocator_signal:
        inst_risk_pct = max(0.0, _to_float(sig_payload.get("risk_budget_pct", 0.0)))
        effective_risk_mult = ai_risk_mult if ai_risk_mult > 0 else 0.0
        if inst_risk_pct <= 0:
            logger.info(
                "Allocator blocked entry on %s: risk_budget_pct=%.5f strategy=%s",
                inst.symbol,
                inst_risk_pct,
                strategy_name,
            )
            return 0, 0.0
        if macro_active and (not macro_block_entries or macro_override_allowed):
            inst_risk_pct *= macro_risk_mult
    else:
        effective_risk_mult = (session_risk_mult if session_policy_enabled else 1.0) * ai_risk_mult
        if _strategy_is_microvol(strategy_name):
            effective_risk_mult *= float(getattr(settings, "MODULE_MICROVOL_RISK_MULT", 0.35) or 0.35)
        if macro_active and (not macro_block_entries or macro_override_allowed):
            effective_risk_mult *= macro_risk_mult
        if effective_risk_mult <= 0:
            logger.info(
                "Session policy blocked entry on %s: risk_mult=%.2f session=%s",
                inst.symbol,
                effective_risk_mult,
                current_session,
            )
            return 0, 0.0
        inst_risk_pct = settings.PER_INSTRUMENT_RISK.get(
            inst.symbol,
            settings.RISK_PER_TRADE_PCT,
        )

    inst_risk_pct = _volatility_adjusted_risk(inst.symbol, atr, inst_risk_pct)
    effective_risk_pct = inst_risk_pct * effective_risk_mult

    # Cross-symbol directional concentration risk reduction
    if corr_risk_mult < 1.0:
        effective_risk_pct *= corr_risk_mult
        logger.info(
            "Cross-symbol corr guard risk reduction on %s: mult=%.2f (%s)",
            inst.symbol, corr_risk_mult, corr_reason,
        )

    # Symbol heat guard — progressive risk reduction on losing streaks
    heat_mult, heat_reason = _symbol_heat_guard(inst.symbol, side=side)
    if heat_mult < 1.0:
        effective_risk_pct *= heat_mult
        logger.info(
            "Symbol heat guard risk reduction on %s: mult=%.2f (%s)",
            inst.symbol, heat_mult, heat_reason,
        )

    # Optional Fractional Kelly Risk Boost (disabled by default via feature flag)
    if getattr(settings, "CONFIDENCE_RISK_BOOST_ENABLED", False):
        try:
            confidence_mult = 1.0
            
            # 1. Boost via High Allocator Confluence Score
            if sig_score > getattr(settings, "CONFIDENCE_SCORE_THRESHOLD", 0.85):
                confidence_mult += getattr(settings, "CONFIDENCE_SCORE_BOOST", 0.25)
                
            # 2. Boost via Machine Learning Probability (if enabled)
            if ml_entry_filter_enabled and ml_prob is not None:
                if ml_prob > getattr(settings, "CONFIDENCE_ML_PROB_THRESHOLD", 0.70):
                    confidence_mult += getattr(settings, "CONFIDENCE_ML_BOOST", 0.25)
                    
            # 3. Cap the multiplier (Fractional Kelly Safety)
            max_boost_mult = getattr(settings, "CONFIDENCE_MAX_MULT", 1.5)
            confidence_mult = min(confidence_mult, max_boost_mult)
            
            if confidence_mult > 1.0:
                logger.info(
                    "Applying Fractional Kelly Boost on %s: mult=%.2f "
                    "(sig_score=%.3f, ml_prob=%s)",
                    inst.symbol, confidence_mult, sig_score, 
                    f"{ml_prob:.3f}" if ml_prob is not None else "None"
                )
                effective_risk_pct *= confidence_mult
                
        except Exception as exc:
            logger.warning("Error calculating Confidence Risk Boost for %s: %s", inst.symbol, exc)

    if allow_scale_entry:
        pyramid_risk_scale = max(
            0.0,
            min(float(getattr(settings, "PYRAMID_RISK_SCALE", 0.6) or 0.6), 1.0),
        )
        effective_risk_pct *= pyramid_risk_scale
    if effective_risk_pct <= 0:
        return 0, 0.0

    lev_ok, lev_set_reason = _ensure_entry_leverage(adapter, symbol, entry_leverage)
    if not lev_ok:
        if entry_leverage > float(leverage or 0):
            logger.warning(
                "Leverage boost fallback to base on %s: target=%.2fx err=%s",
                inst.symbol,
                entry_leverage,
                lev_set_reason,
            )
        entry_leverage = max(1.0, float(leverage or 1.0))
    elif lev_set_reason not in {"cached", "ok"}:
        logger.info("Leverage set status on %s: %s", inst.symbol, lev_set_reason)

    grid_sl_hint, grid_tp_hint, grid_stop_dist = _grid_structural_sl_tp_hints(
        strategy_name=strategy_name,
        sig_payload=sig_payload if isinstance(sig_payload, dict) else {},
        side=side,
        entry_price=last_price,
    )
    stop_dist = float(sl_pct or 0.0) if atr is not None else 0.0
    if grid_stop_dist > 0:
        stop_dist = grid_stop_dist
    min_atr_for_entry = float(getattr(settings, "MIN_ATR_FOR_ENTRY", 0.003) or 0.003)
    if atr is not None and atr < min_atr_for_entry:
        logger.info(
            "ATR too low for %s (%.4f%% < %.4f%%), market compressed Ã¢â‚¬â€ skipping entry",
            symbol,
            atr * 100,
            min_atr_for_entry * 100,
        )
        return 0, 0.0

    if stop_dist and stop_dist > 0:
        qty = _risk_based_qty(
            equity_usdt,
            last_price,
            stop_dist,
            contract_size,
            entry_leverage,
            risk_pct=effective_risk_pct,
        )
    else:
        qty = settings.ORDER_SIZE_USDT / (last_price * contract_size)
        base_risk_pct = max(float(settings.RISK_PER_TRADE_PCT), 1e-9)
        qty *= (effective_risk_pct / base_risk_pct)

    precision_mode = getattr(adapter.client, "precisionMode", None)
    min_qty = _market_min_qty(
        market_info,
        fallback=float(inst.lot_size or 0.0),
        precision_mode=precision_mode,
        last_price=last_price,
        contract_size=contract_size,
    )
    min_qty = _align_min_order_qty(
        adapter,
        symbol,
        min_qty,
        market=market_info,
        precision_mode=precision_mode,
    )
    qty = _normalize_order_qty(adapter, symbol, qty)
    risk_qty = qty
    if min_qty > 0 and qty < min_qty:
        qty = min_qty

    if qty <= 0 or (min_qty > 0 and qty < min_qty):
        logger.warning(
            "Computed qty invalid for %s qty=%.10f min_qty=%.10f",
            inst.symbol,
            qty,
            min_qty,
        )
        return 0, 0.0

    notional_order = last_price * contract_size * qty
    effective_total_notional = total_notional + cycle_notional_added
    if equity_usdt > 0:
        max_new_notional = max(0.0, settings.MAX_EFF_LEVERAGE * equity_usdt - effective_total_notional)
        if notional_order > max_new_notional:
            capped_qty = _normalize_order_qty(
                adapter,
                symbol,
                max_new_notional / (last_price * contract_size),
            )
            if capped_qty < min_qty:
                logger.info(
                    "Pre-trade leverage cap: %s new_notional=%.2f would exceed MAX_EFF_LEVERAGE=%.1f "
                    "(total_notional=%.2f+%.2f equity=%.2f max_new=%.2f min_qty=%s); skipping",
                    symbol,
                    notional_order,
                    settings.MAX_EFF_LEVERAGE,
                    total_notional,
                    cycle_notional_added,
                    equity_usdt,
                    max_new_notional,
                    min_qty,
                )
                return 0, 0.0
            logger.info(
                "Pre-trade leverage cap: %s qty %.10f -> %.10f (max_new_notional=%.2f)",
                symbol,
                qty,
                capped_qty,
                max_new_notional,
            )
            qty = capped_qty
            notional_order = last_price * contract_size * qty

    margin_buffer_pct = max(
        0.0,
        min(
            float(getattr(settings, "ORDER_MARGIN_BUFFER_PCT", 0.03) or 0.03),
            float(getattr(settings, "ORDER_MARGIN_BUFFER_MAX_PCT", 0.20) or 0.20),
        ),
    )
    usable_margin = free_usdt * (1.0 - margin_buffer_pct)
    if usable_margin <= 0:
        logger.info(
            "Margin unavailable %s: free=%.2f buffer=%.2f%%",
            symbol,
            free_usdt,
            margin_buffer_pct * 100,
        )
        return 0, 0.0
    max_qty_margin = _normalize_order_qty(
        adapter,
        symbol,
        (usable_margin * entry_leverage if entry_leverage else usable_margin) / (last_price * contract_size),
    )
    if max_qty_margin < min_qty:
        logger.info(
            "Margin cap too low %s: max_qty=%s < min_qty=%s (free=%.2f buffer=%.2f%%)",
            symbol,
            max_qty_margin,
            min_qty,
            free_usdt,
            margin_buffer_pct * 100,
        )
        return 0, 0.0
    if qty > max_qty_margin:
        logger.info(
            "Margin fit %s qty %s -> %s (free=%.2f buffer=%.2f%%)",
            symbol,
            qty,
            max_qty_margin,
            free_usdt,
            margin_buffer_pct * 100,
        )
        qty = max_qty_margin
        notional_order = last_price * contract_size * qty

    target_risk_amount = max(0.0, equity_usdt * effective_risk_pct)
    actual_stop_risk_amount = _actual_stop_risk_amount(
        qty=qty,
        entry_price=last_price,
        stop_distance_pct=stop_dist,
        contract_size=contract_size,
    )
    actual_risk_mult = (
        actual_stop_risk_amount / target_risk_amount
        if actual_stop_risk_amount > 0 and target_risk_amount > 0
        else 0.0
    )
    absolute_cap_allows, actual_risk_pct, absolute_cap_pct = _min_qty_absolute_risk_cap_allows(
        actual_risk_amount=actual_stop_risk_amount,
        equity_usdt=equity_usdt,
    )
    allowlist_state = _min_qty_dynamic_allowlist_state(actual_risk_mult)
    if bool(getattr(settings, "MIN_QTY_DYNAMIC_ALLOWLIST_ENABLED", True)):
        if allowlist_state == "blocked":
            if absolute_cap_allows:
                logger.info(
                    "Dynamic min-qty allowlist absolute-cap override %s: "
                    "risk_actual=%.5f risk_pct=%.4f%% cap=%.4f%% risk_target=%.5f risk_mult=%.2f",
                    symbol,
                    actual_stop_risk_amount,
                    actual_risk_pct * 100,
                    absolute_cap_pct * 100,
                    target_risk_amount,
                    actual_risk_mult,
                )
            else:
                _record_min_qty_risk_guard_event(
                    inst,
                    qty=qty,
                    risk_qty=risk_qty,
                    min_qty=min_qty,
                    actual_risk_amount=actual_stop_risk_amount,
                    target_risk_amount=target_risk_amount,
                    risk_multiplier=actual_risk_mult,
                    stop_distance_pct=stop_dist,
                )
                logger.info(
                    "Dynamic min-qty allowlist blocked %s: state=%s qty=%.10f risk_qty=%.10f "
                    "min_qty=%.10f risk_actual=%.5f risk_target=%.5f risk_mult=%.2f",
                    symbol,
                    allowlist_state,
                    qty,
                    risk_qty,
                    min_qty,
                    actual_stop_risk_amount,
                    target_risk_amount,
                    actual_risk_mult,
                )
                return 0, 0.0
        if allowlist_state == "watch":
            logger.info(
                "Dynamic min-qty allowlist watch %s: qty=%.10f risk_qty=%.10f min_qty=%.10f risk_mult=%.2f",
                symbol,
                qty,
                risk_qty,
                min_qty,
                actual_risk_mult,
            )

    min_qty_risk_blocked, actual_stop_risk_amount, actual_risk_mult = _min_qty_risk_guard(
        qty=qty,
        risk_qty=risk_qty,
        min_qty=min_qty,
        entry_price=last_price,
        stop_distance_pct=stop_dist,
        contract_size=contract_size,
        target_risk_amount=target_risk_amount,
    )
    if min_qty_risk_blocked:
        absolute_cap_allows, actual_risk_pct, absolute_cap_pct = _min_qty_absolute_risk_cap_allows(
            actual_risk_amount=actual_stop_risk_amount,
            equity_usdt=equity_usdt,
        )
        if absolute_cap_allows:
            logger.info(
                "Min-qty risk guard absolute-cap override %s: "
                "risk_actual=%.5f risk_pct=%.4f%% cap=%.4f%% risk_target=%.5f risk_mult=%.2f",
                symbol,
                actual_stop_risk_amount,
                actual_risk_pct * 100,
                absolute_cap_pct * 100,
                target_risk_amount,
                actual_risk_mult,
            )
        else:
            _record_min_qty_risk_guard_event(
                inst,
                qty=qty,
                risk_qty=risk_qty,
                min_qty=min_qty,
                actual_risk_amount=actual_stop_risk_amount,
                target_risk_amount=target_risk_amount,
                risk_multiplier=actual_risk_mult,
                stop_distance_pct=stop_dist,
            )
            logger.info(
                "Min-qty risk guard blocked %s: qty=%.10f risk_qty=%.10f min_qty=%.10f "
                "risk_actual=%.5f risk_target=%.5f risk_mult=%.2f stop_pct=%.4f%%",
                symbol,
                qty,
                risk_qty,
                min_qty,
                actual_stop_risk_amount,
                target_risk_amount,
                actual_risk_mult,
                stop_dist * 100,
            )
            return 0, 0.0

    required_margin = notional_order / entry_leverage if entry_leverage else notional_order
    if required_margin > usable_margin:
        logger.info(
            "Margin insufficient %s: need=%.2f usable=%.2f free=%.2f",
            symbol,
            required_margin,
            usable_margin,
            free_usdt,
        )
        return 0, 0.0

    if use_allocator_signals and bool(getattr(settings, "SHADOW_TRADING_ENABLED", False)):
        logger.info(
            "Shadow trading ON; skipping live entry %s strategy=%s side=%s qty=%s",
            inst.symbol,
            strategy_name,
            side,
            qty,
        )
        return 0, 0.0

    tp_price, sl_price, _, _ = _compute_tp_sl_prices(
        side,
        last_price,
        atr,
        recommended_bias=btc_recommended_bias,
        strategy_name=strategy_name,
        regime_label=_resolve_regime_label(symbol),
    )
    # --- Grid structural SL/TP override ---
    if grid_sl_hint > 0:
        sl_price = grid_sl_hint
        logger.info("Grid SL override: sl_price=%.6f (structural)", sl_price)
    if grid_tp_hint > 0:
        tp_price = grid_tp_hint
        logger.info("Grid TP override: tp_price=%.6f (structural)", tp_price)
    base_correlation_id = _safe_correlation_id(f"{sig.id}-{inst.symbol}")
    correlation_id = base_correlation_id
    parent_correlation_id = base_correlation_id
    if allow_scale_entry:
        parent_correlation_id = _safe_correlation_id(scale_parent_correlation or base_correlation_id)
        correlation_id = _safe_correlation_id(f"{parent_correlation_id}:add{scale_add_index}")
    if not correlation_id:
        correlation_id = _safe_correlation_id(f"{inst.symbol}-{int(dj_tz.now().timestamp())}")
    if not parent_correlation_id:
        parent_correlation_id = correlation_id

    with transaction.atomic():
        order = Order.objects.create(
            instrument=inst,
            side=side,
            type=Order.OrderType.MARKET,
            qty=qty,
            price=last_price,
            status=Order.OrderStatus.NEW,
            correlation_id=correlation_id,
            leverage=entry_leverage,
            margin_mode=getattr(adapter, "margin_mode", ""),
            notional_usdt=notional_order,
            opened_at=dj_tz.now(),
            parent_correlation_id=parent_correlation_id,
        )

    try:
        order_params = None
        if "kucoin" in str(adapter.__class__.__name__ or "").strip().lower():
            order_params = {
                "leverage": int(max(1, round(float(entry_leverage or 1.0)))),
                "marginMode": str(getattr(adapter, "margin_mode", "cross") or "cross"),
            }
        resp = adapter.create_order(symbol, side, "market", float(order.qty), params=order_params)
        order.status = Order.OrderStatus.FILLED
        order.exchange_order_id = resp.get("id") or resp.get("orderId", "")
        order.raw_response = resp
        entry_fee_notional = max(
            _to_float(notional_order),
            _trade_notional_usdt(
                float(order.qty),
                _to_float(resp.get("average") or resp.get("price") or last_price),
                contract_size,
            ),
        )
        order.fee_usdt = _resolve_order_fee_usdt(resp, entry_fee_notional)
        order.closed_at = dj_tz.now()
        order.status_reason = ""
        order.save(update_fields=[
            "status", "exchange_order_id", "raw_response",
            "fee_usdt", "closed_at", "status_reason",
        ])
        _track_consecutive_errors(symbol, success=True)

        fill_price = _to_float(resp.get("average") or resp.get("price") or last_price)
        if fill_price and fill_price > 0:
            tp_price, sl_price, _, _ = _compute_tp_sl_prices(
                side,
                fill_price,
                atr,
                recommended_bias=btc_recommended_bias,
                strategy_name=strategy_name,
                regime_label=_resolve_regime_label(symbol),
            )
            # --- Grid structural SL/TP override (fill recalc) ---
            fill_grid_sl_hint, fill_grid_tp_hint, _ = _grid_structural_sl_tp_hints(
                strategy_name=strategy_name,
                sig_payload=sig_payload if isinstance(sig_payload, dict) else {},
                side=side,
                entry_price=fill_price,
            )
            if fill_grid_sl_hint > 0:
                sl_price = fill_grid_sl_hint
            if fill_grid_tp_hint > 0:
                tp_price = fill_grid_tp_hint
            logger.info(
                "SL/TP recalculated with fill_price=%.4f (last=%.4f slippage=%.4f%%)",
                fill_price,
                last_price,
                abs(fill_price - last_price) / last_price * 100 if last_price else 0,
            )

        placed = _place_sl_order(adapter, symbol, side, float(qty), sl_price)
        if placed is not None:
            _remember_protective_stop_price(symbol, order.opened_at, fill_price or last_price, sl_price)
        if not allow_scale_entry:
            _increment_daily_trade_count()

        notify_trade_opened(
            inst.symbol, side, float(qty), last_price,
            sl=sl_price, tp=tp_price, leverage=float(entry_leverage or 0),
            notional=notional_order, equity=equity_usdt,
            risk_pct=effective_risk_pct,
            score=float(sig.score) if sig else 0,
            strategy_name=strategy_name,
            active_modules=_signal_active_modules(sig_payload, strategy_name),
            entry_reason=entry_reason,
        )
        logger.info(
            "Opened %s %s qty=%s price=%.4f sl=%.4f tp=%.4f risk_sizing=%s risk_pct=%.5f session=%s mode=%s",
            symbol,
            side,
            qty,
            last_price,
            sl_price,
            tp_price,
            "atr" if stop_dist else "fixed",
            effective_risk_pct,
            current_session if session_policy_enabled else "n/a",
            "pyramid" if allow_scale_entry else ("allocator" if is_allocator_signal else "legacy"),
        )
        return 1, notional_order
    except Exception as exc:
        is_margin_error = _is_insufficient_margin_error(exc)
        discovered_min_qty = _minimum_order_amount_from_error(exc)
        if discovered_min_qty > 0:
            try:
                aligned_min_qty = _align_min_order_qty(
                    adapter,
                    symbol,
                    discovered_min_qty,
                    market=market_info,
                    precision_mode=getattr(adapter.client, "precisionMode", None),
                )
                if aligned_min_qty > 0:
                    discovered_min_qty = aligned_min_qty
                current_lot = _to_float(inst.lot_size or 0.0)
                if discovered_min_qty > current_lot:
                    inst.lot_size = Decimal(str(discovered_min_qty))
                    inst.save(update_fields=["lot_size"])
                    logger.warning(
                        "Updated lot_size from exchange reject for %s: %.10f",
                        inst.symbol,
                        discovered_min_qty,
                    )
            except Exception as min_exc:
                logger.warning("Failed to persist min qty for %s: %s", inst.symbol, min_exc)
        if is_margin_error:
            logger.info("Order rejected by exchange (margin) %s: %s", inst.symbol, exc)
        else:
            logger.warning("Order send failed %s: %s", inst.symbol, exc)
        order.status = Order.OrderStatus.REJECTED
        order.status_reason = str(exc)
        order.closed_at = dj_tz.now()
        order.save(update_fields=["status", "status_reason", "closed_at"])
        err_count = 0
        if is_margin_error:
            _track_consecutive_errors(symbol, success=True)
            record_ai_feedback_event(
                event_type="order_send_error",
                level="warning",
                account_alias=account_alias,
                account_service=account_service,
                symbol=inst.symbol,
                strategy=strategy_name,
                allow=False,
                risk_mult=0.0,
                reason=str(exc)[:255],
                payload={
                    "side": side,
                    "qty": qty,
                    "min_qty_discovered": discovered_min_qty,
                    "is_margin_error": True,
                    "err_count": 0,
                },
            )
            return 0, 0.0
        err_count = _track_consecutive_errors(symbol, success=False)
        notify_error(f"Order failed {inst.symbol}: {exc}")
        record_ai_feedback_event(
            event_type="order_send_error",
            level="error",
            account_alias=account_alias,
            account_service=account_service,
            symbol=inst.symbol,
            strategy=strategy_name,
            allow=False,
            risk_mult=0.0,
            reason=str(exc)[:255],
            payload={
                "side": side,
                "qty": qty,
                "min_qty_discovered": discovered_min_qty,
                "is_margin_error": False,
                "err_count": err_count,
            },
        )
        if err_count >= settings.MAX_CONSECUTIVE_ERRORS:
            _create_risk_event(
                "consecutive_errors",
                "critical",
                instrument=inst,
                details={"count": err_count, "last_error": str(exc)},
            )
            notify_kill_switch(f"{symbol}: {err_count} consecutive errors - pausing")
        return 0, 0.0


def _manage_open_position(
    *,
    adapter,
    inst: Instrument,
    sig: Signal,
    sig_payload: dict[str, Any],
    strategy_name: str,
    symbol: str,
    ticker_used: dict[str, Any] | None,
    last_price: float,
    current_qty: float,
    entry_price: float,
    pos_opened_at: datetime | None,
    signal_direction: str,
    side: str,
    direction_allowed: bool,
    atr: float | None,
    contract_size: float,
    leverage: float,
    equity_usdt: float,
    current_session: str,
    btc_recommended_bias: str,
    account_ai_enabled: bool,
    account_ai_config_id: int | None,
    account_owner_id: int | None,
    account_alias: str,
    account_service: str,
    # Flip pre-validation context (optional for backward compat)
    flip_can_open: bool = True,
    flip_macro_block: bool = False,
    flip_regime_blocked: bool = False,
    flip_session_dead: bool = False,
    flip_score_too_low: bool = False,
    flip_btc_lead_blocked: bool = False,
) -> tuple[bool, bool, str, int]:
    """
    Manage an already-open exchange position.

    Returns:
        skip_symbol: when True caller should continue to next instrument.
        allow_scale_entry, scale_parent_correlation, scale_add_index:
            pyramiding metadata when same-side add is allowed.
    """
    allow_scale_entry = False
    scale_parent_correlation = ""
    scale_add_index = 0

    if current_qty == 0:
        return False, allow_scale_entry, scale_parent_correlation, scale_add_index

    # Refresh live position for this symbol to avoid stale close attempts.
    try:
        live_positions = adapter.fetch_positions([symbol])
        live_qty, live_entry, live_opened_at = _current_position(adapter, symbol, positions=live_positions)
        if abs(live_qty) <= float(getattr(settings, "POSITION_QTY_EPSILON", 1e-12)):
            logger.info("Position already closed on exchange for %s; syncing local state", symbol)
            _mark_position_closed(inst)
            return True, allow_scale_entry, scale_parent_correlation, scale_add_index
        current_qty = live_qty
        if live_entry > 0:
            entry_price = live_entry
        if live_opened_at:
            pos_opened_at = live_opened_at
    except Exception:
        pass

    last = float(ticker_used.get("last")) if ticker_used else last_price
    pos_side = "buy" if current_qty > 0 else "sell"
    pos_direction = "long" if current_qty > 0 else "short"

    # Clear flat-signal timeout tracker when signal is directional again.
    if signal_direction in {"long", "short"}:
        _clear_flat_signal(symbol)

    # Use the actual entry leverage stored in the opening Order (may be boosted above adapter.leverage).
    # This ensures _log_operation reports the real leverage used, not the adapter default.
    try:
        _entry_order_lev = (
            Order.objects.filter(
                instrument=inst,
                side=pos_side,
                status=Order.OrderStatus.FILLED,
            )
            .order_by("-opened_at")
            .values_list("leverage", flat=True)
            .first()
        )
        if _entry_order_lev is not None and float(_entry_order_lev) > 0:
            leverage = float(_entry_order_lev)
    except Exception:
        pass

    origin_signal = _position_origin_signal(inst, pos_side)
    origin_signal_id, origin_correlation_id = _position_origin_refs(inst, pos_side, origin_signal, sig)
    position_strategy_name = str(getattr(origin_signal, "strategy", "") or strategy_name).strip().lower()
    position_sig_payload = (
        getattr(origin_signal, "payload_json", {}) if getattr(origin_signal, "payload_json", None) else sig_payload
    )

    if not pos_opened_at:
        try:
            fallback_open = (
                Order.objects.filter(
                    instrument=inst,
                    side=pos_side,
                    status=Order.OrderStatus.FILLED,
                    opened_at__isnull=False,
                )
                .order_by("-opened_at")
                .values_list("opened_at", flat=True)
                .first()
            )
            max_fallback_hours = max(
                1,
                int(getattr(settings, "POSITION_OPENED_FALLBACK_MAX_HOURS", 72) or 72),
            )
            if fallback_open and (dj_tz.now() - fallback_open) <= timedelta(hours=max_fallback_hours):
                pos_opened_at = fallback_open
        except Exception:
            pass

    # Use TP/SL thresholds for current position direction.
    _, _, tp_pct_pos, sl_pct_pos = _compute_tp_sl_prices(
        pos_side,
        entry_price or last,
        atr,
        recommended_bias=btc_recommended_bias,
        strategy_name=position_strategy_name,
    )

    if entry_price and abs(current_qty) > 0:
        _reconcile_sl(adapter, symbol, pos_side, current_qty, entry_price, atr, pos_opened_at)

    # Trailing stop check (runs before regular TP/SL).
    if last and entry_price:
        was_trailed, trail_fee = _check_trailing_stop(
            adapter,
            symbol,
            pos_side,
            current_qty,
            entry_price,
            last,
            sl_pct_pos,
            pos_opened_at,
            contract_size,
            atr_pct=atr,
            recommended_bias=btc_recommended_bias,
            strategy_name=position_strategy_name,
        )
        if was_trailed:
            pnl_trail = (last - entry_price) / entry_price * (1 if current_qty > 0 else -1)
            pnl_abs_trail = (last - entry_price) * abs(current_qty) * contract_size * (1 if current_qty > 0 else -1)
            trade_side = "buy" if current_qty > 0 else "sell"
            dur_min = (dj_tz.now() - pos_opened_at).total_seconds() / 60 if pos_opened_at else 0
            report = _log_operation(
                inst,
                trade_side,
                abs(current_qty),
                entry_price,
                last,
                reason="trailing_stop",
                signal_id=origin_signal_id,
                correlation_id=origin_correlation_id,
                leverage=leverage,
                equity_before=equity_usdt,
                equity_after=None,
                fee_usdt=trail_fee,
                opened_at=pos_opened_at,
                contract_size=contract_size,
            )
            notify_pnl_pct, notify_pnl_abs = _report_pnl_for_notification(report, pnl_trail, pnl_abs_trail)
            notify_trade_closed(
                inst.symbol,
                "trailing_stop",
                notify_pnl_pct,
                pnl_abs=notify_pnl_abs,
                entry_price=entry_price,
                exit_price=last,
                qty=abs(current_qty),
                equity_before=equity_usdt,
                duration_min=dur_min,
                side=trade_side,
                leverage=leverage,
                strategy_name=position_strategy_name,
                active_modules=_signal_active_modules(position_sig_payload, position_strategy_name),
            )
            _mark_position_closed(inst)
            return True, allow_scale_entry, scale_parent_correlation, scale_add_index

    if _strategy_is_microvol(position_strategy_name) and pos_opened_at:
        age_min = (dj_tz.now() - pos_opened_at).total_seconds() / 60
        max_hold_min = max(1, int(getattr(settings, "MODULE_MICROVOL_MAX_HOLD_MINUTES", 18) or 18))
        if age_min >= max_hold_min and last and entry_price:
            close_side = "sell" if current_qty > 0 else "buy"
            close_qty = abs(current_qty)
            try:
                close_resp = adapter.create_order(symbol, close_side, "market", close_qty, params={"reduceOnly": True})
                close_fee = _resolve_order_fee_usdt(
                    close_resp,
                    _trade_notional_usdt(close_qty, last, contract_size),
                )
                pnl_pct_timeout = (last - entry_price) / entry_price * (1 if current_qty > 0 else -1)
                pnl_abs_timeout = (last - entry_price) * close_qty * contract_size * (1 if current_qty > 0 else -1)
                trade_side = "buy" if current_qty > 0 else "sell"
                logger.info(
                    "Microvol timeout close %s: age=%.1f min >= %d min pnl=%.4f",
                    symbol,
                    age_min,
                    max_hold_min,
                    pnl_pct_timeout,
                )
                report = _log_operation(
                    inst,
                    trade_side,
                    close_qty,
                    entry_price,
                    last,
                    reason="microvol_timeout",
                    signal_id=origin_signal_id,
                    correlation_id=origin_correlation_id,
                    leverage=leverage,
                    equity_before=equity_usdt,
                    equity_after=None,
                    fee_usdt=close_fee,
                    opened_at=pos_opened_at,
                    contract_size=contract_size,
                )
                notify_pnl_pct, notify_pnl_abs = _report_pnl_for_notification(
                    report,
                    pnl_pct_timeout,
                    pnl_abs_timeout,
                )
                notify_trade_closed(
                    inst.symbol,
                    "microvol_timeout",
                    notify_pnl_pct,
                    pnl_abs=notify_pnl_abs,
                    entry_price=entry_price,
                    exit_price=last,
                    qty=close_qty,
                    equity_before=equity_usdt,
                    duration_min=age_min,
                    side=trade_side,
                    leverage=leverage,
                    strategy_name=position_strategy_name,
                    active_modules=_signal_active_modules(position_sig_payload, position_strategy_name),
                )
                _mark_position_closed(inst)
                return True, allow_scale_entry, scale_parent_correlation, scale_add_index
            except Exception as exc:
                if _is_no_position_error(exc):
                    _mark_position_closed(inst)
                    return True, allow_scale_entry, scale_parent_correlation, scale_add_index
                logger.warning("Microvol timeout close failed %s: %s", symbol, exc)

    # Close old positions stuck near breakeven to free margin.
    if last and entry_price:
        pnl_pct_stale = (last - entry_price) / entry_price * (1 if current_qty > 0 else -1)
        if _should_close_stale_position(pos_opened_at, pnl_pct_stale):
            close_side = "sell" if current_qty > 0 else "buy"
            close_qty = abs(current_qty)
            age_h = (dj_tz.now() - pos_opened_at).total_seconds() / 3600 if pos_opened_at else 0
            try:
                close_resp = adapter.create_order(symbol, close_side, "market", close_qty, params={"reduceOnly": True})
                close_fee = _resolve_order_fee_usdt(
                    close_resp,
                    _trade_notional_usdt(close_qty, last, contract_size),
                )
                pnl_abs_stale = (last - entry_price) * close_qty * contract_size * (1 if current_qty > 0 else -1)
                trade_side = "buy" if current_qty > 0 else "sell"
                dur_min = age_h * 60
                logger.info(
                    "Stale position cleanup %s: pnl=%.4f age=%.1fh â€” freeing margin",
                    symbol,
                    pnl_pct_stale,
                    age_h,
                )
                report = _log_operation(
                    inst,
                    trade_side,
                    close_qty,
                    entry_price,
                    last,
                    reason="stale_cleanup",
                    signal_id=origin_signal_id,
                    correlation_id=origin_correlation_id,
                    leverage=leverage,
                    equity_before=equity_usdt,
                    equity_after=None,
                    fee_usdt=close_fee,
                    opened_at=pos_opened_at,
                    contract_size=contract_size,
                )
                notify_pnl_pct, notify_pnl_abs = _report_pnl_for_notification(
                    report,
                    pnl_pct_stale,
                    pnl_abs_stale,
                )
                notify_trade_closed(
                    inst.symbol,
                    "stale_cleanup",
                    notify_pnl_pct,
                    pnl_abs=notify_pnl_abs,
                    entry_price=entry_price,
                    exit_price=last,
                    qty=close_qty,
                    equity_before=equity_usdt,
                    duration_min=dur_min,
                    side=trade_side,
                    leverage=leverage,
                    strategy_name=position_strategy_name,
                    active_modules=_signal_active_modules(position_sig_payload, position_strategy_name),
                )
                _mark_position_closed(inst)
            except Exception as exc:
                if _is_no_position_error(exc):
                    _mark_position_closed(inst)
                else:
                    logger.warning("Stale cleanup failed %s: %s", symbol, exc)
            return True, allow_scale_entry, scale_parent_correlation, scale_add_index

    # Close short positions immediately when HTF turns bullish.
    if (
        current_qty < 0
        and getattr(settings, "UPTREND_SHORT_KILLER_ENABLED", False)
        and last
        and entry_price
    ):
        try:
            from signals.tasks import _trend_from_swings
            htf_candles = list(
                Candle.objects.filter(
                    instrument=inst,
                    timeframe="4h",
                ).order_by("-ts")[:100]
            )
            if htf_candles:
                import pandas as _pd
                htf_df = _pd.DataFrame([
                    {"high": float(c.high), "low": float(c.low), "close": float(c.close)}
                    for c in reversed(htf_candles)
                ])
                htf_trend = _trend_from_swings(htf_df)
                if htf_trend == "bull":
                    close_side = "buy"
                    close_qty = abs(current_qty)
                    pnl_pct_kill = (last - entry_price) / entry_price * -1
                    close_resp = adapter.create_order(symbol, close_side, "market", close_qty, params={"reduceOnly": True})
                    close_fee = _resolve_order_fee_usdt(
                        close_resp,
                        _trade_notional_usdt(close_qty, last, contract_size),
                    )
                    pnl_abs_kill = (last - entry_price) * close_qty * contract_size * -1
                    dur_min = (dj_tz.now() - pos_opened_at).total_seconds() / 60 if pos_opened_at else 0
                    logger.info(
                        "Uptrend short killer closed %s: HTF=bull pnl=%.4f",
                        symbol,
                        pnl_pct_kill,
                    )
                    report = _log_operation(
                        inst,
                        "sell",
                        close_qty,
                        entry_price,
                        last,
                        reason="uptrend_short_kill",
                        signal_id=origin_signal_id,
                        correlation_id=origin_correlation_id,
                        leverage=leverage,
                        equity_before=equity_usdt,
                        equity_after=None,
                        fee_usdt=close_fee,
                        opened_at=pos_opened_at,
                        contract_size=contract_size,
                    )
                    notify_pnl_pct, notify_pnl_abs = _report_pnl_for_notification(
                        report,
                        pnl_pct_kill,
                        pnl_abs_kill,
                    )
                    notify_trade_closed(
                        inst.symbol,
                        "uptrend_short_kill",
                        notify_pnl_pct,
                        pnl_abs=notify_pnl_abs,
                        entry_price=entry_price,
                        exit_price=last,
                        qty=close_qty,
                        equity_before=equity_usdt,
                        duration_min=dur_min,
                        side="sell",
                        leverage=leverage,
                        strategy_name=position_strategy_name,
                        active_modules=_signal_active_modules(position_sig_payload, position_strategy_name),
                    )
                    _mark_position_closed(inst)
                    return True, allow_scale_entry, scale_parent_correlation, scale_add_index
        except Exception as exc:
            logger.debug("Uptrend short killer check failed for %s: %s", symbol, exc)

    # Close long positions immediately when HTF turns bearish.
    if (
        current_qty > 0
        and getattr(settings, "DOWNTREND_LONG_KILLER_ENABLED", False)
        and last
        and entry_price
    ):
        try:
            from signals.tasks import _trend_from_swings
            htf_candles = list(
                Candle.objects.filter(
                    instrument=inst,
                    timeframe="4h",
                ).order_by("-ts")[:100]
            )
            if htf_candles:
                import pandas as _pd
                htf_df = _pd.DataFrame([
                    {"high": float(c.high), "low": float(c.low), "close": float(c.close)}
                    for c in reversed(htf_candles)
                ])
                htf_trend = _trend_from_swings(htf_df)
                if htf_trend == "bear":
                    close_side = "sell"
                    close_qty = abs(current_qty)
                    pnl_pct_kill = (last - entry_price) / entry_price
                    close_resp = adapter.create_order(symbol, close_side, "market", close_qty, params={"reduceOnly": True})
                    close_fee = _resolve_order_fee_usdt(
                        close_resp,
                        _trade_notional_usdt(close_qty, last, contract_size),
                    )
                    pnl_abs_kill = (last - entry_price) * close_qty * contract_size
                    dur_min = (dj_tz.now() - pos_opened_at).total_seconds() / 60 if pos_opened_at else 0
                    logger.info(
                        "Downtrend long killer closed %s: HTF=bear pnl=%.4f",
                        symbol,
                        pnl_pct_kill,
                    )
                    report = _log_operation(
                        inst,
                        "buy",
                        close_qty,
                        entry_price,
                        last,
                        reason="downtrend_long_kill",
                        signal_id=origin_signal_id,
                        correlation_id=origin_correlation_id,
                        leverage=leverage,
                        equity_before=equity_usdt,
                        equity_after=None,
                        fee_usdt=close_fee,
                        opened_at=pos_opened_at,
                        contract_size=contract_size,
                    )
                    notify_pnl_pct, notify_pnl_abs = _report_pnl_for_notification(
                        report,
                        pnl_pct_kill,
                        pnl_abs_kill,
                    )
                    notify_trade_closed(
                        inst.symbol,
                        "downtrend_long_kill",
                        notify_pnl_pct,
                        pnl_abs=notify_pnl_abs,
                        entry_price=entry_price,
                        exit_price=last,
                        qty=close_qty,
                        equity_before=equity_usdt,
                        duration_min=dur_min,
                        side="buy",
                        leverage=leverage,
                        strategy_name=position_strategy_name,
                        active_modules=_signal_active_modules(position_sig_payload, position_strategy_name),
                    )
                    _mark_position_closed(inst)
                    return True, allow_scale_entry, scale_parent_correlation, scale_add_index
        except Exception as exc:
            logger.debug("Downtrend long killer check failed for %s: %s", symbol, exc)

    if last and entry_price and tp_pct_pos > 0 and sl_pct_pos > 0:
        should_close_progress, progress_reason, progress_meta = _evaluate_tp_progress_exit(
            symbol=symbol,
            side=pos_side,
            entry_price=entry_price,
            last_price=last,
            tp_pct=tp_pct_pos,
            sl_pct=sl_pct_pos,
            opened_at=pos_opened_at,
            signal_direction=signal_direction,
            recommended_bias=btc_recommended_bias,
            strategy_name=position_strategy_name,
        )
        if should_close_progress:
            close_side = "sell" if current_qty > 0 else "buy"
            close_qty = abs(current_qty)
            try:
                close_resp = adapter.create_order(
                    symbol,
                    close_side,
                    "market",
                    close_qty,
                    params={"reduceOnly": True},
                )
                close_fee = _resolve_order_fee_usdt(
                    close_resp,
                    _trade_notional_usdt(close_qty, last, contract_size),
                )
                pnl_pct_exit = (last - entry_price) / entry_price * (1 if current_qty > 0 else -1)
                pnl_abs_exit = (last - entry_price) * close_qty * contract_size * (1 if current_qty > 0 else -1)
                trade_side = "buy" if current_qty > 0 else "sell"
                dur_min = (dj_tz.now() - pos_opened_at).total_seconds() / 60 if pos_opened_at else 0
                logger.info(
                    "TP progress exit %s: progress=%.3f giveback=%.3f score=%s reason=%s",
                    symbol,
                    float(progress_meta.get("progress", 0.0) or 0.0),
                    float(progress_meta.get("giveback_ratio", 0.0) or 0.0),
                    progress_meta.get("score", 0),
                    progress_reason,
                )
                report = _log_operation(
                    inst,
                    trade_side,
                    close_qty,
                    entry_price,
                    last,
                    reason="tp_progress_exit",
                    signal_id=origin_signal_id,
                    correlation_id=origin_correlation_id,
                    leverage=leverage,
                    equity_before=equity_usdt,
                    equity_after=None,
                    fee_usdt=close_fee,
                    opened_at=pos_opened_at,
                    contract_size=contract_size,
                    close_sub_reason=progress_reason,
                )
                notify_pnl_pct, notify_pnl_abs = _report_pnl_for_notification(
                    report,
                    pnl_pct_exit,
                    pnl_abs_exit,
                )
                notify_trade_closed(
                    inst.symbol,
                    "tp_progress_exit",
                    notify_pnl_pct,
                    pnl_abs=notify_pnl_abs,
                    entry_price=entry_price,
                    exit_price=last,
                    qty=close_qty,
                    equity_before=equity_usdt,
                    duration_min=dur_min,
                    side=trade_side,
                    leverage=leverage,
                    strategy_name=position_strategy_name,
                    active_modules=_signal_active_modules(position_sig_payload, position_strategy_name),
                )
                _mark_position_closed(inst)
                _track_consecutive_errors(symbol, success=True)
                return True, allow_scale_entry, scale_parent_correlation, scale_add_index
            except Exception as exc:
                if _is_no_position_error(exc):
                    logger.info("TP progress exit skipped %s: no open position (%s)", symbol, exc)
                    _mark_position_closed(inst)
                    _track_consecutive_errors(symbol, success=True)
                    return True, allow_scale_entry, scale_parent_correlation, scale_add_index
                logger.warning("Failed TP progress exit %s: %s", symbol, exc)
                err_count = _track_consecutive_errors(symbol, success=False)
                if err_count >= settings.MAX_CONSECUTIVE_ERRORS:
                    _create_risk_event("consecutive_errors", "critical", instrument=inst, details={"count": err_count})
                    notify_kill_switch(f"{symbol}: {err_count} consecutive errors")

    # AI-assisted early TP exit (conservative): only near TP, never increases risk.
    if (
        get_runtime_bool(
            "AI_EXIT_GATE_ENABLED",
            bool(getattr(settings, "AI_EXIT_GATE_ENABLED", True)),
        )
        and last
        and entry_price
        and tp_pct_pos > 0
        and sl_pct_pos > 0
    ):
        pnl_pct_live_gross = (last - entry_price) / entry_price * (1 if current_qty > 0 else -1)
        pnl_pct_live_gate, _ = _tp_sl_gate_pnl_pct(pnl_pct_live_gross)
        near_tp_ratio = float(getattr(settings, "AI_EXIT_GATE_NEAR_TP_RATIO", 0.88) or 0.88)
        near_tp_ratio = max(0.50, min(0.99, near_tp_ratio))
        near_tp_trigger = tp_pct_pos * near_tp_ratio
        min_r = max(0.0, float(getattr(settings, "AI_EXIT_GATE_MIN_R", 0.8) or 0.8))
        r_multiple_live = pnl_pct_live_gate / sl_pct_pos if sl_pct_pos > 0 else 0.0
        if (
            pnl_pct_live_gate > 0
            and pnl_pct_live_gate >= near_tp_trigger
            and pnl_pct_live_gate < tp_pct_pos
            and r_multiple_live >= min_r
        ):
            eval_allowed = True
            state_key = _position_state_key(symbol, pos_opened_at, entry_price)
            recheck_sec = max(
                10,
                int(getattr(settings, "AI_EXIT_GATE_MIN_RECHECK_SECONDS", 45) or 45),
            )
            client = _redis_client()
            if client is not None:
                eval_key = f"ai_exit:last_eval:{state_key}"
                try:
                    eval_allowed = bool(client.set(eval_key, "1", nx=True, ex=recheck_sec))
                except Exception:
                    eval_allowed = True
            if eval_allowed:
                spread_bps_live = _spread_bps(ticker_used) if ticker_used else None
                pos_age_min = None
                if pos_opened_at:
                    try:
                        pos_age_min = (dj_tz.now() - pos_opened_at).total_seconds() / 60.0
                    except Exception:
                        pos_age_min = None
                should_close_early, ai_exit_reason, ai_exit_meta = evaluate_ai_exit_gate(
                    account_ai_enabled=account_ai_enabled,
                    account_ai_config_id=account_ai_config_id,
                    account_owner_id=account_owner_id,
                    account_alias=account_alias,
                    account_service=account_service,
                    symbol=inst.symbol,
                    strategy_name=strategy_name,
                    position_direction=pos_direction,
                    sig_score=_to_float(getattr(sig, "score", 0.0)),
                    atr=atr,
                    spread_bps=spread_bps_live,
                    tp_pct=tp_pct_pos,
                    sl_pct=sl_pct_pos,
                    pnl_pct_gross=pnl_pct_live_gross,
                    pnl_pct_gate=pnl_pct_live_gate,
                    r_multiple=r_multiple_live,
                    remaining_tp_pct=max(0.0, tp_pct_pos - pnl_pct_live_gate),
                    position_age_min=pos_age_min,
                    session_name=current_session,
                    sig_payload=sig_payload if isinstance(sig_payload, dict) else {},
                )
                if should_close_early:
                    close_side = "sell" if current_qty > 0 else "buy"
                    close_qty = abs(current_qty)
                    try:
                        close_resp = adapter.create_order(
                            symbol,
                            close_side,
                            "market",
                            close_qty,
                            params={"reduceOnly": True},
                        )
                        close_fee = _resolve_order_fee_usdt(
                            close_resp,
                            _trade_notional_usdt(close_qty, last, contract_size),
                        )
                        pnl_abs_exit = (last - entry_price) * close_qty * contract_size * (1 if current_qty > 0 else -1)
                        trade_side = "buy" if current_qty > 0 else "sell"
                        dur_min = (dj_tz.now() - pos_opened_at).total_seconds() / 60 if pos_opened_at else 0
                        logger.info(
                            "AI early TP exit %s: pnl_gate=%.4f tp=%.4f r=%.2f reason=%s cfg=%s",
                            symbol,
                            pnl_pct_live_gate,
                            tp_pct_pos,
                            r_multiple_live,
                            ai_exit_reason,
                            ai_exit_meta.get("cfg_alias", "n/a"),
                        )
                        report = _log_operation(
                            inst,
                            trade_side,
                            close_qty,
                            entry_price,
                            last,
                            reason="ai_tp_early_exit",
                            signal_id=origin_signal_id,
                            correlation_id=origin_correlation_id,
                            leverage=leverage,
                            equity_before=equity_usdt,
                            equity_after=None,
                            fee_usdt=close_fee,
                            opened_at=pos_opened_at,
                            contract_size=contract_size,
                        )
                        notify_pnl_pct, notify_pnl_abs = _report_pnl_for_notification(
                            report,
                            pnl_pct_live_gross,
                            pnl_abs_exit,
                        )
                        notify_trade_closed(
                            inst.symbol,
                            "ai_tp_early_exit",
                            notify_pnl_pct,
                            pnl_abs=notify_pnl_abs,
                            entry_price=entry_price,
                            exit_price=last,
                            qty=close_qty,
                            equity_before=equity_usdt,
                            duration_min=dur_min,
                            side=trade_side,
                            leverage=leverage,
                            strategy_name=position_strategy_name,
                            active_modules=_signal_active_modules(position_sig_payload, position_strategy_name),
                        )
                        _mark_position_closed(inst)
                        _track_consecutive_errors(symbol, success=True)
                        return True, allow_scale_entry, scale_parent_correlation, scale_add_index
                    except Exception as exc:
                        if _is_no_position_error(exc):
                            logger.info("AI early exit skipped %s: no open position (%s)", symbol, exc)
                            _mark_position_closed(inst)
                            _track_consecutive_errors(symbol, success=True)
                            return True, allow_scale_entry, scale_parent_correlation, scale_add_index
                        logger.warning("Failed AI early exit %s: %s", symbol, exc)
                        err_count = _track_consecutive_errors(symbol, success=False)
                        if err_count >= settings.MAX_CONSECUTIVE_ERRORS:
                            _create_risk_event("consecutive_errors", "critical", instrument=inst, details={"count": err_count})
                            notify_kill_switch(f"{symbol}: {err_count} consecutive errors")

    # Regular TP/SL
    if last and entry_price:
        pnl_pct_live_gross = (last - entry_price) / entry_price * (1 if current_qty > 0 else -1)
        pnl_pct_live_gate, fee_pct_estimate = _tp_sl_gate_pnl_pct(pnl_pct_live_gross)
        if pnl_pct_live_gate >= tp_pct_pos or pnl_pct_live_gate <= -sl_pct_pos:
            close_side = "sell" if current_qty > 0 else "buy"
            close_qty = abs(current_qty)
            reason = "tp" if pnl_pct_live_gate >= tp_pct_pos else "sl"
            try:
                close_resp = adapter.create_order(symbol, close_side, "market", close_qty, params={"reduceOnly": True})
                close_fee = _resolve_order_fee_usdt(
                    close_resp,
                    _trade_notional_usdt(close_qty, last, contract_size),
                )
                pnl_abs_tpsl = (last - entry_price) * close_qty * contract_size * (1 if current_qty > 0 else -1)
                trade_side = "buy" if current_qty > 0 else "sell"
                dur_min = (dj_tz.now() - pos_opened_at).total_seconds() / 60 if pos_opened_at else 0
                logger.info(
                    "Closed %s by %s pnl_gross=%.4f pnl_gate=%.4f fee_gate=%.4f tp=%.4f sl=%.4f pnl_usdt=%.4f",
                    symbol,
                    reason,
                    pnl_pct_live_gross,
                    pnl_pct_live_gate,
                    fee_pct_estimate,
                    tp_pct_pos,
                    sl_pct_pos,
                    pnl_abs_tpsl,
                )
                report = _log_operation(
                    inst,
                    trade_side,
                    close_qty,
                    entry_price,
                    last,
                    reason=reason,
                    signal_id=origin_signal_id,
                    correlation_id=origin_correlation_id,
                    leverage=leverage,
                    equity_before=equity_usdt,
                    equity_after=None,
                    fee_usdt=close_fee,
                    opened_at=pos_opened_at,
                    contract_size=contract_size,
                )
                notify_pnl_pct, notify_pnl_abs = _report_pnl_for_notification(
                    report,
                    pnl_pct_live_gross,
                    pnl_abs_tpsl,
                )
                notify_trade_closed(
                    inst.symbol,
                    reason,
                    notify_pnl_pct,
                    pnl_abs=notify_pnl_abs,
                    entry_price=entry_price,
                    exit_price=last,
                    qty=close_qty,
                    equity_before=equity_usdt,
                    duration_min=dur_min,
                    side=trade_side,
                    leverage=leverage,
                    strategy_name=position_strategy_name,
                    active_modules=_signal_active_modules(position_sig_payload, position_strategy_name),
                )
                _mark_position_closed(inst)
                _track_consecutive_errors(symbol, success=True)
            except Exception as exc:
                if _is_no_position_error(exc):
                    logger.info("Close TP/SL skipped %s: no open position (%s)", symbol, exc)
                    _mark_position_closed(inst)
                    _track_consecutive_errors(symbol, success=True)
                    return True, allow_scale_entry, scale_parent_correlation, scale_add_index
                logger.warning("Failed close TP/SL %s: %s", symbol, exc)
                err_count = _track_consecutive_errors(symbol, success=False)
                if err_count >= settings.MAX_CONSECUTIVE_ERRORS:
                    _create_risk_event("consecutive_errors", "critical", instrument=inst, details={"count": err_count})
                    notify_kill_switch(f"{symbol}: {err_count} consecutive errors")
            return True, allow_scale_entry, scale_parent_correlation, scale_add_index

    # Signal flip: close opposite-side position (unless direction policy blocks this signal).
    if (
        signal_direction in {"long", "short"}
        and direction_allowed
        and ((current_qty > 0 and side == "sell") or (current_qty < 0 and side == "buy"))
    ):
        if settings.SIGNAL_FLIP_MIN_AGE_ENABLED and pos_opened_at:
            flip_age_min = (dj_tz.now() - pos_opened_at).total_seconds() / 60
            if flip_age_min < settings.SIGNAL_FLIP_MIN_AGE_MINUTES:
                logger.info(
                    "Signal flip BLOCKED on %s: position age %.1f min < %.1f min threshold",
                    symbol,
                    flip_age_min,
                    settings.SIGNAL_FLIP_MIN_AGE_MINUTES,
                )
                return True, allow_scale_entry, scale_parent_correlation, scale_add_index

        # ── Pre-validate that the new entry can pass critical gates ──────
        # Avoid closing a position for a flip that will be rejected downstream.
        _flip_block_reason = ""
        if not flip_can_open:
            _flip_block_reason = "can_open=false"
        elif flip_macro_block:
            _flip_block_reason = "macro_high_impact"
        elif flip_regime_blocked:
            _flip_block_reason = "regime_adx_gate"
        elif flip_session_dead:
            _flip_block_reason = "session_dead_zone"
        elif flip_score_too_low:
            _flip_block_reason = "score_too_low"
        elif flip_btc_lead_blocked:
            _flip_block_reason = "btc_lead_blocked"
        if _flip_block_reason:
            logger.info(
                "Signal flip BLOCKED on %s: new entry would fail gate (%s); keeping position",
                symbol,
                _flip_block_reason,
            )
            return True, allow_scale_entry, scale_parent_correlation, scale_add_index

        close_side = "sell" if current_qty > 0 else "buy"
        close_qty = abs(current_qty)
        try:
            close_resp = adapter.create_order(symbol, close_side, "market", close_qty, params={"reduceOnly": True})
            close_fee = _resolve_order_fee_usdt(
                close_resp,
                _trade_notional_usdt(close_qty, last, contract_size),
            )
            logger.info("Signal flip close %s qty=%s", symbol, close_qty)
            if last and entry_price:
                pnl_flip = (last - entry_price) / entry_price * (1 if current_qty > 0 else -1)
                pnl_abs_flip = (last - entry_price) * close_qty * contract_size * (1 if current_qty > 0 else -1)
                trade_side = "buy" if current_qty > 0 else "sell"
                dur_min = (dj_tz.now() - pos_opened_at).total_seconds() / 60 if pos_opened_at else 0
                report = _log_operation(
                    inst,
                    trade_side,
                    close_qty,
                    entry_price,
                    last,
                    reason="signal_flip",
                    signal_id=origin_signal_id,
                    correlation_id=origin_correlation_id,
                    leverage=leverage,
                    equity_before=equity_usdt,
                    equity_after=None,
                    fee_usdt=close_fee,
                    opened_at=pos_opened_at,
                    contract_size=contract_size,
                )
                notify_pnl_pct, notify_pnl_abs = _report_pnl_for_notification(
                    report,
                    pnl_flip,
                    pnl_abs_flip,
                )
                notify_trade_closed(
                    inst.symbol,
                    "signal_flip",
                    notify_pnl_pct,
                    pnl_abs=notify_pnl_abs,
                    entry_price=entry_price,
                    exit_price=last,
                    qty=close_qty,
                    equity_before=equity_usdt,
                    duration_min=dur_min,
                    side=trade_side,
                    leverage=leverage,
                    strategy_name=position_strategy_name,
                    active_modules=_signal_active_modules(position_sig_payload, position_strategy_name),
                )
            _mark_position_closed(inst)
            _track_consecutive_errors(symbol, success=True)
            # Preserve existing behavior: after a successful flip close, caller may open new side.
            return False, allow_scale_entry, scale_parent_correlation, scale_add_index
        except Exception as exc:
            if _is_no_position_error(exc):
                logger.info("Flip close skipped %s: no open position (%s)", symbol, exc)
                _mark_position_closed(inst)
                _track_consecutive_errors(symbol, success=True)
                return True, allow_scale_entry, scale_parent_correlation, scale_add_index
            logger.warning("Failed flip close %s: %s", symbol, exc)
            _track_consecutive_errors(symbol, success=False)
            return True, allow_scale_entry, scale_parent_correlation, scale_add_index

    if signal_direction == "flat":
        # -- Flat-signal timeout: close position if flat persists too long --
        flat_timeout_enabled = bool(getattr(settings, "FLAT_SIGNAL_TIMEOUT_ENABLED", True))
        flat_timeout_minutes = max(1.0, float(getattr(settings, "FLAT_SIGNAL_TIMEOUT_MINUTES", 10)))
        if flat_timeout_enabled and last and entry_price and abs(current_qty) > 0:
            flat_seconds = _track_flat_signal(symbol)
            flat_minutes = flat_seconds / 60.0
            if flat_minutes >= flat_timeout_minutes:
                close_side = "sell" if current_qty > 0 else "buy"
                close_qty = abs(current_qty)
                trade_side = "buy" if current_qty > 0 else "sell"
                pnl_flat = (last - entry_price) / entry_price * (1 if current_qty > 0 else -1)
                pnl_abs_flat = (last - entry_price) * close_qty * contract_size * (1 if current_qty > 0 else -1)
                dur_min = (dj_tz.now() - pos_opened_at).total_seconds() / 60 if pos_opened_at else 0
                logger.info(
                    "Flat signal timeout on %s: flat for %.1f min (threshold %.1f min), closing pnl=%.4f%%",
                    symbol,
                    flat_minutes,
                    flat_timeout_minutes,
                    pnl_flat * 100,
                )
                try:
                    close_resp = adapter.create_order(symbol, close_side, "market", close_qty, params={"reduceOnly": True})
                    close_fee = _resolve_order_fee_usdt(
                        close_resp,
                        _trade_notional_usdt(close_qty, last, contract_size),
                    )
                    report = _log_operation(
                        inst,
                        trade_side,
                        close_qty,
                        entry_price,
                        last,
                        reason="flat_signal_timeout",
                        signal_id=origin_signal_id,
                        correlation_id=origin_correlation_id,
                        leverage=leverage,
                        equity_before=equity_usdt,
                        equity_after=None,
                        fee_usdt=close_fee,
                        opened_at=pos_opened_at,
                        contract_size=contract_size,
                    )
                    notify_pnl_pct, notify_pnl_abs = _report_pnl_for_notification(
                        report,
                        pnl_flat,
                        pnl_abs_flat,
                    )
                    notify_trade_closed(
                        inst.symbol,
                        "flat_signal_timeout",
                        notify_pnl_pct,
                        pnl_abs=notify_pnl_abs,
                        entry_price=entry_price,
                        exit_price=last,
                        qty=close_qty,
                        equity_before=equity_usdt,
                        duration_min=dur_min,
                        side=trade_side,
                        leverage=leverage,
                        strategy_name=position_strategy_name,
                        active_modules=_signal_active_modules(position_sig_payload, position_strategy_name),
                    )
                    _mark_position_closed(inst)
                    _clear_flat_signal(symbol)
                except Exception as exc:
                    if _is_no_position_error(exc):
                        _mark_position_closed(inst)
                        _clear_flat_signal(symbol)
                    else:
                        logger.warning("Flat signal timeout close failed %s: %s", symbol, exc)
                return True, allow_scale_entry, scale_parent_correlation, scale_add_index
        else:
            logger.info("Flat signal on %s with open qty=%s; manage-only", symbol, current_qty)
        return True, allow_scale_entry, scale_parent_correlation, scale_add_index

    same_side_signal = (
        signal_direction in {"long", "short"}
        and direction_allowed
        and signal_direction == pos_direction
    )
    if same_side_signal:
        pyramiding_enabled = bool(getattr(settings, "PYRAMIDING_ENABLED", False))
        if not pyramiding_enabled:
            logger.info("Same-side position on %s qty=%s; skip", symbol, current_qty)
            return True, allow_scale_entry, scale_parent_correlation, scale_add_index
        if not (last and entry_price and sl_pct_pos > 0):
            logger.info(
                "Pyramiding skipped on %s: missing price/entry/sl context",
                symbol,
            )
            return True, allow_scale_entry, scale_parent_correlation, scale_add_index
        pnl_pct_now = (last - entry_price) / entry_price * (1 if current_qty > 0 else -1)
        r_now = (pnl_pct_now / sl_pct_pos) if sl_pct_pos > 0 else 0.0
        add_at_r = max(0.0, float(getattr(settings, "PYRAMID_ADD_AT_R", 0.8) or 0.8))
        if pnl_pct_now <= 0 or r_now < add_at_r:
            logger.info(
                "Pyramiding gate on %s: r_now=%.3f < add_at_r=%.3f (pnl=%.3f%%)",
                symbol,
                r_now,
                add_at_r,
                pnl_pct_now * 100,
            )
            return True, allow_scale_entry, scale_parent_correlation, scale_add_index

        root_corr = _position_root_correlation(inst, pos_side)
        if not root_corr:
            fallback_root = f"{sig.id}-{inst.symbol}" if sig else f"pos-{inst.symbol}"
            root_corr = _safe_correlation_id(fallback_root)
        add_count = _count_pyramid_adds(inst, pos_side, root_corr)
        max_adds = max(0, int(getattr(settings, "PYRAMID_MAX_ADDS", 2)))
        if add_count >= max_adds:
            logger.info(
                "Pyramiding max adds reached on %s: adds=%d max=%d",
                symbol,
                add_count,
                max_adds,
            )
            return True, allow_scale_entry, scale_parent_correlation, scale_add_index

        min_add_minutes = max(
            0,
            int(getattr(settings, "PYRAMID_MIN_MINUTES_BETWEEN_ADDS", 3)),
        )
        if min_add_minutes > 0:
            last_add_at = _last_pyramid_add_opened_at(inst, pos_side, root_corr)
            if last_add_at:
                elapsed_add_min = (dj_tz.now() - last_add_at).total_seconds() / 60
                if elapsed_add_min < min_add_minutes:
                    logger.info(
                        "Pyramiding cooldown on %s: %.1f < %d min",
                        symbol,
                        elapsed_add_min,
                        min_add_minutes,
                    )
                    return True, allow_scale_entry, scale_parent_correlation, scale_add_index

        allow_scale_entry = True
        scale_parent_correlation = root_corr
        scale_add_index = add_count + 1
        logger.info(
            "Pyramiding enabled on %s: r_now=%.3f add=%d/%d parent=%s",
            symbol,
            r_now,
            scale_add_index,
            max_adds,
            scale_parent_correlation,
        )
        return False, allow_scale_entry, scale_parent_correlation, scale_add_index

    if not direction_allowed:
        logger.info("Direction policy keeps existing position on %s qty=%s; skip", symbol, current_qty)
        return True, allow_scale_entry, scale_parent_correlation, scale_add_index

    logger.info("Same-side position on %s qty=%s; skip", symbol, current_qty)
    return True, allow_scale_entry, scale_parent_correlation, scale_add_index


@shared_task
def execute_orders():
    """
    Enhanced execution loop with:
    - Signal TTL / expiry
    - Risk-based position sizing (% equity at risk)
    - Exchange stop-market SL orders
    - Trailing stop + partial close
    - Data-staleness kill-switch per instrument
    - Consecutive-error kill-switch
    - Per-instrument exposure cap
    - Weekly drawdown limit
    - Circuit breaker (daily DD / total DD / consecutive losses)
    - Telegram notifications for opens / closes / errors
    """
    if not settings.TRADING_ENABLED or settings.MODE == "paper":
        logger.info("Trading disabled (mode=%s, trading_enabled=%s)", settings.MODE, settings.TRADING_ENABLED)
        return "trading-disabled"

    lock_client = None
    lock_token = ""
    lock_key = str(getattr(settings, "EXECUTION_LOCK_KEY", "lock:execute_orders") or "lock:execute_orders")
    lock_ttl = max(10, int(getattr(settings, "EXECUTION_LOCK_TTL_SECONDS", 90) or 90))
    if bool(getattr(settings, "EXECUTION_LOCK_ENABLED", True)):
        lock_client, lock_token = _acquire_task_lock(lock_key, lock_ttl)
        if lock_client is not None and not lock_token:
            logger.info("execute_orders skipped: lock active key=%s ttl=%ss", lock_key, lock_ttl)
            return "execute_orders:locked"

    adapter = _adapter()
    runtime_ctx = get_runtime_exchange_context()
    risk_ns = str(runtime_ctx.get("risk_namespace") or "global")
    balance_assets = list(runtime_ctx.get("balance_assets") or ["USDT"])
    env_label = str(runtime_ctx.get("label") or "EXCHANGE")
    account_ai_enabled = bool(runtime_ctx.get("ai_enabled", False))
    try:
        account_ai_config_id = int(runtime_ctx.get("ai_provider_config_id") or 0) or None
    except Exception:
        account_ai_config_id = None
    try:
        account_owner_id = int(runtime_ctx.get("owner_id") or 0) or None
    except Exception:
        account_owner_id = None
    account_alias = str(runtime_ctx.get("account_alias") or "")
    account_service = str(runtime_ctx.get("service") or "")
    placed = 0
    positions_snapshot = []
    try:
        positions_snapshot = adapter.fetch_positions()
    except Exception:
        positions_snapshot = []
    cycle_pending_corr_entries: list[dict[str, Any]] = []

    # ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ 1. Sync positions for admin visibility ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬
    _sync_positions(adapter, positions=positions_snapshot, balance_assets=balance_assets)

    # 2. Balance & global guardrails
    can_open, free_usdt, equity_usdt, eff_lev, total_notional, leverage = _evaluate_balance_and_guardrails(
        adapter,
        runtime_ctx,
        risk_ns,
        positions_snapshot=positions_snapshot,
    )

    # 2b. Circuit breaker check
    can_open = _apply_circuit_breaker_gate(can_open, equity_usdt, risk_ns)

    signal_ttl = timedelta(seconds=settings.SIGNAL_TTL_SECONDS)
    session_policy_enabled = bool(getattr(settings, "SESSION_POLICY_ENABLED", False))
    session_dead_zone_block = bool(getattr(settings, "SESSION_DEAD_ZONE_BLOCK", True))
    feature_flags = resolve_runtime_flags()
    multi_strategy_enabled = bool(feature_flags.get(FEATURE_KEYS["multi"], False))
    allocator_enabled = bool(feature_flags.get(FEATURE_KEYS["allocator"], False))
    use_allocator_signals = multi_strategy_enabled and allocator_enabled
    ml_entry_filter_enabled = bool(getattr(settings, "ML_ENTRY_FILTER_ENABLED", False))
    ml_entry_filter_default_min_prob = float(getattr(settings, "ML_ENTRY_FILTER_MIN_PROB", 0.52) or 0.52)
    ml_entry_filter_fail_open = bool(getattr(settings, "ML_ENTRY_FILTER_FAIL_OPEN", True))
    logger.debug(
        "Execution signal source mode=%s (multi=%s allocator=%s)",
        "alloc_*" if use_allocator_signals else "latest_any",
        multi_strategy_enabled,
        allocator_enabled,
    )
    session_now = dj_tz.now()
    current_session = get_current_session(session_now)
    weekday_name = get_weekday_name(session_now)
    session_min_score = float(getattr(settings, "EXECUTION_MIN_SIGNAL_SCORE", 0.80))
    session_risk_mult = 1.0
    if session_policy_enabled:
        session_min_score = get_session_score_min(
            current_session,
            getattr(settings, "SESSION_SCORE_MIN", {}),
        )
        if bool(getattr(settings, "WEEKDAY_CONTEXT_ENABLED", True)):
            session_min_score += get_weekday_score_offset(
                weekday_name,
                getattr(settings, "WEEKDAY_SCORE_OFFSET", {}),
            )
            session_min_score = max(0.0, min(1.0, float(session_min_score)))
        session_risk_mult = get_session_risk_mult(
            current_session,
            getattr(settings, "SESSION_RISK_MULTIPLIER", {}),
        )
        if bool(getattr(settings, "WEEKDAY_CONTEXT_ENABLED", True)):
            session_risk_mult *= get_weekday_risk_mult(
                weekday_name,
                getattr(settings, "WEEKDAY_RISK_MULTIPLIER", {}),
            )
    macro_active, macro_context = _is_macro_high_impact_window(
        now_utc=dj_tz.now(),
        session_name=current_session,
    )
    macro_block_entries = bool(getattr(settings, "MACRO_HIGH_IMPACT_BLOCK_ENTRIES", False))
    macro_risk_mult = float(getattr(settings, "MACRO_HIGH_IMPACT_RISK_MULTIPLIER", 1.0) or 1.0)
    macro_risk_mult = max(0.0, min(macro_risk_mult, 1.0))

    # Running notional accumulator ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â tracks new positions opened THIS cycle
    # so we don't exceed leverage by opening multiple trades before the
    # balance refresh catches up.
    cycle_notional_added = 0.0

    enabled_instruments, latest_signals, now_cycle = _load_enabled_instruments_and_latest_signals(
        use_allocator_signals,
    )
    (
        _regime_adx_by_symbol,
        _regime_blocked_symbols,
        _regime_adx_min,
        _market_regime_adx,
        _regime_bias_by_symbol,
        _regime_adx_min_by_symbol,
    ) = _compute_regime_adx_gate(
        enabled_instruments,
        current_session=current_session,
    )
    _mtf_snapshot_by_symbol, _btc_lead_state, _btc_recommended_bias = _compute_mtf_regime_context(
        enabled_instruments,
    )

    for inst in enabled_instruments:

        # 3a. Data staleness check
        latest_ts = getattr(inst, "latest_1m_ts", None)
        is_stale, emit_stale_event, stale_details = _track_data_staleness_transition(
            inst,
            latest_ts,
            now_ts=now_cycle,
            risk_ns=risk_ns,
        )
        if is_stale:
            log_fn = logger.warning if emit_stale_event else logger.debug
            log_fn("Market data stale for %s, skipping", inst.symbol)
            if emit_stale_event:
                _create_risk_event(
                    "data_stale",
                    "medium",
                    instrument=inst,
                    details=stale_details,
                    risk_ns=risk_ns,
                )
            continue

        # 3b. Latest signal
        sig = latest_signals.get(getattr(inst, "latest_signal_id", None))
        if not sig:
            continue

        # 3c. Signal TTL - ignore expired signals for new entries
        signal_age = dj_tz.now() - sig.ts
        signal_expired = signal_age > signal_ttl
        if signal_expired:
            logger.debug("Signal %s expired (age=%s > ttl=%s)", sig.id, signal_age, signal_ttl)

        strategy_name = str(getattr(sig, "strategy", "") or "").strip().lower()
        sig_payload = sig.payload_json or {}
        signal_direction, side = _resolve_signal_direction(strategy_name)

        direction_mode = get_direction_mode(inst.symbol)
        direction_allowed = True
        if signal_direction in {"long", "short"}:
            direction_allowed = is_direction_allowed(
                signal_direction,
                symbol=inst.symbol,
                mode=direction_mode,
            )
        if signal_direction in {"long", "short"} and not direction_allowed:
            logger.info(
                "Direction policy blocked signal for execution on %s: %s mode=%s",
                inst.symbol,
                signal_direction,
                direction_mode,
            )

        # 3d. Resolve symbol + get ticker + contract size / market limits
        atr = _atr_pct(inst)
        # Blend ATR with GARCH forecast when enabled
        if getattr(settings, "GARCH_ENABLED", False):
            from signals.garch import blended_vol
            blended = blended_vol(inst.symbol, atr)
            if blended is not None:
                atr = blended
        symbol, last_price, ticker_used, spread_bps_selected, contract_size, market_info = (
            _resolve_market_snapshot(
                adapter=adapter,
                inst_symbol=inst.symbol,
                atr_pct=atr,
            )
        )

        if last_price is None:
            logger.warning("No price for %s, skipping", inst.symbol)
            continue

        # 3e. ATR-based TP/SL
        _, _, _, sl_pct = _compute_tp_sl_prices(side, last_price, atr)

        # 3f. Current position from exchange
        current_qty, entry_price, pos_opened_at = _current_position(
            adapter,
            symbol,
            positions=positions_snapshot,
        )
        skip_symbol, allow_scale_entry, scale_parent_correlation, scale_add_index = _manage_open_position(
            adapter=adapter,
            inst=inst,
            sig=sig,
            sig_payload=sig_payload,
            strategy_name=strategy_name,
            symbol=symbol,
            ticker_used=ticker_used,
            last_price=last_price,
            current_qty=current_qty,
            entry_price=entry_price,
            pos_opened_at=pos_opened_at,
            signal_direction=signal_direction,
            side=side,
            direction_allowed=direction_allowed,
            atr=atr,
            contract_size=contract_size,
            leverage=leverage,
            equity_usdt=equity_usdt,
            current_session=current_session,
            btc_recommended_bias=_btc_recommended_bias,
            account_ai_enabled=account_ai_enabled,
            account_ai_config_id=account_ai_config_id,
            account_owner_id=account_owner_id,
            account_alias=account_alias,
            account_service=account_service,
            # Flip pre-validation: pass lightweight gate results so flip
            # does NOT close the current position when the new entry would
            # be immediately rejected downstream.
            flip_can_open=can_open and not signal_expired,
            flip_macro_block=(
                macro_active
                and macro_block_entries
                and not _macro_high_impact_allows_entry(strategy_name=strategy_name, symbol=inst.symbol)
            ),
            flip_regime_blocked=(inst.symbol in _regime_blocked_symbols),
            flip_session_dead=(session_policy_enabled and session_dead_zone_block and is_dead_session(current_session)),
            flip_score_too_low=(
                _to_float(getattr(sig, "score", 0.0))
                < (session_min_score if session_policy_enabled else settings.EXECUTION_MIN_SIGNAL_SCORE)
            ),
            flip_btc_lead_blocked=_btc_lead_directional_risk_mult(
                inst.symbol, signal_direction, _btc_lead_state, _btc_recommended_bias,
            )[1],
        )
        if skip_symbol:
            continue
        # 3h. Open new position (extracted helper to keep loop orchestration-focused).
        corr_positions_snapshot = _corr_guard_positions_snapshot(
            positions_snapshot,
            cycle_pending_corr_entries,
        )
        placed_delta, cycle_notional_delta = _attempt_entry_open(
            adapter=adapter,
            inst=inst,
            sig=sig,
            sig_payload=sig_payload,
            strategy_name=strategy_name,
            side=side,
            signal_direction=signal_direction,
            direction_allowed=direction_allowed,
            signal_expired=signal_expired,
            can_open=can_open,
            macro_active=macro_active,
            macro_context=macro_context,
            macro_block_entries=macro_block_entries,
            macro_risk_mult=macro_risk_mult,
            regime_blocked_symbols=_regime_blocked_symbols,
            regime_adx_by_symbol=_regime_adx_by_symbol,
            regime_adx_min_by_symbol=_regime_adx_min_by_symbol,
            regime_bias_by_symbol=_regime_bias_by_symbol,
            regime_adx_min=_regime_adx_min,
            market_regime_adx=_market_regime_adx,
            mtf_symbol_snapshot=_mtf_snapshot_by_symbol.get(inst.symbol, {}),
            btc_lead_state=_btc_lead_state,
            btc_recommended_bias=_btc_recommended_bias,
            allow_scale_entry=allow_scale_entry,
            scale_parent_correlation=scale_parent_correlation,
            scale_add_index=scale_add_index,
            session_policy_enabled=session_policy_enabled,
            session_dead_zone_block=session_dead_zone_block,
            current_session=current_session,
            session_min_score=session_min_score,
            session_risk_mult=session_risk_mult,
            ml_entry_filter_enabled=ml_entry_filter_enabled,
            ml_entry_filter_default_min_prob=ml_entry_filter_default_min_prob,
            ml_entry_filter_fail_open=ml_entry_filter_fail_open,
            use_allocator_signals=use_allocator_signals,
            symbol=symbol,
            last_price=last_price,
            contract_size=contract_size,
            market_info=market_info,
            atr=atr,
            sl_pct=sl_pct,
            spread_bps_selected=spread_bps_selected,
            free_usdt=free_usdt,
            equity_usdt=equity_usdt,
            leverage=leverage,
            total_notional=total_notional,
            cycle_notional_added=cycle_notional_added,
            account_ai_enabled=account_ai_enabled,
            account_ai_config_id=account_ai_config_id,
            account_owner_id=account_owner_id,
            account_alias=account_alias,
            account_service=account_service,
            positions_snapshot=corr_positions_snapshot,
        )
        placed += placed_delta
        cycle_notional_added += cycle_notional_delta
        if placed_delta > 0 and signal_direction in {"long", "short"}:
            cycle_pending_corr_entries.append(
                {
                    "symbol": symbol,
                    "side": signal_direction,
                    "contracts": 1.0,
                }
            )

    _release_task_lock(lock_client, lock_key, lock_token)
    return f"orders_placed={placed}"



