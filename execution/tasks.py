import logging
import math
import re
import uuid
from pathlib import Path
from io import StringIO
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from datetime import datetime, timezone, timedelta
from statistics import median
from typing import Any

from celery import shared_task
from django.db import transaction
from django.db.models import OuterRef, Subquery
from django.conf import settings
from django.core.management import call_command
from django.utils import timezone as dj_tz

import redis

from adapters import get_default_adapter, get_default_adapter_signature
from core.models import Instrument
from core.exchange_runtime import (
    extract_balance_values,
    get_runtime_exchange_context,
)
from signals.sessions import (
    get_current_session,
    get_session_risk_mult,
    get_session_score_min,
    is_dead_session,
)
from signals.direction_policy import (
    get_direction_mode,
    is_direction_allowed,
)
from signals.feature_flags import FEATURE_KEYS, resolve_runtime_flags
from signals.models import Signal
from marketdata.models import Candle
from risk.models import RiskEvent
from risk.notifications import (
    notify_kill_switch, notify_trade_opened, notify_trade_closed,
    notify_risk_event, notify_error,
)
from execution.ml_entry_filter import load_model, predict_entry_success_probability
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
    Normaliza sÃƒÂ­mbolos provenientes de diferentes exchanges/ccxt
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
    """Devuelve (qty, entry_price) de la posiciÃƒÂ³n abierta para el sÃƒÂ­mbolo normalizado."""
    target = _norm_symbol(symbol)
    if positions is None:
        try:
            # KuCoin ignora a veces el filtro por sÃƒÂ­mbolo, preferimos traer todo y filtrar local.
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


def _check_drawdown(equity: float, risk_ns: str = "global") -> tuple[bool, float]:
    """
    Tracks starting equity of the UTC day in redis and computes drawdown.
    Returns (allowed, dd).
    """
    client = _redis_client()
    if client is None:
        return True, 0.0
    if not _is_valid_equity_value(equity):
        logger.warning("Skipping daily DD check: invalid equity=%.8f ns=%s", equity, risk_ns)
        return True, 0.0
    key = f"risk:equity_start:{risk_ns}:{dj_tz.now().date().isoformat()}"
    start = client.get(key)
    if start is None:
        client.set(key, equity)
        return True, 0.0
    try:
        start_val = float(start)
    except Exception:
        client.set(key, equity)
        return True, 0.0
    if not _is_valid_equity_value(start_val):
        client.set(key, equity)
        return True, 0.0
    dd = (equity - start_val) / start_val if start_val else 0.0
    if not math.isfinite(dd):
        client.set(key, equity)
        return True, 0.0
    if dd <= -settings.DAILY_DD_LIMIT:
        return False, dd
    return True, dd


def _check_weekly_drawdown(equity: float, risk_ns: str = "global") -> tuple[bool, float]:
    """
    Tracks starting equity of the ISO week in redis and computes weekly drawdown.
    Returns (allowed, dd).
    """
    client = _redis_client()
    if client is None:
        return True, 0.0
    if not _is_valid_equity_value(equity):
        logger.warning("Skipping weekly DD check: invalid equity=%.8f ns=%s", equity, risk_ns)
        return True, 0.0
    today = dj_tz.now().date()
    iso_year, iso_week, _ = today.isocalendar()
    key = f"risk:equity_week_start:{risk_ns}:{iso_year}-W{iso_week:02d}"
    start = client.get(key)
    if start is None:
        client.set(key, equity)
        return True, 0.0
    try:
        start_val = float(start)
    except Exception:
        client.set(key, equity)
        return True, 0.0
    if not _is_valid_equity_value(start_val):
        client.set(key, equity)
        return True, 0.0
    dd = (equity - start_val) / start_val if start_val else 0.0
    if not math.isfinite(dd):
        client.set(key, equity)
        return True, 0.0
    if dd <= -settings.WEEKLY_DD_LIMIT:
        return False, dd
    return True, dd


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
    """Create a RiskEvent record and send notification (throttled)."""
    try:
        RiskEvent.objects.create(
            instrument=instrument,
            kind=kind,
            severity=severity,
            details_json=details or {},
        )
        # Throttle Telegram: only notify once per kind+instrument every 30 min
        symbol = instrument.symbol if instrument else "global"
        throttle_key = f"risk:notified:{risk_ns}:{kind}:{symbol}"
        client = _redis_client()
        if client and client.set(throttle_key, "1", nx=True, ex=1800):
            notify_risk_event(kind, severity, str(details) if details else "")
        elif client is None:
            notify_risk_event(kind, severity, str(details) if details else "")
        # else: throttled Ã¢â‚¬â€œ skip Telegram
    except Exception as exc:
        logger.warning("Failed to create risk event: %s", exc)


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

    # Position is old â€” close if PnL is within the breakeven band
    return -pnl_band <= pnl_pct <= pnl_band


def _market_min_qty(market: dict | None, fallback: float = 0.0) -> float:
    if not isinstance(market, dict):
        return max(0.0, _to_float(fallback))
    try:
        limits = market.get("limits") or {}
        amount_limits = limits.get("amount") or {}
        min_qty = _to_float(amount_limits.get("min"))
        if min_qty > 0:
            return min_qty
    except Exception:
        pass
    try:
        precision = market.get("precision") or {}
        amount_precision = _to_float(precision.get("amount"))
        if math.isfinite(amount_precision) and amount_precision >= 0:
            # Common CCXT case: precision.amount is decimal places (0 -> minimum 1).
            if float(amount_precision).is_integer():
                step = 10 ** (-int(amount_precision))
                if step > 0:
                    return step
            # Some exchanges expose amount step directly.
            if amount_precision > 0:
                return amount_precision
    except Exception:
        pass
    return max(0.0, _to_float(fallback))


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


def _precision_step_from_error(exc: Exception) -> float:
    msg = str(exc or "")
    match = re.search(r"minimum amount precision of\s*([0-9]*\.?[0-9]+)", msg, re.IGNORECASE)
    if not match:
        return 0.0
    return max(0.0, _to_float(match.group(1)))


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
            step = _market_min_qty(market, fallback=0.0)
        except Exception:
            step = 0.0
        if step <= 0:
            step = _precision_step_from_error(exc)
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


def _compute_tp_sl_prices(side: str, entry_price: float, atr_pct: float | None):
    """
    Compute TP and SL prices based on ATR or defaults.
    Returns (tp_price, sl_price, tp_pct, sl_pct).
    Applies a minimum SL floor (MIN_SL_PCT) to avoid micro stop-outs from noise.
    """
    tp_pct = settings.TAKE_PROFIT_PCT
    sl_pct = settings.STOP_LOSS_PCT
    min_sl = getattr(settings, 'MIN_SL_PCT', 0.008)
    if atr_pct:
        tp_pct = max(tp_pct, atr_pct * settings.ATR_MULT_TP)
        sl_pct = max(sl_pct, atr_pct * settings.ATR_MULT_SL)

    # Global TP tuning: allow slightly closer TP to secure gains earlier.
    tp_mult = float(getattr(settings, "TAKE_PROFIT_DYNAMIC_MULT", 1.0) or 1.0)
    tp_mult = max(0.1, min(tp_mult, 2.0))
    tp_floor = float(getattr(settings, "TAKE_PROFIT_MIN_PCT", 0.0) or 0.0)
    tp_floor = max(0.0, tp_floor)
    tp_pct = max(tp_floor, tp_pct * tp_mult)

    # Fast-exit mode for high-volatility phases: reduce TP distance to secure gains sooner.
    if (
        bool(getattr(settings, "VOL_FAST_EXIT_ENABLED", False))
        and atr_pct is not None
        and atr_pct >= float(getattr(settings, "VOL_FAST_EXIT_ATR_PCT", 0.012) or 0.012)
    ):
        tp_mult = float(getattr(settings, "VOL_FAST_EXIT_TP_MULT", 0.75) or 0.75)
        min_tp = float(getattr(settings, "VOL_FAST_EXIT_MIN_TP_PCT", 0.006) or 0.006)
        tp_pct = max(min_tp, tp_pct * max(0.1, min(tp_mult, 1.0)))

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
      - long  position Ã¢â€ â€™ close with sell when price drops  Ã¢â€ â€™ stop="down"
      - short position Ã¢â€ â€™ close with buy  when price rises  Ã¢â€ â€™ stop="up"
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
        return True, None, []  # assume exists Ã¢â€ â€™ avoid placing duplicates on API error

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


def _reconcile_sl(
    adapter, symbol: str, side: str, current_qty: float,
    entry_price: float, atr_pct: float | None,
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
            _place_sl_order(adapter, symbol, side, abs(current_qty), sl_price)
        return  # SL exists and is aligned enough or was replaced

    if exists:
        return  # SL exists but couldn't read price Ã¢â‚¬â€ leave it alone

    # SL missing entirely
    logger.warning(
        "SL stop-order missing for %s (side=%s entry=%.4f); placing at %.4f",
        symbol, side, entry_price, sl_price,
    )
    _place_sl_order(adapter, symbol, side, abs(current_qty), sl_price)


def _position_state_key(symbol: str, opened_at, entry_price: float) -> str:
    """
    Build a per-position key for Redis state (HWM, partial-close, etc).
    Prefer exchange-provided opened_at; fallback to entry_price.
    """
    if opened_at:
        try:
            return f"{symbol}:{int(opened_at.timestamp())}"
        except Exception:
            pass
    return f"{symbol}:{round(float(entry_price or 0.0), 6)}"


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


def _check_trailing_stop(
    adapter, symbol: str, side: str, current_qty: float,
    entry_price: float, last_price: float, sl_pct: float,
    opened_at=None,
    contract_size: float = 1.0,
    atr_pct: float | None = None,
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
    trail_state_ttl = max(
        60,
        int(getattr(settings, "TRAILING_STATE_TTL_SECONDS", 172800) or 172800),
    )
    sl_min_move_pct = max(
        0.0,
        float(getattr(settings, "TRAILING_SL_MIN_MOVE_PCT", 0.0002) or 0.0002),
    )

    partial_r_trigger = float(getattr(settings, "PARTIAL_CLOSE_AT_R", 1.0) or 1.0)
    trail_activation_r = float(getattr(settings, "TRAILING_STOP_ACTIVATION_R", 2.5) or 2.5)
    if (
        bool(getattr(settings, "VOL_FAST_EXIT_ENABLED", False))
        and atr_pct is not None
        and atr_pct >= float(getattr(settings, "VOL_FAST_EXIT_ATR_PCT", 0.012) or 0.012)
    ):
        partial_mult = float(getattr(settings, "VOL_FAST_EXIT_PARTIAL_R_MULT", 0.80) or 0.80)
        trail_mult = float(getattr(settings, "VOL_FAST_EXIT_TRAIL_R_MULT", 0.75) or 0.75)
        partial_r_trigger *= max(0.1, min(partial_mult, 1.0))
        trail_activation_r *= max(0.1, min(trail_mult, 1.0))

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
                    min_qty = _market_min_qty(market, fallback=0.0)
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

    # Track max favorable move (HWM) in Redis so trailing uses peak, not current pnl.
    max_fav = max(pnl_pct, 0.0)
    if client:
        hwm_key = f"trail:max_fav:{state_key}"
        try:
            prev = client.get(hwm_key)
            if prev is not None:
                try:
                    max_fav = max(max_fav, float(prev))
                except Exception:
                    pass
            client.set(hwm_key, str(max_fav), ex=trail_state_ttl)
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
                            _place_sl_order(adapter, symbol, side, abs(current_qty), be_price)
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
                                _place_sl_order(adapter, symbol, side, abs(current_qty), be_price)
                                logger.info(
                                    "BE SL updated %s side=%s stop=%.4f prev=%.4f (R=%.2f age=%s window=%s)",
                                    symbol, side, be_price, cur, max_r,
                                    f"{be_age_min:.1f}min" if be_age_min is not None else "n/a",
                                    be_window_min if be_window_min > 0 else "off",
                                )
                    except Exception as exc:
                        logger.debug("BE SL update failed for %s: %s", symbol, exc)

    # Trailing activation gate (based on HWM, not current price)
    if max_fav <= 0:
        return False, 0.0
    if max_r < trail_activation_r:
        return False, 0.0

    lock_in = float(getattr(settings, "TRAILING_STOP_LOCK_IN_PCT", 0.5) or 0.5)
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
            close_fee = _extract_fee_usdt(close_resp)
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
            _place_sl_order(adapter, symbol, side, abs(current_qty), trail_sl)
            logger.info("Trailing SL placed %s side=%s stop=%.4f (hwm=%.4f%%)", symbol, side, trail_sl, max_fav * 100)
        elif cur > 0 and stop_orders:
            should_update = False
            if is_long:
                should_update = trail_sl > cur * (1 + min_move_pct) and trail_sl < last_price
            else:
                should_update = trail_sl < cur * (1 - min_move_pct) and trail_sl > last_price
            if should_update:
                _cancel_stop_orders(adapter, symbol, stop_orders)
                _place_sl_order(adapter, symbol, side, abs(current_qty), trail_sl)
                logger.info(
                    "Trailing SL updated %s side=%s stop=%.4f prev=%.4f (hwm=%.4f%%)",
                    symbol, side, trail_sl, cur, max_fav * 100,
                )
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
    fee_val = max(0.0, _to_float(fee_usdt))
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
    OperationReport.objects.create(
        instrument=inst,
        side=side,
        qty=qty,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl_abs=pnl_abs,
        pnl_pct=pnl_pct,
        notional_usdt=notional,
        margin_used_usdt=margin_used,
        fee_usdt=fee_val,
        leverage=leverage or 0,
        equity_before=equity_before,
        equity_after=equity_after,
        mode=settings.MODE,
        opened_at=opened_at,
        outcome=outcome,
        reason=reason,
        close_sub_reason=close_sub_reason,
        signal_id=str(signal_id or ""),
        correlation_id=correlation_id or "",
        closed_at=dj_tz.now(),
    )
    _queue_ml_retrain_after_operation(inst.symbol, settings.MODE, reason)


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

    # Mark others as closed Ã¢â‚¬â€ detect transitions and notify
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
            sub_reason = _classify_exchange_close(
                adapter,
                sym,
                trade_side,
                entry,
                liq_est,
                exit_price=last,
                sl_pct_hint=sl_pct_hint,
                tp_pct_hint=tp_pct_hint,
            )
            logger.info(
                "exchange_close sub-reason for %s: %s (entry=%.4f exit=%.4f liq_est=%.4f)",
                sym, sub_reason, entry, last, liq_est,
            )

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

        allowed_dd, dd = _check_drawdown(equity_usdt, risk_ns=risk_ns)
        if not allowed_dd:
            logger.warning("Daily DD limit hit dd=%.4f; blocking new trades", dd)
            _client = _redis_client()
            _ks_key = f"risk:ks_notified:daily_dd:{risk_ns}:{dj_tz.now().date().isoformat()}"
            if _client is None or _client.set(_ks_key, "1", nx=True, ex=86400):
                _create_risk_event(
                    "daily_dd_limit",
                    "high",
                    details={"dd": dd},
                    risk_ns=risk_ns,
                )
                notify_kill_switch(f"Daily DD limit: {dd:.4f}")
            can_open = False

        allowed_wdd, wdd = _check_weekly_drawdown(equity_usdt, risk_ns=risk_ns)
        if not allowed_wdd:
            logger.warning("Weekly DD limit hit wdd=%.4f; blocking new trades", wdd)
            _client = _redis_client()
            _ks_key = f"risk:ks_notified:weekly_dd:{risk_ns}:{dj_tz.now().date().isoformat()}"
            if _client is None or _client.set(_ks_key, "1", nx=True, ex=86400):
                _create_risk_event(
                    "weekly_dd_limit",
                    "high",
                    details={"wdd": wdd},
                    risk_ns=risk_ns,
                )
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
        latest_signal_qs = latest_signal_qs.filter(strategy__startswith="alloc_")
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


def _compute_regime_adx_gate(
    enabled_instruments: list[Instrument],
) -> tuple[dict[str, float], set[str], float, float | None]:
    _regime_adx_by_symbol: dict[str, float] = {}
    _regime_adx_min = float(getattr(settings, "MARKET_REGIME_ADX_MIN", 0))
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
    if _regime_adx_min > 0:
        for _sym, _adx_val in _regime_adx_by_symbol.items():
            if _adx_val < _regime_adx_min:
                _regime_blocked_symbols.add(_sym)
        if _regime_blocked_symbols:
            logger.warning(
                "Market regime gate ACTIVE (per-instrument): ADX < %.1f â€” blocked: %s",
                _regime_adx_min,
                ", ".join(f"{s}({_regime_adx_by_symbol[s]:.1f})" for s in sorted(_regime_blocked_symbols)),
            )
    return _regime_adx_by_symbol, _regime_blocked_symbols, _regime_adx_min, _market_regime_adx


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
    regime_adx_min: float,
    market_regime_adx: float | None,
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
    if macro_active and macro_block_entries:
        logger.info(
            "Macro high-impact window blocked entry on %s (session=%s hour=%s weekday=%s)",
            inst.symbol,
            macro_context.get("session"),
            macro_context.get("hour_utc"),
            macro_context.get("weekday"),
        )
        return 0, 0.0

    if inst.symbol in regime_blocked_symbols and not allow_scale_entry:
        logger.info(
            "Market regime gate blocked entry on %s: 1h ADX=%.1f < %.1f",
            inst.symbol,
            regime_adx_by_symbol.get(inst.symbol, 0),
            regime_adx_min,
        )
        return 0, 0.0

    if not allow_scale_entry:
        htf_adx_for_limit = regime_adx_by_symbol.get(inst.symbol, market_regime_adx)
        daily_limit = _max_daily_trades_for_adx(htf_adx_for_limit)
        daily_count = _get_daily_trade_count()
        if daily_count >= daily_limit:
            logger.info(
                "Daily trade limit reached: %d/%d (adx=%.1f) â€” blocking entry on %s",
                daily_count,
                daily_limit,
                htf_adx_for_limit or 0,
                inst.symbol,
            )
            return 0, 0.0

    if session_policy_enabled and session_dead_zone_block and is_dead_session(current_session):
        logger.info(
            "Session dead zone active, skipping new entry for %s (session=%s)",
            inst.symbol,
            current_session,
        )
        return 0, 0.0

    exec_min_score = session_min_score if session_policy_enabled else settings.EXECUTION_MIN_SIGNAL_SCORE
    sig_score = _to_float(getattr(sig, "score", 0.0))
    if sig_score < exec_min_score:
        logger.info(
            "Signal score too low for execution on %s: %.3f < %.3f (session=%s)",
            inst.symbol,
            sig_score,
            exec_min_score,
            current_session if session_policy_enabled else "n/a",
        )
        return 0, 0.0

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

    inst_notional = 0.0
    try:
        pos_obj = Position.objects.filter(instrument=inst, is_open=True).first()
        if pos_obj:
            inst_notional = float(pos_obj.notional_usdt or 0)
    except Exception:
        pass
    max_inst_notional = equity_usdt * settings.MAX_EXPOSURE_PER_INSTRUMENT_PCT * leverage
    if inst_notional >= max_inst_notional and max_inst_notional > 0:
        logger.info("Per-instrument exposure cap reached for %s (%.2f >= %.2f)", symbol, inst_notional, max_inst_notional)
        return 0, 0.0

    is_allocator_signal = use_allocator_signals and strategy_name.startswith("alloc_")
    if is_allocator_signal:
        inst_risk_pct = max(0.0, _to_float(sig_payload.get("risk_budget_pct", 0.0)))
        effective_risk_mult = 1.0
        if inst_risk_pct <= 0:
            logger.info(
                "Allocator blocked entry on %s: risk_budget_pct=%.5f strategy=%s",
                inst.symbol,
                inst_risk_pct,
                strategy_name,
            )
            return 0, 0.0
        if macro_active and not macro_block_entries:
            inst_risk_pct *= macro_risk_mult
    else:
        effective_risk_mult = session_risk_mult if session_policy_enabled else 1.0
        if macro_active and not macro_block_entries:
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
    if allow_scale_entry:
        pyramid_risk_scale = max(
            0.0,
            min(float(getattr(settings, "PYRAMID_RISK_SCALE", 0.6) or 0.6), 1.0),
        )
        effective_risk_pct *= pyramid_risk_scale
    if effective_risk_pct <= 0:
        return 0, 0.0

    stop_dist = float(sl_pct or 0.0) if atr is not None else 0.0
    min_atr_for_entry = float(getattr(settings, "MIN_ATR_FOR_ENTRY", 0.003) or 0.003)
    if atr is not None and atr < min_atr_for_entry:
        logger.info(
            "ATR too low for %s (%.4f%% < %.4f%%), market compressed â€” skipping entry",
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
            leverage,
            risk_pct=effective_risk_pct,
        )
    else:
        qty = settings.ORDER_SIZE_USDT / (last_price * contract_size)
        base_risk_pct = max(float(settings.RISK_PER_TRADE_PCT), 1e-9)
        qty *= (effective_risk_pct / base_risk_pct)

    min_qty = _market_min_qty(market_info, fallback=float(inst.lot_size or 0.0))
    qty = _normalize_order_qty(adapter, symbol, qty)
    if min_qty > 0 and qty < min_qty:
        qty = _normalize_order_qty(adapter, symbol, min_qty)

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
        (usable_margin * leverage if leverage else usable_margin) / (last_price * contract_size),
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

    required_margin = notional_order / leverage if leverage else notional_order
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

    tp_price, sl_price, _, _ = _compute_tp_sl_prices(side, last_price, atr)
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
            leverage=leverage,
            margin_mode=getattr(adapter, "margin_mode", ""),
            notional_usdt=notional_order,
            opened_at=dj_tz.now(),
            parent_correlation_id=parent_correlation_id,
        )

    try:
        resp = adapter.create_order(symbol, side, "market", float(order.qty))
        order.status = Order.OrderStatus.FILLED
        order.exchange_order_id = resp.get("id") or resp.get("orderId", "")
        order.raw_response = resp
        order.fee_usdt = _extract_fee_usdt(resp)
        order.closed_at = dj_tz.now()
        order.status_reason = ""
        order.save(update_fields=[
            "status", "exchange_order_id", "raw_response",
            "fee_usdt", "closed_at", "status_reason",
        ])
        _track_consecutive_errors(symbol, success=True)

        fill_price = _to_float(resp.get("average") or resp.get("price") or last_price)
        if fill_price and fill_price > 0:
            tp_price, sl_price, _, _ = _compute_tp_sl_prices(side, fill_price, atr)
            logger.info(
                "SL/TP recalculated with fill_price=%.4f (last=%.4f slippage=%.4f%%)",
                fill_price,
                last_price,
                abs(fill_price - last_price) / last_price * 100 if last_price else 0,
            )

        _place_sl_order(adapter, symbol, side, float(qty), sl_price)
        if not allow_scale_entry:
            _increment_daily_trade_count()

        notify_trade_opened(
            inst.symbol, side, float(qty), last_price,
            sl=sl_price, tp=tp_price, leverage=float(leverage or 0),
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
        if is_margin_error:
            logger.info("Order rejected by exchange (margin) %s: %s", inst.symbol, exc)
        else:
            logger.warning("Order send failed %s: %s", inst.symbol, exc)
        order.status = Order.OrderStatus.REJECTED
        order.status_reason = str(exc)
        order.closed_at = dj_tz.now()
        order.save(update_fields=["status", "status_reason", "closed_at"])
        if is_margin_error:
            _track_consecutive_errors(symbol, success=True)
            return 0, 0.0
        err_count = _track_consecutive_errors(symbol, success=False)
        notify_error(f"Order failed {inst.symbol}: {exc}")
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
    )

    if entry_price and abs(current_qty) > 0:
        _reconcile_sl(adapter, symbol, pos_side, current_qty, entry_price, atr)

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
        )
        if was_trailed:
            pnl_trail = (last - entry_price) / entry_price * (1 if current_qty > 0 else -1)
            pnl_abs_trail = (last - entry_price) * abs(current_qty) * contract_size * (1 if current_qty > 0 else -1)
            trade_side = "buy" if current_qty > 0 else "sell"
            dur_min = (dj_tz.now() - pos_opened_at).total_seconds() / 60 if pos_opened_at else 0
            _log_operation(
                inst,
                trade_side,
                abs(current_qty),
                entry_price,
                last,
                reason="trailing_stop",
                signal_id=str(sig.id) if sig else "",
                correlation_id=f"{sig.id}-{inst.symbol}" if sig else "",
                leverage=leverage,
                equity_before=equity_usdt,
                equity_after=None,
                fee_usdt=trail_fee,
                opened_at=pos_opened_at,
                contract_size=contract_size,
            )
            notify_trade_closed(
                inst.symbol,
                "trailing_stop",
                pnl_trail,
                pnl_abs=pnl_abs_trail,
                entry_price=entry_price,
                exit_price=last,
                qty=abs(current_qty),
                equity_before=equity_usdt,
                duration_min=dur_min,
                side=trade_side,
                leverage=leverage,
                strategy_name=strategy_name,
                active_modules=_signal_active_modules(sig_payload, strategy_name),
            )
            _mark_position_closed(inst)
            return True, allow_scale_entry, scale_parent_correlation, scale_add_index

    # Close old positions stuck near breakeven to free margin.
    if last and entry_price:
        pnl_pct_stale = (last - entry_price) / entry_price * (1 if current_qty > 0 else -1)
        if _should_close_stale_position(pos_opened_at, pnl_pct_stale):
            close_side = "sell" if current_qty > 0 else "buy"
            close_qty = abs(current_qty)
            age_h = (dj_tz.now() - pos_opened_at).total_seconds() / 3600 if pos_opened_at else 0
            try:
                close_resp = adapter.create_order(symbol, close_side, "market", close_qty, params={"reduceOnly": True})
                close_fee = _extract_fee_usdt(close_resp)
                pnl_abs_stale = (last - entry_price) * close_qty * contract_size * (1 if current_qty > 0 else -1)
                trade_side = "buy" if current_qty > 0 else "sell"
                dur_min = age_h * 60
                logger.info(
                    "Stale position cleanup %s: pnl=%.4f age=%.1fh — freeing margin",
                    symbol,
                    pnl_pct_stale,
                    age_h,
                )
                _log_operation(
                    inst,
                    trade_side,
                    close_qty,
                    entry_price,
                    last,
                    reason="stale_cleanup",
                    signal_id=str(sig.id) if sig else "",
                    correlation_id=f"{sig.id}-{inst.symbol}" if sig else "",
                    leverage=leverage,
                    equity_before=equity_usdt,
                    equity_after=None,
                    fee_usdt=close_fee,
                    opened_at=pos_opened_at,
                    contract_size=contract_size,
                )
                notify_trade_closed(
                    inst.symbol,
                    "stale_cleanup",
                    pnl_pct_stale,
                    pnl_abs=pnl_abs_stale,
                    entry_price=entry_price,
                    exit_price=last,
                    qty=close_qty,
                    equity_before=equity_usdt,
                    duration_min=dur_min,
                    side=trade_side,
                    leverage=leverage,
                    strategy_name=strategy_name,
                    active_modules=_signal_active_modules(sig_payload, strategy_name),
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
                ).order_by("-open_time")[:100]
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
                    close_fee = _extract_fee_usdt(close_resp)
                    pnl_abs_kill = (last - entry_price) * close_qty * contract_size * -1
                    dur_min = (dj_tz.now() - pos_opened_at).total_seconds() / 60 if pos_opened_at else 0
                    logger.info(
                        "Uptrend short killer closed %s: HTF=bull pnl=%.4f",
                        symbol,
                        pnl_pct_kill,
                    )
                    _log_operation(
                        inst,
                        "sell",
                        close_qty,
                        entry_price,
                        last,
                        reason="uptrend_short_kill",
                        signal_id=str(sig.id) if sig else "",
                        correlation_id=f"{sig.id}-{inst.symbol}" if sig else "",
                        leverage=leverage,
                        equity_before=equity_usdt,
                        equity_after=None,
                        fee_usdt=close_fee,
                        opened_at=pos_opened_at,
                        contract_size=contract_size,
                    )
                    notify_trade_closed(
                        inst.symbol,
                        "uptrend_short_kill",
                        pnl_pct_kill,
                        pnl_abs=pnl_abs_kill,
                        entry_price=entry_price,
                        exit_price=last,
                        qty=close_qty,
                        equity_before=equity_usdt,
                        duration_min=dur_min,
                        side="sell",
                        leverage=leverage,
                        strategy_name=strategy_name,
                        active_modules=_signal_active_modules(sig_payload, strategy_name),
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
                ).order_by("-open_time")[:100]
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
                    close_fee = _extract_fee_usdt(close_resp)
                    pnl_abs_kill = (last - entry_price) * close_qty * contract_size
                    dur_min = (dj_tz.now() - pos_opened_at).total_seconds() / 60 if pos_opened_at else 0
                    logger.info(
                        "Downtrend long killer closed %s: HTF=bear pnl=%.4f",
                        symbol,
                        pnl_pct_kill,
                    )
                    _log_operation(
                        inst,
                        "buy",
                        close_qty,
                        entry_price,
                        last,
                        reason="downtrend_long_kill",
                        signal_id=str(sig.id) if sig else "",
                        correlation_id=f"{sig.id}-{inst.symbol}" if sig else "",
                        leverage=leverage,
                        equity_before=equity_usdt,
                        equity_after=None,
                        fee_usdt=close_fee,
                        opened_at=pos_opened_at,
                        contract_size=contract_size,
                    )
                    notify_trade_closed(
                        inst.symbol,
                        "downtrend_long_kill",
                        pnl_pct_kill,
                        pnl_abs=pnl_abs_kill,
                        entry_price=entry_price,
                        exit_price=last,
                        qty=close_qty,
                        equity_before=equity_usdt,
                        duration_min=dur_min,
                        side="buy",
                        leverage=leverage,
                        strategy_name=strategy_name,
                        active_modules=_signal_active_modules(sig_payload, strategy_name),
                    )
                    _mark_position_closed(inst)
                    return True, allow_scale_entry, scale_parent_correlation, scale_add_index
        except Exception as exc:
            logger.debug("Downtrend long killer check failed for %s: %s", symbol, exc)

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
                close_fee = _extract_fee_usdt(close_resp)
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
                _log_operation(
                    inst,
                    trade_side,
                    close_qty,
                    entry_price,
                    last,
                    reason=reason,
                    signal_id=str(sig.id) if sig else "",
                    correlation_id=f"{sig.id}-{inst.symbol}" if sig else "",
                    leverage=leverage,
                    equity_before=equity_usdt,
                    equity_after=None,
                    fee_usdt=close_fee,
                    opened_at=pos_opened_at,
                    contract_size=contract_size,
                )
                notify_trade_closed(
                    inst.symbol,
                    reason,
                    pnl_pct_live_gross,
                    pnl_abs=pnl_abs_tpsl,
                    entry_price=entry_price,
                    exit_price=last,
                    qty=close_qty,
                    equity_before=equity_usdt,
                    duration_min=dur_min,
                    side=trade_side,
                    leverage=leverage,
                    strategy_name=strategy_name,
                    active_modules=_signal_active_modules(sig_payload, strategy_name),
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

        close_side = "sell" if current_qty > 0 else "buy"
        close_qty = abs(current_qty)
        try:
            close_resp = adapter.create_order(symbol, close_side, "market", close_qty, params={"reduceOnly": True})
            close_fee = _extract_fee_usdt(close_resp)
            logger.info("Signal flip close %s qty=%s", symbol, close_qty)
            if last and entry_price:
                pnl_flip = (last - entry_price) / entry_price * (1 if current_qty > 0 else -1)
                pnl_abs_flip = (last - entry_price) * close_qty * contract_size * (1 if current_qty > 0 else -1)
                trade_side = "buy" if current_qty > 0 else "sell"
                dur_min = (dj_tz.now() - pos_opened_at).total_seconds() / 60 if pos_opened_at else 0
                _log_operation(
                    inst,
                    trade_side,
                    close_qty,
                    entry_price,
                    last,
                    reason="signal_flip",
                    signal_id=str(sig.id) if sig else "",
                    correlation_id=f"{sig.id}-{inst.symbol}" if sig else "",
                    leverage=leverage,
                    equity_before=equity_usdt,
                    equity_after=None,
                    fee_usdt=close_fee,
                    opened_at=pos_opened_at,
                    contract_size=contract_size,
                )
                notify_trade_closed(
                    inst.symbol,
                    "signal_flip",
                    pnl_flip,
                    pnl_abs=pnl_abs_flip,
                    entry_price=entry_price,
                    exit_price=last,
                    qty=close_qty,
                    equity_before=equity_usdt,
                    duration_min=dur_min,
                    side=trade_side,
                    leverage=leverage,
                    strategy_name=strategy_name,
                    active_modules=_signal_active_modules(sig_payload, strategy_name),
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
    placed = 0
    positions_snapshot = []
    try:
        positions_snapshot = adapter.fetch_positions()
    except Exception:
        positions_snapshot = []

    # Ã¢â€â‚¬Ã¢â€â‚¬ 1. Sync positions for admin visibility Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
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
    current_session = get_current_session()
    session_min_score = float(getattr(settings, "EXECUTION_MIN_SIGNAL_SCORE", 0.80))
    session_risk_mult = 1.0
    if session_policy_enabled:
        session_min_score = get_session_score_min(
            current_session,
            getattr(settings, "SESSION_SCORE_MIN", {}),
        )
        session_risk_mult = get_session_risk_mult(
            current_session,
            getattr(settings, "SESSION_RISK_MULTIPLIER", {}),
        )
    macro_active, macro_context = _is_macro_high_impact_window(
        now_utc=dj_tz.now(),
        session_name=current_session,
    )
    macro_block_entries = bool(getattr(settings, "MACRO_HIGH_IMPACT_BLOCK_ENTRIES", False))
    macro_risk_mult = float(getattr(settings, "MACRO_HIGH_IMPACT_RISK_MULTIPLIER", 1.0) or 1.0)
    macro_risk_mult = max(0.0, min(macro_risk_mult, 1.0))

    # Running notional accumulator Ã¢â‚¬â€ tracks new positions opened THIS cycle
    # so we don't exceed leverage by opening multiple trades before the
    # balance refresh catches up.
    cycle_notional_added = 0.0

    enabled_instruments, latest_signals, now_cycle = _load_enabled_instruments_and_latest_signals(
        use_allocator_signals,
    )
    _regime_adx_by_symbol, _regime_blocked_symbols, _regime_adx_min, _market_regime_adx = (
        _compute_regime_adx_gate(enabled_instruments)
    )

    for inst in enabled_instruments:

        # 3a. Data staleness check
        latest_ts = getattr(inst, "latest_1m_ts", None)
        if latest_ts is None or (now_cycle - latest_ts).total_seconds() > settings.DATA_STALE_SECONDS:
            logger.warning("Market data stale for %s, skipping", inst.symbol)
            _create_risk_event("data_stale", "medium", instrument=inst)
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
        )
        if skip_symbol:
            continue
        # 3h. Open new position (extracted helper to keep loop orchestration-focused).
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
            regime_adx_min=_regime_adx_min,
            market_regime_adx=_market_regime_adx,
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
        )
        placed += placed_delta
        cycle_notional_added += cycle_notional_delta

    _release_task_lock(lock_client, lock_key, lock_token)
    return f"orders_placed={placed}"

